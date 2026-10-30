defmodule Interpretability.Steering do
  @moduledoc """
  Steering: pushing the neurons along one direction and watching what the
  model says next.

  If a direction in the 128 neuron space really means "the subject is
  plural", then adding it at `the llama who chases the dogs` should make
  the model prefer `flee` to `flees`. That is the test: not whether a
  unit's firings line up with a label, but whether moving it changes the
  model's behaviour the way the label says it should.

  The llama model has one layer, so the neurons at a word only shape the
  prediction made at that word. Pushing every position at once is the
  same as pushing each prediction separately, which makes two clean
  measurements possible:

  - **Flip rate**: of the agreement probes whose subject is singular, the
    share where the model now picks a plural verb.
  - **Damage**: how much the loss rises on every word whose next word is
    not a verb. A direction that means "plural subject" should leave
    those predictions alone.

  Every direction has length one, and `strength` is how far along it to
  push, so different directions are compared at the same size of push.
  """

  alias Interpretability.Compiler
  alias Interpretability.Dataset
  alias Interpretability.Model
  alias Interpretability.Probes
  alias Interpretability.Scoring
  alias Interpretability.Vocab

  @doc """
  `tensor` scaled to length one.
  """
  @spec unit(Nx.Tensor.t()) :: Nx.Tensor.t()
  def unit(tensor), do: Nx.divide(tensor, Nx.sqrt(Nx.sum(Nx.pow(tensor, 2))))

  @doc """
  The average neuron vector where `toward` is true, minus the average
  where `away` is true: the simplest possible "plural minus singular"
  direction, found without any interpretability at all.
  """
  @spec mean_difference(map(), Scoring.label(), Scoring.label()) :: Nx.Tensor.t()
  def mean_difference(activations, toward, away) do
    unit(Nx.subtract(mean_where(activations, toward), mean_where(activations, away)))
  end

  @doc """
  One neuron's direction: turn that neuron up and leave the rest alone.
  """
  @spec neuron(non_neg_integer(), pos_integer()) :: Nx.Tensor.t()
  def neuron(index, width),
    do: Nx.indexed_put(Nx.broadcast(0.0, {width}), Nx.tensor([[index]]), Nx.tensor([1.0]))

  @doc """
  The sum of some SAE features' decoder rows, the pattern each feature
  writes into the neurons.
  """
  @spec features(%{decoder: Nx.Tensor.t()}, [non_neg_integer()]) :: Nx.Tensor.t()
  def features(sae, indices) do
    sae.decoder
    |> Nx.take(Nx.tensor(indices))
    |> Nx.sum(axes: [0])
    |> unit()
  end

  @doc """
  Of the probes whose subject is singular, the share where the model
  picks a plural verb after `direction` is pushed by `strength`.
  """
  @spec flip_rate(Model.params(), [Probes.probe()], Nx.Tensor.t(), number()) :: float()
  def flip_rate(params, probes, direction, strength) do
    prefixes = probes |> Enum.filter(&(&1.number == :singular)) |> Enum.map(& &1.prefix)

    prefixes
    |> margins(params, direction, strength)
    |> Enum.count(&(&1 > 0))
    |> Kernel./(length(prefixes))
  end

  @doc """
  The best plural verb's score minus the best singular verb's after
  `prefix`: above zero means the model now says plural.
  """
  @spec margin(Model.params(), [Vocab.word()], Nx.Tensor.t(), number()) :: float()
  def margin(params, prefix, direction, strength) do
    [margin] = margins([prefix], params, direction, strength)
    margin
  end

  @doc """
  How much the loss rises on words whose next word is not a verb, when
  `direction` is pushed by `strength` everywhere.
  """
  @spec damage(Model.params(), Dataset.t(), Nx.Tensor.t(), number()) :: float()
  def damage(params, dataset, direction, strength) do
    batch = %{tokens: dataset.tokens, targets: dataset.targets, mask: non_verb_mask(dataset)}

    steered_loss =
      Nx.Defn.jit(
        fn params, batch, push -> Model.loss(params, batch, edit_hidden: &Nx.add(&1, push)) end,
        Compiler.options()
      )

    steered = Nx.to_number(steered_loss.(params, batch, Nx.multiply(direction, strength)))
    plain = Nx.to_number(steered_loss.(params, batch, Nx.multiply(direction, 0.0)))
    steered - plain
  end

  ## PRIVATE FUNCTIONS

  defp mean_where(activations, {key, value}) do
    rows =
      activations.labels
      |> Enum.with_index()
      |> Enum.filter(fn {labels, _row} -> labels[key] == value end)
      |> Enum.map(fn {_labels, row} -> row end)

    activations.hidden
    |> Nx.take(Nx.tensor(rows))
    |> Nx.mean(axes: [0])
  end

  # The direction goes in as an argument, not inside a closure, so its
  # tensor can live on any backend.
  defp margins(prefixes, params, direction, strength) do
    context_length = Nx.axis_size(params.positions, 0)
    padding = Vocab.word_to_id(Vocab.end_token())

    {rows, last_positions} =
      prefixes
      |> Enum.map(fn prefix ->
        ids = Vocab.encode([Vocab.start_token() | prefix])
        {ids ++ List.duplicate(padding, context_length - length(ids)), length(ids) - 1}
      end)
      |> Enum.unzip()

    {plural, singular} = verb_ids()

    Nx.Defn.jit(
      fn params, tokens, last_positions, push, plural, singular ->
        logits = Model.forward(params, tokens, edit_hidden: &Nx.add(&1, push))
        {count, _context, vocabulary} = Nx.shape(logits)

        last =
          logits
          |> Nx.take_along_axis(
            Nx.broadcast(Nx.reshape(last_positions, {count, 1, 1}), {count, 1, vocabulary}),
            axis: 1
          )
          |> Nx.squeeze(axes: [1])

        Nx.subtract(
          Nx.reduce_max(Nx.take(last, plural, axis: 1), axes: [1]),
          Nx.reduce_max(Nx.take(last, singular, axis: 1), axes: [1])
        )
      end,
      Compiler.options()
    ).(
      params,
      Nx.tensor(rows, type: :s64),
      Nx.tensor(last_positions, type: :s64),
      Nx.multiply(direction, strength),
      plural,
      singular
    )
    |> Nx.to_list()
  end

  defp verb_ids do
    {plural, singular} = Enum.split_with(Probes.present_verbs(), &(elem(&1, 1) == :plural))
    {ids(plural), ids(singular)}
  end

  defp ids(verbs), do: Nx.tensor(Enum.map(verbs, &Vocab.word_to_id(elem(&1, 0))), type: :s64)

  defp non_verb_mask(dataset) do
    verbs =
      for tense <- [:present, :past],
          number <- [:singular, :plural],
          animacy <- [:animate, :inanimate],
          word <-
            Vocab.transitive_verbs(tense, number) ++
              Vocab.intransitive_verbs(animacy, tense, number) ++ Vocab.copula(tense, number),
          into: MapSet.new(),
          do: Vocab.word_to_id(word)

    keep =
      dataset.targets
      |> Nx.to_list()
      |> Enum.map(fn row -> Enum.map(row, &if(MapSet.member?(verbs, &1), do: 0.0, else: 1.0)) end)
      |> Nx.tensor(type: :f32)

    Nx.multiply(dataset.mask, keep)
  end
end
