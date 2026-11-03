defmodule Interpretability.Model do
  @moduledoc """
  The llama model: the first talk's transformer, rebuilt in Nx.

  Same shape as `TinyLlm.Transformer` and `TinyLlm.Block`, with the same
  parameter names, so the slides can point at the first talk's code and
  say "this, again". What changed is only how the math runs: tensors
  instead of lists of lists, and gradients derived automatically.

  ## Forward

      x       = embeddings[tokens] + positions
      n1      = rmsnorm(x, gain1)
      r       = x + attention(n1)               one causal head
      n2      = rmsnorm(r, gain2)
      hidden  = relu(n2 · weight1 + bias1)      the 128 MLP neurons
      y       = r + hidden · weight2 + bias2
      logits  = rmsnorm(y, gain3) · projection

  ## The neurons

  `hidden` is where this talk looks. It holds 128 numbers per position,
  one per MLP *neuron*, each the output of a ReLU: zero, or some positive
  amount. When people say a neuron "fires", they mean its number is well
  above zero.

  `forward/3` accepts an `:edit_hidden` function that rewrites `hidden`
  before the rest of the model sees it. That one hook is how the talk
  experiments on the model: replacing `hidden` with a sparse
  autoencoder's reconstruction asks "does the SAE capture what matters?",
  and nudging it along one feature's direction is steering.
  """

  import Nx.Defn

  alias Interpretability.Compiler
  alias Interpretability.Dataset
  alias Interpretability.Trainer
  alias Interpretability.Vocab

  defmodule Config do
    @moduledoc """
    The model's size and its training run. `d_model` and `hidden_width`
    are the first talk's values and are not tuned.

    The last three fields exist for the cleanliness experiments, which
    retrain the model one change at a time to see whether any training
    choice makes the neurons cleaner. Their defaults are the talk's model:

    - `activation`: `:relu`, or `:solu` for Anthropic's softmax linear
      unit (see `solu/1`).
    - `weight_decay`: shrinks every parameter a little each step, so the
      model only keeps weights that earn their place.
    - `activation_penalty`: adds this much of the neurons' total
      activity to the loss, pushing each word to use fewer neurons.
    """

    defstruct vocabulary_size: 70,
              d_model: 32,
              hidden_width: 128,
              context_length: 20,
              seed: 1234,
              learning_rate: 3.0e-3,
              batch_size: 128,
              steps: 8_000,
              training_corpus_size: 60_000,
              evaluation_corpus_size: 2_000,
              activation: :relu,
              weight_decay: 0.0,
              activation_penalty: 0.0

    @type t :: %__MODULE__{}
  end

  @typedoc "Named parameter tensors, named as in `TinyLlm.Transformer`."
  @type params :: %{atom() => Nx.Tensor.t()}

  @rmsnorm_epsilon 1.0e-6

  @doc """
  Random starting parameters. Gains start at one and biases at zero. The
  projection starts small, so an untrained model guesses every word about
  equally.
  """
  @spec init(Config.t()) :: params()
  def init(%Config{} = config) do
    vocabulary_size = config.vocabulary_size
    d_model = config.d_model
    hidden_width = config.hidden_width

    shapes = [
      embeddings: {{vocabulary_size, d_model}, 1.0},
      positions: {{config.context_length, d_model}, 1.0},
      query_weight: {{d_model, d_model}, 1 / :math.sqrt(d_model)},
      key_weight: {{d_model, d_model}, 1 / :math.sqrt(d_model)},
      value_weight: {{d_model, d_model}, 1 / :math.sqrt(d_model)},
      output_weight: {{d_model, d_model}, 1 / :math.sqrt(d_model)},
      weight1: {{d_model, hidden_width}, 1 / :math.sqrt(d_model)},
      weight2: {{hidden_width, d_model}, 1 / :math.sqrt(hidden_width)},
      projection: {{d_model, vocabulary_size}, 0.02}
    ]

    {random, _key} =
      Enum.reduce(shapes, {%{}, Nx.Random.key(config.seed)}, fn {name, {shape, scale}},
                                                                {params, key} ->
        {tensor, key} = Nx.Random.normal(key, 0.0, scale, shape: shape, type: :f32)
        {Map.put(params, name, tensor), key}
      end)

    random
    |> Map.merge(%{
      gain1: Nx.broadcast(Nx.tensor(1.0, type: :f32), {d_model}),
      gain2: Nx.broadcast(Nx.tensor(1.0, type: :f32), {d_model}),
      gain3: Nx.broadcast(Nx.tensor(1.0, type: :f32), {d_model}),
      bias1: Nx.broadcast(Nx.tensor(0.0, type: :f32), {hidden_width}),
      bias2: Nx.broadcast(Nx.tensor(0.0, type: :f32), {d_model})
    })
    |> Map.merge(activation_params(config))
  end

  @doc """
  Next word scores for every position: batch × context × vocabulary.
  """
  defn forward(params, tokens, opts \\ []) do
    opts = keyword!(opts, edit_hidden: &Function.identity/1)
    {logits, _hidden} = run(params, tokens, opts[:edit_hidden])
    logits
  end

  @doc """
  The MLP neurons' activations: batch × context × 128.
  """
  defn hidden(params, tokens) do
    {_logits, hidden} = run(params, tokens, &Function.identity/1)
    hidden
  end

  @doc """
  Mean cross entropy over the real (unmasked) positions of a batch, in
  nats. `opts` pass through to `forward/3`.
  """
  defn loss(params, batch, opts \\ []) do
    loss_from(log_probabilities(params, batch.tokens, opts), batch)
  end

  @doc """
  `forward/3` turned into log probabilities of every next word: batch ×
  context × vocabulary. `opts` pass through to `forward/3`.
  """
  defn log_probabilities(params, tokens, opts \\ []) do
    log_softmax(forward(params, tokens, opts))
  end

  @doc """
  The loss training minimizes: `loss/3` plus `:activation_penalty` times
  the mean total activity (sum of absolute values) of the neurons at each
  real position. With no penalty it is `loss/3`.
  """
  defn training_loss(params, batch, opts \\ []) do
    opts = keyword!(opts, activation_penalty: 0.0)
    {logits, hidden} = run(params, batch.tokens, &Function.identity/1)

    loss_from(log_softmax(logits), batch) +
      activity(hidden, batch.mask, opts[:activation_penalty])
  end

  @doc """
  The softmax linear unit from Anthropic's "Softmax Linear Units" (2022):
  each input times its softmax across the layer, `x · softmax(x)`.

  Softmax makes the neurons compete. One large input takes almost all of
  the softmax and passes through nearly unchanged, while the rest are
  squashed toward zero, so a word ends up lighting few neurons. The
  authors found this made more neurons interpretable. In the model the
  unit is followed by a normalization with its own gain, as in the paper,
  because squashing otherwise shrinks the whole layer.
  """
  defn solu(tensor) do
    tensor * softmax(tensor)
  end

  @doc """
  `loss/3` over a whole dataset, compiled with `Interpretability.Compiler`.
  """
  @spec evaluate_loss(params(), Dataset.t(), keyword()) :: Nx.Tensor.t()
  def evaluate_loss(params, dataset, opts \\ []) do
    batch = Map.take(dataset, [:tokens, :targets, :mask])
    Nx.Defn.jit(&loss(&1, &2, opts), Compiler.options()).(params, batch)
  end

  @doc """
  Trains a fresh model on `dataset` with the config's run settings.
  """
  @spec train(Config.t(), Dataset.t()) :: {params(), [Trainer.history_point()]}
  def train(%Config{} = config, dataset) do
    Trainer.run(
      init(config),
      Dataset.batches(dataset, config.batch_size, config.seed),
      &training_loss(&1, &2, activation_penalty: config.activation_penalty),
      steps: config.steps,
      learning_rate: config.learning_rate,
      weight_decay: config.weight_decay
    )
  end

  @doc """
  The training and evaluation datasets for `config`, drawn from the
  grammar in that order from the config's seed. Every task that needs
  "the" evaluation set gets the same sentences this way.

  The grammar repeats short sentences often, so the evaluation set skips
  any sentence that also appears in training. Every evaluation sentence
  is one the model never saw.
  """
  @spec datasets(Config.t()) :: {Dataset.t(), Dataset.t()}
  def datasets(%Config{} = config) do
    Interpretability.Grammar.seed(config.seed)
    training_trees = Interpretability.Grammar.trees(config.training_corpus_size)
    seen = MapSet.new(training_trees, &Interpretability.Grammar.words/1)

    evaluation_trees =
      Stream.repeatedly(&Interpretability.Grammar.tree/0)
      |> Stream.reject(&MapSet.member?(seen, Interpretability.Grammar.words(&1)))
      |> Enum.take(config.evaluation_corpus_size)

    {Dataset.build(training_trees), Dataset.build(evaluation_trees)}
  end

  @doc """
  Whether a sentence appears in the training set for `config`, for
  holding probe sentences out too.
  """
  @spec training_sentences(Config.t()) :: MapSet.t()
  def training_sentences(%Config{} = config) do
    Interpretability.Grammar.seed(config.seed)

    config.training_corpus_size
    |> Interpretability.Grammar.trees()
    |> MapSet.new(&Interpretability.Grammar.words/1)
  end

  @doc """
  A function from the words read so far to a score for every word in the
  vocabulary, for `Interpretability.Probes`.
  """
  @spec predictor(params()) :: ([Vocab.word()] -> [float()])
  def predictor(params) do
    context_length = Nx.axis_size(params.positions, 0)
    padding = Vocab.word_to_id(Vocab.end_token())
    compiled = Nx.Defn.jit(&forward/2, Compiler.options())

    fn prefix ->
      ids = Vocab.encode([Vocab.start_token() | prefix])
      padded = ids ++ List.duplicate(padding, context_length - length(ids))
      logits = compiled.(params, Nx.tensor([padded], type: :s64))
      Nx.to_list(logits[[0, length(ids) - 1]])
    end
  end

  ## PRIVATE FUNCTIONS

  defnp run(params, tokens, edit_hidden) do
    embedded = Nx.take(params.embeddings, tokens) + params.positions

    residual =
      embedded + attention(params, rmsnorm(embedded, params.gain1))

    normalized = rmsnorm(residual, params.gain2)
    hidden = edit_hidden.(activate(params, Nx.dot(normalized, params.weight1) + params.bias1))
    output = residual + Nx.dot(after_activation(params, hidden), params.weight2) + params.bias2
    logits = Nx.dot(rmsnorm(output, params.gain3), params.projection)
    {logits, hidden}
  end

  # A SoLU model is recognized by the gain on its neurons, so every caller
  # of `forward/3` and `hidden/2` handles both kinds without an option.
  deftransformp activate(params, preactivation) do
    if Map.has_key?(params, :gain_hidden),
      do: solu(preactivation),
      else: Nx.max(preactivation, 0)
  end

  deftransformp after_activation(params, hidden) do
    if Map.has_key?(params, :gain_hidden),
      do: rmsnorm(hidden, params.gain_hidden),
      else: hidden
  end

  deftransformp activity(hidden, mask, penalty) do
    if penalty == 0.0 do
      0.0
    else
      total = Nx.sum(Nx.abs(hidden), axes: [2])

      total
      |> Nx.multiply(mask)
      |> Nx.sum()
      |> Nx.divide(Nx.sum(mask))
      |> Nx.multiply(penalty)
    end
  end

  defp activation_params(%Config{activation: :relu}), do: %{}

  defp activation_params(%Config{activation: :solu, hidden_width: hidden_width}),
    do: %{gain_hidden: Nx.broadcast(Nx.tensor(1.0, type: :f32), {hidden_width})}

  # One causal head: each position mixes in the values of itself and the
  # positions before it, weighted by how well its query matches their keys.
  defnp attention(params, normalized) do
    queries = Nx.dot(normalized, params.query_weight)
    keys = Nx.dot(normalized, params.key_weight)
    values = Nx.dot(normalized, params.value_weight)

    d_model = Nx.axis_size(queries, 2)
    context_length = Nx.axis_size(queries, 1)

    scores = Nx.dot(queries, [2], [0], keys, [2], [0]) / Nx.sqrt(d_model)
    rows = Nx.iota({context_length, context_length}, axis: 0)
    columns = Nx.iota({context_length, context_length}, axis: 1)
    scores = Nx.select(Nx.broadcast(columns <= rows, scores), scores, -1.0e9)

    weights = softmax(scores)
    attended = Nx.dot(weights, [2], [0], values, [1], [0])
    Nx.dot(attended, params.output_weight)
  end

  defnp softmax(scores) do
    shifted = scores - Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exponentials = Nx.exp(shifted)
    exponentials / Nx.sum(exponentials, axes: [-1], keep_axes: true)
  end

  defnp log_softmax(logits) do
    shifted = logits - Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted - Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true))
  end

  defnp rmsnorm(tensor, gain) do
    root_mean_square =
      Nx.sqrt(Nx.mean(tensor ** 2, axes: [-1], keep_axes: true) + @rmsnorm_epsilon)

    tensor / root_mean_square * gain
  end

  defnp loss_from(log_probabilities, batch) do
    chosen =
      log_probabilities
      |> Nx.take_along_axis(Nx.new_axis(batch.targets, -1), axis: 2)
      |> Nx.squeeze(axes: [2])

    -Nx.sum(chosen * batch.mask) / Nx.sum(batch.mask)
  end
end
