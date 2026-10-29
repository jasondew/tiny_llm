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
              evaluation_corpus_size: 2_000

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

    Map.merge(random, %{
      gain1: Nx.broadcast(Nx.tensor(1.0, type: :f32), {d_model}),
      gain2: Nx.broadcast(Nx.tensor(1.0, type: :f32), {d_model}),
      gain3: Nx.broadcast(Nx.tensor(1.0, type: :f32), {d_model}),
      bias1: Nx.broadcast(Nx.tensor(0.0, type: :f32), {hidden_width}),
      bias2: Nx.broadcast(Nx.tensor(0.0, type: :f32), {d_model})
    })
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
    log_probabilities = log_softmax(forward(params, batch.tokens, opts))
    loss_from(log_probabilities, batch)
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
      &loss/2,
      steps: config.steps,
      learning_rate: config.learning_rate
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
    hidden = edit_hidden.(Nx.max(Nx.dot(normalized, params.weight1) + params.bias1, 0))
    output = residual + Nx.dot(hidden, params.weight2) + params.bias2
    logits = Nx.dot(rmsnorm(output, params.gain3), params.projection)
    {logits, hidden}
  end

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
