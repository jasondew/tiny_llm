defmodule Interpretability.Toy do
  @moduledoc """
  A toy model of superposition: five features squeezed through two numbers.

  This is the experiment from Anthropic's "Toy Models of Superposition"
  (2022), at its smallest. It answers one question: when a model has more
  things to remember than it has room for, what does it do?

  ## The setup

  There are five *features*, five numbers between 0 and 1. The model must
  squeeze them into two numbers (two *dimensions*) and then rebuild all
  five from those two:

      compressed     = weights · features            five numbers in, two out
      reconstruction = relu(transpose(weights) · compressed + bias)

  `weights` is a 2 × 5 table. Its five columns are the interesting part:
  column `i` is the arrow, in two dimensional space, that feature `i` is
  written along. A column of length near 1 means the model kept that
  feature. A column of length near 0 means it gave up on it.

  Not every feature matters equally. Feature `i` carries an importance of
  `importance_decay^i`, so feature 0 matters most, and getting it wrong
  costs the most.

  ## The knob: sparsity

  Sparsity is how often a feature is simply zero. At sparsity 0.9, each
  feature is zero 90% of the time and only occasionally carries a value.

  - **Dense features** (sparsity 0): every feature is busy on every input.
    Two dimensions can hold two features cleanly at right angles, so the
    model keeps the two most important ones and drops the rest.
  - **Sparse features** (sparsity near 1): features are rarely on at the
    same time. The model can then point several features in nearly the same
    space and accept a little interference on the rare occasions two fire
    together, because it is right most of the time. More features than
    dimensions: that is *superposition*.

  The llama model's MLP neurons face the same trade on a larger scale,
  which is why a single neuron can end up responding to several unrelated
  things.
  """

  import Nx.Defn

  alias Interpretability.Compiler

  defmodule Config do
    @moduledoc """
    One toy training run. Only `sparsity` normally changes between runs.
    """

    defstruct feature_count: 5,
              dimension_count: 2,
              importance_decay: 0.8,
              sparsity: 0.0,
              batch_size: 1024,
              steps: 3_000,
              learning_rate: 1.0e-2,
              seed: 1234

    @type t :: %__MODULE__{}
  end

  @typedoc "A trained toy model and its final loss."
  @type result :: %{weights: Nx.Tensor.t(), bias: Nx.Tensor.t(), loss: float()}

  @doc """
  Trains one toy model from scratch and returns its weights, bias, and the
  loss on a fresh batch.
  """
  @spec train(Config.t()) :: result()
  def train(%Config{} = config) do
    key = Nx.Random.key(config.seed)
    {params, key} = init(key, config)

    {init_optimizer, update_optimizer} =
      Polaris.Optimizers.adam(learning_rate: config.learning_rate)

    options = [
      update_optimizer: update_optimizer,
      steps: config.steps,
      batch_size: config.batch_size,
      sparsity: config.sparsity
    ]

    training = Nx.Defn.jit(&train_loop(&1, &2, &3, &4, options), Compiler.options())
    {params, key} = training.(params, init_optimizer.(params), key, importance(config))

    evaluation =
      Nx.Defn.jit(
        &evaluation_loss(&1, &2, &3, batch_size: config.batch_size, sparsity: config.sparsity),
        Compiler.options()
      )

    loss = evaluation.(params, key, importance(config))

    %{
      weights: Nx.backend_copy(params.weights, Nx.BinaryBackend),
      bias: Nx.backend_copy(params.bias, Nx.BinaryBackend),
      loss: Nx.to_number(loss)
    }
  end

  @doc """
  How many features the model kept: the columns of `weights` longer than
  `threshold`.
  """
  @spec represented(result(), float()) :: non_neg_integer()
  def represented(%{weights: weights}, threshold \\ 0.5) do
    weights
    |> column_lengths()
    |> Nx.greater(threshold)
    |> Nx.sum()
    |> Nx.to_number()
  end

  @doc """
  The length of each feature's arrow, one per feature.
  """
  @spec column_lengths(Nx.Tensor.t()) :: Nx.Tensor.t()
  def column_lengths(weights), do: Nx.sqrt(Nx.sum(Nx.pow(weights, 2), axes: [0]))

  @doc """
  Trains one model per sparsity, everything else taken from `config`.
  """
  @spec sweep([float()], Config.t()) :: [map()]
  def sweep(sparsities, %Config{} = config) do
    Enum.map(sparsities, fn sparsity ->
      result = train(%{config | sparsity: sparsity})
      Map.merge(result, %{sparsity: sparsity, represented: represented(result)})
    end)
  end

  ## PRIVATE FUNCTIONS

  # Xavier normal: small random arrows, scaled so the first outputs are
  # neither huge nor vanishing.
  defp init(key, config) do
    standard_deviation = :math.sqrt(2 / (config.feature_count + config.dimension_count))

    {weights, key} =
      Nx.Random.normal(key, 0.0, standard_deviation,
        shape: {config.dimension_count, config.feature_count},
        type: :f32
      )

    {%{weights: weights, bias: Nx.broadcast(Nx.tensor(0.0, type: :f32), {config.feature_count})},
     key}
  end

  defp importance(config) do
    Nx.pow(config.importance_decay, Nx.iota({config.feature_count}, type: :f32))
  end

  defn train_loop(params, optimizer_state, key, importance, opts \\ []) do
    update_optimizer = opts[:update_optimizer]

    {params, _optimizer_state, key, _importance, _step} =
      while {params, optimizer_state, key, importance, step = 0}, step < opts[:steps] do
        {features, key} =
          batch(key, importance, batch_size: opts[:batch_size], sparsity: opts[:sparsity])

        gradients = grad(params, &loss(&1, features, importance))
        {updates, optimizer_state} = update_optimizer.(gradients, optimizer_state, params)

        {Polaris.Updates.apply_updates(params, updates), optimizer_state, key, importance,
         step + 1}
      end

    {params, key}
  end

  defn evaluation_loss(params, key, importance, opts \\ []) do
    {features, _key} =
      batch(key, importance, batch_size: opts[:batch_size], sparsity: opts[:sparsity])

    loss(params, features, importance)
  end

  # Each feature is uniform on [0, 1], then zeroed with probability
  # `sparsity`. `importance` is only here for its shape.
  defnp batch(key, importance, opts \\ []) do
    shape = {opts[:batch_size], Nx.axis_size(importance, 0)}
    {values, key} = Nx.Random.uniform(key, shape: shape, type: :f32)
    {draws, key} = Nx.Random.uniform(key, shape: shape, type: :f32)
    {values * (draws >= opts[:sparsity]), key}
  end

  defnp loss(params, features, importance) do
    compressed = Nx.dot(features, Nx.transpose(params.weights))
    reconstruction = Nx.max(Nx.dot(compressed, params.weights) + params.bias, 0)
    errors = importance * (features - reconstruction) ** 2
    Nx.mean(Nx.sum(errors, axes: [1]))
  end
end
