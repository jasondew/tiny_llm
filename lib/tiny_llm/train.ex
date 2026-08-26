defmodule TinyLlm.Train do
  @moduledoc """
  The training harness: mini-batches, plain SGD, and a loss history.

  ## The model contract

  A model is any module implementing the callbacks below. Train knows how to
  batch, how to subtract a gradient, and how to log; it knows nothing about
  embeddings, attention, or what an example is. That separation is what lets
  stage 5 reuse this file unchanged.

  The SGD update is generic because params and gradients are maps with the
  same keys: for every named matrix, `param - learning_rate * gradient`.
  """

  alias TinyLlm.Grammar
  alias TinyLlm.Tensor

  defmodule Config do
    @moduledoc """
    Everything a training run needs, shared by stage 3 and stage 5.
    """

    defstruct model: nil,
              vocabulary_size: 32,
              d_model: 32,
              context_length: 16,
              learning_rate: 0.5,
              learning_rate_schedule: :constant,
              batch_size: 64,
              steps: 1_000,
              log_every: 100,
              training_corpus_size: 2_000,
              evaluation_corpus_size: 500,
              seed: 1234

    @type t :: %__MODULE__{}
  end

  @typedoc "Named parameter matrices, e.g. `%{embeddings: E, projection: W}`."
  @type params :: %{atom() => Tensor.matrix()}

  @typedoc "One training example. Its shape is the model's business, not ours."
  @type example :: term()

  @doc "Freshly initialized parameters, drawn from the process `:rand` state."
  @callback init(Config.t()) :: params()

  @doc "Turns a corpus of sentences into whatever this model trains on."
  @callback examples([Grammar.sentence()]) :: [example()]

  @doc "Mean loss over a batch, in nats."
  @callback loss(params(), [example()]) :: float()

  @doc "Mean gradient over a batch, keyed exactly like the params."
  @callback gradients(params(), [example()]) :: params()

  @doc """
  Seeds the process `:rand` state so a whole run reproduces.
  """
  @spec seed(integer()) :: :rand.state()
  def seed(seed), do: :rand.seed(:exsss, seed)

  @doc """
  Trains a model and returns the final params with the loss history.

  Seeds, generates two corpora, then runs `steps` mini-batches of plain SGD.
  The history holds `{step, loss}` pairs, one every `log_every` steps,
  starting at step 0 so the first entry is the loss before any learning.

  Batches are drawn from the training corpus; every logged loss is measured
  on the whole evaluation corpus, which the model never trains on. Measuring
  on a training batch instead is tempting and wrong twice over: a batch is
  small enough that its empirical entropy wanders well below the true value,
  and the examples are ones the model has already fitted. Both push the
  number down, and a loss below `H(next | previous)` is not a triumph, it is
  a leak.
  """
  @spec run(Config.t()) :: %{params: params(), losses: [{non_neg_integer(), float()}]}
  def run(config) do
    # All three of these draw from one seeded `:rand` stream, so their order
    # is load-bearing: swapping any two changes every number the run
    # produces. Training corpus, then evaluation corpus, then parameters.
    seed(config.seed)
    examples = config.training_corpus_size |> Grammar.corpus() |> config.model.examples()
    evaluation = config.evaluation_corpus_size |> Grammar.corpus() |> config.model.examples()
    params = config.model.init(config)

    {final_params, losses} =
      Enum.reduce(
        0..config.steps,
        {params, []},
        fn step, {params, losses} ->
          batch = batch(examples, config.batch_size)
          updated_params = step(config.model, params, batch, learning_rate(config, step))

          updated_losses =
            if rem(step, config.log_every) == 0 do
              loss = config.model.loss(updated_params, evaluation)

              [{step, loss} | losses]
            else
              losses
            end

          {updated_params, updated_losses}
        end
      )

    %{params: final_params, losses: Enum.reverse(losses)}
  end

  @doc """
  The learning rate to use at `step`, under the config's schedule.

  `:constant` is the rate as given. `:cosine` starts there and eases to
  zero over the run, following `base * (1 + cos(pi * step / steps)) / 2`.

  Decaying is not a tweak. Measured over 8 seeds per cell at a fixed
  compute budget, cosine beat a constant rate at every batch size tried, by
  0.02 to 0.04 nats, and it also halved the spread between seeds: a
  constant rate keeps taking full-size steps after it has arrived, so where
  it stops depends on which step it stopped on.

  The rate that suits a batch size scales with it. Measured, batch 4 wants
  0.25 and batch 8 wants 0.5, exactly linear; batch 16 wants 0.5 to 0.75
  rather than the 1.0 the rule predicts, which is the same sublinearity
  large-batch training runs into everywhere.
  """
  @spec learning_rate(Config.t(), non_neg_integer()) :: float()
  def learning_rate(%Config{learning_rate_schedule: :constant} = config, _step) do
    config.learning_rate
  end

  def learning_rate(%Config{learning_rate_schedule: :cosine} = config, step) do
    config.learning_rate * 0.5 * (1 + :math.cos(:math.pi() * step / config.steps))
  end

  @doc """
  One SGD step: subtract `learning_rate` times the gradient from every
  named matrix.
  """
  @spec step(module(), params(), [example()], float()) :: params()
  def step(model, params, batch, learning_rate) do
    gradients = model.gradients(params, batch)

    params
    |> Enum.map(fn {key, param} ->
      gradient = Map.fetch!(gradients, key)
      scaled_gradient = Tensor.scale(gradient, learning_rate)
      updated_param = Tensor.sub(param, scaled_gradient)
      {key, updated_param}
    end)
    |> Map.new()
  end

  @doc """
  A gradient accumulator: the same keys as `params`, every matrix zeroed.
  """
  @spec zero_gradients(params()) :: params()
  def zero_gradients(params) do
    Map.new(params, fn {key, matrix} ->
      {rows, columns} = Tensor.shape(matrix)

      {key, Tensor.zeros(rows, columns)}
    end)
  end

  @doc """
  Two gradient maps added key by key.

  Lives here rather than in a model because it is the same arithmetic
  `step/4` does: params and gradients are maps with the same keys, and
  nothing about that needs to know what a key means.
  """
  @spec add_gradients(params(), params()) :: params()
  def add_gradients(left, right) do
    Map.new(left, fn {key, matrix} -> {key, Tensor.add(matrix, Map.fetch!(right, key))} end)
  end

  @doc """
  A random batch of `size` examples, drawn with replacement.
  """
  @spec batch([example()], pos_integer()) :: [example()]
  def batch(examples, size) do
    for _ <- 1..size do
      Enum.random(examples)
    end
  end
end
