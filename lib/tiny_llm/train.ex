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
          updated_params = step(config.model, params, batch, config.learning_rate)

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
  A random batch of `size` examples, drawn with replacement.
  """
  @spec batch([example()], pos_integer()) :: [example()]
  def batch(examples, size) do
    for _ <- 1..size do
      Enum.random(examples)
    end
  end
end
