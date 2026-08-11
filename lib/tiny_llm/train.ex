defmodule TinyLlm.Train do
  @moduledoc """
  The training harness: mini-batches, plain SGD, and a loss history.

  TODO(stage 3): write the concept paragraph. It should say that training is
  `Enum.reduce(batches, params, &step/2)` and nothing more exotic, and that
  the same harness trains the neural bigram at stage 3 and the full
  transformer at stage 5 because both are just "params in, params out".

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
              hidden_size: 32,
              context_length: 16,
              learning_rate: 0.5,
              batch_size: 64,
              steps: 1_000,
              log_every: 100,
              corpus_size: 2_000,
              seed: 1234

    @type t :: %__MODULE__{}
  end

  @typedoc "Named parameter matrices, e.g. `%{embedding: E, projection: W}`."
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
  def seed(_seed), do: raise("TODO: stage 3")

  @doc """
  Trains a model and returns the final params with the loss history.

  Seeds, generates a corpus, converts it to examples, then runs `steps`
  mini-batches of plain SGD. The history holds `{step, loss}` pairs, one
  every `log_every` steps, starting at step 0 so the first entry is the
  loss before any learning has happened.
  """
  @spec run(Config.t()) :: %{params: params(), losses: [{non_neg_integer(), float()}]}
  def run(_config), do: raise("TODO: stage 3")

  @doc """
  One SGD step: subtract `learning_rate` times the gradient from every
  named matrix.

  Generic over any params map, which is the reason stage 5 gets this free.
  """
  @spec step(module(), params(), [example()], float()) :: params()
  def step(_model, _params, _batch, _learning_rate), do: raise("TODO: stage 3")

  @doc """
  A random batch of `size` examples, drawn with replacement.
  """
  @spec batch([example()], pos_integer()) :: [example()]
  def batch(_examples, _size), do: raise("TODO: stage 3")
end
