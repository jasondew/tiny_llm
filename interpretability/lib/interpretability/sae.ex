defmodule Interpretability.SAE do
  @moduledoc """
  A sparse autoencoder: a second, small network that tries to untangle
  the MLP neurons into features with one meaning each.

  ## The idea

  At every word, the model's MLP produces 128 numbers, one per neuron, and
  the neuron report showed those neurons are tangled: none of them matches
  a single grammar concept. The SAE tries a different set of units:

      features       = keep_top_8(relu((neurons − decoder_bias) · encoder + encoder_bias))
      reconstruction = features · decoder + decoder_bias

  1. **Wider.** 128 neurons become 512 features, so each concept could
     have a slot of its own.
  2. **Rebuild.** The features must reconstruct the original 128 numbers.
  3. **Sparse.** At most 8 of the 512 features may be on at any word.

  The third rule does the work. Without it, the SAE could copy the 128
  neurons into 128 of its features, rebuild them perfectly, and teach us
  nothing. With only 8 allowed at once, copying is impossible; the
  cheapest way to rebuild well is to give each feature one reusable job
  that can be combined with a few others. That is the bet: sparsity forces
  one meaning per feature.

  Each feature's row of `decoder` is its *direction*: the pattern across
  the 128 neurons that the feature writes when it is on. Rows are kept at
  length one, so a feature's activation alone says how strongly it is on.

  ## TopK, not an L1 penalty

  "Towards Monosemanticity" (2023) made features sparse with a penalty on
  their total size, whose strength had to be tuned. This SAE keeps the
  top 8 and zeroes the rest (the TopK variant, from later work), so the
  sparsity is a number you can say on a slide.

  ## Checking the SAE did not cheat

  A good reconstruction is not enough. `report/5` also splices the
  reconstruction into the model in place of the real neurons and measures
  how much worse the model predicts. If the SAE kept what matters, the
  loss barely moves.
  """

  import Nx.Defn

  alias Interpretability.Activations
  alias Interpretability.Compiler
  alias Interpretability.Dataset
  alias Interpretability.Model
  alias Interpretability.Trainer

  defmodule Config do
    @moduledoc """
    One SAE training run.
    """

    defstruct input_width: 128,
              width: 512,
              active_count: 8,
              learning_rate: 1.0e-3,
              batch_size: 1024,
              steps: 5_000,
              seed: 1234

    @type t :: %__MODULE__{}
  end

  @type t :: %{
          encoder: Nx.Tensor.t(),
          encoder_bias: Nx.Tensor.t(),
          decoder: Nx.Tensor.t(),
          decoder_bias: Nx.Tensor.t()
        }

  @type report :: %{
          variance_explained: float(),
          loss_original: float(),
          loss_spliced: float(),
          loss_zero_ablated: float(),
          loss_recovered: float(),
          dead_count: non_neg_integer(),
          dense_count: non_neg_integer()
        }

  @doc """
  Random unit length decoder directions, an encoder that starts as their
  mirror image, and a decoder bias at the mean of the activations, so an
  SAE with no features on already predicts the average.
  """
  @spec init(Config.t(), Nx.Tensor.t()) :: t()
  def init(%Config{} = config, activations) do
    {decoder, _key} =
      Nx.Random.normal(Nx.Random.key(config.seed),
        shape: {config.width, config.input_width},
        type: :f32
      )

    decoder = unit_rows(decoder)

    %{
      encoder: Nx.transpose(decoder),
      encoder_bias: Nx.broadcast(Nx.tensor(0.0, type: :f32), {config.width}),
      decoder: decoder,
      decoder_bias: Nx.mean(Nx.backend_copy(activations, Nx.BinaryBackend), axes: [0])
    }
  end

  @doc """
  Feature activations: at most `active_count` nonzero per input row.
  """
  defn encode(sae, input, opts \\ []) do
    active_count = opts[:active_count]
    pre = Nx.max(Nx.dot(input - sae.decoder_bias, sae.encoder) + sae.encoder_bias, 0)
    {top_values, _indices} = Nx.top_k(pre, k: active_count)
    smallest_kept = Nx.slice_along_axis(top_values, active_count - 1, 1, axis: -1)
    Nx.select(pre >= smallest_kept, pre, 0.0)
  end

  @doc """
  The neuron values rebuilt from feature activations.
  """
  defn(decode(sae, features), do: Nx.dot(features, sae.decoder) + sae.decoder_bias)

  @doc """
  Encode, then decode.
  """
  defn reconstruct(sae, input, opts \\ []) do
    decode(sae, encode(sae, input, opts))
  end

  @doc """
  Reconstruction error as a share of the input's variance: 0 is perfect,
  1 is no better than always predicting the batch mean.
  """
  defn loss(sae, input, opts \\ []) do
    error = Nx.sum((input - reconstruct(sae, input, opts)) ** 2, axes: [-1])
    spread = Nx.sum((input - Nx.mean(input, axes: [0])) ** 2, axes: [-1])
    Nx.mean(error) / Nx.mean(spread)
  end

  @doc """
  The share of the input's variance the reconstruction explains.
  """
  @spec variance_explained(t(), Nx.Tensor.t(), pos_integer()) :: Nx.Tensor.t()
  def variance_explained(sae, input, active_count) do
    compiled = Nx.Defn.jit(&loss(&1, &2, active_count: active_count), Compiler.options())
    Nx.subtract(1, compiled.(sae, input))
  end

  @doc """
  Scales every decoder row back to length one.
  """
  defn(normalize_decoder(sae), do: %{sae | decoder: unit_rows(sae.decoder)})

  @doc """
  Trains an SAE on a tokens × neurons activation tensor.
  """
  @spec train(Config.t(), Nx.Tensor.t()) :: {t(), [Trainer.history_point()]}
  def train(%Config{} = config, activations) do
    activations = Nx.backend_copy(activations, Compiler.backend())
    active_count = config.active_count

    Trainer.run(
      init(config, activations),
      batches(activations, config.batch_size, config.seed),
      fn sae, batch -> loss(sae, batch.input, active_count: active_count) end,
      steps: config.steps,
      learning_rate: config.learning_rate,
      after_step: &normalize_decoder/1
    )
  end

  @doc """
  Reconstruction quality, the cost of splicing the SAE into the model,
  and dead and dense features, all on `dataset` and its `activations`.

  `loss_recovered` is `(zero_ablated − spliced) / (zero_ablated −
  original)`: 1.0 means splicing costs nothing, 0.0 means the SAE is no
  better than switching the neurons off.
  """
  @spec report(t(), pos_integer(), Model.params(), Dataset.t(), Activations.t()) :: report()
  def report(sae, active_count, model_params, dataset, activations) do
    loss_original = Nx.to_number(Model.evaluate_loss(model_params, dataset))

    loss_spliced =
      Nx.to_number(
        Model.evaluate_loss(model_params, dataset,
          edit_hidden: &reconstruct(sae, &1, active_count: active_count)
        )
      )

    loss_zero_ablated =
      Nx.to_number(Model.evaluate_loss(model_params, dataset, edit_hidden: &Nx.multiply(&1, 0.0)))

    firing_rates =
      sae
      |> features(activations.hidden, active_count)
      |> Nx.greater(0)
      |> Nx.mean(axes: [0])
      |> Nx.to_list()

    %{
      variance_explained: Nx.to_number(variance_explained(sae, activations.hidden, active_count)),
      loss_original: loss_original,
      loss_spliced: loss_spliced,
      loss_zero_ablated: loss_zero_ablated,
      loss_recovered: (loss_zero_ablated - loss_spliced) / (loss_zero_ablated - loss_original),
      dead_count: Enum.count(firing_rates, &(&1 == 0.0)),
      dense_count: Enum.count(firing_rates, &(&1 > 0.5))
    }
  end

  @doc """
  Feature activations for a whole tokens × neurons tensor, compiled.
  """
  @spec features(t(), Nx.Tensor.t(), pos_integer()) :: Nx.Tensor.t()
  def features(sae, input, active_count) do
    Nx.Defn.jit(&encode(&1, &2, active_count: active_count), Compiler.options()).(sae, input)
    |> Nx.backend_copy(Nx.BinaryBackend)
  end

  ## PRIVATE FUNCTIONS

  defnp(unit_rows(matrix), do: matrix / Nx.sqrt(Nx.sum(matrix ** 2, axes: [1], keep_axes: true)))

  # An endless stream of random batches of rows, reshuffled every epoch.
  defp batches(activations, batch_size, seed) do
    row_count = Nx.axis_size(activations, 0)

    :rand.seed_s(:exsss, seed)
    |> Stream.unfold(fn random_state ->
      {keys, random_state} =
        Enum.map_reduce(1..row_count, random_state, fn _row, state -> :rand.uniform_s(state) end)

      order = keys |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1))
      {Enum.chunk_every(order, batch_size, batch_size, :discard), random_state}
    end)
    |> Stream.flat_map(& &1)
    |> Stream.map(fn rows ->
      %{input: Nx.take(activations, Nx.tensor(rows, type: :s64, backend: Compiler.backend()))}
    end)
  end
end
