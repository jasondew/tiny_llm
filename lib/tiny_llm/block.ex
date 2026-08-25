defmodule TinyLlm.Block do
  @moduledoc """
  One transformer block: the stage 4 head, wrapped in the three things that
  make it trainable at depth.

  TODO(stage 5): write the concept paragraph. It should say that the head
  on its own is one linear map dressed up, and that what turns it into a layer
  you can stack is normalization before, a residual around, and a
  position-wise MLP after. The residual is the load-bearing one: it leaves a
  path from the loss to the embeddings that passes through no matrix at all.

  ## Forward

      n1  = rmsnorm(x, g1)
      o   = attention(n1)          every line of stage 4, on n1
      r   = x + o                  first residual
      n2  = rmsnorm(r, g2)
      pre = n2 * w1 + b1
      hid = max(pre, 0)            ReLU
      mlp = hid * w2 + b2
      y   = r + mlp                second residual

  **Pre-norm, not post-norm.** The normalization goes before each sublayer
  and the residual skips around both. The 2017 arrangement,
  `r = rmsnorm(x + o)`, trains noticeably worse without a warmup schedule,
  because the residual path is no longer a clean identity back to the
  embeddings.

  ## Backward

  Derived in full in `docs/backprop.md`, sections N through S. The two lines
  worth suspicion:

      dr = dy + dr_from_norm2      a sum of two, arriving from two places
      dx = dr + dx_from_norm1      the same trap, one level up

  Drop the second term of either and the loss still falls, most matrices
  still pass the gradient check, and only `embeddings` and `positions` fail.

  ## Why RMSNorm's backward pass looks like the softmax's

  Both normalize a row, which couples every entry to every other, and the
  price is always one row-scalar subtracted off:

      softmax   dm[j] = a[j] * (da[j] - dot(da, a))
      rmsnorm   dx[j] = (dn[j] - n[j] * mean(dn * n)) / rms

  Same shape, different normalizer. RMSNorm also hands you a free check:
  every row of `dx` comes out orthogonal to the corresponding row of `n`.
  """

  alias TinyLlm.Attention
  alias TinyLlm.Tensor
  alias TinyLlm.Train

  @typedoc "What `rmsnorm/2` kept for the backward pass."
  @type norm_cache :: %{
          output: Tensor.matrix(),
          normalized: Tensor.matrix(),
          rms: [float()]
        }

  @typedoc "Everything one block forward pass produced."
  @type cache :: %{
          input: Tensor.matrix(),
          norm1: norm_cache(),
          attention: Attention.cache(),
          residual: Tensor.matrix(),
          norm2: norm_cache(),
          pre: Tensor.matrix(),
          hidden: Tensor.matrix(),
          mlp: Tensor.matrix(),
          output: Tensor.matrix()
        }

  @doc """
  The guard inside the square root, so a row of exact zeros normalizes to
  zeros rather than to NaN.
  """
  @spec epsilon() :: float()
  def epsilon, do: 1.0e-12

  @doc """
  The MLP's hidden width: four times `d_model`, as the brief specifies.

  Four is the near-universal convention rather than a tuned number, and
  deriving it here keeps it out of `Train.Config`, which stage 3 shares.
  """
  @spec hidden_width(pos_integer()) :: pos_integer()
  def hidden_width(d_model), do: 4 * d_model

  @doc """
  Random starting parameters for one block.

  Gains start at `1.0` and biases at `0.0`, which is not laziness: a gain of
  one makes RMSNorm start as pure normalization, and a zero bias starts the
  ReLU symmetric. Both have a known neutral position, so randomizing them
  would only add noise to a knob that is already where it should be.
  """
  @spec init(Train.Config.t()) :: Train.params()
  def init(%{d_model: d_model}) do
    hidden = hidden_width(d_model)
    square = Tensor.fan_scale(d_model, d_model)
    projection = Tensor.fan_scale(d_model, hidden)

    %{
      gain1: Tensor.ones(1, d_model),
      query_weight: Tensor.random(d_model, d_model, square),
      key_weight: Tensor.random(d_model, d_model, square),
      value_weight: Tensor.random(d_model, d_model, square),
      output_weight: Tensor.random(d_model, d_model, square),
      gain2: Tensor.ones(1, d_model),
      weight1: Tensor.random(d_model, hidden, projection),
      bias1: Tensor.zeros(1, hidden),
      weight2: Tensor.random(hidden, d_model, projection),
      bias2: Tensor.zeros(1, d_model)
    }
  end

  @doc """
  Root-mean-square normalization of every row, scaled by a learned gain.

  No mean subtraction and no bias, which is the difference from LayerNorm.
  The RMSNorm paper's finding is that the recentring does almost nothing and
  costs a pass over the row.

  Returns `normalized` and `rms` alongside the output because the backward
  pass needs both, and recomputing them there is where the two drift apart.
  """
  @spec rmsnorm(Tensor.matrix(), Tensor.matrix()) :: norm_cache()
  def rmsnorm(input, [gain]) do
    rms =
      for row <- input do
        sum_of_squares = Enum.reduce(row, 0.0, fn x, acc -> acc + x * x end)
        :math.sqrt(sum_of_squares / length(row) + epsilon())
      end

    normalized =
      for {row, r} <- Enum.zip(input, rms) do
        Enum.map(row, fn x -> x / r end)
      end

    output =
      for row <- normalized do
        Enum.zip_with(row, gain, fn n, g -> n * g end)
      end

    %{
      output: output,
      normalized: normalized,
      rms: rms
    }
  end

  @doc """
  The gradient of `rmsnorm/2`, as `{dgain, dinput}`.

      dn = dy * g
      c  = mean of (dn * n)                one number per row
      dx = (dn - n * c) / rms

  `dgain` accumulates down every row, because one gain vector is shared by
  every position, exactly as `positions` is in stage 4.
  """
  @spec rmsnorm_backward(norm_cache(), Tensor.matrix(), Tensor.matrix()) ::
          {Tensor.matrix(), Tensor.matrix()}
  def rmsnorm_backward(cache, gain, doutput) do
    [gain_row] = gain
    width = length(gain_row)

    # One gain vector is shared by every position, so its gradient is the sum
    # of dy * n down every row, the same accumulation `positions` gets.
    dgain =
      Enum.zip_reduce(doutput, cache.normalized, List.duplicate(0.0, width), fn drow,
                                                                                normalized_row,
                                                                                totals ->
        contribution = Enum.zip_with(drow, normalized_row, &(&1 * &2))

        Enum.zip_with(totals, contribution, &+/2)
      end)

    # `coupling` is one number per row, not per entry. Normalizing a row ties
    # every entry to every other, so each entry loses the row's average
    # effect, exactly as the softmax Jacobian subtracts its own row scalar.
    dinput =
      Enum.zip_with([doutput, cache.normalized, cache.rms], fn [drow, normalized_row, rms] ->
        dnormalized = Enum.zip_with(drow, gain_row, &(&1 * &2))
        coupling = Tensor.dot(dnormalized, normalized_row) / width

        Enum.zip_with(dnormalized, normalized_row, fn dn, n -> (dn - n * coupling) / rms end)
      end)

    {[dgain], dinput}
  end

  @doc """
  One block forward pass, keeping every intermediate the backward pass needs.
  """
  @spec forward(Train.params(), Tensor.matrix()) :: cache()
  def forward(_params, _input), do: raise("TODO: stage 5")

  @doc """
  The gradient of `forward/2`, as `{gradients, dinput}`.

  `gradients` is keyed like the block's slice of the params. `dinput` is
  what the caller adds to whatever else feeds its input.
  """
  @spec backward(Train.params(), cache(), Tensor.matrix()) ::
          {Train.params(), Tensor.matrix()}
  def backward(_params, _cache, _doutput), do: raise("TODO: stage 5")
end
