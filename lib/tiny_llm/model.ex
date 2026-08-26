defmodule TinyLlm.Model do
  @moduledoc """
  The whole thing: embeddings, one transformer block, and an unembedding.

  TODO(stage 5): write the concept paragraph. It should say that nothing
  here is new. `Model` is a lookup, a `Block`, a normalization and a matrix
  multiply, and every one of those already existed. What makes it a language
  model rather than a pile of layers is that the block leaves its input
  shape unchanged, so the same block could run twice, or twelve times, and
  the only reason it runs once here is that a 32 word grammar does not need
  more.

  ## Forward

      x     = embeddings[input_ids] + positions[0..T-1]
      y     = Block.forward(params, x)
      norm3 = rmsnorm(y, gain3)
      logits = norm3 * projection

  The third normalization is not decoration. Without it the residual stream
  reaches `projection` at whatever scale training happened to leave it,
  which is neither controlled nor stable.

  ## The unembedding is separate, not tied

  `embeddings` is `v x d` and `projection` is `d x v`, and they are distinct
  parameters. Tying them, `projection = transpose(embeddings)`, saves 1024
  parameters and is what most production models do. It is not done here
  because the talk shows the embedding table as a picture, a PCA scatter
  where nouns cluster and singular-plural pairs sit at parallel offsets, and
  a tied table has to serve two masters: it is both "what this word means as
  an input" and "what predicts this word as an output". Untied, the picture
  means one thing.

  ## Backward

  Derived in `docs/backprop.md`, section S. `Model` owns only the two ends:

      dlogits = (p - onehot(y)) / N
      dprojection = norm3ᵀ * dlogits    dnorm3 = dlogits * projectionᵀ
      dgain3, dy = rmsnorm_backward(...)
      then Block.backward, then scatter to embeddings and positions

  Everything between `dy` and the tables is `Block`, and everything inside
  that is `Attention`. Three modules, one chain, no autodiff.
  """

  @behaviour TinyLlm.Train

  alias TinyLlm.Attention
  alias TinyLlm.Block
  alias TinyLlm.Grammar
  alias TinyLlm.Tensor
  alias TinyLlm.Train
  alias TinyLlm.Vocab

  @typedoc "One sequence: the tokens read, and the tokens to predict."
  @type example :: {[Vocab.id()], [Vocab.id()]}

  @typedoc "Everything one forward pass produced."
  @type cache :: %{
          input: Tensor.matrix(),
          block: Block.cache(),
          norm3: Block.norm_cache(),
          logits: Tensor.matrix()
        }

  @doc """
  Random starting parameters: the block's ten, plus the two tables, the
  final gain, and the unembedding.
  """
  @impl Train
  @spec init(Train.Config.t()) :: Train.params()
  def init(_config), do: raise("TODO: stage 5")

  @doc """
  A corpus of sentences as `{input_ids, target_ids}` pairs.

  Identical to `TinyLlm.Attention.examples/1`, because the training problem
  did not change when the architecture did: every position still predicts
  the token after it.
  """
  @impl Train
  @spec examples([Grammar.sentence()]) :: [example()]
  def examples(_corpus), do: raise("TODO: stage 5")

  @doc """
  One forward pass, keeping every intermediate the backward pass needs.
  """
  @spec forward(Train.params(), [Vocab.id()]) :: cache()
  def forward(_params, _input_ids), do: raise("TODO: stage 5")

  @doc """
  The attention matrix on its own, for the stage 7 heatmap.
  """
  @spec weights(Train.params(), [Vocab.id()]) :: Tensor.matrix()
  def weights(_params, _input_ids), do: raise("TODO: stage 5")

  @doc """
  Mean cross-entropy over a batch, in nats, averaged over predicted tokens.
  """
  @impl Train
  @spec loss(Train.params(), [example()]) :: float()
  def loss(_params, _batch), do: raise("TODO: stage 5")

  @doc """
  Mean gradients over a batch, keyed exactly like the params.
  """
  @impl Train
  @spec gradients(Train.params(), [example()]) :: Train.params()
  def gradients(_params, _batch), do: raise("TODO: stage 5")
end
