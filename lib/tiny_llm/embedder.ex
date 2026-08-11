defmodule TinyLlm.Embedder do
  @moduledoc """
  The neural bigram: the same one-word context as `TinyLlm.Bigram`, but with
  the counts replaced by parameters learned from gradients.

  TODO(stage 3): write the concept paragraph. It should say that this model
  can do nothing the count-based bigram cannot, and that the point of
  building it is to introduce embeddings, softmax, cross-entropy, and a
  hand-derived backward pass on a problem whose right answer we already
  know. When its loss lands near the bigram's floor of about 1.90 nats, that
  is confirmation the machinery works, not evidence the model is clever.

  ## Forward

      hidden  = embedding[input_id]        one row, hidden_size wide
      logits  = hidden * projection        one row, vocabulary_size wide
      p       = softmax(logits)

  ## Backward

  Cross-entropy composed with softmax collapses into one line, which is the
  single most useful fact in this whole project:

      dlogits    = p - one_hot(target)
      dprojection = hiddenᵀ * dlogits      an outer product
      dhidden     = dlogits * projectionᵀ
      dembedding[input_id] += dhidden      that row only; every other row is
                                           untouched by this example

  Write the derivation down before implementing it. Stage 4 requires that in
  `docs/backprop.md` and this is the gentle version to practise on.
  """

  @behaviour TinyLlm.Train

  alias TinyLlm.Grammar
  alias TinyLlm.Tensor
  alias TinyLlm.Train
  alias TinyLlm.Vocab

  @typedoc "One `{input, target}` pair of token ids."
  @type example :: {Vocab.id(), Vocab.id()}

  @doc """
  Random starting parameters: an embedding table and a projection.

  Both are drawn at the default `Tensor.random/3` scale, small enough that
  the initial logits are near zero and the first loss lands near
  `ln(vocabulary_size)`.
  """
  @impl Train
  @spec init(Train.Config.t()) :: Train.params()
  def init(_config), do: raise("TODO: stage 3")

  @doc """
  A corpus of sentences as `{input, target}` pairs.

  Each sentence is prepended with the start token, exactly as the counting
  bigram does, so a sentence's first word is predicted from `"<start>"`.
  """
  @impl Train
  @spec examples([Grammar.sentence()]) :: [example()]
  def examples(_corpus), do: raise("TODO: stage 3")

  @doc """
  The next-token distribution given one input token.
  """
  @spec forward(Train.params(), Vocab.id()) :: Tensor.row()
  def forward(_params, _input_id), do: raise("TODO: stage 3")

  @doc """
  Mean cross-entropy over a batch, in nats.

  An untrained model scores about `ln(vocabulary_size)`, because a uniform
  guess over 32 words is the best you can do knowing nothing.
  """
  @impl Train
  @spec loss(Train.params(), [example()]) :: float()
  def loss(_params, _batch), do: raise("TODO: stage 3")

  @doc """
  Mean gradients over a batch, keyed exactly like the params.
  """
  @impl Train
  @spec gradients(Train.params(), [example()]) :: Train.params()
  def gradients(_params, _batch), do: raise("TODO: stage 3")
end
