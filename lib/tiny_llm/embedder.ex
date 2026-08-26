defmodule TinyLlm.Embedder do
  @moduledoc """
  The neural bigram: the same one-word context as `TinyLlm.Bigram`, but with
  the counts replaced by parameters learned from gradients.

  ## Forward

      embedding = embeddings[input_id]     one row, d_model wide
      logits    = embedding * projection   one row, vocabulary_size wide
      p         = softmax(logits)

  The lookup is a list index rather than a matrix multiply, and that is not
  a shortcut. The embedding layer is `one_hot(input_id) * embeddings`, but a
  one-hot row times a matrix selects one row and zeroes the other 31, so
  indexing computes the same thing without the theatre.

  ## Backward

  Cross-entropy composed with softmax collapses into one line, which is the
  single most useful fact in this whole project:

      dlogits     = p - one_hot(target)
      dprojection = embeddingᵀ * dlogits   an outer product
      dembedding  = dlogits * projectionᵀ
      dembeddings[input_id] += dembedding  that row only; every other row is
                                           untouched by this example

  Note the two shapes that look alike and are not: `dembedding` is a single
  row, while `dembeddings` is the whole table, zero everywhere but one row.

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
  def init(config) do
    embeddings = Tensor.random(config.vocabulary_size, config.d_model)
    projection = Tensor.random(config.d_model, config.vocabulary_size)

    %{embeddings: embeddings, projection: projection}
  end

  @doc """
  A corpus of sentences as `{input, target}` pairs.

  Each sentence is prepended with the start token, exactly as the counting
  bigram does, so a sentence's first word is predicted from `"<start>"`.
  """
  @impl Train
  @spec examples([Grammar.sentence()]) :: [example()]
  def examples(corpus) do
    Enum.flat_map(corpus, fn sentence ->
      [Vocab.start_token() | sentence]
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.map(fn [prev, next] ->
        {Vocab.word_to_id(prev), Vocab.word_to_id(next)}
      end)
    end)
  end

  @doc """
  The next-token distribution given one input token.
  """
  @spec forward(Train.params(), Vocab.id()) :: Tensor.row()
  def forward(params, input_id) do
    logits = logits(params, input_id)
    [probabilities] = Tensor.softmax([logits])
    probabilities
  end

  @doc """
  Mean cross-entropy over a batch, in nats.

  An untrained model scores about `ln(vocabulary_size)`, because a uniform
  guess over 32 words is the best you can do knowing nothing.
  """
  @impl Train
  @spec loss(Train.params(), [example()]) :: float()
  def loss(params, batch) do
    Enum.reduce(batch, 0.0, fn {input_id, target_id}, total ->
      total + Tensor.cross_entropy(logits(params, input_id), target_id)
    end) / length(batch)
  end

  @doc """
  Mean gradients over a batch, keyed exactly like the params.

  dlogits     = p - one_hot(target)
  dprojection = embeddingᵀ * dlogits   an outer product
  dembedding  = dlogits * projectionᵀ
  dembeddings[input_id] += dembedding  that row only
  """
  @impl Train
  @spec gradients(Train.params(), [example()]) :: Train.params()
  def gradients(params, batch) do
    vocabulary_size = length(params.embeddings)
    d_model = length(List.first(params.embeddings))

    zero_embeddings = Tensor.zeros(vocabulary_size, d_model)
    zero_projection = Tensor.zeros(d_model, vocabulary_size)

    summed =
      Enum.reduce(
        batch,
        %{embeddings: zero_embeddings, projection: zero_projection},
        fn {input_id, target_id}, acc ->
          probabilities = forward(params, input_id)
          one_hot = Tensor.one_hot(target_id, vocabulary_size)
          dlogits = Enum.zip_with(probabilities, one_hot, &-/2)

          embedding = Enum.at(params.embeddings, input_id)
          dprojection = Tensor.outer_product(embedding, dlogits)
          [dembedding] = Tensor.matmul([dlogits], Tensor.transpose(params.projection))

          dembeddings =
            List.update_at(acc.embeddings, input_id, fn row ->
              Enum.zip_with(row, dembedding, &+/2)
            end)

          %{
            embeddings: dembeddings,
            projection: Tensor.add(acc.projection, dprojection)
          }
        end
      )

    %{
      embeddings: Tensor.scale(summed.embeddings, 1 / length(batch)),
      projection: Tensor.scale(summed.projection, 1 / length(batch))
    }
  end

  ## PRIVATE FUNCTIONS

  defp logits(params, input_id) do
    embedding = Enum.at(params.embeddings, input_id)
    [logits] = Tensor.matmul([embedding], params.projection)
    logits
  end
end
