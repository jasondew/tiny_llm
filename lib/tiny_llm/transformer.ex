defmodule TinyLlm.Transformer do
  @moduledoc """
  The whole thing: embeddings, one transformer block, and an unembedding.

  Nothing here is new. `Transformer` is a lookup, a `Block`, a normalization and
  a matrix multiply, and every one of those already existed by the time this
  file was written.

  What makes it a language model rather than a pile of layers is that the
  block leaves its input shape unchanged. The same block could run twice, or
  twelve times, and the only reason it runs once here is that a 32 word
  grammar does not need more. Scale is the difference between this file and
  a frontier model. Structure is not.

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

  Derived in `docs/backprop.md`, section S. `Transformer` owns only the two ends:

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
  def init(config) do
    v = config.vocabulary_size
    d = config.d_model
    c = config.context_length

    Map.merge(
      Block.init(config),
      %{
        embeddings: Tensor.random(v, d, Tensor.fan_scale(v, d)),
        positions: Tensor.random(c, d, Tensor.fan_scale(c, d)),
        projection: Tensor.random(d, v, Tensor.fan_scale(d, v)),
        gain3: Tensor.ones(1, d)
      }
    )
  end

  @doc """
  A corpus of sentences as `{input_ids, target_ids}` pairs.

  Identical to `TinyLlm.Attention.examples/1`, because the training problem
  did not change when the architecture did: every position still predicts
  the token after it.
  """
  @impl Train
  @spec examples([Grammar.sentence()]) :: [example()]
  def examples(corpus) do
    Attention.examples(corpus)
  end

  @doc """
  One forward pass, keeping every intermediate the backward pass needs.
  """
  @spec forward(Train.params(), [Vocab.id()]) :: cache()
  def forward(params, input_ids) do
    if length(input_ids) > length(params.positions) do
      raise ArgumentError, "Sequence too long: #{length(input_ids)} > #{length(params.positions)}"
    end

    embeddings = Enum.map(input_ids, fn id -> Enum.at(params.embeddings, id) end)
    positions = Enum.take(params.positions, length(input_ids))
    input = Tensor.add(embeddings, positions)

    block = Block.forward(params, input)
    norm3 = Block.rmsnorm(block.output, params.gain3)
    logits = Tensor.matmul(norm3.output, params.projection)

    %{
      input: input,
      block: block,
      norm3: norm3,
      logits: logits
    }
  end

  @doc """
  The attention matrix on its own, for the stage 7 heatmap.
  """
  @spec weights(Train.params(), [Vocab.id()]) :: Tensor.matrix()
  def weights(params, input_ids) do
    forward(params, input_ids).block.attention.weights
  end

  @doc """
  Mean cross-entropy over a batch, in nats, averaged over predicted tokens.
  """
  @impl Train
  @spec loss(Train.params(), [example()]) :: float()
  def loss(params, batch) do
    {summed_loss, token_count} =
      Enum.reduce(
        batch,
        {0.0, 0},
        fn {input_ids, target_ids}, {loss_so_far, tokens_so_far} ->
          logits = forward(params, input_ids).logits

          sequence_loss =
            Enum.zip_reduce(logits, target_ids, 0.0, fn row, target_id, accumulated ->
              accumulated + Tensor.cross_entropy(row, target_id)
            end)

          {loss_so_far + sequence_loss, tokens_so_far + length(target_ids)}
        end
      )

    summed_loss / token_count
  end

  @doc """
  Mean gradients over a batch, keyed exactly like the params.
  """
  @impl Train
  @spec gradients(Train.params(), [example()]) :: Train.params()
  def gradients(params, batch) do
    {summed, token_count} =
      Enum.reduce(
        batch,
        {Train.zero_gradients(params), 0},
        fn {input_ids, target_ids}, {totals, count_so_far} ->
          contribution = example_gradients(params, input_ids, target_ids)

          {Train.add_gradients(totals, contribution), count_so_far + length(target_ids)}
        end
      )

    Map.new(
      summed,
      fn {key, gradient} -> {key, Tensor.scale(gradient, 1 / token_count)} end
    )
  end

  ## PRIVATE FUNCTIONS

  # One example's contribution. Transformer owns only the two ends of the chain:
  # the loss and the unembedding on one side, the tables on the other.
  # Everything between is Block, and everything inside that is Attention.
  defp example_gradients(params, input_ids, target_ids) do
    cache = forward(params, input_ids)
    {vocabulary_size, _d_model} = Tensor.shape(params.embeddings)

    dlogits =
      Enum.zip_with(cache.logits, target_ids, fn row, target_id ->
        [probabilities] = Tensor.softmax([row])

        Enum.zip_with(probabilities, Tensor.one_hot(target_id, vocabulary_size), &-/2)
      end)

    dprojection = Tensor.matmul(Tensor.transpose(cache.norm3.output), dlogits)
    dnorm3 = Tensor.matmul(dlogits, Tensor.transpose(params.projection))

    {dgain3, dblock} = Block.rmsnorm_backward(cache.norm3, params.gain3, dnorm3)
    {block_gradients, dinput} = Block.backward(params, cache.block, dblock)
    {dembeddings, dpositions} = scatter(params, input_ids, dinput)

    Map.merge(block_gradients, %{
      embeddings: dembeddings,
      positions: dpositions,
      gain3: dgain3,
      projection: dprojection
    })
  end

  # Row t of dinput is filed twice: under the token that sat there, and
  # under the position it sat in. Summing either table down its columns
  # gives the identical row, which is the free check.
  defp scatter(params, input_ids, dinput) do
    {vocabulary_size, d_model} = Tensor.shape(params.embeddings)
    {context_length, _d_model} = Tensor.shape(params.positions)
    empty = {Tensor.zeros(vocabulary_size, d_model), Tensor.zeros(context_length, d_model)}

    input_ids
    |> Enum.zip(dinput)
    |> Enum.with_index()
    |> Enum.reduce(empty, fn {{input_id, drow}, position}, {dembeddings, dpositions} ->
      {Tensor.add_row(dembeddings, input_id, drow), Tensor.add_row(dpositions, position, drow)}
    end)
  end
end
