defmodule TinyLlm.Attention do
  @moduledoc """
  A single causal self-attention head, and the first model here that can
  read more than one word.

  TODO(stage 4): write the concept paragraph. It should say that the bigram
  and the Embedder both answer "what usually follows this word", and that
  every question worth asking about language needs more: `the llama who
  chases the dogs ___` needs the verb to agree with `llama`, four words
  back, and not with `dogs`, which is adjacent. Attention is the mechanism
  for reaching back. Every position emits a query, every position emits a
  key, and the softmax over their dot products decides how much each
  position gets to draw on each earlier one.

  ## Forward

      X    = embeddings[input_ids] + positions[0..T-1]   T x d
      Q    = X * query_weight       K = X * key_weight       V = X * value_weight    T x d
      U    = Q * Kᵀ                                      T x T
      S    = U / sqrt(d)
      M    = S with -1.0e9 added above the diagonal
      A    = softmax of each row of M                    T x T
      C    = A * V                                       T x d
      O    = C * output_weight                                      T x d
      Z    = O * projection                              T x v

  Three details do all the work. **Positions are added, not concatenated**,
  which is why `positions` is `d` wide rather than something extra. **The
  mask goes on before the softmax**, so rows still sum to 1; masking after
  would leave rows that do not. And **the scores are divided by sqrt(d)**,
  which holds their variance fixed as `d` grows and keeps the softmax from
  saturating.

  ## Backward

  Derived in full in `docs/backprop.md`, which the brief requires be written
  before this file is. The short version, in the order it runs:

      dZ = (p - onehot(y)) / N          N is total tokens, not sentences
      dprojection = Oᵀ * dZ             dO = dZ * projectionᵀ
      doutput_weight = Cᵀ * dO                     dC = dO * woᵀ
      dA = dC * Vᵀ                      dV = Aᵀ * dC
      dM[t][j] = A[t][j] * (dA[t][j] - dot(dA[t], A[t]))      (the Jacobian)
      dS = dM                           the mask is additive, so it vanishes
      dU = dS / sqrt(d)
      dQ = dU * K                       dK = dUᵀ * Q          (mind the ᵀ)
      dquery_weight = Xᵀ * dQ    dkey_weight = Xᵀ * dK    dvalue_weight = Xᵀ * dV
      dX = dQ*wqᵀ + dK*wkᵀ + dV*wvᵀ     three paths, summed
      dembeddings[a_t] += dX[t]         dpositions[t] += dX[t]

  Two lines deserve suspicion while implementing. `dK = dUᵀ * Q` has the
  same shape as `dU * Q`, so nothing but the gradient check will catch a
  missing transpose. And `dX` is a sum of three terms; drop one and two
  thirds of the gradient is still correct, the loss still falls, and only
  `embeddings` and `positions` fail the check.

  ## Why the init scale is a calculation

  Stage 3 drew every matrix from `-0.02..0.02` and trained fine, because its
  path was two factors deep. The score path here is four factors deep before
  the gradient turns around, and at 0.02 the gradient reaching `query_weight` is
  1.1e-12: the loss sits on `ln(32)` unchanged for 500 steps, with a
  completely correct backward pass underneath it. Scale each matrix by its
  own shape instead, `sqrt(6 / (fan_in + fan_out))`, and the same code
  crosses the bigram floor of 1.9021 nats around step 120.
  """

  @behaviour TinyLlm.Train

  alias TinyLlm.Grammar
  alias TinyLlm.Tensor
  alias TinyLlm.Train
  alias TinyLlm.Vocab

  @typedoc "One sequence: the tokens read, and the tokens to predict."
  @type example :: {[Vocab.id()], [Vocab.id()]}

  @typedoc """
  Everything the head produced. The backward pass reads almost all of it,
  and `:weights` is what the stage 7 heatmap plots.
  """
  @type head_cache :: %{
          input: Tensor.matrix(),
          queries: Tensor.matrix(),
          keys: Tensor.matrix(),
          values: Tensor.matrix(),
          weights: Tensor.matrix(),
          context: Tensor.matrix(),
          output: Tensor.matrix()
        }

  @typedoc "The head's cache, plus the logits `forward/2` adds on the end."
  @type cache :: %{
          input: Tensor.matrix(),
          queries: Tensor.matrix(),
          keys: Tensor.matrix(),
          values: Tensor.matrix(),
          weights: Tensor.matrix(),
          context: Tensor.matrix(),
          output: Tensor.matrix(),
          logits: Tensor.matrix()
        }

  @doc """
  The mask added to scores at positions a query is not allowed to see.

  Large enough that `exp/1` underflows to exactly `0.0`, which is what makes
  the mask cost nothing in the backward pass. A merely large number would
  leave a subnormal weight on the future and leak gradient into the past.
  """
  @spec mask_value() :: float()
  def mask_value, do: -1.0e9

  @doc """
  Random starting parameters: two lookup tables and five matrices.

  `positions` gets one row per position the context can hold, not one per
  position in any particular sentence. Rows past the longest sequence in a
  batch simply receive no gradient.
  """
  @impl Train
  @spec init(Train.Config.t()) :: Train.params()
  def init(config) do
    d = config.d_model
    v = config.vocabulary_size
    c = config.context_length

    %{
      embeddings: Tensor.random(v, d, Tensor.fan_scale(v, d)),
      positions: Tensor.random(c, d, Tensor.fan_scale(c, d)),
      query_weight: Tensor.random(d, d, Tensor.fan_scale(d, d)),
      key_weight: Tensor.random(d, d, Tensor.fan_scale(d, d)),
      value_weight: Tensor.random(d, d, Tensor.fan_scale(d, d)),
      output_weight: Tensor.random(d, d, Tensor.fan_scale(d, d)),
      projection: Tensor.random(d, v, Tensor.fan_scale(d, v))
    }
  end

  @doc """
  A corpus of sentences as `{input_ids, target_ids}` pairs.

  One example per sentence, unlike the Embedder's one per token, because
  here the sequence is the unit: every position predicts the next token
  while seeing every position before it.

  Sequences are not padded. A short sentence is a short sequence.
  """
  @impl Train
  @spec examples([Grammar.sentence()]) :: [example()]
  def examples(corpus) do
    Enum.map(corpus, fn sentence ->
      sentence_with_start_token = [Vocab.start_token() | sentence]
      ids = Enum.map(sentence_with_start_token, &Vocab.word_to_id/1)

      {Enum.drop(ids, -1), Enum.drop(ids, 1)}
    end)
  end

  @doc """
  One forward pass, keeping every intermediate the backward pass needs.

  Raises if the sequence is longer than the position table, which is the
  model saying it has no embedding for that position rather than silently
  reading past the end of the table.
  """
  @spec forward(Train.params(), [Vocab.id()]) :: cache()
  def forward(params, input_ids) do
    if length(input_ids) > length(params.positions) do
      raise ArgumentError, "Sequence too long: #{length(input_ids)} > #{length(params.positions)}"
    end

    embeddings = Enum.map(input_ids, fn id -> Enum.at(params.embeddings, id) end)
    positions = Enum.take(params.positions, length(input_ids))
    input = Tensor.add(embeddings, positions)

    cache = attend(params, input)

    Map.put(cache, :logits, Tensor.matmul(cache.output, params.projection))
  end

  @doc """
  The head on its own, over an arbitrary `T` by `d` matrix.

  `forward/2` is this plus a lookup on the front and an unembedding on the
  back. Stage 5 wraps it in a normalization and a residual instead, and the
  head cannot tell the difference: it takes rows in and gives rows out.

  Nothing here mentions token ids or the vocabulary, which is the point.
  """
  @spec attend(Train.params(), Tensor.matrix()) :: head_cache()
  def attend(params, input) do
    d_model = length(hd(input))

    queries = Tensor.matmul(input, params.query_weight)
    keys = Tensor.matmul(input, params.key_weight)
    values = Tensor.matmul(input, params.value_weight)

    weights =
      queries
      |> Tensor.matmul(Tensor.transpose(keys))
      |> Tensor.scale(1.0 / :math.sqrt(d_model))
      |> Enum.with_index(fn row_values, row ->
        Enum.with_index(row_values, fn score, column ->
          if column > row, do: score + mask_value(), else: score
        end)
      end)
      |> Tensor.softmax()

    context = Tensor.matmul(weights, values)

    %{
      input: input,
      queries: queries,
      keys: keys,
      values: values,
      weights: weights,
      context: context,
      output: Tensor.matmul(context, params.output_weight)
    }
  end

  @doc """
  The gradient of `attend/2`, as `{gradients, dinput}`.

  `gradients` covers the four projections and nothing else. `dinput` is what
  the caller adds to whatever else feeds the head's input: an embedding
  table in stage 4, a residual stream in stage 5.
  """
  @spec attend_backward(Train.params(), head_cache(), Tensor.matrix()) ::
          {Train.params(), Tensor.matrix()}
  def attend_backward(params, cache, doutput) do
    d_model = length(hd(cache.input))

    doutput_weight = Tensor.matmul(Tensor.transpose(cache.context), doutput)
    dcontext = Tensor.matmul(doutput, Tensor.transpose(params.output_weight))

    dweights = Tensor.matmul(dcontext, Tensor.transpose(cache.values))
    dvalues = Tensor.matmul(Tensor.transpose(cache.weights), dcontext)

    # The mask was additive and the masked weights are exactly 0.0, so the
    # mask needs no code here at all: those entries come back exactly 0.0.
    dmasked = softmax_backward(cache.weights, dweights)
    dscores = Tensor.scale(dmasked, 1 / :math.sqrt(d_model))

    # Mind the transpose. `matmul(dscores, queries)` has the same shape and
    # only the gradient check can tell the two apart.
    dqueries = Tensor.matmul(dscores, cache.keys)
    dkeys = Tensor.matmul(Tensor.transpose(dscores), cache.queries)

    input_transposed = Tensor.transpose(cache.input)

    # The input fed all three projections, so all three send gradient back
    # and the three contributions add.
    from_queries = Tensor.matmul(dqueries, Tensor.transpose(params.query_weight))
    from_keys = Tensor.matmul(dkeys, Tensor.transpose(params.key_weight))
    from_values = Tensor.matmul(dvalues, Tensor.transpose(params.value_weight))
    dinput = from_queries |> Tensor.add(from_keys) |> Tensor.add(from_values)

    gradients = %{
      query_weight: Tensor.matmul(input_transposed, dqueries),
      key_weight: Tensor.matmul(input_transposed, dkeys),
      value_weight: Tensor.matmul(input_transposed, dvalues),
      output_weight: doutput_weight
    }

    {gradients, dinput}
  end

  @doc """
  The attention matrix on its own: how much each position drew on each
  earlier one.

  Row `t` is a distribution over positions 0 through `t`, and everything
  above the diagonal is exactly zero. This is the term the stage 7 heatmap
  plots, and the probe sentence `the llama who chases the dogs` is the one
  the talk turns on.
  """
  @spec weights(Train.params(), [Vocab.id()]) :: Tensor.matrix()
  def weights(params, input_ids) do
    forward(params, input_ids).weights
  end

  @doc """
  Mean cross-entropy over a batch, in nats.

  Averaged over predicted **tokens**, not over sequences. Averaging each
  sequence and then averaging those averages weights a short sentence's
  tokens more heavily than a long one's, and the result stops being
  comparable to the bigram's 1.9021 nats.
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

  See the module doc for the chain, and `docs/backprop.md` for where each
  line comes from. Two invariants are worth asserting while debugging, and
  both are cheaper than a finite difference run: every row of `dM` sums to
  zero, and `dembeddings` and `dpositions` have identical column sums.
  """
  @impl Train
  @spec gradients(Train.params(), [example()]) :: Train.params()
  def gradients(params, batch) do
    {summed, token_count} =
      Enum.reduce(batch, {Train.zero_gradients(params), 0}, fn {input_ids, target_ids},
                                                               {totals, count_so_far} ->
        contribution = example_gradients(params, input_ids, target_ids)

        {Train.add_gradients(totals, contribution), count_so_far + length(target_ids)}
      end)

    Map.new(summed, fn {key, gradient} -> {key, Tensor.scale(gradient, 1 / token_count)} end)
  end

  ## PRIVATE FUNCTIONS

  # One example's contribution, keyed like the params. The chain runs in the
  # reverse of the forward order, each step naming both what it produces for
  # a parameter and what it passes further back.
  defp example_gradients(params, input_ids, target_ids) do
    cache = forward(params, input_ids)
    {vocabulary_size, _d_model} = Tensor.shape(params.embeddings)

    # Cross-entropy through softmax collapses to p - onehot, exactly as in
    # the Embedder, one row per predicted position.
    dlogits =
      Enum.zip_with(cache.logits, target_ids, fn row, target_id ->
        [probabilities] = Tensor.softmax([row])

        Enum.zip_with(probabilities, Tensor.one_hot(target_id, vocabulary_size), &-/2)
      end)

    dprojection = Tensor.matmul(Tensor.transpose(cache.output), dlogits)
    doutput = Tensor.matmul(dlogits, Tensor.transpose(params.projection))

    {head_gradients, dinput} = attend_backward(params, cache, doutput)
    {dembeddings, dpositions} = scatter(params, input_ids, dinput)

    Map.merge(head_gradients, %{
      embeddings: dembeddings,
      positions: dpositions,
      projection: dprojection
    })
  end

  # dM[t][j] = A[t][j] * (dA[t][j] - dot(dA[t], A[t]))
  #
  # The subtracted term is one number per row, not per entry. Softmax
  # outputs are coupled, so pushing one up pushes the rest down, and the
  # gradient into each entry has to lose the row's average effect. Every row
  # of the result sums to exactly zero, which is a free check while
  # debugging.
  defp softmax_backward(weights, dweights) do
    Enum.zip_with(weights, dweights, fn weight_row, dweight_row ->
      row_total = Tensor.dot(dweight_row, weight_row)

      Enum.zip_with(weight_row, dweight_row, fn weight, dweight ->
        weight * (dweight - row_total)
      end)
    end)
  end

  # Row t of dinput is filed twice: under the token that sat there, and
  # under the position it sat in. Adding, never assigning, because a token
  # can appear more than once in one sequence. Summing either table down its
  # columns gives the identical row, which is the other free check.
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
