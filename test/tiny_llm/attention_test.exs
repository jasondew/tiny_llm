defmodule TinyLlm.AttentionTest do
  @moduledoc """
  Stage 4 acceptance tests, written from the criteria in docs/build-brief.md
  and the derivation in docs/backprop.md.

  These are a specification, not a description of existing code.

  The brief fixes the architecture but not every shape, so these tests
  assume:

    * params are the seven matrices `%{embeddings:, positions:, query_weight:, key_weight:,
      value_weight:, output_weight:, projection:}`. `embeddings` and `projection` keep the names
      and roles they had in stage 3; `positions` is new and is what makes
      this the first model in the project that can tell word order apart.
    * an example is `{input_ids, target_ids}`, two equal length lists, one
      per sentence. Sequences are not padded to a fixed width. A short
      sentence is a short sequence, and `T` varies from example to example.
    * `forward/2` returns a cache holding every intermediate the backward
      pass needs, including `:weights` (the `T` by `T` attention matrix) and
      `:logits`.
    * `loss/2` averages over predicted *tokens*, not over sequences, so the
      number stays comparable to the bigram's 1.9021 nats.

  The gradient check runs on a deliberately larger init scale than training
  would use. See "the gradient check has a hole" in docs/backprop.md: at the
  old default of 0.02 the gradients into `query_weight` and `key_weight` land five orders of
  magnitude below `GradCheck.guard()`, and every bug in the softmax
  Jacobian, the scale and the QK transpose passes. Two tests below exist
  only to make sure that can never quietly happen again.
  """

  use ExUnit.Case, async: true

  alias TinyLlm.Attention
  alias TinyLlm.GradCheck
  alias TinyLlm.Grammar
  alias TinyLlm.Tensor
  alias TinyLlm.Train
  alias TinyLlm.Vocab

  @config %Train.Config{model: Attention, vocabulary_size: 32, d_model: 32, context_length: 16}
  @tiny %Train.Config{model: Attention, vocabulary_size: 6, d_model: 8, context_length: 4}

  @matrices [
    :embeddings,
    :key_weight,
    :output_weight,
    :positions,
    :projection,
    :query_weight,
    :value_weight
  ]

  # Three sequences of different lengths, so a batch exercises the two things
  # a single fixed-width batch cannot: that T varies, and that dP's later
  # rows receive fewer contributions than its earlier ones.
  @batch [
    {[0, 3, 1, 3], [3, 1, 3, 5]},
    {[2, 2, 4], [2, 4, 5]},
    {[1], [5]}
  ]

  defp id(word), do: Vocab.word_to_id(word)

  defp tiny_params do
    Train.seed(4)
    Attention.init(@tiny)
  end

  describe "init/1" do
    test "builds the seven matrices the derivation names" do
      params = tiny_params()

      assert Map.keys(params) |> Enum.sort() == @matrices
    end

    test "shapes each matrix from the config" do
      params = tiny_params()

      assert Tensor.shape(params.embeddings) == {6, 8}
      assert Tensor.shape(params.positions) == {4, 8}
      assert Tensor.shape(params.query_weight) == {8, 8}
      assert Tensor.shape(params.key_weight) == {8, 8}
      assert Tensor.shape(params.value_weight) == {8, 8}
      assert Tensor.shape(params.output_weight) == {8, 8}
      assert Tensor.shape(params.projection) == {8, 6}
    end

    test "gives positions one row per position the context can hold" do
      # Not one row per position in some sentence. The table has to cover the
      # longest sequence the model will ever see, and rows past the longest
      # sentence in a batch simply receive no gradient.
      assert Tensor.shape(Attention.init(@config).positions) == {16, 32}
    end

    test "draws the same parameters twice from the same seed" do
      Train.seed(7)
      first = Attention.init(@config)

      Train.seed(7)
      second = Attention.init(@config)

      assert first == second
    end

    test "scales each matrix by its own fan in and fan out, not by one constant" do
      # The stage 3 default of 0.02 makes this model untrainable: the score
      # path is four factors deep, dWq lands at 1.1e-12, and the loss sits on
      # ln(32) unchanged for 500 steps. Scale has to come from the shape.
      params = Attention.init(@config)

      for {key, {fan_in, fan_out}} <- [
            embeddings: {32, 32},
            positions: {16, 32},
            query_weight: {32, 32},
            projection: {32, 32}
          ] do
        expected = :math.sqrt(6 / (fan_in + fan_out))
        largest = params |> Map.fetch!(key) |> List.flatten() |> Enum.map(&abs/1) |> Enum.max()

        assert largest <= expected, "#{key} drew outside its fan scale"
        assert largest > expected * 0.9, "#{key} is not filling its fan scale"
      end
    end
  end

  describe "examples/1" do
    test "turns a sentence into inputs and the same sentence shifted by one" do
      assert Attention.examples([~w(the llama flees .)]) == [
               {
                 [id(Vocab.start_token()), id("the"), id("llama"), id("flees")],
                 [id("the"), id("llama"), id("flees"), id(".")]
               }
             ]
    end

    test "predicts every token of the sentence, including the first and the period" do
      [{inputs, targets}] = Attention.examples([~w(a dog sees a goose .)])

      assert length(inputs) == length(targets)
      assert List.first(inputs) == id(Vocab.start_token())
      assert List.last(targets) == id(".")
    end

    test "keeps one example per sentence, unlike the Embedder's one per token" do
      # The whole point of stage 4: a sequence is the unit, because every
      # position can see every earlier position.
      assert length(Attention.examples(Grammar.corpus(50))) == 50
    end

    test "never builds a sequence longer than the context the brief fixes" do
      # The grammar's longest sentence is 16 tokens, which with <start>
      # prepended is exactly 16 predicted positions. That is a knife edge
      # worth pinning: one more and the position table would be indexed past
      # its last row.
      Grammar.seed(7)

      longest =
        Grammar.corpus(20_000)
        |> Attention.examples()
        |> Enum.map(fn {inputs, _targets} -> length(inputs) end)
        |> Enum.max()

      assert longest <= @config.context_length
    end
  end

  describe "forward/2" do
    setup do
      {:ok, params: tiny_params(), cache: Attention.forward(tiny_params(), [0, 3, 1, 3])}
    end

    test "produces one row of logits per position", %{cache: cache} do
      assert Tensor.shape(cache.logits) == {4, 6}
    end

    test "produces a square attention matrix, one row and column per position", %{cache: cache} do
      assert Tensor.shape(cache.weights) == {4, 4}
    end

    test "every row of attention weights is a distribution", %{cache: cache} do
      for row <- cache.weights do
        assert_in_delta Enum.sum(row), 1.0, 1.0e-12
      end
    end

    test "no position attends to the future, and not merely a little", %{cache: cache} do
      # exp(-1.0e9) underflows to exactly 0.0, which is what makes the mask
      # free in the backward pass too. A "large enough looking" -100.0 would
      # leave 3.7e-44 here and leak gradient into the past forever.
      for {row, query_position} <- Enum.with_index(cache.weights),
          {weight, key_position} <- Enum.with_index(row),
          key_position > query_position do
        assert weight === 0.0
      end
    end

    test "the first position can only attend to itself", %{cache: cache} do
      assert List.first(cache.weights) == [1.0, 0.0, 0.0, 0.0]
    end

    test "attention depends on position, not only on the tokens", %{params: params} do
      # The same token in two places must be able to behave differently,
      # otherwise the position embeddings are doing nothing and the model is
      # a bag of words with extra steps.
      forwards = Attention.forward(params, [3, 3])

      [first, second] = forwards.logits

      refute first == second
    end

    test "shortens gracefully: a one token sequence is a one by one matrix", %{params: params} do
      assert Tensor.shape(Attention.forward(params, [2]).weights) == {1, 1}
    end

    test "refuses a sequence longer than the position table", %{params: params} do
      assert_raise ArgumentError, fn -> Attention.forward(params, [0, 1, 2, 3, 4]) end
    end
  end

  describe "weights/2" do
    test "hands back the attention matrix on its own, for the stage 7 heatmap" do
      params = tiny_params()

      assert Attention.weights(params, [0, 3, 1]) == Attention.forward(params, [0, 3, 1]).weights
    end
  end

  describe "loss/2" do
    test "an untrained model pays about ln(vocabulary_size) per token" do
      Train.seed(1)
      params = Attention.init(@config)
      examples = Attention.examples(Grammar.corpus(50))

      assert_in_delta Attention.loss(params, examples), :math.log(32), 0.25
    end

    test "averages over tokens, not over sequences" do
      # A four token sentence and a one token sentence do not carry equal
      # weight. Getting this wrong still produces a falling loss curve, but
      # the number stops being comparable to the bigram's 1.9021 nats and
      # the one measurement this project is built around means nothing.
      params = tiny_params()
      long = Enum.at(@batch, 0)
      short = Enum.at(@batch, 2)

      {_inputs, long_targets} = long
      {_inputs, short_targets} = short

      long_count = length(long_targets)
      short_count = length(short_targets)

      weighted =
        (Attention.loss(params, [long]) * long_count +
           Attention.loss(params, [short]) * short_count) / (long_count + short_count)

      assert_in_delta Attention.loss(params, [long, short]), weighted, 1.0e-12
    end
  end

  describe "gradients/2" do
    test "returns a gradient for every parameter, keyed exactly like the params" do
      params = tiny_params()

      assert Map.keys(Attention.gradients(params, @batch)) |> Enum.sort() == @matrices
    end

    test "every gradient has the shape of the thing it is the gradient of" do
      params = tiny_params()
      gradients = Attention.gradients(params, @batch)

      for key <- @matrices do
        assert Tensor.shape(Map.fetch!(gradients, key)) == Tensor.shape(Map.fetch!(params, key))
      end
    end

    test "leaves rows of the embedding table alone when their token is absent" do
      # Token 5 never appears as an input in @batch, only as a target, so no
      # path from it to the loss exists and its row is exactly zero.
      gradients = Attention.gradients(tiny_params(), @batch)

      assert Enum.at(gradients.embeddings, 5) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    end

    test "the embedding and position tables receive the same total gradient" do
      # From (M) in docs/backprop.md: both tables are scatter adds of the
      # same rows of dX, filed under different indices, so summing each down
      # its columns has to give the identical row. This catches a scatter add
      # that dropped a position or counted one twice, and it costs nothing.
      gradients = Attention.gradients(tiny_params(), @batch)

      embedding_totals = gradients.embeddings |> Tensor.transpose() |> Enum.map(&Enum.sum/1)
      position_totals = gradients.positions |> Tensor.transpose() |> Enum.map(&Enum.sum/1)

      for {from_embeddings, from_positions} <- Enum.zip(embedding_totals, position_totals) do
        assert_in_delta from_embeddings, from_positions, 1.0e-12
      end
    end
  end

  describe "the gradient check" do
    test "every parameter matrix agrees with finite differences" do
      # The brief's stage 4 acceptance criterion, on its 4 token, d = 8
      # config. T = 4 rather than 1 on purpose: at T = 1 the mask is empty,
      # the attention matrix is the scalar 1.0, and the softmax Jacobian
      # reports zero no matter what was written.
      errors = GradCheck.check(Attention, tiny_params(), @batch)

      for {key, error} <- errors do
        assert error < 1.0e-3, "#{key} disagrees by #{error}"
      end
    end

    test "and the check is capable of failing, which is not automatic" do
      # GradCheck divides by max(|analytic| + |numeric|, guard). When both
      # gradients sit far below the guard it reports agreement whatever the
      # derivation says. At the stage 3 init scale of 0.02 that is exactly
      # what happens to query_weight and key_weight, and a swapped transpose, a dropped r_t
      # and a missing sqrt(d) all pass. Assert the gradients are real before
      # trusting the test above.
      gradients = Attention.gradients(tiny_params(), @batch)

      for key <- @matrices do
        largest = gradients |> Map.fetch!(key) |> List.flatten() |> Enum.map(&abs/1) |> Enum.max()

        assert largest > 1_000 * GradCheck.guard(),
               "#{key}'s largest gradient is #{largest}, too close to the guard to check"
      end
    end
  end
end
