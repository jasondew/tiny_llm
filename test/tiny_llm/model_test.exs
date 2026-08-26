defmodule TinyLlm.ModelTest do
  @moduledoc """
  Stage 5 acceptance tests for the whole model, written from the criteria in
  docs/build-brief.md and the derivation in docs/backprop.md section S.

  These are a specification, not a description of existing code.

  Assumptions the brief leaves open:

    * the unembedding is **separate** from the embedding table, not tied.
      The brief asks for a choice and for it to be documented; see the
      Model moduledoc for why an untied table makes the stage 7 PCA
      picture mean one thing instead of two.
    * params are one flat map: the two tables, the block's ten, the final
      gain, and the projection. Flat because `Train.step/4` iterates it and
      knows nothing about structure.
    * `examples/1` is identical to `Attention.examples/1`. The training
      problem did not change when the architecture did.
  """

  use ExUnit.Case, async: true

  alias TinyLlm.Attention
  alias TinyLlm.GradCheck
  alias TinyLlm.Grammar
  alias TinyLlm.Model
  alias TinyLlm.Tensor
  alias TinyLlm.Train
  alias TinyLlm.Vocab

  @config %Train.Config{model: Model, vocabulary_size: 32, d_model: 32, context_length: 16}
  @tiny %Train.Config{model: Model, vocabulary_size: 6, d_model: 8, context_length: 4}

  @keys [
    :bias1,
    :bias2,
    :embeddings,
    :gain1,
    :gain2,
    :gain3,
    :key_weight,
    :output_weight,
    :positions,
    :projection,
    :query_weight,
    :value_weight,
    :weight1,
    :weight2
  ]

  @batch [
    {[0, 3, 1, 3], [3, 1, 3, 5]},
    {[2, 2, 4], [2, 4, 5]},
    {[1], [5]}
  ]

  defp tiny_params do
    Train.seed(4)
    Model.init(@tiny)
  end

  describe "init/1" do
    test "builds one flat map of every parameter in the model" do
      # Flat because Train.step/4 iterates it and knows nothing about which
      # layer a matrix belongs to. Nesting would mean teaching it.
      assert Map.keys(tiny_params()) |> Enum.sort() == @keys
    end

    test "shapes each matrix from the config" do
      params = tiny_params()

      assert Tensor.shape(params.embeddings) == {6, 8}
      assert Tensor.shape(params.positions) == {4, 8}
      assert Tensor.shape(params.gain3) == {1, 8}
      assert Tensor.shape(params.projection) == {8, 6}
    end

    test "keeps the unembedding separate from the embedding table" do
      # Tying them would make projection == transpose(embeddings) and save
      # 1024 parameters. The brief asks for a choice; this is it, and the
      # reason is the stage 7 PCA picture.
      params = Model.init(@config)

      refute params.projection == Tensor.transpose(params.embeddings)
    end

    test "is the size the brief budgeted for" do
      # 1024 embeddings + 512 positions + 4096 attention + 8192 MLP
      # + 1024 unembedding + 128 bias1 + 32 bias2 + 96 gains.
      count =
        Model.init(@config)
        |> Map.values()
        |> Enum.map(fn matrix -> matrix |> List.flatten() |> length() end)
        |> Enum.sum()

      assert count == 15_104
    end

    test "draws the same parameters twice from the same seed" do
      Train.seed(7)
      first = Model.init(@config)

      Train.seed(7)
      second = Model.init(@config)

      assert first == second
    end
  end

  describe "examples/1" do
    test "builds the same examples the Attention model trains on" do
      Grammar.seed(3)
      corpus = Grammar.corpus(20)

      assert Model.examples(corpus) == Attention.examples(corpus)
    end
  end

  describe "forward/2" do
    setup do
      {:ok, params: tiny_params(), cache: Model.forward(tiny_params(), [0, 3, 1, 3])}
    end

    test "produces one row of logits per position", %{cache: cache} do
      assert Tensor.shape(cache.logits) == {4, 6}
    end

    test "keeps the block's cache so the backward pass can walk back through it", %{cache: cache} do
      assert Map.has_key?(cache, :block)
      assert Tensor.shape(cache.block.output) == {4, 8}
    end

    test "normalizes once more before the unembedding", %{cache: cache} do
      # Without this the residual stream reaches the projection at whatever
      # scale training left it at.
      for row <- cache.norm3.normalized do
        mean_square = Enum.sum(Enum.map(row, &(&1 * &1))) / length(row)

        assert_in_delta :math.sqrt(mean_square), 1.0, 1.0e-6
      end
    end

    test "refuses a sequence longer than the position table", %{params: params} do
      assert_raise ArgumentError, fn -> Model.forward(params, [0, 1, 2, 3, 4]) end
    end

    test "still cannot see the future", %{params: params} do
      # The residual and the MLP are both position-wise, so wrapping the head
      # cannot have leaked anything. Cheap to assert, catastrophic to lose.
      for {row, query_position} <- Enum.with_index(Model.weights(params, [0, 3, 1, 3])),
          {weight, key_position} <- Enum.with_index(row),
          key_position > query_position do
        assert weight === 0.0
      end
    end
  end

  describe "loss/2" do
    test "an untrained model pays about ln(vocabulary_size) per token" do
      Train.seed(1)
      params = Model.init(@config)
      examples = Model.examples(Grammar.corpus(50))

      assert_in_delta Model.loss(params, examples), :math.log(32), 0.3
    end

    test "averages over tokens, not over sequences" do
      params = tiny_params()
      long = Enum.at(@batch, 0)
      short = Enum.at(@batch, 2)

      {_inputs, long_targets} = long
      {_inputs, short_targets} = short

      long_count = length(long_targets)
      short_count = length(short_targets)

      weighted =
        (Model.loss(params, [long]) * long_count + Model.loss(params, [short]) * short_count) /
          (long_count + short_count)

      assert_in_delta Model.loss(params, [long, short]), weighted, 1.0e-12
    end
  end

  describe "gradients/2" do
    test "returns a gradient for every parameter, keyed exactly like the params" do
      assert Map.keys(Model.gradients(tiny_params(), @batch)) |> Enum.sort() == @keys
    end

    test "every gradient has the shape of the thing it is the gradient of" do
      params = tiny_params()
      gradients = Model.gradients(params, @batch)

      for key <- @keys do
        assert Tensor.shape(Map.fetch!(gradients, key)) == Tensor.shape(Map.fetch!(params, key))
      end
    end

    test "leaves rows of the embedding table alone when their token is absent" do
      gradients = Model.gradients(tiny_params(), @batch)

      assert Enum.at(gradients.embeddings, 5) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    end

    test "the embedding and position tables receive the same total gradient" do
      gradients = Model.gradients(tiny_params(), @batch)

      embedding_totals = gradients.embeddings |> Tensor.transpose() |> Enum.map(&Enum.sum/1)
      position_totals = gradients.positions |> Tensor.transpose() |> Enum.map(&Enum.sum/1)

      for {from_embeddings, from_positions} <- Enum.zip(embedding_totals, position_totals) do
        assert_in_delta from_embeddings, from_positions, 1.0e-12
      end
    end
  end

  describe "the gradient check" do
    test "every parameter matrix agrees with finite differences" do
      # The brief's stage 5 acceptance criterion: a full-model check on a
      # tiny config, covering all fourteen parameters at once.
      errors = GradCheck.check(Model, tiny_params(), @batch)

      for {key, error} <- errors do
        assert error < 1.0e-3, "#{key} disagrees by #{error}"
      end
    end

    test "and the check is capable of failing, which is not automatic" do
      # GradCheck divides by max(|analytic| + |numeric|, guard), so gradients
      # far below the guard report agreement whatever the derivation says.
      # Stage 4 found this the hard way; assert the gradients are real before
      # trusting the test above.
      gradients = Model.gradients(tiny_params(), @batch)

      for key <- @keys do
        largest = gradients |> Map.fetch!(key) |> List.flatten() |> Enum.map(&abs/1) |> Enum.max()

        assert largest > 1_000 * GradCheck.guard(),
               "#{key}'s largest gradient is #{largest}, too close to the guard to check"
      end
    end
  end

  describe "training" do
    test "the loss falls on a tiny config, which the gradient check alone does not prove" do
      # A correct gradient and a model that learns are different claims: the
      # stage 4 model had the first without the second for 500 steps.
      Train.seed(9)
      params = Model.init(@tiny)
      batch = @batch

      before = Model.loss(params, batch)

      trained =
        Enum.reduce(1..30, params, fn _step, params ->
          Train.step(Model, params, batch, 0.5)
        end)

      assert Model.loss(trained, batch) < before - 0.5
    end
  end
end
