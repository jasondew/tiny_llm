defmodule TinyLlm.BlockTest do
  @moduledoc """
  Stage 5 acceptance tests for the block, written from the criteria in
  docs/build-brief.md and the derivation in docs/backprop.md sections N
  through S.

  These are a specification, not a description of existing code.

  The brief fixes the architecture but not every shape, so these tests
  assume:

    * pre-norm: `rmsnorm` before each sublayer, residual around each, and a
      block that is the identity when both sublayer outputs are zero.
    * the block's params are `%{g1:, wq:, wk:, wv:, wo:, g2:, w1:, b1:,
      w2:, b2:}`. Gains and biases are single-row matrices rather than bare
      lists, so `Train.step/4` keeps working on them unchanged.
    * `rmsnorm/2` returns `normalized` and `rms` alongside `output`, because
      the backward pass needs both and recomputing is where they drift.
    * `backward/3` returns `{gradients, dinput}` rather than folding the
      input gradient into the map, since `dinput` is not a parameter.

  The finite-difference checks here are hand-rolled rather than going
  through `GradCheck`, because a block has no loss of its own. They pick a
  fixed random probe matrix `p` and differentiate `sum(output * p)`, whose
  gradient with respect to the output is exactly `p`.
  """

  use ExUnit.Case, async: true

  alias TinyLlm.Block
  alias TinyLlm.Tensor
  alias TinyLlm.Train

  @tiny %Train.Config{vocabulary_size: 6, d_model: 8, context_length: 4}

  @epsilon 1.0e-5

  defp tiny_params do
    Train.seed(5)
    Block.init(@tiny)
  end

  defp probe(rows, columns) do
    Train.seed(11)
    Tensor.random(rows, columns, 1.0)
  end

  # sum(m * p), the scalar whose gradient with respect to m is p.
  defp scalar(matrix, probe) do
    Enum.zip_reduce(matrix, probe, 0.0, fn row, probe_row, total ->
      total + Tensor.dot(row, probe_row)
    end)
  end

  defp perturb(matrix, row_index, column_index, delta) do
    List.update_at(matrix, row_index, fn row ->
      List.update_at(row, column_index, &(&1 + delta))
    end)
  end

  # (f(x+e) - f(x-e)) / 2e, entry by entry.
  defp numeric_gradient(matrix, fun) do
    for {row, row_index} <- Enum.with_index(matrix) do
      for {_entry, column_index} <- Enum.with_index(row) do
        up = fun.(perturb(matrix, row_index, column_index, @epsilon))
        down = fun.(perturb(matrix, row_index, column_index, -@epsilon))

        (up - down) / (2 * @epsilon)
      end
    end
  end

  defp max_difference(left, right) do
    Enum.zip_reduce(List.flatten(left), List.flatten(right), 0.0, fn a, b, worst ->
      max(worst, abs(a - b))
    end)
  end

  describe "hidden_width/1" do
    test "is four times d_model, which is 128 at the brief's 32" do
      assert Block.hidden_width(32) == 128
      assert Block.hidden_width(8) == 32
    end
  end

  describe "init/1" do
    test "builds the ten matrices the block needs" do
      assert Map.keys(tiny_params()) |> Enum.sort() ==
               [:b1, :b2, :g1, :g2, :w1, :w2, :wk, :wo, :wq, :wv]
    end

    test "shapes each matrix from the config" do
      params = tiny_params()

      assert Tensor.shape(params.g1) == {1, 8}
      assert Tensor.shape(params.g2) == {1, 8}
      assert Tensor.shape(params.wq) == {8, 8}
      assert Tensor.shape(params.wk) == {8, 8}
      assert Tensor.shape(params.wv) == {8, 8}
      assert Tensor.shape(params.wo) == {8, 8}
      assert Tensor.shape(params.w1) == {8, 32}
      assert Tensor.shape(params.b1) == {1, 32}
      assert Tensor.shape(params.w2) == {32, 8}
      assert Tensor.shape(params.b2) == {1, 8}
    end

    test "starts both gains at exactly one" do
      # A gain of 1 makes RMSNorm start as pure normalization. Randomizing a
      # knob whose neutral position is known only adds noise.
      params = tiny_params()

      assert List.flatten(params.g1) == List.duplicate(1.0, 8)
      assert List.flatten(params.g2) == List.duplicate(1.0, 8)
    end

    test "starts both biases at exactly zero" do
      params = tiny_params()

      assert List.flatten(params.b1) == List.duplicate(0.0, 32)
      assert List.flatten(params.b2) == List.duplicate(0.0, 8)
    end

    test "scales the MLP matrices by their own fan in and fan out" do
      # w1 is 8x32 and w2 is 32x8, so they share a fan sum and a scale, but
      # neither shares it with the square attention matrices.
      params = Block.init(@tiny)
      expected = :math.sqrt(6 / (8 + 32))

      for key <- [:w1, :w2] do
        largest = params |> Map.fetch!(key) |> List.flatten() |> Enum.map(&abs/1) |> Enum.max()

        assert largest <= expected, "#{key} drew outside its fan scale"
        assert largest > expected * 0.9, "#{key} is not filling its fan scale"
      end
    end

    test "draws the same parameters twice from the same seed" do
      Train.seed(7)
      first = Block.init(@tiny)

      Train.seed(7)
      second = Block.init(@tiny)

      assert first == second
    end
  end

  describe "rmsnorm/2" do
    setup do
      input = [[3.0, 4.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
      ones = [[1.0, 1.0, 1.0, 1.0]]

      {:ok, input: input, ones: ones, cache: Block.rmsnorm(input, ones)}
    end

    test "gives every row a root mean square of one", %{cache: cache} do
      for row <- cache.normalized do
        mean_square = Enum.sum(Enum.map(row, &(&1 * &1))) / length(row)

        assert_in_delta :math.sqrt(mean_square), 1.0, 1.0e-6
      end
    end

    test "reports the divisor it used", %{cache: cache} do
      # Row one is [3,4,0,0]: mean square 25/4, so rms 2.5.
      [first, second] = cache.rms

      assert_in_delta first, 2.5, 1.0e-6
      assert_in_delta second, 1.0, 1.0e-6
    end

    test "rescales without recentring, unlike LayerNorm", %{cache: cache} do
      # [1,1,1,1] has a nonzero mean, and RMSNorm leaves it alone. LayerNorm
      # would return four zeros here, which is the whole difference.
      assert Enum.at(cache.normalized, 1) == [1.0, 1.0, 1.0, 1.0]
    end

    test "applies the gain per column", %{input: input} do
      gain = [[2.0, 0.5, 1.0, 1.0]]
      cache = Block.rmsnorm(input, gain)
      [scaled_first | _] = cache.output
      [normalized_first | _] = cache.normalized

      assert_in_delta Enum.at(scaled_first, 0), Enum.at(normalized_first, 0) * 2.0, 1.0e-9
      assert_in_delta Enum.at(scaled_first, 1), Enum.at(normalized_first, 1) * 0.5, 1.0e-9
    end

    test "survives a row of exact zeros", %{ones: ones} do
      # Without the epsilon this divides zero by zero and every entry is NaN,
      # which then poisons every parameter through the backward pass.
      cache = Block.rmsnorm([[0.0, 0.0, 0.0, 0.0]], ones)

      assert cache.output == [[0.0, 0.0, 0.0, 0.0]]
    end

    test "is scale invariant, which is the point of normalizing", %{input: input, ones: ones} do
      # Ten times the input gives the same normalized rows. That invariance
      # is why the residual stream cannot blow the block up.
      quiet = Block.rmsnorm(input, ones)
      loud = Block.rmsnorm(Tensor.scale(input, 10.0), ones)

      assert max_difference(quiet.normalized, loud.normalized) < 1.0e-9
    end
  end

  describe "rmsnorm_backward/3" do
    setup do
      input = [[3.0, 4.0, 0.5, 2.0], [1.0, 1.0, 1.0, 2.0], [0.25, 3.0, 1.0, 0.5]]
      gain = [[1.3, 0.7, 1.0, 2.1]]
      cache = Block.rmsnorm(input, gain)
      doutput = probe(3, 4)

      {:ok, input: input, gain: gain, cache: cache, doutput: doutput}
    end

    test "returns a gain gradient shaped like the gain and an input gradient shaped like the input",
         %{cache: cache, gain: gain, doutput: doutput} do
      {dgain, dinput} = Block.rmsnorm_backward(cache, gain, doutput)

      assert Tensor.shape(dgain) == {1, 4}
      assert Tensor.shape(dinput) == {3, 4}
    end

    test "every row of dinput is orthogonal to the same row of the normalized input", %{
      cache: cache,
      gain: gain,
      doutput: doutput
    } do
      # The free check from docs/backprop.md (O2). Sum of n squared is d, so
      # the subtracted term cancels exactly. Costs one dot product per row
      # and catches a dropped or mis-scaled c immediately.
      {_dgain, dinput} = Block.rmsnorm_backward(cache, gain, doutput)

      for {drow, nrow} <- Enum.zip(dinput, cache.normalized) do
        assert_in_delta Tensor.dot(drow, nrow), 0.0, 1.0e-6
      end
    end

    test "the gain gradient matches finite differences", %{
      input: input,
      gain: gain,
      doutput: doutput
    } do
      {dgain, _dinput} = Block.rmsnorm_backward(Block.rmsnorm(input, gain), gain, doutput)

      numeric =
        numeric_gradient(gain, fn perturbed ->
          scalar(Block.rmsnorm(input, perturbed).output, doutput)
        end)

      assert max_difference(dgain, numeric) < 1.0e-6
    end

    test "the input gradient matches finite differences", %{
      input: input,
      gain: gain,
      doutput: doutput
    } do
      {_dgain, dinput} = Block.rmsnorm_backward(Block.rmsnorm(input, gain), gain, doutput)

      numeric =
        numeric_gradient(input, fn perturbed ->
          scalar(Block.rmsnorm(perturbed, gain).output, doutput)
        end)

      assert max_difference(dinput, numeric) < 1.0e-6
    end

    test "accumulates the gain gradient over every row, not just the first" do
      # One gain vector is shared by every position, so a backward pass that
      # forgets to sum down the rows is off by a factor that grows with T.
      gain = [[1.0, 1.0]]
      input = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
      doutput = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]

      {dgain, _dinput} = Block.rmsnorm_backward(Block.rmsnorm(input, gain), gain, doutput)

      # Column 0 of normalized is [sqrt(2), 0, 1], and doutput selects it in
      # every row, so dgain[0] is their sum.
      assert_in_delta dgain |> List.first() |> Enum.at(0), :math.sqrt(2) + 0.0 + 1.0, 1.0e-6
    end
  end

  describe "forward/2" do
    setup do
      input = probe(3, 8)

      {:ok, params: tiny_params(), input: input, cache: Block.forward(tiny_params(), input)}
    end

    test "returns an output the same shape as its input", %{cache: cache} do
      assert Tensor.shape(cache.output) == {3, 8}
    end

    test "widens through the MLP and comes back", %{cache: cache} do
      assert Tensor.shape(cache.hidden) == {3, 32}
      assert Tensor.shape(cache.mlp) == {3, 8}
    end

    test "the ReLU leaves nothing negative", %{cache: cache} do
      assert Enum.all?(List.flatten(cache.hidden), &(&1 >= 0.0))
    end

    test "the ReLU actually fires, so the test above is not vacuous", %{cache: cache} do
      assert Enum.any?(List.flatten(cache.pre), &(&1 < 0.0))
    end

    test "is the identity when both sublayers output zero", %{input: input} do
      # Zero wo and w2/b2 and the two sublayers contribute nothing, so the
      # residuals must hand the input straight through. This is the sharpest
      # test of the residual wiring there is: nothing else in the block can
      # make it pass.
      params = tiny_params()

      silenced = %{
        params
        | wo: Tensor.zeros(8, 8),
          w2: Tensor.zeros(32, 8),
          b2: Tensor.zeros(1, 8)
      }

      assert max_difference(Block.forward(silenced, input).output, input) < 1.0e-12
    end
  end

  describe "backward/3" do
    setup do
      input = probe(3, 8)
      params = tiny_params()
      cache = Block.forward(params, input)
      doutput = Tensor.random(3, 8, 1.0)

      {:ok, params: params, input: input, cache: cache, doutput: doutput}
    end

    test "returns a gradient for every block parameter, keyed like the params", %{
      params: params,
      cache: cache,
      doutput: doutput
    } do
      {gradients, _dinput} = Block.backward(params, cache, doutput)

      assert Enum.sort(Map.keys(gradients)) == Enum.sort(Map.keys(params))
    end

    test "every gradient has the shape of the thing it is the gradient of", %{
      params: params,
      cache: cache,
      doutput: doutput
    } do
      {gradients, dinput} = Block.backward(params, cache, doutput)

      for {key, param} <- params do
        assert Tensor.shape(Map.fetch!(gradients, key)) == Tensor.shape(param)
      end

      assert Tensor.shape(dinput) == {3, 8}
    end

    test "passes the output gradient straight through when both sublayers are silenced", %{
      input: input,
      doutput: doutput
    } do
      # The counterpart of the identity test above. With no sublayer
      # contribution the only surviving path is the residual, which is a
      # bare addition, so dinput must equal doutput exactly. Drop either
      # residual term in the backward pass and this fails while almost
      # everything else still passes.
      params = tiny_params()

      silenced = %{
        params
        | wo: Tensor.zeros(8, 8),
          w2: Tensor.zeros(32, 8),
          b2: Tensor.zeros(1, 8)
      }

      cache = Block.forward(silenced, input)
      {_gradients, dinput} = Block.backward(silenced, cache, doutput)

      assert max_difference(dinput, doutput) < 1.0e-12
    end

    test "the input gradient matches finite differences", %{
      params: params,
      input: input,
      doutput: doutput
    } do
      {_gradients, dinput} = Block.backward(params, Block.forward(params, input), doutput)

      numeric =
        numeric_gradient(input, fn perturbed ->
          scalar(Block.forward(params, perturbed).output, doutput)
        end)

      assert max_difference(dinput, numeric) < 1.0e-5
    end

    test "every parameter gradient matches finite differences", %{
      params: params,
      input: input,
      doutput: doutput
    } do
      {gradients, _dinput} = Block.backward(params, Block.forward(params, input), doutput)

      for {key, param} <- params do
        numeric =
          numeric_gradient(param, fn perturbed ->
            scalar(Block.forward(Map.put(params, key, perturbed), input).output, doutput)
          end)

        assert max_difference(Map.fetch!(gradients, key), numeric) < 1.0e-5,
               "#{key} disagrees with finite differences"
      end
    end

    test "the biases receive the column sums of what reaches them", %{
      params: params,
      input: input
    } do
      # A bias is added identically to every row, so its gradient is the sum
      # down the rows. Summing down the wrong axis produces a shape error
      # only when T happens to differ from d, which it does not here.
      doutput = [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        List.duplicate(0.0, 8),
        List.duplicate(0.0, 8)
      ]

      cache = Block.forward(params, input)
      {gradients, _dinput} = Block.backward(params, cache, doutput)

      assert Tensor.shape(gradients.b2) == {1, 8}
      refute List.flatten(gradients.b2) == List.duplicate(0.0, 8)
    end
  end
end
