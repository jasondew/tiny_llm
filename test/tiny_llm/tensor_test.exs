defmodule TinyLlm.TensorTest do
  @moduledoc """
  Stage 1c acceptance tests, written from the criteria in docs/build-brief.md.

  These are a specification, not a description of existing code.

  The brief fixes the function names but not every shape. Where it was
  silent, these tests assume:

    * `one_hot/2` takes `(index, size)` and returns one row, so a plain
      list of floats rather than a 1-row matrix.
    * `dot/2` takes two rows and returns a float.
    * `argmax/1` takes one row and returns an integer index.
    * `random/3` takes `(rows, cols, scale)` and draws uniformly from the
      closed interval `-scale..scale`, with `scale` defaulting to 0.02 and
      the draw coming from the process `:rand` state.

  Any of those can be flipped; they are assumptions, not requirements.
  """

  use ExUnit.Case, async: true

  alias TinyLlm.Tensor

  test "builds a zero matrix of the requested shape" do
    assert Tensor.zeros(2, 3) == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
  end

  test "builds an empty matrix when asked for zero rows" do
    assert Tensor.zeros(0, 5) == []
    assert Tensor.random(0, 5) == []
  end

  test "reports the shape as rows and columns" do
    assert Tensor.shape([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) == {2, 3}
  end

  test "draws random entries within plus or minus the scale" do
    entries = 8 |> Tensor.random(8, 0.02) |> List.flatten()

    assert length(entries) == 64
    assert Enum.all?(entries, &(&1 >= -0.02 and &1 <= 0.02))
  end

  test "defaults the random scale to the initialization scale of 0.02" do
    entries = 4 |> Tensor.random(4) |> List.flatten()

    assert length(entries) == 16
    assert Enum.all?(entries, &(&1 >= -0.02 and &1 <= 0.02))
  end

  test "draws the same random matrix twice from the same rand seed" do
    :rand.seed(:exsss, {1, 2, 3})
    first = Tensor.random(4, 4, 0.02)

    :rand.seed(:exsss, {1, 2, 3})
    second = Tensor.random(4, 4, 0.02)

    assert first == second
  end

  test "builds a one hot row" do
    assert Tensor.one_hot(2, 5) == [0.0, 0.0, 1.0, 0.0, 0.0]
  end

  test "transposes rows into columns" do
    assert Tensor.transpose([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) ==
             [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
  end

  test "adds matrices entry by entry" do
    assert Tensor.add([[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]) ==
             [[11.0, 22.0], [33.0, 44.0]]
  end

  test "subtracts matrices entry by entry" do
    assert Tensor.sub([[10.0, 20.0], [30.0, 40.0]], [[1.0, 2.0], [3.0, 4.0]]) ==
             [[9.0, 18.0], [27.0, 36.0]]
  end

  test "multiplies matrices entry by entry" do
    assert Tensor.hadamard([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]) ==
             [[5.0, 12.0], [21.0, 32.0]]
  end

  test "scales every entry by a constant" do
    assert Tensor.scale([[1.0, 2.0], [3.0, 4.0]], 3.0) == [[3.0, 6.0], [9.0, 12.0]]
  end

  test "maps a function over every entry" do
    assert Tensor.map([[1.0, -2.0], [-3.0, 4.0]], &abs/1) == [[1.0, 2.0], [3.0, 4.0]]
  end

  test "takes the dot product of two rows" do
    assert Tensor.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == 32.0
  end

  test "multiplies a 2x3 by a 3x2 into a 2x2" do
    left = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    right = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]

    assert Tensor.matmul(left, right) == [[58.0, 64.0], [139.0, 154.0]]
  end

  test "multiplies non-square shapes in the other order" do
    left = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    right = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]

    assert Tensor.matmul(left, right) ==
             [[27.0, 30.0, 33.0], [61.0, 68.0, 75.0], [95.0, 106.0, 117.0]]
  end

  test "softmaxes each row to sum to one" do
    for row <- Tensor.softmax([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]) do
      assert_in_delta Enum.sum(row), 1.0, 1.0e-12
    end
  end

  test "softmaxes a uniform row into a uniform distribution" do
    assert Tensor.softmax([[0.0, 0.0, 0.0, 0.0]]) == [[0.25, 0.25, 0.25, 0.25]]
  end

  test "softmaxes ordering-preservingly" do
    [row] = Tensor.softmax([[1.0, 3.0, 2.0]])

    assert Enum.at(row, 1) > Enum.at(row, 2)
    assert Enum.at(row, 2) > Enum.at(row, 0)
  end

  test "survives logits of 1000 without overflowing" do
    [row] = Tensor.softmax([[1000.0, 1000.0, 1000.0]])

    assert Enum.all?(row, &is_float/1)
    assert_in_delta Enum.sum(row), 1.0, 1.0e-12

    for probability <- row do
      assert_in_delta probability, 1.0 / 3.0, 1.0e-12
    end
  end

  test "collapses to near certainty on a dominant logit" do
    [row] = Tensor.softmax([[1000.0, 0.0]])

    assert_in_delta Enum.at(row, 0), 1.0, 1.0e-12
    assert_in_delta Enum.at(row, 1), 0.0, 1.0e-12
  end

  test "finds the index of the largest entry in a row" do
    assert Tensor.argmax([0.1, 0.7, 0.2]) == 1
  end

  test "breaks argmax ties toward the first index" do
    assert Tensor.argmax([0.5, 0.5, 0.1]) == 0
  end
end
