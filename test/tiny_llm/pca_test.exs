defmodule TinyLlm.PCATest do
  @moduledoc """
  Stage 7 acceptance tests for PCA, written from the criteria in
  docs/build-brief.md.

  These are a specification, not a description of existing code.

  Most tests use data whose principal components are known in advance by
  construction, so a wrong answer is wrong against arithmetic rather than
  against a previous run. A component is only defined up to sign, since
  `v` and `-v` describe the same direction, so every check is written to
  tolerate a flip.
  """

  use ExUnit.Case, async: true

  alias TinyLlm.PCA
  alias TinyLlm.Tensor

  defp unit_ish?(row), do: abs(Tensor.dot(row, row) - 1.0) < 1.0e-9

  # Compares directions, not vectors: v and -v are the same component.
  defp aligned?(row, expected) do
    abs(abs(Tensor.dot(row, expected)) - 1.0) < 1.0e-6
  end

  describe "center/1" do
    test "gives every column a mean of zero" do
      centered = PCA.center([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])

      for column <- Tensor.transpose(centered) do
        assert_in_delta Enum.sum(column) / length(column), 0.0, 1.0e-12
      end
    end

    test "shifts, and does not scale" do
      # Dividing by the standard deviation as well would be standardizing,
      # which is a different and defensible choice. It is not this one: the
      # embedding dimensions share units, and rescaling them would throw
      # away the fact that some carry more signal than others.
      centered = PCA.center([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])

      assert centered == [[-2.0, -10.0], [0.0, 0.0], [2.0, 10.0]]
    end
  end

  describe "covariance/1" do
    test "is square, with one row per feature" do
      centered = PCA.center([[1.0, 2.0, 3.0], [4.0, 5.0, 7.0], [7.0, 8.0, 11.0]])

      assert Tensor.shape(PCA.covariance(centered)) == {3, 3}
    end

    test "is symmetric" do
      centered = PCA.center([[1.0, 2.0], [4.0, 9.0], [7.0, 3.0], [2.0, 5.0]])
      matrix = PCA.covariance(centered)

      assert matrix == Tensor.transpose(matrix)
    end

    test "puts each feature's variance on the diagonal" do
      centered = PCA.center([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
      [[first, _], [_, second]] = PCA.covariance(centered)

      # Sample variance of 1, 3, 5 is 4.0, over n - 1.
      assert_in_delta first, 4.0, 1.0e-12
      assert_in_delta second, 0.0, 1.0e-12
    end
  end

  describe "dominant_eigenvector/2" do
    test "finds the axis a diagonal matrix stretches most" do
      PCA.seed(1)
      found = PCA.dominant_eigenvector([[9.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 4.0]])

      assert unit_ish?(found)
      assert aligned?(found, [1.0, 0.0, 0.0])
    end

    test "finds a diagonal direction when the data lies along one" do
      # Points strung along y = x, so the top component is [1, 1] / sqrt(2).
      PCA.seed(2)
      diagonal = 1.0 / :math.sqrt(2)
      centered = PCA.center([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

      assert aligned?(PCA.dominant_eigenvector(PCA.covariance(centered)), [diagonal, diagonal])
    end

    test "reproduces from a seed, since it starts somewhere random" do
      matrix = PCA.covariance(PCA.center([[1.0, 4.0], [2.0, 1.0], [5.0, 3.0], [3.0, 9.0]]))

      PCA.seed(7)
      first = PCA.dominant_eigenvector(matrix)

      PCA.seed(7)
      assert PCA.dominant_eigenvector(matrix) == first
    end
  end

  describe "components/1" do
    test "returns two unit rows, one per feature" do
      PCA.seed(3)
      {first, second} = PCA.components([[1.0, 2.0, 1.0], [4.0, 1.0, 7.0], [7.0, 8.0, 2.0]])

      assert length(first) == 3
      assert unit_ish?(first)
      assert unit_ish?(second)
    end

    test "the second component is orthogonal to the first" do
      # This is what deflation buys. Without subtracting the first component
      # out, the second iteration finds the same direction again.
      PCA.seed(4)
      {first, second} = PCA.components([[1.0, 2.0, 1.0], [4.0, 1.0, 7.0], [7.0, 8.0, 2.0]])

      assert_in_delta Tensor.dot(first, second), 0.0, 1.0e-6
    end

    test "orders them by how much the data varies along each" do
      # Chosen so the covariance is exactly diagonal: x and y are
      # uncorrelated, x varies 500 times as much, so the components are the
      # axes themselves and the only question is which comes first.
      #
      # Collinear data would not test this. Points along y = x/100 have a
      # top component of [0.99995, 0.01], the direction of the line, and no
      # second component worth the name.
      PCA.seed(5)
      rows = [[-3.0, -0.1], [-1.0, 0.1], [1.0, 0.1], [3.0, -0.1]]
      {first, second} = PCA.components(rows)

      assert aligned?(first, [1.0, 0.0])
      assert aligned?(second, [0.0, 1.0])
    end
  end

  describe "project/1" do
    test "returns one point per row" do
      PCA.seed(6)
      points = PCA.project([[1.0, 2.0, 1.0], [4.0, 1.0, 7.0], [7.0, 8.0, 2.0]])

      assert length(points) == 3
      assert Enum.all?(points, &match?({x, y} when is_float(x) and is_float(y), &1))
    end

    test "spreads the points along x more than along y" do
      # The first component carries the most variance by construction, so a
      # projection that came back with the axes swapped would show up here.
      PCA.seed(7)
      rows = for step <- 1..10, do: [step * 10.0, step * 0.1]
      points = PCA.project(rows)

      spread = fn values -> Enum.max(values) - Enum.min(values) end
      horizontal = spread.(Enum.map(points, fn {x, _y} -> x end))
      vertical = spread.(Enum.map(points, fn {_x, y} -> y end))

      assert horizontal > vertical
    end

    test "centres the cloud on the origin" do
      PCA.seed(8)
      points = PCA.project([[1.0, 2.0, 1.0], [4.0, 1.0, 7.0], [7.0, 8.0, 2.0], [2.0, 5.0, 4.0]])

      assert_in_delta Enum.sum(Enum.map(points, fn {x, _y} -> x end)), 0.0, 1.0e-9
      assert_in_delta Enum.sum(Enum.map(points, fn {_x, y} -> y end)), 0.0, 1.0e-9
    end

    test "keeps a straight line straight" do
      # Points on a line in 3D must land on a line in 2D. A projection is
      # linear, and this is the cheapest way to notice if it stopped being.
      PCA.seed(9)
      rows = for step <- 0..5, do: [step * 1.0, step * 2.0, step * 3.0]
      points = PCA.project(rows)

      [{first_x, _}, {second_x, _} | _] = points
      step_size = second_x - first_x

      for {{x, _y}, index} <- Enum.with_index(points) do
        assert_in_delta x, first_x + index * step_size, 1.0e-6
      end
    end
  end
end
