defmodule TinyLlm.Tensor do
  @moduledoc """
  The entire math library. A matrix is a list of rows of floats.
  """

  @typedoc "One row of a matrix, and the shape `dot/2` and `argmax/1` take."
  @type row :: [float()]

  @typedoc "A matrix: a list of equal-length rows."
  @type matrix :: [row()]

  @default_scale 0.02

  @doc """
  A `rows` by `columns` matrix of zeros.
  """
  @spec zeros(non_neg_integer(), non_neg_integer()) :: matrix()
  def zeros(rows, columns) do
    for _row <- 1..rows//1 do
      for _column <- 1..columns//1, do: 0.0
    end
  end

  @doc """
  A `rows` by `columns` matrix drawn uniformly from `-scale..scale`.

  Draws from the process `:rand` state, so seeding the process makes the
  matrix reproducible.
  """
  @spec random(non_neg_integer(), non_neg_integer(), float()) :: matrix()
  def random(rows, columns, scale \\ @default_scale) do
    for _row <- 1..rows//1 do
      for _column <- 1..columns//1, do: :rand.uniform() * 2 * scale - scale
    end
  end

  @doc """
  A row of `size` floats, zero everywhere but `index`.
  """
  @spec one_hot(non_neg_integer(), pos_integer()) :: row()
  def one_hot(index, size) do
    for i <- 0..(size - 1) do
      if i == index, do: 1.0, else: 0.0
    end
  end

  @doc """
  The `{rows, columns}` shape of a matrix.
  """
  @spec shape(matrix()) :: {non_neg_integer(), non_neg_integer()}
  def shape([]), do: {0, 0}
  def shape([first_row | _] = matrix), do: {length(matrix), length(first_row)}

  @doc """
  A matrix with its rows and columns exchanged.
  """
  @spec transpose(matrix()) :: matrix()
  def transpose([]), do: []
  def transpose(matrix), do: Enum.zip_with(matrix, & &1)

  @doc """
  Two matrices added entry by entry.
  """
  @spec add(matrix(), matrix()) :: matrix()
  def add(left, right) do
    apply_binary(left, right, fn left_entry, right_entry ->
      left_entry + right_entry
    end)
  end

  @doc """
  The right matrix subtracted from the left, entry by entry.
  """
  @spec sub(matrix(), matrix()) :: matrix()
  def sub(left, right) do
    apply_binary(left, right, fn left_entry, right_entry ->
      left_entry - right_entry
    end)
  end

  @doc """
  Two matrices multiplied entry by entry.
  """
  @spec hadamard(matrix(), matrix()) :: matrix()
  def hadamard(left, right) do
    apply_binary(left, right, fn left_entry, right_entry ->
      left_entry * right_entry
    end)
  end

  @doc """
  Every entry of a matrix multiplied by a constant.
  """
  @spec scale(matrix(), float()) :: matrix()
  def scale(matrix, factor) do
    apply_unary(matrix, &(&1 * factor))
  end

  @doc """
  A function applied to every entry of a matrix.
  """
  @spec map(matrix(), (float() -> float())) :: matrix()
  def map(matrix, fun) do
    apply_unary(matrix, fun)
  end

  @doc """
  The dot product of two rows.
  """
  @spec dot(row(), row()) :: float()
  def dot(left_row, right_row) do
    Enum.zip_reduce(left_row, right_row, 0.0, fn left_entry, right_entry, sum ->
      sum + left_entry * right_entry
    end)
  end

  @doc """
  The matrix product of an `m` by `n` and an `n` by `p` matrix.
  """
  @spec matmul(matrix(), matrix()) :: matrix()
  def matmul(left, right) do
    right_transposed = transpose(right)

    for left_row <- left do
      for right_column <- right_transposed do
        dot(left_row, right_column)
      end
    end
  end

  @doc """
  Each row of a matrix turned into a probability distribution.

  Subtracts the row maximum before exponentiating, which changes nothing
  mathematically and keeps `exp/1` from overflowing on large logits.
  """
  @spec softmax(matrix()) :: matrix()
  def softmax(matrix) do
    for row <- matrix do
      max = Enum.max(row)
      exponentials = Enum.map(row, fn entry -> :math.exp(entry - max) end)
      sum = Enum.sum(exponentials)

      Enum.map(exponentials, &(&1 / sum))
    end
  end

  @doc """
  The index of the largest entry in a row, earliest index winning ties.
  """
  @spec argmax(row()) :: non_neg_integer()
  def argmax(row) do
    {_max_value, max_index} =
      row
      |> Enum.with_index()
      |> Enum.max_by(fn {entry, _index} -> entry end)

    max_index
  end

  ## PRIVATE FUNCTIONS

  defp apply_unary(matrix, fun) do
    for row <- matrix do
      for entry <- row do
        fun.(entry)
      end
    end
  end

  defp apply_binary(left_matrix, right_matrix, fun) do
    for {left_row, right_row} <- Enum.zip(left_matrix, right_matrix) do
      for {left_entry, right_entry} <- Enum.zip(left_row, right_row) do
        fun.(left_entry, right_entry)
      end
    end
  end
end
