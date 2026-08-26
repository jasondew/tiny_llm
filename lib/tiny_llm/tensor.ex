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
  A `rows` by `columns` matrix of ones.
  """
  @spec ones(non_neg_integer(), non_neg_integer()) :: matrix()
  def ones(rows, columns) do
    for _row <- 1..rows//1 do
      for _column <- 1..columns//1, do: 1.0
    end
  end

  @doc """
  The scale to draw a `fan_in` by `fan_out` matrix at.

  `sqrt(6 / (fan_in + fan_out))` gives a uniform draw the variance that
  keeps activations from growing or shrinking as they pass through a layer.
  It is the same argument the `sqrt(d)` in attention's scores makes, applied
  to the parameters instead of the scores.

  A constant scale works while a network is shallow and stops working when
  it is not. See "the same number kills training outright" in
  docs/backprop.md for the measurement.
  """
  @spec fan_scale(pos_integer(), pos_integer()) :: float()
  def fan_scale(fan_in, fan_out) do
    :math.sqrt(6 / (fan_in + fan_out))
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
  Every pairing of an entry from one row with an entry from another.

  The expanding counterpart to `dot/2`. Those are the only two ways to
  multiply a pair of vectors: `dot/2` contracts them into one number, this
  expands them into a `length(left)` by `length(right)` matrix where
  `result[i][j]` is `left[i] * right[j]`.

  A backward pass reaches for it whenever a parameter connects exactly one
  input to exactly one output, because then its gradient is simply how
  active that input was times how wrong that output was. Equivalent to
  `matmul(transpose([left]), [right])`, and easier to read as what it is.
  """
  @spec outer_product(row(), row()) :: matrix()
  def outer_product(left_row, right_row) do
    for left_entry <- left_row do
      for right_entry <- right_row, do: left_entry * right_entry
    end
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

  @spec add_bias(matrix(), matrix()) :: matrix()
  def add_bias(matrix, [bias]) do
    for row <- matrix do
      Enum.zip_with(row, bias, &+/2)
    end
  end

  @doc """
  Each row divided by its own sum, so every row becomes a distribution.

  This is the other way to turn a row into probabilities, and the one to
  reach for when the entries are already counts rather than logits.

  A row summing to zero is returned untouched. There is no distribution to
  normalize it to, and dividing would turn an honestly empty row into a row
  of NaNs.
  """
  @spec normalize(matrix()) :: matrix()
  def normalize(matrix) do
    for row <- matrix do
      sum = Enum.sum(row)

      if sum == 0, do: row, else: Enum.map(row, &(&1 / sum))
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

  @doc """
  An index drawn at random, each entry weighted by its own value.

  The stochastic sibling of `argmax/1`: same input, same kind of output,
  but `[0.7, 0.3]` returns 0 about seven times in ten rather than always.
  Stage 6's temperature knob interpolates between the two, and at `T = 0`
  this collapses into `argmax/1`.

  Weights, not probabilities. The draw is scaled by the row's own total,
  which buys two things: the row need not already sum to 1, so raw counts
  sample correctly without normalizing first; and the draw can never land
  above the last cumulative value, so there is no float-drift gap where
  nothing matches.

  Raises on a row summing to zero, which weights nothing. The period's row
  in a bigram matrix is exactly that, so this catches a sampler that has
  walked somewhere it should not.
  """
  @spec weighted_random_index(row()) :: non_neg_integer()
  def weighted_random_index([]) do
    raise(ArgumentError, "cannot sample from an empty row")
  end

  def weighted_random_index(row) do
    cumulative_weights = Enum.scan(row, &+/2)
    total = List.last(cumulative_weights)

    if total == 0 do
      raise ArgumentError, "cannot sample from a row that sums to zero"
    end

    draw = :rand.uniform() * total

    Enum.find_index(cumulative_weights, &(&1 >= draw))
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
