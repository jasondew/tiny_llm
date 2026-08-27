defmodule TinyLlm.PCA do
  @moduledoc """
  Principal components by power iteration, so a 32-dimensional embedding
  can be drawn on a page.

  The embedding table is the only part of this model that can be *looked
  at* rather than reasoned about, and that is worth twenty lines of
  arithmetic.

  Projected onto its two most variable directions, the table has structure
  nobody put there: nouns, adjectives and determiners land in tight
  separate groups, and the first component comes out close to a noun
  detector. The model was only ever asked to predict the next word. Parts
  of speech are what that turned out to require.

  What is *not* in the picture is as interesting. Verbs do not cluster, and
  their singular-to-plural offsets point in no shared direction at all,
  while nouns' do. Verb number lives in the unembedding instead, because
  nothing following a verb depends on its number: this model has to write
  verb number without ever needing to read it. See the stage 7 section of
  `docs/build-brief.md` for the measurements.

  ## Power iteration

  The first principal component is the direction the data varies most
  along, which is the top eigenvector of the covariance matrix. Finding it
  needs no eigensolver:

      start from any vector that is not orthogonal to the answer
      multiply by the covariance matrix
      normalize
      repeat

  Each multiplication stretches the vector further along the directions the
  data varies most, so the largest one wins and everything else shrinks
  relative to it. Convergence is geometric in the ratio between the top two
  eigenvalues, which for real data is fast.

  The second component is the same procedure run again on the data with the
  first component projected out, which is what "the next most variable
  direction, at right angles to the first" means arithmetically.

  Twenty lines of stdlib, no library, and the same algorithm that ranked
  the early web.
  """

  alias TinyLlm.Tensor

  @doc """
  Seeds the process `:rand` state, so a projection reproduces.

  Power iteration starts from a random vector, which is the one place this
  module is not deterministic.
  """
  @spec seed(integer()) :: :rand.state()
  def seed(seed), do: :rand.seed(:exsss, seed)

  @doc """
  Every column shifted to have mean zero.

  Principal components are directions of *variance*, which is defined
  around the mean, so skipping this finds the direction of the data's
  centroid instead and the picture is nonsense.
  """
  @spec center(Tensor.matrix()) :: Tensor.matrix()
  def center(matrix) do
    rows = length(matrix)
    means = matrix |> Tensor.transpose() |> Enum.map(fn column -> Enum.sum(column) / rows end)

    for row <- matrix, do: Enum.zip_with(row, means, &-/2)
  end

  @doc """
  The covariance matrix of centered data: `Xᵀ X` over `rows - 1`.
  """
  @spec covariance(Tensor.matrix()) :: Tensor.matrix()
  def covariance(centered) do
    rows = length(centered)

    centered
    |> Tensor.transpose()
    |> Tensor.matmul(centered)
    |> Tensor.scale(1 / (rows - 1))
  end

  @doc """
  The dominant eigenvector of a symmetric matrix, by power iteration.

  Returns a unit row. Runs a fixed number of iterations rather than testing
  for convergence, because a fixed count is reproducible and 100 is far
  more than a 32 by 32 covariance needs.
  """
  @spec dominant_eigenvector(Tensor.matrix(), pos_integer()) :: Tensor.row()
  def dominant_eigenvector(matrix, iterations \\ 100) do
    [start] = Tensor.random(1, length(matrix), 1.0)

    Enum.reduce(1..iterations, Tensor.unit(start), fn _iteration, vector ->
      [stretched] = Tensor.matmul([vector], matrix)

      Tensor.unit(stretched)
    end)
  end

  @doc """
  The two leading principal components, as unit rows.

  The second is found by removing the first from the covariance matrix,
  `C - λ₁ v₁ᵀ v₁`, and iterating again. That subtraction is deflation, and
  it is why the second component comes out orthogonal to the first rather
  than being the same direction over again.
  """
  @spec components(Tensor.matrix()) :: {Tensor.row(), Tensor.row()}
  def components(matrix) do
    covariance = matrix |> center() |> covariance()
    first = dominant_eigenvector(covariance)

    # Deflation: strip the first component out of the covariance so the next
    # iteration cannot rediscover it. The eigenvalue is how far the matrix
    # stretches its own eigenvector, which is what this quadratic form says.
    [stretched] = Tensor.matmul([first], covariance)
    eigenvalue = Tensor.dot(first, stretched)

    deflated =
      Tensor.sub(covariance, Tensor.scale(Tensor.outer_product(first, first), eigenvalue))

    {first, dominant_eigenvector(deflated)}
  end

  @doc """
  Every row of a matrix projected onto its own two leading components.

  This is what the stage 7 scatter plots: one `{x, y}` per vocabulary word,
  ready to hand to VegaLite alongside the words themselves.
  """
  @spec project(Tensor.matrix()) :: [{float(), float()}]
  def project(matrix) do
    {first, second} = components(matrix)

    for row <- center(matrix) do
      {Tensor.dot(row, first), Tensor.dot(row, second)}
    end
  end
end
