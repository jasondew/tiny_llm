defmodule TinyLlm.PCA do
  @moduledoc """
  Principal components by power iteration, so a 32-dimensional embedding
  can be drawn on a page.

  TODO(stage 7): write the concept paragraph. It should say that the
  embedding table is the only part of this model that can be *looked at*
  rather than reasoned about, and that the picture is worth the twenty
  lines: nouns cluster, verbs cluster, and the singular-to-plural offset is
  roughly the same vector for every pair. Nobody put those there. They are
  what "predict the next word" turns out to require.

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
  def center(_matrix), do: raise("TODO: stage 7")

  @doc """
  The covariance matrix of centered data: `Xᵀ X` over `rows - 1`.
  """
  @spec covariance(Tensor.matrix()) :: Tensor.matrix()
  def covariance(_centered), do: raise("TODO: stage 7")

  @doc """
  The dominant eigenvector of a symmetric matrix, by power iteration.

  Returns a unit row. Runs a fixed number of iterations rather than testing
  for convergence, because a fixed count is reproducible and 100 is far
  more than a 32 by 32 covariance needs.
  """
  @spec dominant_eigenvector(Tensor.matrix(), pos_integer()) :: Tensor.row()
  def dominant_eigenvector(_matrix, _iterations \\ 100), do: raise("TODO: stage 7")

  @doc """
  The two leading principal components, as unit rows.

  The second is found by removing the first from the covariance matrix,
  `C - λ₁ v₁ᵀ v₁`, and iterating again. That subtraction is deflation, and
  it is why the second component comes out orthogonal to the first rather
  than being the same direction over again.
  """
  @spec components(Tensor.matrix()) :: {Tensor.row(), Tensor.row()}
  def components(_matrix), do: raise("TODO: stage 7")

  @doc """
  Every row of a matrix projected onto its own two leading components.

  This is what the stage 7 scatter plots: one `{x, y}` per vocabulary word,
  ready to hand to VegaLite alongside the words themselves.
  """
  @spec project(Tensor.matrix()) :: [{float(), float()}]
  def project(_matrix), do: raise("TODO: stage 7")
end
