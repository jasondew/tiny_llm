defmodule TinyLlm.GradCheck do
  @moduledoc """
  The safety net for every explicitly written gradient in this project.

  A plausible looking wrong gradient is the most likely way this whole
  project fails, and it is the one bug that hides completely. A model with a
  subtly wrong backward pass still trains. The loss still falls, the
  sentences still come out, nothing raises, and the only symptom is that it
  ends up somewhat worse than it should have, which is invisible without
  something correct to compare against.

  Finite differences are how you find out anyway. Nudge one parameter by a
  tiny amount in each direction, watch what the loss does, and compare that
  to what the derivation claimed. It is far too slow to train with and
  exactly right for checking.

  ## How it works

  The derivative of a function is its slope, and a slope can be measured
  without any calculus at all by nudging the input and seeing what happens:

      numeric = (loss(theta + epsilon) - loss(theta - epsilon)) / (2 * epsilon)

  Do that for one parameter entry at a time and compare against what the
  analytic gradient claims. The two-sided form is used rather than the
  cheaper one-sided version because its error shrinks with `epsilon`
  squared, which buys several digits for free.

  This is far too slow to train with, which is the point: it is a test, run
  once on a tiny config, never in the training loop.

  ## Reading the result

  Agreement is measured as relative error, so the comparison means the same
  thing for a gradient of 1000 as for one of 0.001:

      |analytic - numeric| / max(|analytic| + |numeric|, guard)

  Below `1.0e-3` means the derivation is right. Above `1.0e-2` means it is
  wrong. In between usually means `epsilon` is fighting floating point
  rather than that the math is subtly off.
  """

  alias TinyLlm.Train

  @default_epsilon 1.0e-4
  @guard 1.0e-8

  @doc """
  The default nudge used when no `:epsilon` option is given.
  """
  @spec default_epsilon() :: float()
  def default_epsilon, do: @default_epsilon

  @doc """
  The floor that keeps `relative_error/2` finite for vanishing gradients.
  """
  @spec guard() :: float()
  def guard, do: @guard

  @doc """
  The largest relative error for each named parameter matrix.

  Perturbs every entry of every matrix twice, so cost grows with the total
  parameter count. Use the smallest config that still exercises the shapes.

  ## Options

    * `:epsilon` - the nudge, defaulting to `#{@default_epsilon}`.
  """
  @spec check(module(), Train.params(), [Train.example()], keyword()) :: %{atom() => float()}
  def check(model, params, batch, opts \\ []) do
    epsilon = Keyword.get(opts, :epsilon, @default_epsilon)
    analytics = model.gradients(params, batch)

    for {key, matrix} <- analytics, into: %{} do
      row_count = length(matrix)
      column_count = length(hd(matrix))

      max_error =
        for row <- 0..(row_count - 1), column <- 0..(column_count - 1) do
          plus = model.loss(perturb(params, key, row, column, +epsilon), batch)
          minus = model.loss(perturb(params, key, row, column, -epsilon), batch)
          numeric = (plus - minus) / (2 * epsilon)
          analytic = get_in(matrix, [Access.at(row), Access.at(column)])

          relative_error(analytic, numeric)
        end
        |> Enum.max()

      {key, max_error}
    end
  end

  @doc """
  The worst relative error across every parameter, as one number.
  """
  @spec max_relative_error(module(), Train.params(), [Train.example()], keyword()) :: float()
  def max_relative_error(model, params, batch, opts \\ []) do
    check(model, params, batch, opts) |> Map.values() |> Enum.max()
  end

  @doc """
  The relative difference between two numbers, guarded against dividing by
  zero when both gradients are legitimately tiny.
  """
  @spec relative_error(float(), float()) :: float()
  def relative_error(analytic, numeric) do
    abs(analytic - numeric) / max(abs(analytic) + abs(numeric), guard())
  end

  ## PRIVATE FUNCTIONS

  defp perturb(params, key, row, column, delta) do
    update_in(params, [key, Access.at(row), Access.at(column)], &(&1 + delta))
  end
end
