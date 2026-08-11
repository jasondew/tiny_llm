defmodule TinyLlm.GradCheckTest do
  @moduledoc """
  Stage 3 tests for the gradient checker itself.

  These come first, before any model is checked with it, because a checker
  that always reports agreement would pass every backward pass in the
  project including the wrong ones. So it is pointed at two models with
  exactly known gradients: one right, one subtly wrong.
  """

  use ExUnit.Case, async: true

  alias TinyLlm.GradCheck
  alias TinyLlm.Test.BrokenModel
  alias TinyLlm.Test.QuadraticModel
  alias TinyLlm.Train

  @config %Train.Config{model: QuadraticModel}

  describe "relative_error/2" do
    test "is zero when the numbers agree" do
      assert GradCheck.relative_error(2.0, 2.0) == 0.0
    end

    test "grows as the numbers diverge" do
      close = GradCheck.relative_error(1.0, 1.001)
      far = GradCheck.relative_error(1.0, 2.0)

      assert close < far
    end

    test "scales, so the same verdict holds for large and small gradients" do
      small = GradCheck.relative_error(0.001, 0.002)
      large = GradCheck.relative_error(1000.0, 2000.0)

      assert_in_delta small, large, 1.0e-12
    end

    test "stays finite when both gradients vanish" do
      assert GradCheck.relative_error(0.0, 0.0) == 0.0
    end

    test "reports total disagreement on a flipped sign" do
      assert GradCheck.relative_error(2.0, -2.0) == 1.0
    end
  end

  describe "check/4" do
    test "agrees with a gradient that is exactly right" do
      params = QuadraticModel.init(@config)
      errors = GradCheck.check(QuadraticModel, params, [:unused])

      assert Map.keys(errors) == [:weights]
      assert errors.weights < 1.0e-3
    end

    test "catches a gradient with one wrong sign" do
      params = BrokenModel.init(@config)
      errors = GradCheck.check(BrokenModel, params, [:unused])

      assert errors.weights > 1.0e-2
    end

    test "reports one entry per named parameter matrix" do
      params = QuadraticModel.init(@config)

      assert QuadraticModel |> GradCheck.check(params, [:unused]) |> map_size() == 1
    end

    test "accepts a different epsilon" do
      params = QuadraticModel.init(@config)
      errors = GradCheck.check(QuadraticModel, params, [:unused], epsilon: 1.0e-3)

      assert errors.weights < 1.0e-3
    end
  end

  describe "max_relative_error/4" do
    test "collapses every parameter into the worst single number" do
      params = QuadraticModel.init(@config)

      assert GradCheck.max_relative_error(QuadraticModel, params, [:unused]) < 1.0e-3
    end

    test "surfaces a broken gradient" do
      params = BrokenModel.init(@config)

      assert GradCheck.max_relative_error(BrokenModel, params, [:unused]) > 1.0e-2
    end
  end
end
