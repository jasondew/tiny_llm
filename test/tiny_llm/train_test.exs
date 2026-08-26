defmodule TinyLlm.TrainTest do
  @moduledoc """
  Stage 3 acceptance tests for the training harness.

  These are a specification, not a description of existing code.

  The harness is deliberately ignorant of the model: it batches, subtracts
  gradients, and logs. Everything model-shaped arrives through the callbacks
  in `TinyLlm.Train`. Stage 5 reuses this file unchanged, so anything here
  that mentions embeddings is a bug.

  Assumed shapes where the brief is silent:

    * `run/1` returns `%{params: params, losses: [{step, loss}]}`.
    * the history starts at step 0, so the first entry is the loss before
      any learning, which should land near `ln(32)`.
    * `batch/2` samples with replacement, so a batch may repeat an example.
  """

  use ExUnit.Case, async: true

  alias TinyLlm.Embedder
  alias TinyLlm.Tensor
  alias TinyLlm.Test.QuadraticModel
  alias TinyLlm.Train

  # Module level, because ExUnit only allows setup_all outside a describe. A
  # training run is by far the most expensive thing in this suite and every
  # test in "run/1" reads the same result, so it happens once for the file
  # rather than once per test.
  #
  # 800 steps rather than the config default of 1000. The Embedder converges
  # by 800 and then wanders: measured 1.9436 at 800 and 1.9546 at 2000, so
  # the extra steps cost 9 seconds of suite time and buy a slightly worse
  # number.
  setup_all do
    config = %Train.Config{
      model: Embedder,
      learning_rate: 1.0,
      batch_size: 64,
      steps: 800,
      log_every: 200,
      training_corpus_size: 2_000,
      evaluation_corpus_size: 500,
      seed: 1234
    }

    {:ok, config: config, result: Train.run(config)}
  end

  describe "learning_rate/2" do
    test "a constant schedule ignores the step" do
      config = %Train.Config{learning_rate: 0.5, steps: 100}

      assert Train.learning_rate(config, 0) == 0.5
      assert Train.learning_rate(config, 50) == 0.5
      assert Train.learning_rate(config, 100) == 0.5
    end

    test "cosine starts at the full rate, halves at the midpoint, and ends at zero" do
      # Ending at zero is the point: a constant rate keeps taking full-size
      # steps after it has arrived, so where it stops depends on which step
      # it stopped on. Measured over 8 seeds, decaying halved the spread.
      config = %Train.Config{learning_rate: 0.5, learning_rate_schedule: :cosine, steps: 100}

      assert_in_delta Train.learning_rate(config, 0), 0.5, 1.0e-12
      assert_in_delta Train.learning_rate(config, 50), 0.25, 1.0e-12
      assert_in_delta Train.learning_rate(config, 100), 0.0, 1.0e-12
    end

    test "cosine never rises" do
      config = %Train.Config{learning_rate: 1.0, learning_rate_schedule: :cosine, steps: 40}
      rates = Enum.map(0..40, fn step -> Train.learning_rate(config, step) end)

      assert rates == Enum.sort(rates, :desc)
    end

    test "defaults to constant, so stage 3 is unaffected" do
      assert %Train.Config{}.learning_rate_schedule == :constant
    end
  end

  describe "Config" do
    test "defaults to the architecture the brief fixes" do
      config = %Train.Config{}

      assert config.vocabulary_size == 32
      assert config.d_model == 32
      assert config.context_length == 16
    end

    test "carries the model it trains" do
      assert %Train.Config{model: Embedder}.model == Embedder
    end
  end

  describe "batch/2" do
    test "draws the requested number of examples" do
      Train.seed(1)

      assert length(Train.batch([1, 2, 3, 4, 5], 3)) == 3
    end

    test "draws only from the examples it was given" do
      Train.seed(1)
      examples = [:a, :b, :c]

      assert Train.batch(examples, 20)
             |> Enum.uniq()
             |> Enum.sort()
             |> Enum.all?(&(&1 in examples))
    end

    test "draws the same batch twice from the same seed" do
      Train.seed(42)
      first = Train.batch(Enum.to_list(1..100), 10)

      Train.seed(42)
      second = Train.batch(Enum.to_list(1..100), 10)

      assert first == second
    end
  end

  describe "step/4" do
    test "subtracts the learning rate times the gradient" do
      params = %{weights: [[1.0, 2.0], [3.0, 4.0]]}

      # QuadraticModel's gradient is 2 * theta, so a rate of 0.1 leaves 0.8 * theta
      stepped = Train.step(QuadraticModel, params, [:unused], 0.1)

      assert stepped.weights == [[0.8, 1.6], [2.4, 3.2]]
    end

    test "lowers the loss it was asked to descend" do
      params = %{weights: [[1.0, 2.0], [3.0, 4.0]]}

      before = QuadraticModel.loss(params, [:unused])

      after_step =
        QuadraticModel.loss(Train.step(QuadraticModel, params, [:unused], 0.1), [:unused])

      assert after_step < before
    end

    test "leaves the parameter shapes alone" do
      params = %{weights: [[1.0, 2.0], [3.0, 4.0]]}

      assert Tensor.shape(Train.step(QuadraticModel, params, [:unused], 0.1).weights) == {2, 2}
    end
  end

  describe "run/1" do
    test "returns the trained parameters and the loss history", %{result: result} do
      assert Map.keys(result) |> Enum.sort() == [:losses, :params]
      assert Map.keys(result.params) |> Enum.sort() == [:embeddings, :projection]
    end

    test "logs a loss every log_every steps, starting before any learning", %{
      config: config,
      result: result
    } do
      steps = Enum.map(result.losses, fn {step, _loss} -> step end)

      assert List.first(steps) == 0
      assert steps == Enum.sort(steps)
      assert steps == Enum.to_list(0..config.steps//config.log_every)
    end

    test "starts at the cost of knowing nothing, ln(32)", %{result: result} do
      {0, first_loss} = List.first(result.losses)

      assert_in_delta first_loss, :math.log(32), 0.1
    end

    test "learns: the loss falls a long way from where it started", %{result: result} do
      {_step, first_loss} = List.first(result.losses)
      {_step, last_loss} = List.last(result.losses)

      assert last_loss < first_loss - 1.0
    end

    test "never claims to beat the bigram floor, which would mean a leak", %{result: result} do
      # No model conditioned on one word can score below H(next | previous),
      # which measures 1.9021 nats on this grammar. A loss under it is not a
      # better model, it is an evaluation set that is too small or not held
      # out. This test exists because both happened.
      {_step, last_loss} = List.last(result.losses)

      assert last_loss > 1.85
    end

    test "approaches the bigram floor of about 1.90 nats", %{result: result} do
      # The brief's acceptance criterion. Note the headroom is thin: no
      # model with one word of context can beat H(next | previous), which
      # measures 1.9021 nats on this grammar. Landing under 2.0 means
      # getting within 5% of the theoretical optimum.
      {_step, last_loss} = List.last(result.losses)

      assert last_loss < 2.0
    end

    test "keeps the parameter shapes intact through training", %{result: result} do
      assert Tensor.shape(result.params.embeddings) == {32, 32}
      assert Tensor.shape(result.params.projection) == {32, 32}
    end

    test "reproduces exactly from the same seed" do
      # Its own small config. Reproducibility does not need convergence, and
      # this test is the only one that pays for a second training run.
      config = %Train.Config{
        model: Embedder,
        learning_rate: 1.0,
        batch_size: 16,
        steps: 40,
        log_every: 10,
        training_corpus_size: 200,
        evaluation_corpus_size: 50,
        seed: 99
      }

      assert Train.run(config).losses == Train.run(config).losses
    end
  end
end
