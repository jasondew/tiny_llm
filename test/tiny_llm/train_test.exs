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

  describe "Config" do
    test "defaults to the architecture the brief fixes" do
      config = %Train.Config{}

      assert config.vocabulary_size == 32
      assert config.hidden_size == 32
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
    setup do
      config = %Train.Config{
        model: Embedder,
        learning_rate: 0.5,
        batch_size: 32,
        steps: 300,
        log_every: 50,
        corpus_size: 1_000,
        seed: 1234
      }

      {:ok, config: config, result: Train.run(config)}
    end

    test "returns the trained parameters and the loss history", %{result: result} do
      assert Map.keys(result) |> Enum.sort() == [:losses, :params]
      assert Map.keys(result.params) |> Enum.sort() == [:embedding, :projection]
    end

    test "logs a loss every log_every steps, starting before any learning", %{result: result} do
      steps = Enum.map(result.losses, fn {step, _loss} -> step end)

      assert List.first(steps) == 0
      assert steps == Enum.sort(steps)
      assert length(steps) >= 6
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

    test "approaches the bigram floor of about 1.90 nats", %{result: result} do
      # The brief's acceptance criterion. Note the headroom is thin: no
      # model with one word of context can beat H(next | previous), which
      # measures 1.9021 nats on this grammar. Landing under 2.0 means
      # getting within 5% of the theoretical optimum.
      {_step, last_loss} = List.last(result.losses)

      assert last_loss < 2.0
    end

    test "keeps the parameter shapes intact through training", %{result: result} do
      assert Tensor.shape(result.params.embedding) == {32, 32}
      assert Tensor.shape(result.params.projection) == {32, 32}
    end

    test "reproduces exactly from the same seed", %{config: config, result: result} do
      assert Train.run(config).losses == result.losses
    end
  end
end
