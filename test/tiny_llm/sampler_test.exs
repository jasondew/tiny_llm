defmodule TinyLlm.SamplerTest do
  @moduledoc """
  Stage 6 acceptance tests for generation, written from the criteria in
  docs/build-brief.md.

  These are a specification, not a description of existing code.

  Assumptions the brief leaves open:

    * the sampler works in **words**, not ids. It is the boundary between
      the model and everything that reads its output, and every consumer of
      it, Eval and the notebook, wants words.
    * `sentence/2` drops the leading `"<start>"` from what it returns, so a
      generated sentence has the same shape as a `Grammar.sentence/0` and
      the two can be compared with `==`.
    * options are `:temperature` (default 1.0) and `:max_tokens` (default
      the context length the params imply).

  Most tests run against a **trained** model, because an untrained one
  emits noise and cannot distinguish "the sampler works" from "the sampler
  returns something". Training happens once in setup_all.
  """

  use ExUnit.Case, async: true

  alias TinyLlm.Grammar
  alias TinyLlm.Model
  alias TinyLlm.Sampler
  alias TinyLlm.Train
  alias TinyLlm.Vocab

  setup_all do
    # Small and short: these tests need a model that has learned the shape
    # of the grammar, not one that has converged.
    config = %Train.Config{
      model: Model,
      batch_size: 8,
      steps: 60,
      log_every: 60,
      learning_rate: 0.5,
      learning_rate_schedule: :cosine,
      training_corpus_size: 400,
      evaluation_corpus_size: 50,
      seed: 1234
    }

    {:ok, params: Train.run(config).params}
  end

  describe "distribution/3" do
    test "returns a probability row over the whole vocabulary", %{params: params} do
      row = Sampler.distribution(params, ["<start>"], 1.0)

      assert length(row) == 32
      assert_in_delta Enum.sum(row), 1.0, 1.0e-9
      assert Enum.all?(row, &(&1 >= 0.0))
    end

    test "a temperature of zero is one-hot on the most likely word", %{params: params} do
      # Special-cased rather than computed, since dividing by zero is what
      # the arithmetic would otherwise do. The limit as temperature falls is
      # exactly this, so the special case is not a different rule.
      row = Sampler.distribution(params, ["<start>", "the"], 0.0)

      assert Enum.count(row, &(&1 == 1.0)) == 1
      assert Enum.count(row, &(&1 == 0.0)) == 31
    end

    test "a low temperature sharpens and a high one flattens", %{params: params} do
      # Entropy is the direct measurement of "how spread out is this", and
      # it has to fall monotonically as temperature falls.
      entropy = fn row ->
        row
        |> Enum.reject(&(&1 <= 0.0))
        |> Enum.map(fn probability -> -probability * :math.log(probability) end)
        |> Enum.sum()
      end

      prefix = ["<start>", "the", "llama"]

      cold = entropy.(Sampler.distribution(params, prefix, 0.5))
      warm = entropy.(Sampler.distribution(params, prefix, 1.0))
      hot = entropy.(Sampler.distribution(params, prefix, 2.0))

      assert cold < warm
      assert warm < hot
    end

    test "temperature divides the logits, it does not reweight the probabilities", %{
      params: params
    } do
      # Two formulations are correct and identical: softmax(z / t), and
      # raising the probabilities to the power 1/t and renormalizing. The
      # exp(z_i)/Z substitution makes the Z^(1/t) cancel. Dividing the
      # probabilities by t and renormalizing is the actual mistake, and it
      # is a silent one: linear rescaling cancels completely and returns the
      # distribution unchanged, so temperature appears to do nothing.
      #
      # Dividing the logits is the implementation to prefer anyway: it
      # reuses softmax's max-subtraction guard and normalizes once.
      prefix = ["<start>", "the"]
      logits = Model.forward(params, Vocab.encode(prefix)).logits |> List.last()

      [expected] = TinyLlm.Tensor.softmax([Enum.map(logits, &(&1 / 0.7))])

      for {actual, wanted} <- Enum.zip(Sampler.distribution(params, prefix, 0.7), expected) do
        assert_in_delta actual, wanted, 1.0e-12
      end
    end
  end

  describe "sentence/2" do
    test "generates words from the vocabulary and nothing else", %{params: params} do
      Sampler.seed(1)

      for word <- Sampler.sentence(params) do
        assert word in Vocab.words()
      end
    end

    test "does not include the start token it was primed with", %{params: params} do
      # So a generated sentence has the same shape as a Grammar.sentence/0
      # and the two can be compared with ==, which stage 6b depends on.
      Sampler.seed(2)

      refute Vocab.start_token() in Sampler.sentence(params)
    end

    test "stops at the first period rather than running on", %{params: params} do
      Sampler.seed(3)
      words = Sampler.sentence(params)

      assert List.last(words) == "."
      assert Enum.count(words, &(&1 == ".")) == 1
    end

    test "never exceeds the context the model was trained on", %{params: params} do
      # A sequence longer than the position table has no embedding for its
      # last position, so this is not a preference.
      Sampler.seed(4)

      for _ <- 1..20 do
        assert length(Sampler.sentence(params)) <= 16
      end
    end

    test "honours a shorter max_tokens, and then does not end in a period", %{params: params} do
      Sampler.seed(5)
      words = Sampler.sentence(params, max_tokens: 3)

      assert length(words) <= 3
      refute List.last(words) == "."
    end

    test "reproduces exactly from the same seed", %{params: params} do
      Sampler.seed(11)
      first = Sampler.sentence(params)

      Sampler.seed(11)
      second = Sampler.sentence(params)

      assert first == second
    end

    test "at temperature zero it says the same thing every time", %{params: params} do
      # No randomness left, so no seeding needed. Two calls with different
      # process state must still agree.
      Sampler.seed(1)
      first = Sampler.sentence(params, temperature: 0.0)

      Sampler.seed(99)
      second = Sampler.sentence(params, temperature: 0.0)

      assert first == second
    end

    test "at temperature one it does not", %{params: params} do
      Sampler.seed(7)
      sentences = for _ <- 1..10, do: Sampler.sentence(params)

      assert length(Enum.uniq(sentences)) > 1
    end
  end

  describe "stream/2" do
    test "is lazy, so taking two costs two", %{params: params} do
      Sampler.seed(8)

      assert params |> Sampler.stream() |> Enum.take(2) |> length() == 2
    end

    test "gives the same sentences sentence/2 would, in the same order", %{params: params} do
      Sampler.seed(12)
      streamed = params |> Sampler.stream() |> Enum.take(3)

      Sampler.seed(12)
      one_by_one = for _ <- 1..3, do: Sampler.sentence(params)

      assert streamed == one_by_one
    end
  end

  describe "trace/2" do
    test "records one step per generated word", %{params: params} do
      Sampler.seed(13)
      steps = Sampler.trace(params)

      Sampler.seed(13)
      assert length(steps) == length(Sampler.sentence(params))
    end

    test "keeps the whole distribution at each step, not just the winner", %{params: params} do
      # The stage 7 notebook draws probability bars from this.
      Sampler.seed(14)
      [first | _] = Sampler.trace(params)

      assert length(first.distribution) == 32
      assert_in_delta Enum.sum(first.distribution), 1.0, 1.0e-9
    end

    test "each step's prefix is the start token plus everything chosen before it", %{
      params: params
    } do
      Sampler.seed(15)
      steps = Sampler.trace(params)

      for {step, index} <- Enum.with_index(steps) do
        chosen_so_far = steps |> Enum.take(index) |> Enum.map(& &1.chosen)

        assert step.prefix == [Vocab.start_token() | chosen_so_far]
      end
    end
  end
end
