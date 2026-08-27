defmodule TinyLlm.EvalTest do
  @moduledoc """
  Stage 6 acceptance tests for the three measurements, written from the
  criteria in docs/build-brief.md.

  These are a specification, not a description of existing code.

  Assumptions the brief leaves open:

    * a probe is `%{prefix: words, number: :singular | :plural}`, the
      sentence cut just before its main verb.
    * accuracy is measured over **verb forms only**, so a model that fails
      to predict a verb at all is not scored as an agreement error. The
      bigram must lose because it gets the number wrong, not because it
      wandered off into nouns.
    * a predictor is any `[word] -> distribution` function, which is what
      lets one accuracy function score three models with three different
      interfaces.

  The interesting tests here do not need a trained model: they check the
  measurements are measuring the right thing, using hand-built
  distributions where the correct answer is known by construction. A
  trained model is used only where nothing else will do.
  """

  use ExUnit.Case, async: true

  alias TinyLlm.Bigram
  alias TinyLlm.Eval
  alias TinyLlm.Grammar
  alias TinyLlm.Transformer
  alias TinyLlm.Tensor
  alias TinyLlm.Train
  alias TinyLlm.Vocab

  # A predictor that puts all its mass on one word, whatever it is asked.
  defp always(word) do
    fn _prefix -> Tensor.one_hot(Vocab.word_to_id(word), Vocab.size()) end
  end

  describe "probes/1" do
    test "cuts a relative-clause sentence just before its main verb" do
      sentence = ~w(the llama who chases the dogs flees .)

      assert Eval.probes([sentence]) == [
               %{prefix: ~w(the llama who chases the dogs), number: :singular}
             ]
    end

    test "takes the number from the head noun, not the adjacent one" do
      # The whole point. `dogs` sits next to the blank and is plural; the
      # answer is singular because `llama` is the subject.
      [probe] = Eval.probes([~w(the llama who chases the dogs flees .)])

      assert probe.number == :singular
      assert List.last(probe.prefix) == "dogs"
    end

    test "handles the mirror case" do
      [probe] = Eval.probes([~w(the dogs who chase the llama flee .)])

      assert probe.number == :plural
      assert List.last(probe.prefix) == "llama"
    end

    test "keeps the adjectives, which lengthen the dependency" do
      [probe] = Eval.probes([~w(the grumpy small llama who chases the dogs flees .)])

      assert probe.prefix == ~w(the grumpy small llama who chases the dogs)
      assert probe.number == :singular
    end

    test "skips sentences with no relative clause" do
      assert Eval.probes([~w(the llama flees .)]) == []
    end

    test "skips compound subjects, which test a different rule" do
      # A compound subject is plural whatever its conjuncts are, so it
      # measures "did it see the word and", not head-noun tracking.
      assert Eval.probes([~w(the llama and the dog who chases the geese flee .)]) == []
    end

    test "builds a probe from most relative-clause sentences in a real corpus" do
      Grammar.seed(5)
      corpus = Grammar.corpus(300)
      relative = Enum.filter(corpus, fn words -> "who" in words and "and" not in words end)

      assert length(Eval.probes(corpus)) == length(relative)
      assert length(Eval.probes(corpus)) > 20
    end
  end

  describe "agreement/2" do
    setup do
      {:ok,
       probes: [
         %{prefix: ~w(the llama who chases the dogs), number: :singular},
         %{prefix: ~w(the dogs who chase the llama), number: :plural}
       ]}
    end

    test "scores a predictor that is always singular at exactly half", %{probes: probes} do
      assert Eval.agreement(always("flees"), probes) == 0.5
    end

    test "scores a predictor that is always plural at exactly half", %{probes: probes} do
      assert Eval.agreement(always("flee"), probes) == 0.5
    end

    test "ignores mass outside the verbs entirely", %{probes: probes} do
      # A model that mostly wants to say `llama` is not making an agreement
      # error. Rank among the verbs only, or the bigram loses for the wrong
      # reason and the contrast slide says nothing.
      distribution = fn word ->
        fn _prefix ->
          Vocab.words()
          |> Enum.map(fn candidate ->
            cond do
              candidate == "llama" -> 0.9
              candidate == word -> 0.1
              true -> 0.0
            end
          end)
        end
      end

      correct = [
        %{prefix: ~w(the llama who chases the dogs), number: :singular}
      ]

      assert Eval.agreement(distribution.("flees"), correct) == 1.0
      assert Eval.agreement(distribution.("flee"), correct) == 0.0
    end

    test "counts the copula as a verb", %{probes: probes} do
      # `is` and `are` carry number exactly as the others do, and the
      # grammar generates them 30% of the time.
      assert Eval.agreement(always("is"), probes) == 0.5
      assert Eval.agreement(always("are"), probes) == 0.5
    end

    test "has nothing to say about an empty probe set" do
      assert Eval.agreement(always("flees"), []) == 0.0
    end
  end

  describe "the three predictors" do
    test "a bigram predictor gives the same answer for any two prefixes ending alike" do
      # This is the whole of what one word of context buys, and it is less
      # of a handicap than it sounds: measured, the bigram scores 66.7% on
      # held-out probes, not 50%. A third of them end in the relative
      # clause's own verb, which already agrees with the head, so the cue
      # is adjacent and free. On the probes where a distractor noun
      # intervenes it scores 55.3%, and so does the Embedder, to the
      # decimal. That is the number the contrast slide should quote.
      Grammar.seed(2)
      matrix = Grammar.corpus(2_000) |> Bigram.counts() |> Bigram.matrix()
      predict = Eval.bigram_predictor(matrix)

      singular = predict.(~w(the llama who chases the dogs))
      plural = predict.(~w(the dogs who chase the llama who chases the dogs))

      assert singular == plural
    end

    test "a model predictor sees the whole prefix, so the same last word can differ" do
      Train.seed(1)
      params = Transformer.init(%Train.Config{model: Transformer})
      predict = Eval.model_predictor(params)

      refute predict.(~w(the llama who chases the dogs)) ==
               predict.(~w(the dogs who chase the dogs))
    end
  end

  describe "novelty/2" do
    test "is one when nothing was memorized" do
      assert Eval.novelty([~w(the llama flees .)], [~w(the dog flees .)]) == 1.0
    end

    test "is zero when everything was" do
      sentence = ~w(the llama flees .)

      assert Eval.novelty([sentence], [sentence]) == 0.0
    end

    test "counts duplicates in the generated set separately" do
      # Three generations of the same memorized sentence is three failures,
      # not one, because the number reported is a fraction of generations.
      sentence = ~w(the llama flees .)
      novel = ~w(the mice flee .)

      assert Eval.novelty([sentence, sentence, novel], [sentence]) == 1 / 3
    end
  end

  describe "grammatical?/1" do
    test "accepts what the grammar generates, all of it" do
      Grammar.seed(9)

      for sentence <- Grammar.corpus(500) do
        assert Eval.grammatical?(sentence), "rejected #{Enum.join(sentence, " ")}"
      end
    end

    test "rejects a word that is not in the vocabulary" do
      refute Eval.grammatical?(~w(the wombat flees .))
    end

    test "rejects a sentence that does not end in a period" do
      refute Eval.grammatical?(~w(the llama flees))
    end

    test "rejects a plural noun after the determiner a" do
      refute Eval.grammatical?(~w(a llamas flee .))
    end

    test "rejects an object after an intransitive verb" do
      refute Eval.grammatical?(~w(the llama flees the dog .))
    end

    test "rejects simple disagreement" do
      refute Eval.grammatical?(~w(the llama flee .))
    end

    test "rejects a compound subject with a singular verb" do
      refute Eval.grammatical?(~w(the llama and the dog flees .))
    end

    test "rejects a main verb agreeing with the distractor instead of the head" do
      # The case the whole talk is about, and the one the stage 1 checker
      # explicitly deferred to here.
      refute Eval.grammatical?(~w(the llama who chases the dogs flee .))
      refute Eval.grammatical?(~w(the dogs who chase the llama flees .))
    end

    test "rejects a relative-clause verb disagreeing with its head" do
      refute Eval.grammatical?(~w(the llama who chase the dogs flees .))
    end

    test "accepts a relative clause where both verbs agree with the head" do
      assert Eval.grammatical?(~w(the llama who chases the dogs flees .))
      assert Eval.grammatical?(~w(the dogs who chase the llama flee .))
    end
  end

  describe "grammaticality/1" do
    test "is the fraction that pass" do
      sentences = [
        ~w(the llama flees .),
        ~w(the llama flee .),
        ~w(the mice flee .),
        ~w(a llamas flee .)
      ]

      assert Eval.grammaticality(sentences) == 0.5
    end

    test "is one on the grammar's own output" do
      Grammar.seed(4)

      assert Eval.grammaticality(Grammar.corpus(200)) == 1.0
    end
  end
end
