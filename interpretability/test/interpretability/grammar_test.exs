defmodule Interpretability.GrammarTest do
  use ExUnit.Case, async: true

  alias Interpretability.Grammar
  alias Interpretability.Trees
  alias Interpretability.Vocab

  test "the same seed generates the same trees" do
    Grammar.seed(7)
    first = Grammar.trees(200)
    Grammar.seed(7)

    assert Grammar.trees(200) == first
  end

  test "every sentence fits the context and ends in a period" do
    Grammar.seed(1)

    for tree <- Grammar.trees(5_000) do
      words = Grammar.words(tree)
      assert length(words) <= Grammar.max_tokens()
      assert List.last(words) == "."
      assert Enum.all?(words, &(&1 in Vocab.words()))
    end
  end

  test "agreement holds by construction" do
    Grammar.seed(2)

    for tree <- Grammar.trees(5_000),
        %{tense: :present} = verb_phrase <- Trees.verb_phrases(tree) do
      expected =
        case verb_phrase.type do
          :copula -> Vocab.copula(:present, verb_phrase.number)
          :transitive -> Vocab.transitive_verbs(:present, verb_phrase.number)
          :intransitive -> present_intransitive(verb_phrase.number)
        end

      assert verb_phrase.verb in expected
    end
  end

  test "can produce the probe sentence" do
    assert Grammar.words(Trees.probe_sentence()) == ~w(the llama who chases the dogs flees .)

    Grammar.seed(3)

    assert Enum.any?(Grammar.trees(5_000), fn tree ->
             match?(%{subject: %{type: :relative, verb_phrase: %{type: :transitive}}}, tree) and
               tree.subject.head.number != tree.subject.verb_phrase.object.number
           end)
  end

  test "subject constructions appear at their target rates" do
    Grammar.seed(4)
    counts = Grammar.trees(5_000) |> Enum.map(& &1.subject.type) |> Enum.frequencies()

    for {type, target} <- [simple: 50, compound: 12, relative: 18, prepositional: 12, pronoun: 8] do
      percentage = 100 * Map.get(counts, type, 0) / 5_000
      assert abs(percentage - target) <= 3, "#{type}: #{percentage}% against #{target}%"
    end
  end

  ## PRIVATE FUNCTIONS

  defp present_intransitive(number),
    do:
      Vocab.intransitive_verbs(:animate, :present, number) ++
        Vocab.intransitive_verbs(:inanimate, :present, number)
end
