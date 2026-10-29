defmodule Interpretability.LabelerTest do
  use ExUnit.Case, async: true

  alias Interpretability.Grammar
  alias Interpretability.Labeler
  alias Interpretability.Trees

  @keys [
    :word,
    :part_of_speech,
    :word_number,
    :agreement_number,
    :role,
    :depth,
    :tense,
    :animacy,
    :after_main_verb,
    :can_end
  ]

  describe "the llama who chases the dogs flees ." do
    setup do
      %{tokens: Labeler.label(Trees.probe_sentence())}
    end

    test "words come out in sentence order", %{tokens: tokens} do
      assert Enum.map(tokens, & &1.word) == ~w(the llama who chases the dogs flees .)
    end

    test "the next verb's number comes from left context only", %{tokens: tokens} do
      assert column(tokens, :agreement_number) ==
               [nil, :singular, :singular, :singular, :singular, :singular, nil, nil]
    end

    test "depth, role and main verb position", %{tokens: tokens} do
      assert column(tokens, :depth) == [0, 0, 1, 1, 1, 1, 0, 0]

      assert column(tokens, :role) ==
               [:subject, :subject, nil, nil, :object, :object, nil, nil]

      assert column(tokens, :after_main_verb) ==
               [false, false, false, false, false, false, true, true]
    end

    test "the sentence could end only after the main verb", %{tokens: tokens} do
      assert column(tokens, :can_end) ==
               [false, false, false, false, false, false, true, false]
    end

    test "parts of speech, numbers, tense and animacy", %{tokens: tokens} do
      assert column(tokens, :part_of_speech) == [
               :determiner,
               :noun,
               :relative_pronoun,
               :transitive_verb,
               :determiner,
               :noun,
               :intransitive_verb,
               :end
             ]

      assert column(tokens, :word_number) ==
               [nil, :singular, nil, :singular, nil, :plural, :singular, nil]

      assert column(tokens, :tense) == [nil, nil, nil, :present, nil, nil, :present, nil]

      assert column(tokens, :animacy) ==
               [nil, :animate, nil, nil, nil, :animate, nil, nil]
    end
  end

  test "a relative clause on the object opens its own agreement" do
    tokens = Labeler.label(Trees.object_relative_sentence())

    assert column(tokens, :agreement_number) ==
             [nil, :singular, nil, nil, nil, :plural, nil, nil]

    assert Enum.at(tokens, 4).labels.role == :object
    assert Enum.at(tokens, 6).labels.depth == 1

    # "the fox sees the dogs" is already a sentence; "the fox sees the" is not.
    assert column(tokens, :can_end) == [false, false, false, false, true, false, true, false]
  end

  test "a copula's adjective can end the sentence, an adjective before a noun cannot" do
    tree = %{
      subject: %{
        type: :simple,
        determiner: "the",
        adjectives: ["big"],
        noun: "rock",
        number: :singular,
        animacy: :inanimate
      },
      verb_phrase: %{
        type: :copula,
        verb: "is",
        tense: :present,
        number: :singular,
        adjective: "wet"
      }
    }

    assert column(Labeler.label(tree), :can_end) == [false, false, false, false, true, false]
  end

  test "past tense verbs carry no number" do
    tokens = Labeler.label(Trees.past_tense_sentence())

    assert Enum.at(tokens, 2).labels.word_number == nil
    assert Enum.at(tokens, 2).labels.tense == :past
  end

  test "start labels mark only the start" do
    assert Labeler.start_labels() == %{
             word: "<start>",
             part_of_speech: :start,
             word_number: nil,
             agreement_number: nil,
             role: nil,
             depth: 0,
             tense: nil,
             animacy: nil,
             after_main_verb: false,
             can_end: false
           }
  end

  test "every token of every generated sentence carries exactly the ten labels" do
    Grammar.seed(5)

    for tree <- Grammar.trees(2_000) do
      tokens = Labeler.label(tree)
      assert Enum.map(tokens, & &1.word) == Grammar.words(tree)
      assert Enum.all?(tokens, &(Enum.sort(Map.keys(&1.labels)) == Enum.sort(@keys)))
    end
  end

  ## PRIVATE FUNCTIONS

  defp column(tokens, key), do: Enum.map(tokens, & &1.labels[key])
end
