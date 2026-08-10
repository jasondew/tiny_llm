defmodule TinyLlm.GrammarTest do
  @moduledoc """
  Stage 1b acceptance tests, written from the criteria in docs/build-brief.md.

  These are a specification, not a description of existing code.

  The brief fixes the function names but not every shape. Where it was
  silent, these tests assume that `sentence/0` returns a list of word
  strings rather than token ids. That is an assumption, not a requirement.

  The structural checks come from `TinyLlm.Test.Structure`, which re-derives
  the word classes from the brief instead of importing them, so a bug in the
  generator cannot hide behind the same bug in the checker.
  """

  use ExUnit.Case, async: true

  alias TinyLlm.Grammar
  alias TinyLlm.Test.Structure
  alias TinyLlm.Test.Vocabulary

  setup do
    Grammar.seed(1234)
    :ok
  end

  test "draws one sentence as a list of vocabulary words ending in a period" do
    sentence = Grammar.sentence()

    assert is_list(sentence)
    assert Enum.all?(sentence, &(&1 in Vocabulary.words()))
    assert List.last(sentence) == "."
  end

  test "draws a corpus of the requested size" do
    assert length(Grammar.corpus(37)) == 37
  end

  test "draws the same corpus twice from the same seed" do
    Grammar.seed(99)
    first = Grammar.corpus(50)

    Grammar.seed(99)
    second = Grammar.corpus(50)

    assert first == second
  end

  test "builds every sentence out of vocabulary words" do
    unknown =
      10_000
      |> Grammar.corpus()
      |> Enum.flat_map(fn sentence -> Enum.reject(sentence, &(&1 in Vocabulary.words())) end)
      |> Enum.uniq()

    assert unknown == []
  end

  test "ends every sentence with a period" do
    refute Enum.any?(Grammar.corpus(10_000), &(List.last(&1) != "."))
  end

  test "keeps every sentence within the 16 token context length" do
    longest = 10_000 |> Grammar.corpus() |> Enum.map(&length/1) |> Enum.max()

    assert longest <= 16
  end

  test "agrees subject and verb in number in simple clauses" do
    disagreeing =
      10_000
      |> Grammar.corpus()
      |> Enum.filter(&Structure.simple?/1)
      |> Enum.reject(&Structure.agreement_ok?/1)

    assert disagreeing == []
  end

  test "never places the determiner a before a plural noun" do
    offending =
      10_000
      |> Grammar.corpus()
      |> Enum.reject(&Structure.determiners_ok?/1)

    assert offending == []
  end

  test "agrees a compound subject with a plural verb" do
    disagreeing =
      10_000
      |> Grammar.corpus()
      |> Enum.filter(&Structure.compound?/1)
      |> Enum.reject(&Structure.compound_agreement_ok?/1)

    assert disagreeing == []
  end

  test "never gives an intransitive verb an object" do
    offending =
      10_000
      |> Grammar.corpus()
      |> Enum.reject(&Structure.intransitive_ok?/1)

    assert offending == []
  end

  test "produces relative clauses and compound subjects" do
    words = 10_000 |> Grammar.corpus() |> List.flatten() |> MapSet.new()

    assert "who" in words
    assert "and" in words
  end

  test "keeps relative clauses and compound subjects near their stated rates" do
    corpus = Grammar.corpus(10_000)
    relative_clauses = Enum.count(corpus, &("who" in &1)) / 10_000
    compounds = Enum.count(corpus, &("and" in &1)) / 10_000

    assert relative_clauses > 0.05 and relative_clauses < 0.30
    assert compounds > 0.05 and compounds < 0.30
  end

  test "generates mostly distinct sentences rather than memorizable few" do
    corpus = Grammar.corpus(5_000)
    unique_fraction = corpus |> Enum.uniq() |> length() |> Kernel./(5_000)

    assert unique_fraction > 0.6
  end
end
