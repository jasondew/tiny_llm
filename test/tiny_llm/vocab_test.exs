defmodule TinyLlm.VocabTest do
  @moduledoc """
  Stage 1a acceptance tests, written from the criteria in docs/build-brief.md.

  These are a specification, not a description of existing code.

  The brief fixes the function names but not every shape. Where it was
  silent, these tests pin `word_to_id/1` and `id_to_word/1` to rejecting
  anything outside the vocabulary with a `RuntimeError` naming the offending
  word or id, rather than letting a bad lookup return `nil` and ride into
  the embedding table.

  The message text is asserted on purpose. A descriptive failure is the
  whole reason for raising here rather than pattern-matching, so it is
  worth holding onto.
  """

  use ExUnit.Case, async: true

  alias TinyLlm.Test.Vocabulary
  alias TinyLlm.Vocab

  test "holds exactly 32 words" do
    assert Vocab.size() == 32
  end

  test "lists the 32 words in the order the brief fixes" do
    assert Vocab.words() == Vocabulary.words()
  end

  test "assigns every word a distinct id covering 0 through 31" do
    ids = Enum.map(Vocabulary.words(), &Vocab.word_to_id/1)

    assert Enum.sort(ids) == Enum.to_list(0..31)
  end

  test "round-trips every word through its id" do
    for word <- Vocabulary.words() do
      assert word |> Vocab.word_to_id() |> Vocab.id_to_word() == word
    end
  end

  test "anchors the first and last ids to the fixed ordering" do
    assert Vocab.word_to_id("the") == 0
    assert Vocab.word_to_id(".") == 31
  end

  test "round-trips a sentence through encode and decode" do
    sentence = ~w(the big llama sees a mouse .)

    assert sentence |> Vocab.encode() |> Vocab.decode() == sentence
  end

  test "refuses a word outside the vocabulary rather than inventing an id" do
    assert_raise RuntimeError, "Unknown word: elephant", fn -> Vocab.word_to_id("elephant") end
  end

  test "refuses an id outside the vocabulary" do
    assert_raise RuntimeError, "Unknown id: 32", fn -> Vocab.id_to_word(32) end
  end
end
