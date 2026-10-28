defmodule Interpretability.VocabTest do
  use ExUnit.Case, async: true

  alias Interpretability.Vocab

  test "holds 70 unique words" do
    assert Vocab.size() == 70
    assert Enum.uniq(Vocab.words()) == Vocab.words()
  end

  test "every word round trips through its id" do
    for word <- Vocab.words() do
      assert word |> Vocab.word_to_id() |> Vocab.id_to_word() == word
    end

    assert Vocab.decode(Vocab.encode(~w(the llama flees))) == ~w(the llama flees)
  end

  test "category accessors pick by animacy, tense and number" do
    assert "llamas" in Vocab.nouns(:animate, :plural)
    assert "rocks" in Vocab.nouns(:inanimate, :plural)
    assert Vocab.copula(:past, :plural) == ["were"]
    assert Vocab.copula(:present, :singular) == ["is"]
    assert "chases" in Vocab.transitive_verbs(:present, :singular)
    assert Vocab.transitive_verbs(:past, :singular) == Vocab.transitive_verbs(:past, :plural)
    assert Vocab.intransitive_verbs(:inanimate, :present, :plural) == ["fall"]
    assert Vocab.intransitive_verbs(:animate, :past, :singular) == ~w(fled slept)
  end

  test "unknown words and ids raise" do
    assert_raise RuntimeError, fn -> Vocab.word_to_id("zebra") end
    assert_raise RuntimeError, fn -> Vocab.id_to_word(70) end
  end
end
