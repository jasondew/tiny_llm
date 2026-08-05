defmodule TinyLlm.Test.Structure do
  @moduledoc """
  A structural checker for generated sentences, used to assert that the
  grammar is grammatical by construction.

  This is deliberately independent of `TinyLlm.Grammar`: it re-derives the
  word classes from the brief rather than importing them, so a bug in the
  generator cannot hide behind the same bug in the checker.

  It only judges simple clauses, meaning those with no `who` and no `and`.
  Relative clauses and compound subjects need the head-noun tracking that
  stage 6 introduces, and are skipped here rather than guessed at.
  """

  @determiners ~w(the a)
  @singular_nouns ~w(cat dog bird fox mouse)
  @plural_nouns ~w(cats dogs birds foxes mice)
  @singular_verbs ~w(sees chases eats sleeps is)
  @plural_verbs ~w(see chase eat sleep are)
  @adjectives ~w(big small hungry happy fast old)

  @doc """
  True when a sentence has no relative clause and no compound subject.
  """
  def simple?(words), do: "who" not in words and "and" not in words

  @doc """
  True when the subject noun and its verb match in number.

  Returns false for anything it cannot parse, so a malformed sentence
  fails the test rather than passing it by default.
  """
  def agreement_ok?(words) do
    with [determiner | after_determiner] <- strip_period(words),
         true <- determiner in @determiners,
         {noun, [verb | _rest]} <- take_noun(after_determiner),
         subject_number when not is_nil(subject_number) <- noun_number(noun),
         predicate_number when not is_nil(predicate_number) <- verb_number(verb) do
      subject_number == predicate_number
    else
      _unparseable -> false
    end
  end

  @doc """
  True when every occurrence of `a` introduces a singular noun.

  The determiner `the` works for both numbers, so it is not constrained.
  """
  def determiners_ok?(words) do
    words |> strip_period() |> check_determiners()
  end

  @doc """
  The number of a noun, or nil when the word is not a noun.
  """
  def noun_number(word) when word in @singular_nouns, do: :sg
  def noun_number(word) when word in @plural_nouns, do: :pl
  def noun_number(_word), do: nil

  @doc """
  The number of a verb or copula, or nil when the word is not one.
  """
  def verb_number(word) when word in @singular_verbs, do: :sg
  def verb_number(word) when word in @plural_verbs, do: :pl
  def verb_number(_word), do: nil

  ## PRIVATE FUNCTIONS

  defp strip_period(words) do
    case List.last(words) do
      "." -> Enum.drop(words, -1)
      _other -> words
    end
  end

  defp check_determiners([]), do: true

  defp check_determiners(["a" | remaining]) do
    case take_noun(remaining) do
      {noun, rest} -> noun_number(noun) == :sg and check_determiners(rest)
      :error -> false
    end
  end

  defp check_determiners([_word | remaining]), do: check_determiners(remaining)

  defp take_noun([word | remaining]) when word in @adjectives, do: take_noun(remaining)
  defp take_noun([word | remaining]), do: {word, remaining}
  defp take_noun([]), do: :error
end
