defmodule TinyLlm.Test.Structure do
  @moduledoc """
  A structural checker for generated sentences, used to assert that the
  grammar is grammatical by construction.

  This is deliberately independent of `TinyLlm.Grammar`: it re-derives the
  word classes from the brief rather than importing them, so a bug in the
  generator cannot hide behind the same bug in the checker.

  Agreement is judged only in simple clauses, meaning those with no `who`
  and no `and`, plus the one extra case of a compound subject without a
  relative clause. Relative-clause agreement needs the head-noun tracking
  that stage 6 introduces, and is skipped here rather than guessed at.
  """

  @determiners ~w(the a)
  @singular_nouns ~w(llama dog goose fox mouse)
  @plural_nouns ~w(llamas dogs geese foxes mice)
  @singular_verbs ~w(sees chases ignores flees is)
  @plural_verbs ~w(see chase ignore flee are)
  @intransitive_verbs ~w(flees flee)
  @adjectives ~w(big small hungry grumpy fast sleepy)

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
  True when a sentence has a compound subject and no relative clause.
  """
  def compound?(words), do: "and" in words and "who" not in words

  @doc """
  True when the verb after a compound subject is plural.

  A compound subject is `:plural` no matter what its conjuncts are, so the verb
  following the last conjunct settles it. Everything after that verb is the
  object, which cannot contain `and`, so the last `and` in the sentence is
  reliably the one joining the subject.
  """
  def compound_agreement_ok?(words) do
    with [determiner | after_determiner] <- last_conjunct(words),
         true <- determiner in @determiners,
         {_noun, [verb | _rest]} <- take_noun(after_determiner),
         predicate_number when not is_nil(predicate_number) <- verb_number(verb) do
      predicate_number == :plural
    else
      _unparseable -> false
    end
  end

  @doc """
  True when no intransitive verb is followed by an object.

  `flees` and `flee` never take one, so the word after them can never be
  the determiner that would open an object noun phrase.
  """
  def intransitive_ok?(words) do
    words
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.all?(fn [word, next_word] ->
      word not in @intransitive_verbs or next_word not in @determiners
    end)
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
  def noun_number(word) when word in @singular_nouns, do: :singular
  def noun_number(word) when word in @plural_nouns, do: :plural
  def noun_number(_word), do: nil

  @doc """
  The number of a verb or copula, or nil when the word is not one.
  """
  def verb_number(word) when word in @singular_verbs, do: :singular
  def verb_number(word) when word in @plural_verbs, do: :plural
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
      {noun, rest} -> noun_number(noun) == :singular and check_determiners(rest)
      :error -> false
    end
  end

  defp check_determiners([_word | remaining]), do: check_determiners(remaining)

  defp last_conjunct(words) do
    reversed = words |> strip_period() |> Enum.reverse()

    if "and" in reversed do
      reversed |> Enum.take_while(&(&1 != "and")) |> Enum.reverse()
    else
      :error
    end
  end

  defp take_noun([word | remaining]) when word in @adjectives, do: take_noun(remaining)
  defp take_noun([word | remaining]), do: {word, remaining}
  defp take_noun([]), do: :error
end
