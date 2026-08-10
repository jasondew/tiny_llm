defmodule TinyLlm.Grammar do
  @moduledoc """
  A probabilistic context-free grammar that writes the training corpus.

  The rules, from docs/build-brief.md:

      Sentence        -> NounPhrase VerbPhrase "."

      NounPhrase      -> Determiner AdjectivePhrase Noun (70%)
                       | NounPhrase "and" NounPhrase (15%)
                       | Determiner AdjectivePhrase Noun "who" VerbPhrase (15%)

      AdjectivePhrase -> nothing (55%)
                       | Adjective (30%)
                       | Adjective Adjective (15%)

      VerbPhrase      -> IntransitiveVerb (35%)
                       | TransitiveVerb NounPhrase (35%)
                       | Copula Adjective (30%)

  The last two noun phrase rules, the compound and the relative clause,
  apply only in subject position, where those percentages hold. An object
  noun phrase is always the plain first rule, at 100%.

  Agreement is enforced at sampling time, so every sentence is grammatical
  by construction: a `:singular | :plural` flag threads from the subject
  noun phrase into its verb phrase, a relative-clause verb agrees with the
  head noun, and a compound subject is always `:plural`.

  Number is decided in exactly one place, when a noun phrase picks its noun.
  It then travels up to the caller, which hands it down to the verb phrase.
  A noun phrase in object position picks a number too, but nobody upstream
  ever hears about it, and that discarded number is what makes the object a
  distractor in `the llama who chases the dogs flees`.
  """

  alias TinyLlm.Vocab

  @typedoc "The agreement feature threaded from a subject into its verb."
  @type grammatical_number :: :singular | :plural

  @typedoc "A generated sentence: vocabulary words, always ending in a period."
  @type sentence :: [Vocab.word()]

  @typedoc "A phrase and the number it committed to."
  @type numbered_phrase :: {[Vocab.word()], grammatical_number()}

  @max_tokens 16

  @doc """
  Seeds the process `:rand` state so a corpus reproduces.
  """
  @spec seed(integer()) :: :rand.state()
  def seed(seed), do: :rand.seed(:exsss, seed)

  @doc """
  One sentence, as a list of vocabulary words ending in `"."`.

  Resamples anything longer than the 16 token context length.
  """
  @spec sentence() :: sentence()
  def sentence do
    {subject, number} = subject_noun_phrase()
    candidate = List.flatten([subject, verb_phrase(number), "."])

    if length(candidate) > @max_tokens, do: sentence(), else: candidate
  end

  @doc """
  A corpus of `count` sentences.
  """
  @spec corpus(non_neg_integer()) :: [sentence()]
  def corpus(count), do: Enum.map(1..count//1, fn _index -> sentence() end)

  ## PRIVATE FUNCTIONS

  # Only subject position may compound or take a relative clause.
  @spec subject_noun_phrase() :: numbered_phrase()
  defp subject_noun_phrase do
    case :rand.uniform(100) do
      n when n <= 70 -> simple_noun_phrase()
      n when n <= 85 -> compound_noun_phrase()
      _ -> relative_noun_phrase()
    end
  end

  # Object position is always simple, and its number is discarded.
  @spec object_noun_phrase() :: [Vocab.word()]
  defp object_noun_phrase do
    {words, _discarded_number} = simple_noun_phrase()
    words
  end

  @spec simple_noun_phrase() :: numbered_phrase()
  defp simple_noun_phrase do
    number = number()
    {[determiner(number), adjective_phrase(), noun(number)], number}
  end

  # Conjuncts are simple noun phrases, which keeps sentences inside the
  # context length. The compound is plural whatever its conjuncts chose.
  @spec compound_noun_phrase() :: numbered_phrase()
  defp compound_noun_phrase do
    {left, _left_number} = simple_noun_phrase()
    {right, _right_number} = simple_noun_phrase()

    {[left, "and", right], :plural}
  end

  # The head noun's number goes into the relative clause verb AND back up to
  # the main verb. This is the sentence the talk is built around.
  @spec relative_noun_phrase() :: numbered_phrase()
  defp relative_noun_phrase do
    number = number()
    head = [determiner(number), adjective_phrase(), noun(number)]

    {[head, "who", verb_phrase(number)], number}
  end

  @spec verb_phrase(grammatical_number()) :: [Vocab.word()]
  defp verb_phrase(number) do
    case :rand.uniform(100) do
      n when n <= 35 -> [intransitive_verb(number)]
      n when n <= 70 -> [transitive_verb(number), object_noun_phrase()]
      _ -> [copula(number), adjective()]
    end
  end

  @spec adjective_phrase() :: [Vocab.word()]
  defp adjective_phrase do
    case :rand.uniform(100) do
      n when n <= 55 -> []
      n when n <= 85 -> [adjective()]
      _ -> [adjective(), adjective()]
    end
  end

  @spec number() :: grammatical_number()
  defp number, do: Enum.random([:singular, :plural])

  # "a" introduces a singular noun only; "the" works for both.
  @spec determiner(grammatical_number()) :: Vocab.word()
  defp determiner(:singular), do: Enum.random(Vocab.determiners())
  defp determiner(:plural), do: "the"

  @spec noun(grammatical_number()) :: Vocab.word()
  defp noun(number), do: Enum.random(Vocab.nouns(number))

  @spec intransitive_verb(grammatical_number()) :: Vocab.word()
  defp intransitive_verb(number), do: Enum.random(Vocab.intransitive_verbs(number))

  @spec transitive_verb(grammatical_number()) :: Vocab.word()
  defp transitive_verb(number), do: Enum.random(Vocab.transitive_verbs(number))

  @spec copula(grammatical_number()) :: Vocab.word()
  defp copula(number), do: Enum.random(Vocab.copula(number))

  @spec adjective() :: Vocab.word()
  defp adjective, do: Enum.random(Vocab.adjectives())
end
