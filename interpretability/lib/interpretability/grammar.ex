defmodule Interpretability.Grammar do
  @moduledoc """
  A probabilistic grammar that writes the training corpus, one parse tree
  at a time.

  The first talk's grammar returned lists of words. This one returns the
  *tree* behind each sentence, because the tree is the ground truth: it
  says which noun each verb agrees with, which words sit inside a relative
  clause, and what role every noun plays. `Interpretability.Labeler` reads
  those answers off the tree for every token. The model only ever sees
  the words.

  ## The rules

      Sentence          -> SubjectPhrase VerbPhrase "."

      SubjectPhrase     -> Simple                           (50%)
                         | Simple "and" Simple              (12%, always plural)
                         | Simple RelativePronoun VerbPhrase (18%)
                         | Simple Preposition Simple        (12%)
                         | Pronoun                          (8%)

      ObjectPhrase      -> Simple                           (80%)
                         | Simple RelativePronoun VerbPhrase (20%, depth < 2)

      Simple            -> Determiner Adjective* Noun       (0, 1 or 2 adjectives)

      VerbPhrase        -> IntransitiveVerb                 (animate 30%, inanimate 40%)
                         | TransitiveVerb ObjectPhrase      (animate 40%)
                         | Copula Adjective                 (animate 30%, inanimate 60%)

  Tense is drawn for every verb phrase: present 70%, past 30%. Subjects are
  animate 80% of the time, objects 70%. Only animate subjects take
  transitive verbs, `flee` or `sleep`; rocks can only fall or be something.
  The relative pronoun is `who` for an animate noun and `that` for an
  inanimate one.

  ## The distractors

  Agreement is decided when a noun phrase picks its noun, and it travels
  only to that noun phrase's own verb. Every other noun between a subject
  and its verb is a distractor whose number the verb must ignore:

  - `the llama who chases the dogs flees`: the object of a relative clause.
  - `the llama near the dogs flees`: the object of a preposition.
  - `the fox sees the dogs who flee`: here the object `dogs` governs the
    relative clause's own verb, so the same noun is a distractor for one
    verb and the subject of another.

  Sentences longer than 20 tokens are thrown away and drawn again.
  """

  alias Interpretability.Vocab

  @type grammatical_number :: Vocab.grammatical_number()
  @type animacy :: Vocab.animacy()
  @type tense :: Vocab.tense()

  @type simple_phrase :: %{
          type: :simple,
          determiner: Vocab.word(),
          adjectives: [Vocab.word()],
          noun: Vocab.word(),
          number: grammatical_number(),
          animacy: animacy()
        }

  @type noun_phrase ::
          simple_phrase()
          | %{type: :compound, left: simple_phrase(), right: simple_phrase()}
          | %{
              type: :relative,
              head: simple_phrase(),
              relative_pronoun: Vocab.word(),
              verb_phrase: verb_phrase()
            }
          | %{
              type: :prepositional,
              head: simple_phrase(),
              preposition: Vocab.word(),
              object: simple_phrase()
            }
          | %{
              type: :pronoun,
              pronoun: Vocab.word(),
              number: grammatical_number(),
              animacy: animacy()
            }

  @type verb_phrase ::
          %{type: :intransitive, verb: Vocab.word(), tense: tense(), number: grammatical_number()}
          | %{
              type: :transitive,
              verb: Vocab.word(),
              tense: tense(),
              number: grammatical_number(),
              object: noun_phrase()
            }
          | %{
              type: :copula,
              verb: Vocab.word(),
              tense: tense(),
              number: grammatical_number(),
              adjective: Vocab.word()
            }

  @type sentence_tree :: %{subject: noun_phrase(), verb_phrase: verb_phrase()}

  @max_tokens 20
  @max_depth 2

  @doc """
  Seeds the process `:rand` state so a corpus reproduces.
  """
  @spec seed(integer()) :: :rand.state()
  def seed(seed), do: :rand.seed(:exsss, seed)

  @doc """
  The longest sentence allowed, counting the final period.
  """
  @spec max_tokens() :: pos_integer()
  def max_tokens, do: @max_tokens

  @doc """
  One sentence's parse tree. Resamples anything longer than `max_tokens/0`.
  """
  @spec tree() :: sentence_tree()
  def tree do
    {subject, number, animacy} = subject_phrase()
    candidate = %{subject: subject, verb_phrase: verb_phrase(number, animacy, 0)}

    if length(words(candidate)) > @max_tokens, do: tree(), else: candidate
  end

  @doc """
  `count` parse trees.
  """
  @spec trees(non_neg_integer()) :: [sentence_tree()]
  def trees(count), do: Enum.map(1..count//1, fn _index -> tree() end)

  @doc """
  The words of a sentence, ending in `"."`.
  """
  @spec words(sentence_tree()) :: [Vocab.word()]
  def words(%{subject: subject, verb_phrase: verb_phrase}),
    do: List.flatten([noun_phrase_words(subject), verb_phrase_words(verb_phrase), "."])

  ## PRIVATE FUNCTIONS

  defp subject_phrase do
    case :rand.uniform(100) do
      roll when roll <= 50 ->
        phrase = simple_phrase(animacy(80))
        {phrase, phrase.number, phrase.animacy}

      roll when roll <= 62 ->
        {%{type: :compound, left: simple_phrase(:animate), right: simple_phrase(:animate)},
         :plural, :animate}

      roll when roll <= 80 ->
        head = simple_phrase(animacy(80))
        {relative_phrase(head, 0), head.number, head.animacy}

      roll when roll <= 92 ->
        head = simple_phrase(animacy(80))

        {%{
           type: :prepositional,
           head: head,
           preposition: Enum.random(Vocab.prepositions()),
           object: simple_phrase(animacy(70))
         }, head.number, head.animacy}

      _roll ->
        phrase = pronoun_phrase()
        {phrase, phrase.number, phrase.animacy}
    end
  end

  # `depth` is the depth of the clause this object belongs to. A relative
  # clause on the object opens a clause one deeper.
  defp object_phrase(depth) do
    head = simple_phrase(animacy(70))

    if depth < @max_depth and :rand.uniform(100) <= 20,
      do: relative_phrase(head, depth),
      else: head
  end

  defp relative_phrase(head, depth) do
    %{
      type: :relative,
      head: head,
      relative_pronoun: relative_pronoun(head.animacy),
      verb_phrase: verb_phrase(head.number, head.animacy, depth + 1)
    }
  end

  defp simple_phrase(animacy) do
    number = Enum.random([:singular, :plural])

    %{
      type: :simple,
      determiner: determiner(number),
      adjectives: adjectives(),
      noun: Enum.random(Vocab.nouns(animacy, number)),
      number: number,
      animacy: animacy
    }
  end

  defp pronoun_phrase do
    case Enum.random(Vocab.pronouns()) do
      "it" -> %{type: :pronoun, pronoun: "it", number: :singular, animacy: :inanimate}
      "they" -> %{type: :pronoun, pronoun: "they", number: :plural, animacy: animacy(80)}
      pronoun -> %{type: :pronoun, pronoun: pronoun, number: :singular, animacy: :animate}
    end
  end

  defp verb_phrase(number, :animate, depth) do
    tense = tense()

    case :rand.uniform(100) do
      roll when roll <= 30 ->
        intransitive(Vocab.intransitive_verbs(:animate, tense, number), tense, number)

      roll when roll <= 70 ->
        %{
          type: :transitive,
          verb: Enum.random(Vocab.transitive_verbs(tense, number)),
          tense: tense,
          number: number,
          object: object_phrase(depth)
        }

      _roll ->
        copula(tense, number)
    end
  end

  defp verb_phrase(number, :inanimate, _depth) do
    tense = tense()

    if :rand.uniform(100) <= 40,
      do: intransitive(Vocab.intransitive_verbs(:inanimate, tense, number), tense, number),
      else: copula(tense, number)
  end

  defp intransitive(verbs, tense, number),
    do: %{type: :intransitive, verb: Enum.random(verbs), tense: tense, number: number}

  defp copula(tense, number) do
    %{
      type: :copula,
      verb: hd(Vocab.copula(tense, number)),
      tense: tense,
      number: number,
      adjective: Enum.random(Vocab.adjectives())
    }
  end

  defp noun_phrase_words(%{type: :simple} = phrase),
    do: [phrase.determiner, phrase.adjectives, phrase.noun]

  defp noun_phrase_words(%{type: :compound, left: left, right: right}),
    do: [noun_phrase_words(left), "and", noun_phrase_words(right)]

  defp noun_phrase_words(%{type: :relative} = phrase),
    do: [
      noun_phrase_words(phrase.head),
      phrase.relative_pronoun,
      verb_phrase_words(phrase.verb_phrase)
    ]

  defp noun_phrase_words(%{type: :prepositional} = phrase),
    do: [noun_phrase_words(phrase.head), phrase.preposition, noun_phrase_words(phrase.object)]

  defp noun_phrase_words(%{type: :pronoun, pronoun: pronoun}), do: [pronoun]

  defp verb_phrase_words(%{type: :intransitive, verb: verb}), do: [verb]

  defp verb_phrase_words(%{type: :transitive, verb: verb, object: object}),
    do: [verb, noun_phrase_words(object)]

  defp verb_phrase_words(%{type: :copula, verb: verb, adjective: adjective}),
    do: [verb, adjective]

  defp animacy(animate_percentage),
    do: if(:rand.uniform(100) <= animate_percentage, do: :animate, else: :inanimate)

  defp tense, do: if(:rand.uniform(100) <= 70, do: :present, else: :past)

  # "a" introduces a singular noun only; "the" works for both.
  defp determiner(:singular), do: Enum.random(Vocab.determiners())
  defp determiner(:plural), do: "the"

  defp adjectives do
    case :rand.uniform(100) do
      roll when roll <= 55 -> []
      roll when roll <= 85 -> [Enum.random(Vocab.adjectives())]
      _roll -> [Enum.random(Vocab.adjectives()), Enum.random(Vocab.adjectives())]
    end
  end

  defp relative_pronoun(:animate), do: "who"
  defp relative_pronoun(:inanimate), do: "that"
end
