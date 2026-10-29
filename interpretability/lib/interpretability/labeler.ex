defmodule Interpretability.Labeler do
  @moduledoc """
  The answer key: every token of a sentence, with the facts the grammar
  knows about it.

  The model reads only words. The grammar knows much more about each word:
  whether it belongs to the subject, how deep inside relative clauses it
  sits, and which number the next verb will need. This module walks a parse
  tree from `Interpretability.Grammar` and writes those facts onto every
  token as *labels*.

  Labels are never shown to the model. They exist so that, once we find a
  unit inside the model that seems to mean something, we can check it
  against the truth: does it switch on exactly where a label is true? That
  check is what turns "this looks like a plural detector" into a precision
  and a recall.

  ## The labels

  | Label | Values |
  | --- | --- |
  | `:word` | the word itself |
  | `:part_of_speech` | `:determiner`, `:noun`, `:pronoun`, `:transitive_verb`, `:intransitive_verb`, `:copula`, `:adjective`, `:relative_pronoun`, `:conjunction`, `:preposition`, `:start`, `:end` |
  | `:word_number` | the word's own number, where it has one |
  | `:agreement_number` | the number the next verb needs, from the words so far |
  | `:role` | `:subject`, `:object` or `:prepositional_object`, on the determiners, adjectives, nouns and pronouns of a noun phrase |
  | `:depth` | how many relative clauses enclose the word |
  | `:tense` | `:present` or `:past`, on verbs |
  | `:animacy` | `:animate` or `:inanimate`, on nouns and pronouns |
  | `:after_main_verb` | whether the sentence's main verb has been read |
  | `:can_end` | whether the words so far already make a whole sentence |

  ## Only the words so far

  `:agreement_number` answers: if a verb came next, which number would it
  need? It may use only the words already read, because that is all the
  model has. In `the fox sees the dogs who flee`, the label at `dogs` is
  `nil`: the main verb is done, and nothing yet says a relative clause is
  coming. It turns `:plural` at `who`, which opens a clause whose subject
  is `dogs`. A label that peeked ahead would reward the model for knowing
  the future.

  Internally this is a stack of clauses still waiting for their verb. A
  clause goes on the stack when it opens and comes off when its verb is
  read. The label is the number on top of the stack, once that number is
  known.

  `:can_end` is the same idea: the sentence could stop here when no clause
  is still waiting for its verb and this word finishes a phrase (a noun, a
  pronoun, an intransitive verb, or the adjective after `is`). It was
  added after a neuron turned up that fires on exactly those words, a
  concept the first nine labels did not have.
  """

  alias Interpretability.Grammar
  alias Interpretability.Vocab

  @typedoc "The ten facts known about one token."
  @type labels :: %{
          word: Vocab.word(),
          part_of_speech: atom(),
          word_number: Vocab.grammatical_number() | nil,
          agreement_number: Vocab.grammatical_number() | nil,
          role: :subject | :object | :prepositional_object | nil,
          depth: non_neg_integer(),
          tense: Vocab.tense() | nil,
          animacy: Vocab.animacy() | nil,
          after_main_verb: boolean(),
          can_end: boolean()
        }

  @type token :: %{word: Vocab.word(), labels: labels()}

  @doc """
  Every word of the sentence, `"."` included, with its labels.
  """
  @spec label(Grammar.sentence_tree()) :: [token()]
  def label(%{subject: subject, verb_phrase: verb_phrase}) do
    state = %{stack: [:unknown], after_main_verb: false, tokens: []}
    main = %{depth: 0, role: nil}

    state
    |> noun_phrase(subject, %{main | role: :subject}, :clause_subject)
    |> verb_phrase(verb_phrase, main, :main)
    |> emit(".", :end, %{}, main)
    |> Map.fetch!(:tokens)
    |> Enum.reverse()
  end

  @doc """
  The labels of the `<start>` token that begins every input.
  """
  @spec start_labels() :: labels()
  def start_labels do
    %{
      word: Vocab.start_token(),
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

  ## PRIVATE FUNCTIONS

  # `subject_of` says whether this noun phrase's head noun decides the
  # number of the clause on top of the stack (:clause_subject) or not
  # (:not_subject).
  defp noun_phrase(state, %{type: :simple} = phrase, context, subject_of) do
    state
    |> emit(phrase.determiner, :determiner, %{word_number: determiner_number(phrase)}, context)
    |> then(fn state ->
      Enum.reduce(phrase.adjectives, state, &emit(&2, &1, :adjective, %{}, context))
    end)
    |> settle_subject(phrase.number, subject_of)
    |> emit(phrase.noun, :noun, %{word_number: phrase.number, animacy: phrase.animacy}, context)
  end

  defp noun_phrase(state, %{type: :compound, left: left, right: right}, context, subject_of) do
    state
    |> noun_phrase(left, context, subject_of)
    |> settle_subject(:plural, subject_of)
    |> emit("and", :conjunction, %{}, %{context | role: nil})
    |> noun_phrase(right, context, :not_subject)
  end

  defp noun_phrase(state, %{type: :relative} = phrase, context, subject_of) do
    clause = %{depth: context.depth + 1, role: nil}

    state
    |> noun_phrase(phrase.head, context, subject_of)
    |> Map.update!(:stack, &[phrase.head.number | &1])
    |> emit(phrase.relative_pronoun, :relative_pronoun, %{}, clause)
    |> verb_phrase(phrase.verb_phrase, clause, :relative)
  end

  defp noun_phrase(state, %{type: :prepositional} = phrase, context, subject_of) do
    state
    |> noun_phrase(phrase.head, context, subject_of)
    |> emit(phrase.preposition, :preposition, %{}, %{context | role: nil})
    |> noun_phrase(phrase.object, %{context | role: :prepositional_object}, :not_subject)
  end

  defp noun_phrase(state, %{type: :pronoun} = phrase, context, subject_of) do
    state
    |> settle_subject(phrase.number, subject_of)
    |> emit(
      phrase.pronoun,
      :pronoun,
      %{word_number: phrase.number, animacy: phrase.animacy},
      context
    )
  end

  # Reading a clause's verb takes that clause off the stack, so the label
  # at the verb already looks ahead to whichever verb is still owed.
  defp verb_phrase(state, phrase, context, clause_kind) do
    state =
      state
      |> Map.update!(:stack, &tl/1)
      |> Map.update!(:after_main_verb, &(&1 or clause_kind == :main))
      |> emit(phrase.verb, verb_part_of_speech(phrase.type), verb_labels(phrase), context)

    case phrase do
      %{type: :transitive, object: object} ->
        noun_phrase(state, object, %{context | role: :object}, :not_subject)

      %{type: :copula, adjective: adjective} ->
        emit(state, adjective, :adjective, %{}, context, true)

      %{type: :intransitive} ->
        state
    end
  end

  defp settle_subject(%{stack: [:unknown | rest]} = state, number, :clause_subject),
    do: %{state | stack: [number | rest]}

  defp settle_subject(%{stack: [_number | rest]} = state, :plural, :clause_subject),
    do: %{state | stack: [:plural | rest]}

  defp settle_subject(state, _number, _subject_of), do: state

  # `finishes_phrase` says whether this word completes a phrase, so that
  # the sentence could stop after it. Nouns, pronouns and intransitive
  # verbs do; a copula's adjective is passed in explicitly.
  defp emit(
         state,
         word,
         part_of_speech,
         extra,
         context,
         finishes_phrase \\ nil
       ) do
    finishes_phrase =
      if finishes_phrase == nil,
        do: part_of_speech in [:noun, :pronoun, :intransitive_verb],
        else: finishes_phrase

    labels =
      Map.merge(
        %{
          word: word,
          part_of_speech: part_of_speech,
          word_number: nil,
          agreement_number: agreement_number(state.stack),
          role: context.role,
          depth: context.depth,
          tense: nil,
          animacy: nil,
          after_main_verb: state.after_main_verb,
          can_end: finishes_phrase and state.stack == []
        },
        extra
      )

    %{state | tokens: [%{word: word, labels: labels} | state.tokens]}
  end

  defp agreement_number([number | _rest]) when number in [:singular, :plural], do: number
  defp agreement_number(_stack), do: nil

  defp determiner_number(%{determiner: "a"}), do: :singular
  defp determiner_number(_phrase), do: nil

  defp verb_part_of_speech(:transitive), do: :transitive_verb
  defp verb_part_of_speech(:intransitive), do: :intransitive_verb
  defp verb_part_of_speech(:copula), do: :copula

  # Past tense verbs lose their number, except the copula: was and were.
  defp verb_labels(%{type: :copula} = phrase),
    do: %{tense: phrase.tense, word_number: phrase.number}

  defp verb_labels(%{tense: :present} = phrase),
    do: %{tense: :present, word_number: phrase.number}

  defp verb_labels(%{tense: :past}), do: %{tense: :past}
end
