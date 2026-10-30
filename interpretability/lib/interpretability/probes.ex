defmodule Interpretability.Probes do
  @moduledoc """
  Agreement probes: did the model pick a verb with the right number?

  A probe is a sentence cut off just before a present tense verb. The
  model is asked what comes next, and the probe passes when its favourite
  present tense verb has the right number. Past tense verbs are skipped,
  because `chased` looks the same for one llama or many.

  Probes are grouped by *category*, the construction the verb sits in,
  because the interesting question is not "is the model right?" but
  "where is it right?":

  | Category | Example, cut before the verb |
  | --- | --- |
  | `:simple_subject` | `the llamas` |
  | `:compound_subject` | `the llama and the dog` |
  | `:pronoun_subject` | `they` |
  | `:relative_distractor` | `the llama who chases the dogs` |
  | `:prepositional_distractor` | `the llama near the dogs` |
  | `:inside_relative` | `the llama who` |
  | `:object_relative` | `the fox sees the dogs who` |

  Each probe also records `mismatch`: whether the nearest noun before the
  verb has the *other* number. A model that just copies the nearest noun's
  number gets every mismatch probe wrong, so accuracy on mismatches is the
  honest test of whether it tracks the real subject.
  """

  alias Interpretability.Grammar
  alias Interpretability.Labeler
  alias Interpretability.Model
  alias Interpretability.Vocab

  @type probe :: %{
          prefix: [Vocab.word()],
          number: Vocab.grammatical_number(),
          category: atom(),
          mismatch: boolean()
        }

  @typedoc "Anything that scores every vocabulary word as the next word."
  @type predictor :: ([Vocab.word()] -> [float()])

  @finite_verbs [:transitive_verb, :intransitive_verb, :copula]

  @doc """
  One probe per present tense verb in the sentence.
  """
  @spec from_tree(Grammar.sentence_tree()) :: [probe()]
  def from_tree(tree) do
    tokens = Labeler.label(tree)

    verb_positions =
      tokens
      |> Enum.with_index()
      |> Enum.filter(fn {token, _position} -> token.labels.part_of_speech in @finite_verbs end)
      |> Enum.map(fn {_token, position} -> position end)

    verb_positions
    |> Enum.zip(categories(tree))
    |> Enum.flat_map(fn {position, {verb_phrase, category}} ->
      if verb_phrase.tense == :present,
        do: [probe(tokens, position, verb_phrase.number, category)],
        else: []
    end)
  end

  @doc """
  Probes from `count` fresh grammar sentences drawn from the seed after
  the config's, skipping any sentence that also appears in training, so
  every probe is new to the model. The cold open case is rare, about one
  sentence in forty, so probe corpora are large: 20,000 by default.
  """
  @spec held_out(Model.Config.t(), pos_integer()) :: [probe()]
  def held_out(config, count \\ 20_000),
    do: Enum.flat_map(held_out_trees(config, count), &from_tree/1)

  @doc """
  The sentences `held_out/2` takes its probes from.
  """
  @spec held_out_trees(Model.Config.t(), pos_integer()) :: [Grammar.sentence_tree()]
  def held_out_trees(config, count) do
    seen = Model.training_sentences(config)
    Grammar.seed(config.seed + 1)

    count
    |> Grammar.trees()
    |> Enum.reject(&MapSet.member?(seen, Grammar.words(&1)))
  end

  @doc """
  Accuracy per `{category, mismatch}` group.
  """
  @spec accuracy(predictor(), [probe()]) :: %{{atom(), boolean()} => float()}
  def accuracy(predict, probes) do
    probes
    |> Enum.group_by(&{&1.category, &1.mismatch})
    |> Map.new(fn {group, group_probes} ->
      correct = Enum.count(group_probes, &(predicted_number(predict, &1.prefix) == &1.number))
      {group, correct / length(group_probes)}
    end)
  end

  @doc """
  Whether the highest scoring present tense verb after `prefix` is singular
  or plural.
  """
  @spec predicted_number(predictor(), [Vocab.word()]) :: Vocab.grammatical_number()
  def predicted_number(predict, prefix) do
    scores = predict.(prefix)

    {_score, number} =
      present_verbs()
      |> Enum.map(fn {word, number} -> {Enum.at(scores, Vocab.word_to_id(word)), number} end)
      |> Enum.max_by(fn {score, _number} -> score end)

    number
  end

  @doc """
  Every present tense verb with its number: the words whose number shows.
  """
  @spec present_verbs() :: [{Vocab.word(), Vocab.grammatical_number()}]
  def present_verbs do
    for number <- [:singular, :plural],
        word <-
          Vocab.transitive_verbs(:present, number) ++
            Vocab.intransitive_verbs(:animate, :present, number) ++
            Vocab.intransitive_verbs(:inanimate, :present, number) ++
            Vocab.copula(:present, number),
        do: {word, number}
  end

  ## PRIVATE FUNCTIONS

  defp probe(tokens, position, number, category) do
    preceding = Enum.take(tokens, position)

    nearest_noun =
      preceding
      |> Enum.reverse()
      |> Enum.find(&(&1.labels.part_of_speech in [:noun, :pronoun]))

    %{
      prefix: Enum.map(preceding, & &1.word),
      number: number,
      category: category,
      mismatch: nearest_noun.labels.word_number != number
    }
  end

  # Every verb phrase with its category, in the order its verb is read.
  defp categories(%{subject: subject, verb_phrase: verb_phrase}) do
    subject_verbs(subject) ++
      [{verb_phrase, main_category(subject.type)}] ++ object_verbs(verb_phrase)
  end

  defp subject_verbs(%{type: :relative, verb_phrase: verb_phrase}),
    do: [{verb_phrase, :inside_relative} | object_verbs(verb_phrase)]

  defp subject_verbs(_subject), do: []

  defp object_verbs(%{type: :transitive, object: %{type: :relative, verb_phrase: verb_phrase}}),
    do: [{verb_phrase, :object_relative} | object_verbs(verb_phrase)]

  defp object_verbs(_verb_phrase), do: []

  defp main_category(:simple), do: :simple_subject
  defp main_category(:compound), do: :compound_subject
  defp main_category(:pronoun), do: :pronoun_subject
  defp main_category(:relative), do: :relative_distractor
  defp main_category(:prepositional), do: :prepositional_distractor
end
