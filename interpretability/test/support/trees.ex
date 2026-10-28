defmodule Interpretability.Trees do
  @moduledoc """
  Hand built parse trees for tests, written out in full so a test reads
  the exact sentence it checks.
  """

  @doc "the llama who chases the dogs flees ."
  def probe_sentence do
    %{
      subject: %{
        type: :relative,
        head: simple("the", "llama", :singular, :animate),
        relative_pronoun: "who",
        verb_phrase: %{
          type: :transitive,
          verb: "chases",
          tense: :present,
          number: :singular,
          object: simple("the", "dogs", :plural, :animate)
        }
      },
      verb_phrase: %{type: :intransitive, verb: "flees", tense: :present, number: :singular}
    }
  end

  @doc "the fox sees the dogs who flee ."
  def object_relative_sentence do
    %{
      subject: simple("the", "fox", :singular, :animate),
      verb_phrase: %{
        type: :transitive,
        verb: "sees",
        tense: :present,
        number: :singular,
        object: %{
          type: :relative,
          head: simple("the", "dogs", :plural, :animate),
          relative_pronoun: "who",
          verb_phrase: %{type: :intransitive, verb: "flee", tense: :present, number: :plural}
        }
      }
    }
  end

  @doc "the llama chased the dogs ."
  def past_tense_sentence do
    %{
      subject: simple("the", "llama", :singular, :animate),
      verb_phrase: %{
        type: :transitive,
        verb: "chased",
        tense: :past,
        number: :singular,
        object: simple("the", "dogs", :plural, :animate)
      }
    }
  end

  @doc """
  Every verb phrase in a tree, outermost first.
  """
  def verb_phrases(%{subject: subject, verb_phrase: verb_phrase}),
    do: noun_phrase_verb_phrases(subject) ++ verb_phrase_tree(verb_phrase)

  ## PRIVATE FUNCTIONS

  defp simple(determiner, noun, number, animacy),
    do: %{
      type: :simple,
      determiner: determiner,
      adjectives: [],
      noun: noun,
      number: number,
      animacy: animacy
    }

  defp verb_phrase_tree(%{type: :transitive, object: object} = verb_phrase),
    do: [verb_phrase | noun_phrase_verb_phrases(object)]

  defp verb_phrase_tree(verb_phrase), do: [verb_phrase]

  defp noun_phrase_verb_phrases(%{type: :relative, verb_phrase: verb_phrase}),
    do: verb_phrase_tree(verb_phrase)

  defp noun_phrase_verb_phrases(_noun_phrase), do: []
end
