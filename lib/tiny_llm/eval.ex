defmodule TinyLlm.Eval do
  @moduledoc """
  The three measurements the talk is built on.

  A loss curve going down is not evidence of anything in particular. It is
  compatible with learning the language, with memorizing the corpus, and
  with exploiting a cue that happens to correlate with the answer.

  The question anyone watching actually has is which of those happened, and
  the three checks below are what answer it. None of them is a loss.

  ## a. Agreement

  Held-out `who`-clause sentences with the verb removed. `the llama who
  chases the dogs ___` needs a singular verb, agreeing with `llama` four
  words back, while the adjacent noun `dogs` is plural. A bigram sees only
  `dogs` and must fail. This is the contrast slide.

  Accuracy is measured over **verb forms only**: of the ten verbs in the
  vocabulary, which does the model rank highest, and does its number match
  the subject's. Ranking over the whole vocabulary instead would conflate
  "picked the wrong number" with "did not predict a verb at all", and the
  bigram would score near zero for the second reason rather than the first.

  ## b. Memorization

  What fraction of generated sentences never appeared in the training
  corpus. The grammar has enough combinations that a model which memorized
  would be caught: high grammaticality with low novelty means a lookup
  table, and novelty without grammaticality means noise. Only both together
  mean it learned the rules.

  ## c. Grammaticality

  A structural checker over generated sentences, extending the stage 1 one
  in `test/support/structure.ex` with the relative-clause agreement that
  file explicitly defers to here.

  The checker re-derives the word classes from the brief rather than
  importing them from `TinyLlm.Vocab`, for the same reason the stage 1 one
  does: a checker that shares a bug with what it checks proves nothing.
  """

  alias TinyLlm.Embedder
  alias TinyLlm.Grammar
  alias TinyLlm.Sampler
  alias TinyLlm.Tensor
  alias TinyLlm.Vocab

  # Re-derived from the brief, not imported from Vocab. A checker that
  # shares a definition with the thing it checks cannot catch a bug in it.
  @determiners ~w(the a)
  @singular_nouns ~w(llama dog goose fox mouse)
  @plural_nouns ~w(llamas dogs geese foxes mice)
  @singular_verbs ~w(sees chases ignores flees is)
  @plural_verbs ~w(see chase ignore flee are)
  @verbs @singular_verbs ++ @plural_verbs
  @intransitive_verbs ~w(flees flee)
  @adjectives ~w(big small hungry grumpy fast sleepy)
  @known ~w(<start> . who and) ++
           @determiners ++ @singular_nouns ++ @plural_nouns ++ @verbs ++ @adjectives

  @typedoc """
  A held-out sentence with its main verb removed, and the number that verb
  has to have.
  """
  @type probe :: %{prefix: [Vocab.word()], number: :singular | :plural}

  @typedoc "Anything that answers 'what comes next', given the words so far."
  @type predictor :: ([Vocab.word()] -> Tensor.row())

  @doc """
  Turns a corpus into agreement probes, one per usable sentence.

  Keeps only sentences with a relative clause and no compound subject,
  since a compound subject is plural whatever its conjuncts are and would
  test a different rule. The main verb is the last verb in the sentence:
  the relative clause contributes exactly one verb and the main verb phrase
  contributes exactly one, and the main one comes second.
  """
  @spec probes([Grammar.sentence()]) :: [probe()]
  def probes(corpus) do
    corpus
    |> Enum.filter(fn words -> "who" in words and "and" not in words end)
    |> Enum.flat_map(fn words ->
      case probe(words) do
        nil -> []
        probe -> [probe]
      end
    end)
  end

  @doc """
  The fraction of probes whose highest-ranked verb has the right number.
  """
  @spec agreement(predictor(), [probe()]) :: float()
  def agreement(_predict, []), do: 0.0

  def agreement(predict, probes) do
    correct =
      Enum.count(probes, fn probe ->
        verb_number(highest_ranked_verb(predict.(probe.prefix))) == probe.number
      end)

    correct / length(probes)
  end

  @doc """
  A predictor for the stage 5 model, which sees the whole prefix.
  """
  @spec model_predictor(map()) :: predictor()
  def model_predictor(params) do
    fn words -> Sampler.distribution(params, words, 1.0) end
  end

  @doc """
  A predictor for the stage 3 Embedder, which sees only the last word.
  """
  @spec embedder_predictor(map()) :: predictor()
  def embedder_predictor(params) do
    fn words -> Embedder.forward(params, Vocab.word_to_id(List.last(words))) end
  end

  @doc """
  A predictor for the stage 2 counting Bigram, which also sees only the
  last word, and which cannot in principle score above chance here.
  """
  @spec bigram_predictor(Tensor.matrix()) :: predictor()
  def bigram_predictor(matrix) do
    fn words -> Enum.at(matrix, Vocab.word_to_id(List.last(words))) end
  end

  @doc """
  The fraction of `sentences` that do not appear in `training`.
  """
  @spec novelty([Grammar.sentence()], [Grammar.sentence()]) :: float()
  def novelty([], _training), do: 0.0

  def novelty(sentences, training) do
    seen = MapSet.new(training)

    Enum.count(sentences, fn sentence -> not MapSet.member?(seen, sentence) end) /
      length(sentences)
  end

  @doc """
  True when a sentence obeys every rule the grammar enforces.

  Checks, in the order they are cheapest to fail: every word is in the
  vocabulary, the sentence ends in a period, `a` introduces only singular
  nouns, no intransitive verb takes an object, and every verb agrees with
  the noun it belongs to.
  """
  @spec grammatical?([Vocab.word()]) :: boolean()
  def grammatical?(words) do
    Enum.all?(words, &(&1 in @known)) and List.last(words) == "." and
      determiners_ok?(words) and intransitive_ok?(words) and agreement_ok?(words)
  end

  @doc """
  The fraction of `sentences` that are grammatical.
  """
  @spec grammaticality([Grammar.sentence()]) :: float()
  def grammaticality([]), do: 0.0

  def grammaticality(sentences) do
    Enum.count(sentences, &grammatical?/1) / length(sentences)
  end

  ## PRIVATE FUNCTIONS

  # The main verb is the last verb in the sentence: the relative clause
  # contributes exactly one and the main verb phrase contributes exactly
  # one, in that order. The number comes from the head noun, which is the
  # first noun, and is what the adjacent distractor is not.
  defp probe(words) do
    body = strip_period(words)
    verb_positions = for {word, index} <- Enum.with_index(body), word in @verbs, do: index

    with [_ | _] <- verb_positions,
         head when not is_nil(head) <- Enum.find_value(body, &noun_number/1) do
      %{prefix: Enum.take(body, List.last(verb_positions)), number: head}
    else
      _unparseable -> nil
    end
  end

  # Rank among the verbs only. A model that mostly wants to say `llama` is
  # not making an agreement error, and scoring it as one would let the
  # bigram lose for the wrong reason.
  defp highest_ranked_verb(distribution) do
    @verbs
    |> Enum.max_by(fn verb -> Enum.at(distribution, Vocab.word_to_id(verb)) end)
  end

  # Three shapes of agreement. A compound subject is plural whatever its
  # conjuncts are; a relative clause makes the head noun govern both its own
  # verb and the main one; anything else is a simple clause.
  defp agreement_ok?(words) do
    body = strip_period(words)

    cond do
      "who" in body -> relative_agreement_ok?(body)
      "and" in body -> compound_agreement_ok?(body)
      true -> simple_agreement_ok?(body)
    end
  end

  defp simple_agreement_ok?(body) do
    with subject when not is_nil(subject) <- Enum.find_value(body, &noun_number/1),
         predicate when not is_nil(predicate) <- Enum.find_value(body, &verb_number/1) do
      subject == predicate
    else
      _unparseable -> false
    end
  end

  # Every verb in the sentence belongs to the head noun: the relative
  # clause's verb by construction, and the main verb because the head is the
  # subject of the whole sentence. So both must match it, and the distractor
  # sitting next to the blank must not.
  defp relative_agreement_ok?(body) do
    head = Enum.find_value(body, &noun_number/1)
    numbers = body |> Enum.map(&verb_number/1) |> Enum.reject(&is_nil/1)

    not is_nil(head) and length(numbers) == 2 and Enum.all?(numbers, &(&1 == head))
  end

  # The last `and` joins the subject's conjuncts, because an object noun
  # phrase is always simple and can never contain one.
  defp compound_agreement_ok?(body) do
    last_conjunct = body |> Enum.reverse() |> Enum.take_while(&(&1 != "and")) |> Enum.reverse()

    case Enum.find_value(last_conjunct, &verb_number/1) do
      nil -> false
      predicate -> predicate == :plural
    end
  end

  defp determiners_ok?(words) do
    words |> strip_period() |> check_determiners()
  end

  defp check_determiners([]), do: true

  defp check_determiners(["a" | remaining]) do
    case take_noun(remaining) do
      {noun, rest} -> noun_number(noun) == :singular and check_determiners(rest)
      :error -> false
    end
  end

  defp check_determiners([_word | remaining]), do: check_determiners(remaining)

  # `flees` and `flee` never take an object, so the word after one can never
  # be the determiner that would open an object noun phrase.
  defp intransitive_ok?(words) do
    words
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.all?(fn [word, next_word] ->
      word not in @intransitive_verbs or next_word not in @determiners
    end)
  end

  defp noun_number(word) when word in @singular_nouns, do: :singular
  defp noun_number(word) when word in @plural_nouns, do: :plural
  defp noun_number(_word), do: nil

  defp verb_number(word) when word in @singular_verbs, do: :singular
  defp verb_number(word) when word in @plural_verbs, do: :plural
  defp verb_number(_word), do: nil

  defp strip_period(words) do
    case List.last(words) do
      "." -> Enum.drop(words, -1)
      _other -> words
    end
  end

  defp take_noun([word | remaining]) when word in @adjectives, do: take_noun(remaining)
  defp take_noun([word | remaining]), do: {word, remaining}
  defp take_noun([]), do: :error
end
