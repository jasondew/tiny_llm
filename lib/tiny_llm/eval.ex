defmodule TinyLlm.Eval do
  @moduledoc """
  The three measurements the talk is built on.

  TODO(stage 6): write the concept paragraph. It should say that a loss
  curve going down is not evidence of anything in particular, and that the
  question the audience actually has is "did it learn language or did it
  memorize the corpus". These three checks answer it.

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

  alias TinyLlm.Grammar
  alias TinyLlm.Tensor
  alias TinyLlm.Vocab

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
  def probes(_corpus), do: raise("TODO: stage 6")

  @doc """
  The fraction of probes whose highest-ranked verb has the right number.
  """
  @spec agreement(predictor(), [probe()]) :: float()
  def agreement(_predict, _probes), do: raise("TODO: stage 6")

  @doc """
  A predictor for the stage 5 model, which sees the whole prefix.
  """
  @spec model_predictor(map()) :: predictor()
  def model_predictor(_params), do: raise("TODO: stage 6")

  @doc """
  A predictor for the stage 3 Embedder, which sees only the last word.
  """
  @spec embedder_predictor(map()) :: predictor()
  def embedder_predictor(_params), do: raise("TODO: stage 6")

  @doc """
  A predictor for the stage 2 counting Bigram, which also sees only the
  last word, and which cannot in principle score above chance here.
  """
  @spec bigram_predictor(Tensor.matrix()) :: predictor()
  def bigram_predictor(_matrix), do: raise("TODO: stage 6")

  @doc """
  The fraction of `sentences` that do not appear in `training`.
  """
  @spec novelty([Grammar.sentence()], [Grammar.sentence()]) :: float()
  def novelty(_sentences, _training), do: raise("TODO: stage 6")

  @doc """
  True when a sentence obeys every rule the grammar enforces.

  Checks, in the order they are cheapest to fail: every word is in the
  vocabulary, the sentence ends in a period, `a` introduces only singular
  nouns, no intransitive verb takes an object, and every verb agrees with
  the noun it belongs to.
  """
  @spec grammatical?([Vocab.word()]) :: boolean()
  def grammatical?(_words), do: raise("TODO: stage 6")

  @doc """
  The fraction of `sentences` that are grammatical.
  """
  @spec grammaticality([Grammar.sentence()]) :: float()
  def grammaticality(_sentences), do: raise("TODO: stage 6")
end
