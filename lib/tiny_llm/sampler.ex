defmodule TinyLlm.Sampler do
  @moduledoc """
  Generation: running the model forward over and over, feeding it what it
  just said.

  TODO(stage 6): write the concept paragraph. It should say that nothing
  about the model is generative. It answers one question, "what comes
  next", and generation is just asking it repeatedly and believing the
  answer. Every impressive thing a language model appears to do is this
  loop around that one question.

  ## The loop

      "<start>"                 -> distribution -> draw "the"
      "<start> the"             -> distribution -> draw "llama"
      "<start> the llama"       -> distribution -> draw "flees"
      "<start> the llama flees" -> distribution -> draw "."   stop

  Note what is *not* happening: there is no state carried between steps and
  no plan. Each step re-reads the whole prefix from scratch. That is
  wasteful and it is also why the model can never contradict something it
  has already said without seeing it.

  ## Temperature

  Temperature divides the logits before the softmax, which is the only
  place it can go: dividing after would need renormalizing, and that is the
  softmax again.

      temperature -> 0    argmax, the single most likely word every time
      temperature = 1     the model's own distribution, unmodified
      temperature > 1     flattened, so unlikely words get a real chance

  Below 1 the gaps between logits widen and the distribution sharpens;
  above 1 they narrow and it flattens. At exactly 0 the arithmetic divides
  by zero, so `0.0` is special-cased to `argmax`, which is what the limit
  approaches anyway.

  This is the knob the stage 7 Livebook puts a slider on, because watching
  a grammatical sentence turn to noise as it rises says more about what a
  distribution is than any amount of explanation.
  """

  alias TinyLlm.Tensor
  alias TinyLlm.Train
  alias TinyLlm.Vocab

  @doc """
  Seeds the process `:rand` state so a generation reproduces.
  """
  @spec seed(integer()) :: :rand.state()
  def seed(seed), do: :rand.seed(:exsss, seed)

  @doc """
  The next-token distribution given a prefix, at the given temperature.

  Returns a row over the whole vocabulary. At `0.0` it is one-hot on the
  argmax, which keeps the return type the same whatever the temperature.
  """
  @spec distribution(Train.params(), [Vocab.word()], float()) :: Tensor.row()
  def distribution(_params, _words, _temperature), do: raise("TODO: stage 6")

  @doc """
  One word, drawn from `distribution/3`.
  """
  @spec next_word(Train.params(), [Vocab.word()], float()) :: Vocab.word()
  def next_word(_params, _words, _temperature), do: raise("TODO: stage 6")

  @doc """
  One generated sentence, as a list of words ending in `"."`.

  Starts from `"<start>"`, which is dropped from the result, and stops at
  the first `"."` or when the context is full, whichever comes first. A
  sentence cut off by the context length does not end in `"."`, and callers
  that care should check rather than assume.
  """
  @spec sentence(Train.params(), keyword()) :: [Vocab.word()]
  def sentence(_params, _options \\ []), do: raise("TODO: stage 6")

  @doc """
  A stream of generated sentences.

  Lazy, so `Enum.take/2` costs only what it takes and the stage 7 notebook
  can pull one sentence per click.
  """
  @spec stream(Train.params(), keyword()) :: Enumerable.t()
  def stream(_params, _options \\ []), do: raise("TODO: stage 6")

  @doc """
  Every step of one generation: what was read, and what came of it.

  The stage 7 notebook draws the per-step probability bars from this, so it
  keeps the whole distribution at each position rather than just the word
  that won.
  """
  @spec trace(Train.params(), keyword()) :: [
          %{prefix: [Vocab.word()], distribution: Tensor.row(), chosen: Vocab.word()}
        ]
  def trace(_params, _options \\ []), do: raise("TODO: stage 6")
end
