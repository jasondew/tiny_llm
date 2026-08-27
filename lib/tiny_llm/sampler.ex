defmodule TinyLlm.Sampler do
  @moduledoc """
  Generation: running the model forward over and over, feeding it what it
  just said.

  Nothing about the model is generative. It answers exactly one question,
  what comes next, and it answers it with a probability for each of 32
  words. Generation is asking that question over and over and believing the
  answer.

  Every impressive thing a language model appears to do is this loop around
  that one question. There is no plan, no state carried between steps, and
  nothing that knows a sentence is being written.

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

  Temperature divides the logits before the softmax:

      p_i = exp(z_i / t) / sum of exp(z_j / t)

      temperature -> 0    argmax, the single most likely word every time
      temperature = 1     the model's own distribution, unmodified
      temperature > 1     flattened, so unlikely words get a real chance

  Below 1 the gaps between logits widen and the distribution sharpens;
  above 1 they narrow and it flattens. At exactly 0 the arithmetic divides
  by zero, so `0.0` is special-cased to `argmax`, which is what the limit
  approaches anyway.

  There is a second, identical formulation that works on the probabilities
  instead: raise each to the power `1/t` and renormalize. Substituting
  `p_i = exp(z_i)/Z` makes the `Z^(1/t)` cancel, leaving the expression
  above. What does *not* work is dividing the probabilities by `t` and
  renormalizing, and it fails silently: linear rescaling cancels completely
  and hands back the distribution unchanged, so temperature appears to have
  no effect at all.

  Dividing the logits is the one to implement regardless, since it reuses
  the max-subtraction guard inside `Tensor.softmax/1` and normalizes once
  rather than twice.

  This is the knob the stage 7 Livebook puts a slider on, because watching
  a grammatical sentence turn to noise as it rises says more about what a
  distribution is than any amount of explanation.
  """

  alias TinyLlm.Transformer
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
  def distribution(params, words, temperature) do
    tokens = Vocab.encode(words)
    logits = Transformer.forward(params, tokens).logits |> List.last()

    if temperature == 0.0 do
      Tensor.one_hot(Tensor.argmax(logits), length(logits))
    else
      scaled_logits = Enum.map(logits, fn z -> z / temperature end)
      [probabilities] = Tensor.softmax([scaled_logits])
      probabilities
    end
  end

  @doc """
  One word, drawn from `distribution/3`.
  """
  @spec next_word(Train.params(), [Vocab.word()], float()) :: Vocab.word()
  def next_word(params, words, temperature) do
    params
    |> distribution(words, temperature)
    |> Tensor.weighted_random_index()
    |> Vocab.id_to_word()
  end

  @doc """
  One generated sentence, as a list of words ending in `"."`.

  Starts from `"<start>"`, which is dropped from the result, and stops at
  the first `"."` or when the context is full, whichever comes first. A
  sentence cut off by the context length does not end in `"."`, and callers
  that care should check rather than assume.
  """
  @spec sentence(Train.params(), keyword()) :: [Vocab.word()]
  def sentence(params, options \\ []) do
    params |> trace(options) |> Enum.map(& &1.chosen)
  end

  @doc """
  A stream of generated sentences.

  Lazy, so `Enum.take/2` costs only what it takes and the stage 7 notebook
  can pull one sentence per click.
  """
  @spec stream(Train.params(), keyword()) :: Enumerable.t()
  def stream(params, options \\ []) do
    Stream.repeatedly(fn -> sentence(params, options) end)
  end

  @doc """
  Every step of one generation: the distribution, and what was drawn from it.

  The stage 7 notebook draws the per-step probability bars from this, so it
  keeps the whole distribution at each position rather than just the word
  that won.

  The prefix each step read is deliberately absent: it is the start token
  plus every earlier `:chosen`, so storing it would make the trace
  quadratic in a sentence's length to say nothing new.
  """
  @spec trace(Train.params(), keyword()) :: [
          %{distribution: Tensor.row(), chosen: Vocab.word()}
        ]
  def trace(params, options \\ []) do
    temperature = Keyword.get(options, :temperature, 1.0)
    max_tokens = Keyword.get(options, :max_tokens, length(params.positions))

    {[Vocab.start_token()], false}
    |> Stream.unfold(fn
      {_reversed_prefix, true} ->
        nil

      {reversed_prefix, false} ->
        prefix = Enum.reverse(reversed_prefix)
        distribution = distribution(params, prefix, temperature)
        chosen = Vocab.id_to_word(Tensor.weighted_random_index(distribution))
        step = %{distribution: distribution, chosen: chosen}
        done? = chosen == Vocab.end_token()

        {step, {[chosen | reversed_prefix], done?}}
    end)
    |> Enum.take(max_tokens)
  end
end
