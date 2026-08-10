defmodule TinyLlm.Bigram do
  @moduledoc """
  The dumbest language model that works: count which word follows which.

  ## Sequences are bounded, not wrapped

  A bigram needs somewhere to start, so each sentence is counted with
  `Vocab.start_token/0` prepended and its own period already at the end.
  Sampling begins from the start row and stops at a period.

  That leaves two gaps in the matrix, both meaningful: nothing precedes
  `"<start>"`, so its column is empty, and nothing follows `"."`, so its
  row is empty. Those two stripes are the sequence boundaries made visible
  on the heatmap slide.
  """

  alias TinyLlm.Grammar
  alias TinyLlm.Tensor
  alias TinyLlm.Vocab

  @typedoc "How often each ordered pair of tokens was observed."
  @type counts :: %{{Vocab.id(), Vocab.id()} => pos_integer()}

  @max_tokens 16

  @doc """
  The context length a sample may not exceed, counting its final period.
  """
  @spec max_tokens() :: pos_integer()
  def max_tokens, do: @max_tokens

  @doc """
  Seeds the process `:rand` state so sampling reproduces.
  """
  @spec seed(integer()) :: :rand.state()
  def seed(seed), do: :rand.seed(:exsss, seed)

  @doc """
  Counts every adjacent pair of tokens in a corpus.

  Each sentence is counted with the start token prepended, so a sentence's
  first word is recorded as following `"<start>"`.
  """
  @spec counts([Grammar.sentence()]) :: counts()
  def counts(corpus) do
    Enum.reduce(corpus, %{}, fn sentence, counts ->
      sentence_with_start_token = [Vocab.start_token() | sentence]
      adjacent_pairs = Enum.chunk_every(sentence_with_start_token, 2, 1, :discard)

      Enum.reduce(adjacent_pairs, counts, fn
        [prev, next], counts ->
          prev_id = Vocab.word_to_id(prev)
          next_id = Vocab.word_to_id(next)

          Map.update(counts, {prev_id, next_id}, 1, &(&1 + 1))
      end)
    end)
  end

  @doc """
  Counts as a 32 by 32 matrix, each row normalized to sum to 1.

  Row `i` column `j` is the probability that token `j` follows token `i`.
  A row whose token was never a predecessor stays all zeros rather than
  being smoothed into a uniform distribution.
  """
  @spec matrix(counts()) :: Tensor.matrix()
  def matrix(counts) do
    zero_matrix = Tensor.zeros(Vocab.size(), Vocab.size())

    counts
    |> Enum.reduce(zero_matrix, fn {{prev_id, next_id}, count}, matrix ->
      List.update_at(matrix, prev_id, fn row ->
        List.update_at(row, next_id, fn _ -> count end)
      end)
    end)
    |> Tensor.normalize()
  end

  @doc """
  One sampled sentence, as words, always ending in `"."`.

  Starts from the start-token row and samples until it draws a period or
  reaches the 16 token context length, whichever comes first. On reaching
  the limit the final token is a period regardless, so every sample is a
  well formed sentence. The start token itself is not part of the result.
  """
  @spec sentence(Tensor.matrix()) :: [Vocab.word()]
  def sentence(matrix) do
    [_last_token | reversed_tokens] =
      1..@max_tokens
      |> Enum.reduce_while(
        [Vocab.start_token()],
        fn _index, [last_token | _] = tokens ->
          prev_id = Vocab.word_to_id(last_token)
          row = Enum.at(matrix, prev_id)
          next_id = Tensor.weighted_random_index(row)

          next_word = Vocab.id_to_word(next_id)
          updated_tokens = [next_word | tokens]

          if next_word == "." do
            {:halt, updated_tokens}
          else
            {:cont, updated_tokens}
          end
        end
      )

    # force a period at the end and strip the start token from the front
    [_start_token | tokens] = Enum.reverse(["." | reversed_tokens])

    tokens
  end
end
