defmodule TinyLlm.Test.Vocabulary do
  @moduledoc """
  The 32 words the brief fixes, written out independently of `TinyLlm.Vocab`.

  Both the vocabulary tests and the grammar tests need this list, and both
  need it as a literal rather than as a call into `lib/`. Asserting that
  `Vocab.words/0` equals `Vocab.words/0` would prove nothing.
  """

  @words ~w(
    the a
    llama llamas dog dogs goose geese fox foxes mouse mice
    sees see chases chase ignores ignore flees flee
    is are
    big small hungry grumpy fast sleepy
    and then who
    .
  )

  @doc """
  Every word the model may ever see, in the order the brief fixes.
  """
  def words, do: @words
end
