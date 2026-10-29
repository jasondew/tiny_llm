defmodule Interpretability.Activations do
  @moduledoc """
  Recording what the MLP neurons do on every word of a corpus.

  To find out what a neuron means, run lots of sentences through the
  model and write down the neuron's value at every word, next to that
  word's labels. The result is one big table: a row per word read, a
  column per neuron. Everything after this, from top activating examples
  to precision and recall to training the sparse autoencoder, reads from
  that table.

  Padding positions are left out, so every row is a word the model really
  read.
  """

  alias Interpretability.Compiler
  alias Interpretability.Dataset
  alias Interpretability.Labeler
  alias Interpretability.Model

  @type t :: %{
          hidden: Nx.Tensor.t(),
          labels: [Labeler.labels()],
          contexts: [{non_neg_integer(), non_neg_integer()}]
        }

  @type example :: %{
          activation: float(),
          word: String.t(),
          words: [String.t()],
          position: non_neg_integer()
        }

  @doc """
  The hidden layer at every real input position of `dataset`, in dataset
  order: a tokens × 128 tensor, with each row's labels and its
  `{sentence_index, position}`.
  """
  @spec collect(Model.params(), Dataset.t()) :: t()
  def collect(params, dataset) do
    hidden = Nx.Defn.jit(&Model.hidden/2, Compiler.options()).(params, dataset.tokens)
    {sentence_count, context_length, width} = Nx.shape(hidden)

    contexts =
      dataset.labels
      |> Enum.with_index()
      |> Enum.flat_map(fn {sentence_labels, sentence} ->
        for position <- 0..(length(sentence_labels) - 1), do: {sentence, position}
      end)

    rows =
      Nx.tensor(
        Enum.map(contexts, fn {sentence, position} -> sentence * context_length + position end),
        type: :s64
      )

    %{
      hidden:
        hidden
        |> Nx.reshape({sentence_count * context_length, width})
        |> Nx.take(rows)
        |> Nx.backend_copy(Nx.BinaryBackend),
      labels: List.flatten(dataset.labels),
      contexts: contexts
    }
  end

  @doc """
  The `count` words on which `unit` is most active, strongest first, each
  with the input it appeared in.
  """
  @spec top_examples(t(), Dataset.t(), non_neg_integer(), pos_integer()) :: [example()]
  def top_examples(activations, dataset, unit, count) do
    {values, indices} = Nx.top_k(activations.hidden[[.., unit]], k: count)

    [Nx.to_list(values), Nx.to_list(indices)]
    |> Enum.zip_with(fn [value, row] ->
      {sentence, position} = Enum.at(activations.contexts, row)
      words = dataset.labels |> Enum.at(sentence) |> Enum.map(& &1.word)
      %{activation: value, word: Enum.at(words, position), words: words, position: position}
    end)
  end
end
