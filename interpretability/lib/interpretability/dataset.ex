defmodule Interpretability.Dataset do
  @moduledoc """
  Sentences turned into the rectangles of numbers a model trains on, with
  each position's labels kept alongside.

  A language model learns to predict the next word. For one sentence, the
  input is `<start>` followed by every word but the last, and the target
  at each position is the word that actually came next:

      position   0        1      2      3      4       5      6      7
      input      <start>  the    llama  who    chases  the    dogs   flees
      target     the      llama  who    chases the     dogs   flees  .

  Sentences have different lengths, but a tensor is a rectangle, so every
  row is padded out to the context length. A *mask* of 1s and 0s marks the
  real positions, so padding never counts toward the loss.

  `labels` holds, for each sentence, the labels of each real *input*
  position: the labels at position t describe the word the model has just
  read there, which is the word whose activations we will later inspect.
  Keeping them aligned is the whole job of this module; an off by one here
  would score every feature against the wrong word.
  """

  alias Interpretability.Grammar
  alias Interpretability.Labeler
  alias Interpretability.Vocab

  @type t :: %{
          tokens: Nx.Tensor.t(),
          targets: Nx.Tensor.t(),
          mask: Nx.Tensor.t(),
          labels: [[Labeler.labels()]],
          trees: [Grammar.sentence_tree()]
        }

  @type batch :: %{tokens: Nx.Tensor.t(), targets: Nx.Tensor.t(), mask: Nx.Tensor.t()}

  @doc """
  Encodes, pads and labels a list of parse trees.
  """
  @spec build([Grammar.sentence_tree()], pos_integer()) :: t()
  def build(trees, context_length \\ Grammar.max_tokens()) do
    padding = Vocab.word_to_id(Vocab.end_token())

    rows =
      Enum.map(trees, fn tree ->
        tokens = Labeler.label(tree)
        targets = Vocab.encode(Enum.map(tokens, & &1.word))
        inputs = [Vocab.word_to_id(Vocab.start_token()) | Enum.drop(targets, -1)]
        labels = [Labeler.start_labels() | Enum.map(Enum.drop(tokens, -1), & &1.labels)]
        padding_count = context_length - length(targets)

        %{
          tokens: inputs ++ List.duplicate(padding, padding_count),
          targets: targets ++ List.duplicate(padding, padding_count),
          mask: List.duplicate(1.0, length(targets)) ++ List.duplicate(0.0, padding_count),
          labels: labels
        }
      end)

    %{
      tokens: Nx.tensor(Enum.map(rows, & &1.tokens), type: :s64),
      targets: Nx.tensor(Enum.map(rows, & &1.targets), type: :s64),
      mask: Nx.tensor(Enum.map(rows, & &1.mask), type: :f32),
      labels: Enum.map(rows, & &1.labels),
      trees: trees
    }
  end

  @doc """
  An endless stream of shuffled batches. Every epoch is a fresh shuffle,
  and the final partial batch of an epoch is dropped. The same seed gives
  the same stream.
  """
  @spec batches(t(), pos_integer(), integer()) :: Enumerable.t()
  def batches(dataset, batch_size, seed) do
    row_count = Nx.axis_size(dataset.tokens, 0)

    :rand.seed_s(:exsss, seed)
    |> Stream.unfold(fn random_state ->
      {order, random_state} = shuffle(Enum.to_list(0..(row_count - 1)), random_state)
      {Enum.chunk_every(order, batch_size, batch_size, :discard), random_state}
    end)
    |> Stream.flat_map(& &1)
    |> Stream.map(fn indices ->
      indices = Nx.tensor(indices, type: :s64)

      %{
        tokens: Nx.take(dataset.tokens, indices),
        targets: Nx.take(dataset.targets, indices),
        mask: Nx.take(dataset.mask, indices)
      }
    end)
  end

  ## PRIVATE FUNCTIONS

  # A shuffle that threads its random state explicitly, so the stream does
  # not depend on whatever else has used the process's random state.
  defp shuffle(list, random_state) do
    {keyed, random_state} =
      Enum.map_reduce(list, random_state, fn element, state ->
        {key, state} = :rand.uniform_s(state)
        {{key, element}, state}
      end)

    {keyed |> Enum.sort() |> Enum.map(fn {_key, element} -> element end), random_state}
  end
end
