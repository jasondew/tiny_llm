defmodule Interpretability.DatasetTest do
  use ExUnit.Case, async: true

  alias Interpretability.Dataset
  alias Interpretability.Grammar
  alias Interpretability.Labeler
  alias Interpretability.Trees
  alias Interpretability.Vocab

  setup do
    Grammar.seed(6)
    trees = [Trees.probe_sentence() | Grammar.trees(300)]
    %{dataset: Dataset.build(trees)}
  end

  test "tensors are one row per sentence, twenty positions wide", %{dataset: dataset} do
    assert Nx.shape(dataset.tokens) == {301, 20}
    assert Nx.shape(dataset.targets) == {301, 20}
    assert Nx.shape(dataset.mask) == {301, 20}
  end

  test "the input is <start> plus the sentence, the target is the sentence", %{dataset: dataset} do
    tokens = dataset.tokens[0] |> Nx.to_list() |> Enum.take(8) |> Vocab.decode()
    targets = dataset.targets[0] |> Nx.to_list() |> Enum.take(8) |> Vocab.decode()

    assert tokens == ~w(<start> the llama who chases the dogs flees)
    assert targets == ~w(the llama who chases the dogs flees .)
    assert dataset.mask[0] |> Nx.to_list() |> Enum.sum() == 8
  end

  test "labels align with input tokens", %{dataset: dataset} do
    token_rows = Nx.to_list(dataset.tokens)
    mask_rows = Nx.to_list(dataset.mask)

    for {labels, token_row, mask_row} <- Enum.zip([dataset.labels, token_rows, mask_rows]) do
      assert length(labels) == round(Enum.sum(mask_row))
      assert hd(labels) == Labeler.start_labels()

      for {position_labels, token} <- Enum.zip(labels, token_row) do
        assert position_labels.word == Vocab.id_to_word(token)
      end
    end
  end

  test "targets are the inputs shifted left by one", %{dataset: dataset} do
    for {token_row, target_row, mask_row} <-
          Enum.zip([
            Nx.to_list(dataset.tokens),
            Nx.to_list(dataset.targets),
            Nx.to_list(dataset.mask)
          ]) do
      length = round(Enum.sum(mask_row))
      assert Enum.slice(token_row, 1, length - 1) == Enum.take(target_row, length - 1)
    end
  end

  test "the same seed gives the same batches", %{dataset: dataset} do
    first = dataset |> Dataset.batches(32, 9) |> Enum.take(3) |> Enum.map(&Nx.to_list(&1.tokens))
    second = dataset |> Dataset.batches(32, 9) |> Enum.take(3) |> Enum.map(&Nx.to_list(&1.tokens))

    assert first == second
    assert Nx.shape(hd(Enum.take(Dataset.batches(dataset, 32, 9), 1)).tokens) == {32, 20}
  end

  test "batches keep going past one epoch", %{dataset: dataset} do
    assert length(Enum.take(Dataset.batches(dataset, 100, 9), 7)) == 7
  end
end
