defmodule Interpretability.ActivationsTest do
  use ExUnit.Case, async: true

  alias Interpretability.Activations
  alias Interpretability.Dataset
  alias Interpretability.Grammar
  alias Interpretability.Model

  setup do
    Grammar.seed(10)
    dataset = Dataset.build(Grammar.trees(40))
    params = Model.init(%Model.Config{})
    %{dataset: dataset, params: params, activations: Activations.collect(params, dataset)}
  end

  test "one row per real input position after <start>", %{
    dataset: dataset,
    activations: activations
  } do
    # <start> is the same vector in every sentence, so it is left out.
    real_positions =
      (dataset.mask |> Nx.sum() |> Nx.to_number() |> round()) - length(dataset.labels)

    assert Nx.shape(activations.hidden) == {real_positions, 128}
    assert length(activations.labels) == real_positions
    assert length(activations.contexts) == real_positions
    refute Enum.any?(activations.labels, &(&1.part_of_speech == :start))
  end

  test "each row is the model's hidden layer at its context", %{
    dataset: dataset,
    params: params,
    activations: activations
  } do
    hidden = Model.hidden(params, dataset.tokens)

    for row <- [0, 17, length(activations.contexts) - 1] do
      {sentence, position} = Enum.at(activations.contexts, row)
      expected = hidden[[sentence, position]]

      assert Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(activations.hidden[row], expected)))) <
               1.0e-5

      assert Enum.at(activations.labels, row) ==
               dataset.labels |> Enum.at(sentence) |> Enum.at(position)
    end
  end

  test "top examples come strongest first, with the word they fired on", %{
    dataset: dataset,
    activations: activations
  } do
    examples = Activations.top_examples(activations, dataset, 3, 5)
    values = Enum.map(examples, & &1.activation)

    assert length(examples) == 5
    assert values == Enum.sort(values, :desc)

    for example <- examples do
      assert Enum.at(example.words, example.position) == example.word
    end
  end
end
