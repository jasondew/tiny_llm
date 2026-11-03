defmodule Interpretability.ExportTest do
  use ExUnit.Case, async: true

  alias Interpretability.Activations
  alias Interpretability.Dataset
  alias Interpretability.Export
  alias Interpretability.Grammar
  alias Interpretability.Model
  alias Interpretability.Scoring

  test "weights round trip through the manifest and binary" do
    first = Nx.tensor([[1.5, -2.0, 3.25], [0.0, 4.0, -0.5]], type: :f32)
    second = Nx.tensor([9.0, 8.0, 7.0, 6.0], type: :f32)
    {manifest, binary} = Export.weights(%{b: second, a: first})

    assert Enum.map(manifest["tensors"], & &1["name"]) == ["a", "b"]
    assert Enum.map(manifest["tensors"], & &1["shape"]) == [[2, 3], [4]]
    assert Enum.map(manifest["tensors"], & &1["offset"]) == [0, 6]

    decoded = Nx.from_binary(binary, :f32)
    assert decoded[0..5] |> Nx.reshape({2, 3}) == first
    assert decoded[6..9] == second
  end

  test "label names read as plain words" do
    assert Export.label_name({:agreement_number, :plural}) == "agreement number: plural"
    assert Export.label_name({:word, "the"}) == "word: the"
  end

  describe "units" do
    setup do
      Grammar.seed(15)
      dataset = Dataset.build(Grammar.trees(50))
      activations = Activations.collect(Model.init(%Model.Config{}), dataset)
      {rows, _width} = Nx.shape(activations.hidden)

      hidden =
        Nx.put_slice(
          activations.hidden,
          [0, 3],
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {rows, 1})
        )

      activations = %{activations | hidden: hidden}
      scores = Scoring.score(hidden, activations.labels)
      %{units: Export.units(activations, dataset, scores, 5)}
    end

    test "a unit that never fires has no examples", %{units: units} do
      dead = Enum.at(units, 3)
      assert dead["unit"] == 3
      assert dead["firing_rate"] == 0.0
      assert dead["examples"] == []
    end

    test "a live unit has its strongest examples first", %{units: units} do
      live = Enum.at(units, 0)
      activations = Enum.map(live["examples"], & &1["activation"])

      assert length(live["examples"]) == 5
      assert activations == Enum.sort(activations, :desc)
      assert Enum.all?(live["top"], &is_binary(&1["label"]))
      assert hd(live["examples"])["words"] |> Enum.all?(&is_binary/1)
    end

    test "the whole export encodes as JSON", %{units: units} do
      {manifest, _binary} = Export.weights(%{a: Nx.tensor([1.0])})
      assert is_binary(JSON.encode!(units))
      assert is_binary(JSON.encode!(manifest))
    end
  end
end
