defmodule Interpretability.CheckpointTest do
  use ExUnit.Case, async: true

  alias Interpretability.Checkpoint

  @moduletag :tmp_dir

  test "a saved term loads back equal, tensors included", %{tmp_dir: directory} do
    path = Path.join([directory, "nested", "checkpoint.etf"])
    term = %{weights: Nx.tensor([[1.0, 2.0], [3.0, 4.0]]), config: %{seed: 1}}

    Checkpoint.save(term, path)
    loaded = Checkpoint.load(path)

    assert loaded.config == %{seed: 1}
    assert Nx.to_list(loaded.weights) == [[1.0, 2.0], [3.0, 4.0]]
  end
end
