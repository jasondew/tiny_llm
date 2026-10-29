defmodule Interpretability.CleanlinessTest do
  use ExUnit.Case, async: true

  alias Interpretability.Cleanliness
  alias Interpretability.Dataset
  alias Interpretability.Grammar
  alias Interpretability.Model
  alias Interpretability.Probes

  test "every variant changes one thing from the talk's model, which is the baseline" do
    [baseline | others] = Cleanliness.variants()

    assert baseline.name == "baseline"
    assert Cleanliness.config(baseline) == %Model.Config{}
    assert Enum.all?(others, &(length(&1.changes) == 1))
    assert Enum.uniq_by(others, & &1.name) == others
  end

  test "measure sorts every neuron into exactly one kind, at every bar" do
    Grammar.seed(3)
    evaluation = Dataset.build(Grammar.trees(200))
    probes = Enum.flat_map(evaluation.trees, &Probes.from_tree/1)
    params = Model.init(%Model.Config{hidden_width: 16})

    result = Cleanliness.measure(params, evaluation, probes)

    assert result.neuron_count == 16

    for bar <- [0.8, 0.9] do
      kinds = result.kinds[bar]
      assert kinds.clean + kinds.coarse + kinds.mixed + kinds.dead == 16
    end

    assert Map.keys(result.rubric) ==
             for(threshold <- [0.1, 0.2, 0.3, 0.5], bar <- [0.8, 0.9], do: {threshold, bar})

    assert result.probe_accuracy >= 0.0 and result.probe_accuracy <= 1.0
    assert is_float(result.evaluation_loss)
  end
end
