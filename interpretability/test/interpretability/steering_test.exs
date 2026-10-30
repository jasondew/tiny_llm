defmodule Interpretability.SteeringTest do
  use ExUnit.Case, async: true

  import Nx.Testing, only: [assert_all_close: 2]

  alias Interpretability.Dataset
  alias Interpretability.Grammar
  alias Interpretability.Model
  alias Interpretability.Probes
  alias Interpretability.Steering

  describe "directions" do
    test "every direction has length one" do
      assert_all_close(Nx.sum(Nx.pow(Steering.unit(Nx.tensor([3.0, 4.0])), 2)), Nx.tensor(1.0))
    end

    test "the mean difference points from singular words toward plural ones" do
      activations = %{
        hidden: Nx.tensor([[2.0, 0.0], [4.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
        labels: [
          %{agreement_number: :plural},
          %{agreement_number: :plural},
          %{agreement_number: :singular},
          %{agreement_number: nil}
        ]
      }

      direction =
        Steering.mean_difference(
          activations,
          {:agreement_number, :plural},
          {:agreement_number, :singular}
        )

      assert_all_close(direction, Steering.unit(Nx.tensor([3.0, -2.0])))
    end

    test "a neuron's direction is that neuron alone" do
      assert Nx.to_list(Steering.neuron(2, 4)) == [0.0, 0.0, 1.0, 0.0]
    end

    test "features add their decoder rows" do
      sae = %{decoder: Nx.tensor([[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])}

      assert_all_close(Steering.features(sae, [0, 1]), Steering.unit(Nx.tensor([1.0, 1.0])))
    end
  end

  describe "measuring a push" do
    setup do
      Grammar.seed(5)
      dataset = Dataset.build(Grammar.trees(100))
      probes = Enum.flat_map(dataset.trees, &Probes.from_tree/1)
      %{params: Model.init(%Model.Config{}), dataset: dataset, probes: probes}
    end

    test "with no push, the flip rate is how often the plain model calls a singular subject plural",
         %{params: params, probes: probes} do
      predictor = Model.predictor(params)
      singular = Enum.filter(probes, &(&1.number == :singular))

      expected =
        Enum.count(singular, &(Probes.predicted_number(predictor, &1.prefix) == :plural)) /
          length(singular)

      assert_in_delta Steering.flip_rate(params, probes, Steering.neuron(0, 128), 0.0),
                      expected,
                      1.0e-9
    end

    test "with no push there is no damage, and a big random push does damage",
         %{params: params, dataset: dataset} do
      direction = Steering.unit(Nx.Random.normal(Nx.Random.key(1), shape: {128}) |> elem(0))

      assert_in_delta Steering.damage(params, dataset, direction, 0.0), 0.0, 1.0e-6
      assert Steering.damage(params, dataset, direction, 50.0) > 0.01
    end

    test "the verb score margin after a prefix moves with the push", %{params: params} do
      direction = Steering.neuron(3, 128)

      assert Steering.margin(params, ~w(the llama), direction, 0.0) !=
               Steering.margin(params, ~w(the llama), direction, 20.0)
    end
  end
end
