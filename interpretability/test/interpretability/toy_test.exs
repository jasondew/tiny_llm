defmodule Interpretability.ToyTest do
  use ExUnit.Case, async: true

  alias Interpretability.Toy
  alias Interpretability.Toy.Config

  test "dense features: only the two most important fit in two dimensions" do
    assert Toy.represented(Toy.train(%Config{sparsity: 0.0})) == 2
  end

  test "sparse features: more than two share the two dimensions" do
    assert Toy.represented(Toy.train(%Config{sparsity: 0.95})) >= 4
  end

  test "the same seed trains the same weights" do
    config = %Config{sparsity: 0.9, steps: 200}

    assert Nx.to_list(Toy.train(config).weights) == Nx.to_list(Toy.train(config).weights)
  end
end
