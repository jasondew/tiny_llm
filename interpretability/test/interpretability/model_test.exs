defmodule Interpretability.ModelTest do
  use ExUnit.Case, async: true

  alias Interpretability.Dataset
  alias Interpretability.Grammar
  alias Interpretability.Model
  alias Interpretability.Model.Config

  setup do
    Grammar.seed(8)
    dataset = Dataset.build(Grammar.trees(500))
    %{params: Model.init(%Config{}), dataset: dataset, tokens: dataset.tokens[0..1]}
  end

  test "logits and hidden activations have the expected shapes", %{params: params, tokens: tokens} do
    assert Nx.shape(Model.forward(params, tokens)) == {2, 20, 70}
    assert Nx.shape(Model.hidden(params, tokens)) == {2, 20, 128}
  end

  test "a position never sees the positions after it", %{params: params, tokens: tokens} do
    changed = Nx.indexed_put(tokens, Nx.tensor([[0, 5], [1, 5]]), Nx.tensor([3, 3]))
    before = Model.forward(params, tokens)[[.., 0..4]]
    after_change = Model.forward(params, changed)[[.., 0..4]]

    assert Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(before, after_change)))) < 1.0e-6
  end

  test "edit_hidden reaches the MLP activations", %{params: params, tokens: tokens} do
    plain = Model.forward(params, tokens)
    identity = Model.forward(params, tokens, edit_hidden: &Function.identity/1)
    zeroed = Model.forward(params, tokens, edit_hidden: &Nx.multiply(&1, 0.0))

    assert Nx.to_list(identity) == Nx.to_list(plain)
    assert Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(zeroed, plain)))) > 1.0e-3
  end

  test "an untrained model is close to guessing uniformly", %{params: params, dataset: dataset} do
    loss = Nx.to_number(Model.evaluate_loss(params, dataset))

    assert_in_delta loss, :math.log(70), 0.5
  end

  test "a short training run lowers the loss by at least 30%", %{dataset: dataset} do
    config = %Config{steps: 200, batch_size: 64}
    initial = Nx.to_number(Model.evaluate_loss(Model.init(config), dataset))
    {params, _history} = Model.train(config, dataset)

    assert Nx.to_number(Model.evaluate_loss(params, dataset)) < 0.7 * initial
  end

  test "the predictor scores every word after a prefix", %{params: params} do
    scores = Model.predictor(params).(~w(the llama))

    assert length(scores) == 70
  end
end
