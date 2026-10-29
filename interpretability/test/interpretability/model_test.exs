defmodule Interpretability.ModelTest do
  use ExUnit.Case, async: true

  alias Interpretability.Dataset
  alias Interpretability.Grammar
  alias Interpretability.Model
  alias Interpretability.Model.Config

  import Nx.Testing, only: [assert_all_close: 2]

  setup do
    Grammar.seed(8)
    dataset = Dataset.build(Grammar.trees(500))
    %{params: Model.init(%Config{}), dataset: dataset, tokens: dataset.tokens[0..1]}
  end

  test "logits and hidden activations have the expected shapes", %{params: params, tokens: tokens} do
    assert Nx.shape(Model.forward(params, tokens)) == {2, 20, 70}
    assert Nx.shape(Model.hidden(params, tokens)) == {2, 20, 128}
  end

  test "a wider MLP has more neurons", %{tokens: tokens} do
    params = Model.init(%Config{hidden_width: 512})

    assert Nx.shape(Model.hidden(params, tokens)) == {2, 20, 512}
  end

  test "SoLU multiplies each input by its softmax" do
    input = Nx.tensor([1.0, 2.0, 3.0])
    softmax = Nx.divide(Nx.exp(input), Nx.sum(Nx.exp(input)))
    expected = Nx.multiply(input, softmax)

    assert_all_close(Model.solu(input), expected)
  end

  test "only a SoLU model has a gain after its neurons", %{tokens: tokens} do
    relu = Model.init(%Config{})
    solu = Model.init(%Config{activation: :solu})

    refute Map.has_key?(relu, :gain_hidden)
    assert Map.has_key?(solu, :gain_hidden)
    assert Nx.shape(Model.hidden(solu, tokens)) == {2, 20, 128}
    assert Nx.to_number(Nx.reduce_min(Model.hidden(solu, tokens))) < 0.0
  end

  test "the activation penalty adds the mean L1 of the neurons at real positions",
       %{params: params, dataset: dataset} do
    batch = %{
      tokens: dataset.tokens[0..7],
      targets: dataset.targets[0..7],
      mask: dataset.mask[0..7]
    }

    plain = Nx.to_number(Model.loss(params, batch))

    l1 =
      params
      |> Model.hidden(batch.tokens)
      |> Nx.abs()
      |> Nx.sum(axes: [2])
      |> Nx.multiply(batch.mask)
      |> Nx.sum()
      |> Nx.divide(Nx.sum(batch.mask))
      |> Nx.to_number()

    assert_in_delta Nx.to_number(Model.training_loss(params, batch)), plain, 1.0e-5

    assert_in_delta Nx.to_number(Model.training_loss(params, batch, activation_penalty: 0.1)),
                    plain + 0.1 * l1,
                    1.0e-4
  end

  test "a SoLU model trains too", %{dataset: dataset} do
    config = %Config{steps: 200, batch_size: 64, activation: :solu}
    initial = Nx.to_number(Model.evaluate_loss(Model.init(config), dataset))
    {params, _history} = Model.train(config, dataset)

    assert Nx.to_number(Model.evaluate_loss(params, dataset)) < 0.7 * initial
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

  test "no evaluation sentence also appears in training" do
    config = %Config{training_corpus_size: 3_000, evaluation_corpus_size: 300}
    {training, evaluation} = Model.datasets(config)
    seen = MapSet.new(training.trees, &Grammar.words/1)

    assert length(evaluation.trees) == 300
    refute Enum.any?(evaluation.trees, &MapSet.member?(seen, Grammar.words(&1)))
  end

  test "the predictor scores every word after a prefix", %{params: params} do
    scores = Model.predictor(params).(~w(the llama))

    assert length(scores) == 70
  end
end
