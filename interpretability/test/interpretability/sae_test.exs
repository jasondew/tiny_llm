defmodule Interpretability.SAETest do
  use ExUnit.Case, async: true

  alias Interpretability.Dataset
  alias Interpretability.Grammar
  alias Interpretability.Model
  alias Interpretability.SAE

  # Inputs built from a known dictionary: each is the sum of 3 of 16 random
  # unit directions in 32 dimensions, with random positive strengths.
  defp synthetic_inputs(count) do
    key = Nx.Random.key(42)
    {directions, key} = Nx.Random.normal(key, shape: {16, 32}, type: :f32)
    directions = Nx.divide(directions, Nx.LinAlg.norm(directions, axes: [1], keep_axes: true))
    {scores, key} = Nx.Random.uniform(key, shape: {count, 16}, type: :f32)
    {strengths, _key} = Nx.Random.uniform(key, 0.5, 1.5, shape: {count, 16}, type: :f32)
    {_values, chosen} = Nx.top_k(scores, k: 3)

    coefficients =
      chosen
      |> Nx.new_axis(-1)
      |> Nx.equal(Nx.iota({1, 1, 16}))
      |> Nx.any(axes: [1])
      |> Nx.multiply(strengths)

    Nx.dot(coefficients, directions)
  end

  test "at most active_count features are on for any input" do
    config = %SAE.Config{input_width: 32, width: 64, active_count: 3}
    inputs = synthetic_inputs(200)
    sae = SAE.init(config, inputs)

    nonzero =
      sae
      |> SAE.encode(inputs, active_count: 3)
      |> Nx.not_equal(0)
      |> Nx.sum(axes: [1])
      |> Nx.to_list()

    assert Enum.max(nonzero) <= 3
  end

  test "decoder rows have unit length after normalizing" do
    config = %SAE.Config{input_width: 32, width: 64}
    sae = SAE.init(config, synthetic_inputs(50))
    stretched = %{sae | decoder: Nx.multiply(sae.decoder, 3.0)}

    lengths =
      stretched |> SAE.normalize_decoder() |> Map.fetch!(:decoder) |> Nx.LinAlg.norm(axes: [1])

    assert Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(lengths, 1.0)))) < 1.0e-5
  end

  test "recovers inputs built from a sparse dictionary" do
    config = %SAE.Config{
      input_width: 32,
      width: 64,
      active_count: 3,
      steps: 2_000,
      batch_size: 256,
      learning_rate: 3.0e-3
    }

    inputs = synthetic_inputs(4_000)
    {sae, _history} = SAE.train(config, inputs)

    assert Nx.to_number(SAE.variance_explained(sae, inputs, 3)) > 0.9
  end

  test "an SAE that rebuilds the activations exactly recovers all of the loss" do
    Grammar.seed(15)
    dataset = Dataset.build(Grammar.trees(100))
    params = Model.init(%Model.Config{})
    activations = Interpretability.Activations.collect(params, dataset)

    identity = %{
      encoder: Nx.eye(128, type: :f32),
      encoder_bias: Nx.broadcast(Nx.tensor(0.0, type: :f32), {128}),
      decoder: Nx.eye(128, type: :f32),
      decoder_bias: Nx.broadcast(Nx.tensor(0.0, type: :f32), {128})
    }

    report = SAE.report(identity, 128, params, dataset, activations)

    assert_in_delta report.loss_recovered, 1.0, 1.0e-3

    assert_in_delta Nx.to_number(SAE.variance_explained(identity, activations.hidden, 128)),
                    1.0,
                    1.0e-5
  end

  defp end_to_end_loss(sae, batch, options) do
    Nx.to_number(
      Nx.Defn.jit(&SAE.end_to_end_loss(&1, &2, options), Interpretability.Compiler.options()).(
        sae,
        batch
      )
    )
  end

  describe "end to end training" do
    setup do
      Grammar.seed(15)
      dataset = Dataset.build(Grammar.trees(200))
      params = Model.init(%Model.Config{})
      # An untrained model guesses every word about equally, which leaves
      # nothing for a splice to disturb. A larger projection gives it real
      # opinions to keep.
      params = %{params | projection: Nx.multiply(params.projection, 100.0)}
      batch = Map.put(Map.take(dataset, [:tokens, :targets, :mask]), :model, params)
      %{dataset: dataset, params: params, batch: batch}
    end

    test "an SAE that rebuilds the neurons exactly leaves the predictions unchanged",
         %{batch: batch} do
      identity = %{
        encoder: Nx.eye(128, type: :f32),
        encoder_bias: Nx.broadcast(Nx.tensor(0.0, type: :f32), {128}),
        decoder: Nx.eye(128, type: :f32),
        decoder_bias: Nx.broadcast(Nx.tensor(0.0, type: :f32), {128})
      }

      assert_in_delta end_to_end_loss(identity, batch, active_count: 128),
                      0.0,
                      1.0e-4
    end

    test "an SAE that rebuilds nothing moves the predictions", %{batch: batch} do
      blank = %{
        encoder: Nx.broadcast(Nx.tensor(0.0, type: :f32), {128, 128}),
        encoder_bias: Nx.broadcast(Nx.tensor(0.0, type: :f32), {128}),
        decoder: Nx.broadcast(Nx.tensor(0.0, type: :f32), {128, 128}),
        decoder_bias: Nx.broadcast(Nx.tensor(0.0, type: :f32), {128})
      }

      assert end_to_end_loss(blank, batch, active_count: 8) > 0.01
    end

    test "training brings the spliced predictions closer to the model's",
         %{dataset: dataset, params: params, batch: batch} do
      config = %SAE.Config{width: 64, active_count: 4, steps: 300, batch_size: 32}
      activations = Interpretability.Activations.collect(params, dataset)
      before = end_to_end_loss(SAE.init(config, activations.hidden), batch, active_count: 4)
      {sae, history} = SAE.train_end_to_end(config, params, dataset)

      assert length(history) == 3

      assert end_to_end_loss(sae, batch, active_count: 4) <
               0.5 * before

      lengths = Nx.LinAlg.norm(sae.decoder, axes: [1])
      assert Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(lengths, 1.0)))) < 1.0e-4
    end
  end
end
