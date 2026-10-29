defmodule Interpretability.TrainerTest do
  use ExUnit.Case, async: true

  import Nx.Defn

  alias Interpretability.Trainer

  defn line_loss(params, batch) do
    predictions = params.slope * batch.inputs + params.intercept
    Nx.mean((predictions - batch.outputs) ** 2)
  end

  setup do
    inputs = Nx.linspace(-1.0, 1.0, n: 256)

    %{
      batches:
        Stream.repeatedly(fn -> %{inputs: inputs, outputs: Nx.add(Nx.multiply(inputs, 3), 1)} end)
    }
  end

  test "fits a straight line", %{batches: batches} do
    params = %{slope: Nx.tensor(0.0), intercept: Nx.tensor(0.0)}

    {params, history} =
      Trainer.run(params, batches, &line_loss/2, steps: 500, learning_rate: 0.05, log_every: 100)

    assert_in_delta Nx.to_number(params.slope), 3.0, 0.05
    assert_in_delta Nx.to_number(params.intercept), 1.0, 0.05
    assert List.last(history).loss < 1.0e-3
    assert Enum.map(history, & &1.step) == [100, 200, 300, 400, 500]
  end

  test "applies after_step to the parameters after every update", %{batches: batches} do
    params = %{slope: Nx.tensor(0.0), intercept: Nx.tensor(0.0)}

    {params, _history} =
      Trainer.run(params, batches, &line_loss/2,
        steps: 50,
        learning_rate: 0.05,
        after_step: fn params -> %{params | intercept: Nx.tensor(0.0)} end
      )

    assert Nx.to_number(params.intercept) == 0.0
  end
end
