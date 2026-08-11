defmodule TinyLlm.Test.QuadraticModel do
  @moduledoc """
  A model whose loss is the sum of its squared parameters.

  Its gradient is known exactly, `2 * theta`, with no derivation required
  and no floating point subtlety. That makes it the right thing to point
  `TinyLlm.GradCheck` at when the question is whether the *checker* works,
  rather than whether some model's backward pass does.
  """

  @behaviour TinyLlm.Train

  @impl TinyLlm.Train
  def init(_config), do: %{weights: [[0.5, -1.5], [2.0, 0.25]]}

  @impl TinyLlm.Train
  def examples(_corpus), do: [:unused]

  @impl TinyLlm.Train
  def loss(%{weights: rows}, _batch) do
    rows |> List.flatten() |> Enum.reduce(0.0, fn value, sum -> sum + value * value end)
  end

  @impl TinyLlm.Train
  def gradients(%{weights: rows}, _batch) do
    %{weights: Enum.map(rows, fn row -> Enum.map(row, &(2 * &1)) end)}
  end
end

defmodule TinyLlm.Test.BrokenModel do
  @moduledoc """
  `TinyLlm.Test.QuadraticModel` with one wrong sign in its gradient.

  This is what a subtly broken backward pass looks like: the loss is still
  correct, the gradient still points roughly downhill, and training would
  still appear to work. Only finite differences catch it, which is the
  entire argument for `TinyLlm.GradCheck` existing.
  """

  @behaviour TinyLlm.Train

  @impl TinyLlm.Train
  defdelegate init(config), to: TinyLlm.Test.QuadraticModel

  @impl TinyLlm.Train
  defdelegate examples(corpus), to: TinyLlm.Test.QuadraticModel

  @impl TinyLlm.Train
  defdelegate loss(params, batch), to: TinyLlm.Test.QuadraticModel

  @impl TinyLlm.Train
  def gradients(%{weights: [first_row | rest]}, _batch) do
    [first | others] = first_row

    %{
      weights: [
        [-2 * first | Enum.map(others, &(2 * &1))]
        | Enum.map(rest, fn row -> Enum.map(row, &(2 * &1)) end)
      ]
    }
  end
end
