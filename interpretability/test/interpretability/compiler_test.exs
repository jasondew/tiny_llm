defmodule Interpretability.CompilerTest do
  use ExUnit.Case, async: true

  alias Interpretability.Compiler

  test "jitted functions run with the chosen compiler" do
    double = Nx.Defn.jit(fn tensor -> Nx.multiply(tensor, 2) end, Compiler.options())

    assert Nx.to_list(double.(Nx.tensor([1.0, 2.0]))) == [2.0, 4.0]
  end

  test "picks EXLA whenever it is loaded" do
    expected = if Code.ensure_loaded?(EXLA), do: [compiler: EXLA], else: []

    assert Compiler.options() == expected
  end
end
