defmodule Interpretability.Compiler do
  @moduledoc """
  Which compiler turns our `Nx.Defn` functions into machine code.

  Nx describes the math once and lets a compiler decide how to run it. The
  default, `Nx.Defn.Evaluator`, walks the expression in plain Elixir: slow,
  but it runs anywhere the BEAM does. EXLA hands the same expression to
  Google's XLA, which compiles it for the CPU and runs training many times
  faster.

  EXLA is only a dependency in `:dev` and `:test`. The deck depends on this
  project without it, because it loads trained checkpoints rather than
  training anything heavy. So nothing here names EXLA at compile time; this
  module looks for it at runtime instead.
  """

  @doc """
  Options for `Nx.Defn.jit/2`: EXLA when it is loaded, the default otherwise.
  """
  @spec options() :: keyword()
  def options do
    if Code.ensure_loaded?(EXLA), do: [compiler: EXLA], else: []
  end
end
