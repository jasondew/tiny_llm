defmodule Mix.Tasks.Interp.Toy do
  @shortdoc "Trains the toy superposition model across sparsities"

  @moduledoc """
  Trains one toy model per sparsity and prints how many of the five
  features each one kept in its two dimensions.

      mix interp.toy

  Writes every trained model to `priv/results/toy_sweep.etf` for the deck.
  """

  use Mix.Task

  alias Interpretability.Checkpoint
  alias Interpretability.Toy

  @sparsities [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]
  @path "priv/results/toy_sweep.etf"

  @impl Mix.Task
  def run(_arguments) do
    Mix.Task.run("app.start")

    {microseconds, sweep} = :timer.tc(fn -> Toy.sweep(@sparsities, %Toy.Config{}) end)

    Mix.shell().info("sparsity  kept  loss      feature lengths")

    Enum.each(sweep, fn run ->
      lengths = run.weights |> Toy.column_lengths() |> Nx.to_list() |> Enum.map(&format/1)

      Mix.shell().info(
        "#{format(run.sparsity)}      #{run.represented}     #{format(run.loss, 6)}  #{Enum.join(lengths, " ")}"
      )
    end)

    Mix.shell().info("\n#{format(microseconds / 1.0e6, 1)} seconds")
    Checkpoint.save(sweep, @path)
    Mix.shell().info("Wrote #{@path}")
  end

  ## PRIVATE FUNCTIONS

  defp format(number, decimals \\ 2),
    do: :erlang.float_to_binary(number * 1.0, decimals: decimals)
end
