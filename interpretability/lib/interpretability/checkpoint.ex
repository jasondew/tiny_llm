defmodule Interpretability.Checkpoint do
  @moduledoc """
  Saving and loading trained models and results, as the first talk did.

  A checkpoint is any Elixir term written with `:erlang.term_to_binary/1`.
  Tensors are first copied to `Nx.BinaryBackend`, so a checkpoint trained
  with EXLA loads in a project that does not have EXLA, such as the deck.
  """

  @doc """
  Writes `term` to `path`, creating directories as needed.
  """
  @spec save(term(), Path.t()) :: :ok
  def save(term, path) do
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, :erlang.term_to_binary(portable(term)))
  end

  @doc """
  Reads a term written by `save/2`.
  """
  @spec load(Path.t()) :: term()
  def load(path), do: path |> File.read!() |> :erlang.binary_to_term()

  ## PRIVATE FUNCTIONS

  defp portable(%Nx.Tensor{} = tensor), do: Nx.backend_copy(tensor, Nx.BinaryBackend)

  defp portable(%_{} = struct), do: struct

  defp portable(map) when is_map(map),
    do: Map.new(map, fn {key, value} -> {key, portable(value)} end)

  defp portable(list) when is_list(list), do: Enum.map(list, &portable/1)

  defp portable(tuple) when is_tuple(tuple),
    do: tuple |> Tuple.to_list() |> Enum.map(&portable/1) |> List.to_tuple()

  defp portable(other), do: other
end
