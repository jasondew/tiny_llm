defmodule Interpretability.Export do
  @moduledoc """
  Packing the model up for the browser explorer.

  The explorer is a plain web page that runs the llama model itself, in
  JavaScript, on the viewer's phone. To do that it needs three things
  from this side:

  1. **The weights**, exactly. They go out as raw 32 bit floats, the same
     bytes Nx holds, plus a small JSON *manifest* saying where each named
     tensor starts and what shape it has. Writing them as decimal numbers
     would round them, and a rounded model is a slightly different model:
     the test that the page computes what Nx computes would be testing
     two different things.
  2. **What each unit is about**: every neuron's and every SAE feature's
     best matching grammar labels, with precision and recall, as plain
     strings the page can show.
  3. **What each unit fired on most strongly** in the evaluation
     sentences, so a click can show real examples, not just a label.
  """

  alias Interpretability.Activations
  alias Interpretability.Dataset
  alias Interpretability.Scoring

  @type manifest :: %{String.t() => [map()]}

  @doc """
  The tensors as one binary of float32s, in name order, and a manifest
  giving each one's name, shape and offset (counted in floats, not bytes).
  """
  @spec weights(%{atom() => Nx.Tensor.t()}) :: {manifest(), binary()}
  def weights(tensors) do
    sorted = Enum.sort_by(tensors, fn {name, _tensor} -> Atom.to_string(name) end)

    {entries, {binaries, _offset}} =
      Enum.map_reduce(sorted, {[], 0}, fn {name, tensor}, {binaries, offset} ->
        entry = %{
          "name" => Atom.to_string(name),
          "shape" => Tuple.to_list(Nx.shape(tensor)),
          "offset" => offset
        }

        binary = tensor |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.as_type(:f32) |> Nx.to_binary()
        {entry, {[binary | binaries], offset + Nx.size(tensor)}}
      end)

    {%{"tensors" => entries}, binaries |> Enum.reverse() |> IO.iodata_to_binary()}
  end

  @doc """
  A label as the page shows it: `{:agreement_number, :plural}` becomes
  `"agreement number: plural"`.
  """
  @spec label_name(Scoring.label()) :: String.t()
  def label_name({key, value}), do: "#{words(key)}: #{words(value)}"

  @doc """
  Every unit's firing rate, top labels, and `example_count` strongest
  examples, as plain maps ready for JSON. A unit that never fires has no
  examples.
  """
  @spec units(Activations.t(), Dataset.t(), [Scoring.score()], pos_integer()) :: [map()]
  def units(activations, dataset, scores, example_count) do
    scores
    |> Enum.sort_by(& &1.unit)
    |> Enum.map(fn score ->
      %{
        "unit" => score.unit,
        "firing_rate" => score.firing_rate,
        "top" => Enum.map(score.top, &grade/1),
        "examples" => examples(activations, dataset, score, example_count)
      }
    end)
  end

  ## PRIVATE FUNCTIONS

  defp examples(_activations, _dataset, %{firing_rate: rate}, _count) when rate == 0.0, do: []

  defp examples(activations, dataset, score, count) do
    activations
    |> Activations.top_examples(dataset, score.unit, count)
    # A unit that fired on fewer than `count` words gets padded with
    # words where it did not fire at all; those are not examples.
    |> Enum.filter(&(&1.activation > 0))
    |> Enum.map(fn example ->
      %{
        "words" => example.words,
        "position" => example.position,
        "activation" => example.activation
      }
    end)
  end

  defp grade(grade) do
    %{
      "label" => label_name(grade.label),
      "precision" => grade.precision,
      "recall" => grade.recall,
      "f1" => grade.f1
    }
  end

  defp words(value) when is_atom(value), do: value |> Atom.to_string() |> String.replace("_", " ")
  defp words(value), do: to_string(value)
end
