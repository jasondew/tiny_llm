defmodule Mix.Tasks.Interp.Neurons do
  @shortdoc "Scores every MLP neuron of the trained model against the labels"

  @moduledoc """
  How clean are the raw neurons? Scores each of the 128 MLP neurons of
  the trained model against every label on the evaluation sentences.

      mix interp.neurons

  Prints how many neurons are monosemantic (precision and recall both at
  least 0.9), the spread of best F1 scores, the monosemantic neurons, and
  the three messiest neurons with their strongest examples. Writes
  `priv/results/neurons.etf`.
  """

  use Mix.Task

  alias Interpretability.Activations
  alias Interpretability.Checkpoint
  alias Interpretability.Model
  alias Interpretability.Scoring

  @model_path "priv/checkpoints/model.etf"
  @results_path "priv/results/neurons.etf"

  @impl Mix.Task
  def run(_arguments) do
    Mix.Task.run("app.start")
    %{config: config, params: params} = Checkpoint.load(@model_path)
    {_training, evaluation} = Model.datasets(config)

    activations = Activations.collect(params, evaluation)
    scores = Scoring.score(activations.hidden, activations.labels)
    Checkpoint.save(%{scores: scores}, @results_path)

    live = Enum.reject(scores, &(&1.firing_rate == 0.0))
    monosemantic = Enum.filter(scores, &Scoring.monosemantic?/1)

    info("#{length(activations.labels)} words read, #{length(scores)} neurons")
    info("Never fire: #{length(scores) - length(live)}")
    info("Monosemantic at 0.9: #{length(monosemantic)}")
    info("\nBest F1 per neuron:")
    print_histogram(live)

    info("\nMonosemantic neurons:")
    Enum.each(monosemantic, &info("  neuron #{&1.unit}: #{describe(&1.best)}"))

    info("\nThe three messiest neurons that fire:")

    live
    |> Enum.sort_by(& &1.best.f1)
    |> Enum.take(3)
    |> Enum.each(&print_neuron(&1, activations, evaluation))

    info("\nWrote #{@results_path}")
  end

  ## PRIVATE FUNCTIONS

  defp print_histogram(scores) do
    buckets = Enum.frequencies_by(scores, &min(trunc(&1.best.f1 * 10), 9))

    for bucket <- 0..9 do
      count = Map.get(buckets, bucket, 0)
      range = "#{format(bucket / 10, 1)}-#{format((bucket + 1) / 10, 1)}"

      info(
        "  #{range}  #{String.pad_leading(Integer.to_string(count), 3)} #{String.duplicate("█", count)}"
      )
    end
  end

  defp print_neuron(score, activations, dataset) do
    info("\n  neuron #{score.unit}, fires on #{format(score.firing_rate * 100, 1)}% of words")
    Enum.each(score.top, &info("    #{describe(&1)}"))

    for example <- Activations.top_examples(activations, dataset, score.unit, 10) do
      words =
        example.words
        |> Enum.with_index()
        |> Enum.map_join(" ", fn {word, index} ->
          if index == example.position, do: "[#{word}]", else: word
        end)

      info("    #{format(example.activation, 2)}  #{words}")
    end
  end

  defp describe(grade) do
    {key, value} = grade.label

    "#{key} = #{inspect(value)}  precision #{format(grade.precision, 2)}  recall #{format(grade.recall, 2)}"
  end

  defp info(message), do: Mix.shell().info(message)

  defp format(number, decimals), do: :erlang.float_to_binary(number * 1.0, decimals: decimals)
end
