defmodule Mix.Tasks.Interp.Sae do
  @shortdoc "Trains a sparse autoencoder and compares its features with the neurons"

  @moduledoc """
  Trains a sparse autoencoder on the trained model's MLP activations, then
  scores its features against the labels and puts them head to head with
  the raw neurons.

      mix interp.sae

  The SAE trains on the activations of the first 20,000 training
  sentences and is evaluated on the 2,000 evaluation sentences. Writes
  `priv/checkpoints/sae_512.etf` as `%{config, sae, report, scores}`.
  """

  use Mix.Task

  alias Interpretability.Activations
  alias Interpretability.Checkpoint
  alias Interpretability.Dataset
  alias Interpretability.Model
  alias Interpretability.SAE
  alias Interpretability.Scoring

  @model_path "priv/checkpoints/model.etf"
  @sae_path "priv/checkpoints/sae_512.etf"
  @training_sentence_count 20_000
  @margin 0.05

  @impl Mix.Task
  def run(_arguments) do
    Mix.Task.run("app.start")
    %{config: model_config, params: params} = Checkpoint.load(@model_path)
    {training, evaluation} = Model.datasets(model_config)
    config = %SAE.Config{}

    training_activations =
      params
      |> Activations.collect(Dataset.build(Enum.take(training.trees, @training_sentence_count)))
      |> Map.fetch!(:hidden)

    {microseconds, {sae, history}} = :timer.tc(fn -> SAE.train(config, training_activations) end)

    activations = Activations.collect(params, evaluation)
    report = SAE.report(sae, config.active_count, params, evaluation, activations)
    features = SAE.features(sae, activations.hidden, config.active_count)
    scores = Scoring.score(features, activations.labels)

    Checkpoint.save(%{config: config, sae: sae, report: report, scores: scores}, @sae_path)

    info("Trained #{config.steps} steps in #{format(microseconds / 1.0e6, 1)} seconds")
    info("Loss curve: " <> Enum.map_join(Enum.take_every(history, 10), "  ", &format(&1.loss, 3)))
    print_report(report, config)
    print_comparison(activations, features, scores)
    info("\nWrote #{@sae_path}")
  end

  ## PRIVATE FUNCTIONS

  defp print_report(report, config) do
    info("\nReconstruction: #{format(report.variance_explained * 100, 1)}% of variance explained")
    info("Model loss: #{format(report.loss_original, 4)} with its own neurons")
    info("            #{format(report.loss_spliced, 4)} with the SAE spliced in")
    info("            #{format(report.loss_zero_ablated, 4)} with the neurons switched off")
    info("Loss recovered: #{format(report.loss_recovered * 100, 1)}%")
    info("Dead features: #{report.dead_count} of #{config.width}")
    info("Dense features (on for more than half of all words): #{report.dense_count}")
  end

  defp print_comparison(activations, features, scores) do
    neuron_best = Scoring.by_label(activations.hidden, activations.labels)
    feature_best = Scoring.by_label(features, activations.labels)
    neuron_scores = Scoring.score(activations.hidden, activations.labels)

    info(
      "\nMonosemantic at 0.9: #{count_clean(neuron_scores)} of 128 neurons, " <>
        "#{count_clean(scores)} of #{length(scores)} features"
    )

    info("\nBest features:")

    scores
    |> Enum.sort_by(& &1.best.f1, :desc)
    |> Enum.take(15)
    |> Enum.each(&info("  feature #{&1.unit}: #{describe(&1.best)}"))

    labels = Map.keys(neuron_best)
    feature_wins = Enum.count(labels, &(feature_best[&1].f1 - neuron_best[&1].f1 >= @margin))
    neuron_wins = Enum.count(labels, &(neuron_best[&1].f1 - feature_best[&1].f1 >= @margin))

    info("\nHead to head over #{length(labels)} labels (a win needs #{@margin} more F1):")

    info(
      "  features win #{feature_wins}, neurons win #{neuron_wins}, " <>
        "ties #{length(labels) - feature_wins - neuron_wins}"
    )

    info(
      "  mean best F1: neurons #{format(mean_f1(neuron_best), 3)}, features #{format(mean_f1(feature_best), 3)}"
    )

    info("\nGrammar labels (words left out), best neuron against best feature:")
    info("  label                                   neuron F1 (P / R)       feature F1 (P / R)")

    for {key, _value} = label <- Enum.sort(labels), key != :word do
      info(
        "  " <>
          String.pad_trailing(label_name(label), 40) <>
          String.pad_trailing(grade(neuron_best[label]), 24) <> grade(feature_best[label])
      )
    end
  end

  defp count_clean(scores), do: Enum.count(scores, &Scoring.monosemantic?/1)

  defp mean_f1(best),
    do: best |> Map.values() |> Enum.map(& &1.f1) |> then(&(Enum.sum(&1) / length(&1)))

  defp grade(%{f1: f1, precision: precision, recall: recall}),
    do: "#{format(f1, 2)} (#{format(precision, 2)} / #{format(recall, 2)})"

  defp describe(grade_with_label) do
    "#{label_name(grade_with_label.label)}  precision #{format(grade_with_label.precision, 2)}  " <>
      "recall #{format(grade_with_label.recall, 2)}"
  end

  defp label_name({key, value}), do: "#{key} = #{inspect(value)}"

  defp info(message), do: Mix.shell().info(message)

  defp format(number, decimals), do: :erlang.float_to_binary(number * 1.0, decimals: decimals)
end
