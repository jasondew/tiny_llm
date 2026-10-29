defmodule Interpretability.Cleanliness do
  @moduledoc """
  Does how a model is trained decide how clean its neurons are?

  Goodfire's "Neural Networks Are Already Clean" (2026) argues that
  polysemantic neurons are mostly a training problem, not a law of
  nature: modern architectures and regularization give neurons with one
  job each, and a layer squeezed too narrow makes its neurons *coarse*
  (one broad job) rather than *mixed* (unrelated jobs). Superposition
  says the opposite: a layer with more concepts than neurons packs
  several concepts into each one, however it is trained.

  This module tests that on the llama model. Each variant retrains it
  with one change, and every variant is measured the same way:

  - **Is the model still good?** Evaluation loss and agreement probe
    accuracy. A change that cleans the neurons by breaking the model
    does not count.
  - **How clean are the neurons?** Each neuron is one of five kinds:
    *clean* (one label matches it, precision and recall at the bar),
    *coarse* (no single label does, but a family of one label's values
    does, see `Interpretability.Scoring.coarse/3`), *combined* (two labels
    together do, such as "plural noun", see
    `Interpretability.Scoring.combined/3`), *mixed* (none of these), or
    *dead* (never fires). The first match in that order wins.
  - **How many are precise?** Neurons right at least 90% of the time
    when they fire, whether or not they catch every instance, which is
    closer to how Goodfire judges neurons. See
    `Interpretability.Scoring.precise/3`.
  - **How much does the rubric matter?** The clean count at every pairing
    of firing threshold and bar, since Goodfire stresses that "clean"
    depends on where the lines are drawn.

  Three seeds of the unchanged model show how much the numbers move by
  chance alone; a variant has to beat that spread to mean anything.
  """

  alias Interpretability.Activations
  alias Interpretability.Dataset
  alias Interpretability.Model
  alias Interpretability.Probes
  alias Interpretability.Scoring

  @type variant :: %{name: String.t(), changes: keyword()}

  @type kinds :: %{
          clean: non_neg_integer(),
          coarse: non_neg_integer(),
          combined: non_neg_integer(),
          mixed: non_neg_integer(),
          dead: non_neg_integer()
        }

  @type result :: %{
          evaluation_loss: float(),
          probe_accuracy: float(),
          mismatch_accuracy: float(),
          neuron_count: pos_integer(),
          precise: non_neg_integer(),
          kinds: %{float() => kinds()},
          rubric: %{{float(), float()} => non_neg_integer()},
          median_f1: float(),
          median_family_f1: float()
        }

  @thresholds [0.1, 0.2, 0.3, 0.5]
  @bars [0.8, 0.9]

  @doc """
  The talk's model first, then one variant per change.
  """
  @spec variants() :: [variant()]
  def variants do
    [
      %{name: "baseline", changes: []},
      %{name: "seed 1235", changes: [seed: 1235]},
      %{name: "seed 1236", changes: [seed: 1236]},
      %{name: "weight decay 0.01", changes: [weight_decay: 0.01]},
      %{name: "weight decay 0.1", changes: [weight_decay: 0.1]},
      %{name: "weight decay 0.1, seed 1235", changes: [weight_decay: 0.1, seed: 1235]},
      %{name: "weight decay 0.1, seed 1236", changes: [weight_decay: 0.1, seed: 1236]},
      %{name: "activity penalty 0.001", changes: [activation_penalty: 1.0e-3]},
      %{name: "activity penalty 0.01", changes: [activation_penalty: 1.0e-2]},
      %{name: "SoLU", changes: [activation: :solu]},
      %{name: "width 512", changes: [hidden_width: 512]}
    ]
  end

  @doc """
  The model config a variant trains with.
  """
  @spec config(variant()) :: Model.Config.t()
  def config(variant), do: struct!(%Model.Config{}, variant.changes)

  @doc """
  Measures a trained model on the evaluation sentences and probes.
  """
  @spec measure(Model.params(), Dataset.t(), [Probes.probe()]) :: result()
  def measure(params, evaluation, probes) do
    activations = Activations.collect(params, evaluation)
    {_token_count, neuron_count} = Nx.shape(activations.hidden)
    scores = Scoring.score(activations.hidden, activations.labels)
    families = Scoring.coarse(activations.hidden, activations.labels)
    combinations = Scoring.combined(activations.hidden, activations.labels)
    precise = Scoring.precise(activations.hidden, activations.labels)
    live = Enum.reject(scores, &(&1.firing_rate == 0.0))
    live_units = MapSet.new(live, & &1.unit)

    %{
      evaluation_loss: Nx.to_number(Model.evaluate_loss(params, evaluation)),
      neuron_count: neuron_count,
      precise: Enum.count(precise, &(&1 != nil and &1.precision >= 0.9)),
      kinds: Map.new(@bars, &{&1, kinds(scores, families, combinations, &1)}),
      rubric: rubric(activations),
      median_f1: median(Enum.map(live, & &1.best.f1)),
      median_family_f1:
        families
        |> Enum.filter(&MapSet.member?(live_units, &1.unit))
        |> Enum.map(& &1.f1)
        |> median()
    }
    |> Map.merge(probe_accuracies(params, probes))
  end

  ## PRIVATE FUNCTIONS

  defp kinds(scores, families, combinations, bar) do
    [scores, families, combinations]
    |> Enum.zip_with(fn [score, family, combination] -> kind(score, family, combination, bar) end)
    |> Enum.frequencies()
    |> then(&Map.merge(%{clean: 0, coarse: 0, combined: 0, mixed: 0, dead: 0}, &1))
  end

  defp kind(%{firing_rate: rate}, _family, _combination, _bar) when rate == 0.0, do: :dead

  defp kind(score, family, combination, bar) do
    cond do
      Scoring.monosemantic?(score, bar) -> :clean
      meets?(family, bar) -> :coarse
      meets?(combination, bar) -> :combined
      true -> :mixed
    end
  end

  defp meets?(grade, bar), do: grade.precision >= bar and grade.recall >= bar

  defp rubric(activations) do
    for threshold <- @thresholds, into: %{} do
      scores =
        Scoring.score(activations.hidden, activations.labels, threshold_fraction: threshold)

      {threshold, scores}
    end
    |> Enum.flat_map(fn {threshold, scores} ->
      for bar <- @bars,
          do: {{threshold, bar}, Enum.count(scores, &Scoring.monosemantic?(&1, bar))}
    end)
    |> Map.new()
  end

  defp probe_accuracies(params, probes) do
    predictor = Model.predictor(params)

    correct =
      Enum.map(probes, &{&1.mismatch, Probes.predicted_number(predictor, &1.prefix) == &1.number})

    mismatches = Enum.filter(correct, fn {mismatch, _correct} -> mismatch end)

    %{probe_accuracy: share(correct), mismatch_accuracy: share(mismatches)}
  end

  defp share([]), do: 0.0
  defp share(outcomes), do: Enum.count(outcomes, &elem(&1, 1)) / length(outcomes)

  defp median([]), do: 0.0

  defp median(values) do
    sorted = Enum.sort(values)
    count = length(sorted)
    middle = div(count, 2)

    if rem(count, 2) == 1,
      do: Enum.at(sorted, middle),
      else: (Enum.at(sorted, middle - 1) + Enum.at(sorted, middle)) / 2
  end
end
