defmodule Interpretability.Scoring do
  @moduledoc """
  Grading a unit against the answer key: precision and recall.

  A *unit* is anything inside the model that produces one number per
  token: an MLP neuron, or later a sparse autoencoder feature. To ask what
  a unit means, we compare where it fires with where each label is true.

  - **Precision**: of the tokens where the unit fired, the fraction where
    the label was true. When it says "plural", how often is it right?
  - **Recall**: of the tokens where the label was true, the fraction where
    the unit fired. Of all the plural moments, how many did it catch?

  A unit that fires everywhere has perfect recall and poor precision. A
  unit that fires once, correctly, has perfect precision and poor recall.
  Their balance, the *F1 score* `2 · precision · recall / (precision +
  recall)`, is high only when both are. Each unit's best label is the one
  with the highest F1, and a unit is *monosemantic* when both precision and
  recall reach 0.9 for that label.

  ## When does a unit "fire"?

  A unit fires on a token when its activation is above 20% of the largest
  activation that unit reaches anywhere in the dataset. The same rule
  applies to neurons and to SAE features. A raw ReLU neuron is above zero
  on a large share of tokens, often weakly, so "above zero" would make
  neurons look messier than they are and rig the comparison in the SAE's
  favour.
  """

  @typedoc "One label value, such as `{:part_of_speech, :noun}`."
  @type label :: {atom(), term()}

  @type grade :: %{label: label(), precision: float(), recall: float(), f1: float()}

  @type score :: %{
          unit: non_neg_integer(),
          firing_rate: float(),
          best: grade(),
          top: [grade()]
        }

  import Nx.Defn

  alias Interpretability.Compiler

  @top_count 3

  @doc """
  For every label value that is true somewhere, a 0 or 1 per token.
  """
  @spec indicators([map()]) :: %{label() => Nx.Tensor.t()}
  def indicators(labels) do
    labels
    |> Enum.flat_map(fn token_labels ->
      for {key, value} <- token_labels, value != nil, do: {key, value}
    end)
    |> Enum.uniq()
    |> Map.new(fn {key, value} = label ->
      {label, Nx.tensor(Enum.map(labels, &if(&1[key] == value, do: 1, else: 0)), type: :u8)}
    end)
  end

  @doc """
  Grades every unit (column) of `activations`, a tokens × units tensor,
  against every label. Each score carries the unit's best label and its
  `#{@top_count}` best labels in `top`.

  Options: `:threshold_fraction`, the share of a unit's maximum it must
  exceed to count as firing. Defaults to 0.2.
  """
  @spec score(Nx.Tensor.t(), [map()], keyword()) :: [score()]
  def score(activations, labels, options \\ []) do
    {token_count, _unit_count} = Nx.shape(activations)
    %{grades: grades, fire_counts: fire_counts} = grade_table(activations, labels, options)

    [grades, fire_counts]
    |> Enum.zip()
    |> Enum.with_index()
    |> Enum.map(fn {{unit_grades, fire_count}, unit} ->
      top = unit_grades |> Enum.sort_by(& &1.f1, :desc) |> Enum.take(@top_count)
      %{unit: unit, firing_rate: fire_count / token_count, best: hd(top), top: top}
    end)
  end

  @doc """
  The other way round: for every label, the unit that matches it best
  (highest F1). This is how neurons and SAE features are compared head to
  head: for each concept, how clean is the best unit either side has?

  Takes the same options as `score/3`, plus `:max_base_rate`: leave out
  labels true for more than this share of tokens. A unit that fires on
  every token already scores well on a label that is true almost
  everywhere, so such labels say nothing about how clean a unit is, and a
  sparse feature can never win them. Comparisons use 0.5.
  """
  @spec by_label(Nx.Tensor.t(), [map()], keyword()) :: %{
          label() => %{
            unit: non_neg_integer(),
            precision: float(),
            recall: float(),
            f1: float()
          }
        }
  def by_label(activations, labels, options \\ []) do
    %{grades: grades, label_rates: label_rates} = grade_table(activations, labels, options)
    max_base_rate = Keyword.get(options, :max_base_rate, 1.0)

    grades
    |> Enum.with_index()
    |> Enum.flat_map(fn {unit_grades, unit} ->
      Enum.map(unit_grades, &Map.put(&1, :unit, unit))
    end)
    |> Enum.filter(&(label_rates[&1.label] <= max_base_rate))
    |> Enum.group_by(& &1.label)
    |> Map.new(fn {label, candidates} ->
      best = Enum.max_by(candidates, & &1.f1, fn -> nil end)
      {label, Map.delete(best, :label)}
    end)
  end

  @doc """
  Whether a unit's best label reaches `threshold` on both precision and
  recall.
  """
  @spec monosemantic?(%{best: map()}, float()) :: boolean()
  def monosemantic?(%{best: best}, threshold \\ 0.9),
    do: best.precision >= threshold and best.recall >= threshold

  ## PRIVATE FUNCTIONS

  # Every unit graded against every label: one list of grades per unit, in
  # label order.
  defp grade_table(activations, labels, options) do
    threshold_fraction = Keyword.get(options, :threshold_fraction, 0.2)
    {label_names, label_matrix} = label_matrix(indicators(labels))

    {true_positives, fire_counts, label_counts} =
      Nx.Defn.jit(&tallies(&1, &2, threshold_fraction: threshold_fraction), Compiler.options()).(
        activations,
        label_matrix
      )

    true_positives = counts(true_positives)
    fire_counts = counts(fire_counts)
    label_counts = counts(label_counts)

    grades =
      Enum.zip_with(true_positives, fire_counts, fn unit_true_positives, fire_count ->
        Enum.zip_with([label_names, unit_true_positives, label_counts], fn [
                                                                             label,
                                                                             true_positive_count,
                                                                             label_count
                                                                           ] ->
          grade(label, true_positive_count, fire_count, label_count)
        end)
      end)

    {token_count, _unit_count} = Nx.shape(activations)

    label_rates =
      [label_names, label_counts]
      |> Enum.zip()
      |> Map.new(fn {label, count} -> {label, count / token_count} end)

    %{grades: grades, fire_counts: fire_counts, label_rates: label_rates}
  end

  # The heavy part, compiled: which tokens each unit fires on, and how many
  # of those carry each label. With 512 features and about 200 labels over
  # 15,000 tokens, this is billions of multiply adds.
  defnp tallies(activations, label_matrix, opts \\ []) do
    maximum = Nx.reduce_max(activations, axes: [0], keep_axes: true)

    fires =
      (activations > maximum * opts[:threshold_fraction] and maximum > 0)
      |> Nx.as_type(:f32)

    {Nx.dot(fires, [0], label_matrix, [0]), Nx.sum(fires, axes: [0]),
     Nx.sum(label_matrix, axes: [0])}
  end

  defp label_matrix(indicators) do
    label_names = indicators |> Map.keys() |> Enum.sort()
    columns = Enum.map(label_names, &Nx.as_type(indicators[&1], :f32))
    {label_names, Nx.stack(columns, axis: 1)}
  end

  # Counts come back from f32 sums; round them to exact integers so every
  # ratio is computed exactly.
  defp counts(tensor), do: tensor |> Nx.round() |> Nx.as_type(:s64) |> Nx.to_list()

  defp grade(label, true_positive_count, fire_count, label_count) do
    precision = ratio(true_positive_count, fire_count)
    recall = ratio(true_positive_count, label_count)
    f1 = ratio(2 * precision * recall, precision + recall)
    %{label: label, precision: precision, recall: recall, f1: f1}
  end

  defp ratio(_numerator, denominator) when denominator == 0, do: 0.0
  defp ratio(numerator, denominator), do: numerator / denominator
end
