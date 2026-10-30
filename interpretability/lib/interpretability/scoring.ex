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

  @type family :: %{
          unit: non_neg_integer(),
          key: atom(),
          values: [term()],
          precision: float(),
          recall: float(),
          f1: float()
        }

  @type combination :: %{
          unit: non_neg_integer(),
          labels: [label()],
          precision: float(),
          recall: float(),
          f1: float()
        }

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
  For every unit, the best *family* of labels: one label key and a set of
  its values, such as `part_of_speech` in `[:noun, :verb]`, graded as if
  the family were a single label that is true when any of its values is.

  This separates two kinds of messy unit. A *coarse* unit does one broad
  job, firing on every noun and every verb: no single label matches it,
  but a family does. A *mixed* unit does unrelated jobs, and no family
  matches it either. Goodfire's "Neural Networks Are Already Clean"
  argues that a layer with too few neurons makes them coarse, not mixed.

  Families never use the `word` key, because any set of words is a
  family of words: a unit firing on "llama" and "happy" would count as
  coarse, which is exactly the mixing this is meant to catch.

  For each key, values are added in order of how often the unit fires on
  them, and the best prefix by F1 is kept. Ties go to the smaller family.
  Takes the same options as `score/3`.
  """
  @spec coarse(Nx.Tensor.t(), [map()], keyword()) :: [family()]
  def coarse(activations, labels, options \\ []) do
    %{label_names: label_names, true_positives: true_positives, label_counts: label_counts} =
      table = grade_table(activations, labels, options)

    families =
      [label_names, label_counts]
      |> Enum.zip()
      |> Enum.with_index()
      |> Enum.reject(fn {{{key, _value}, _count}, _index} -> key == :word end)
      |> Enum.group_by(fn {{{key, _value}, _count}, _index} -> key end)

    [true_positives, table.fire_counts]
    |> Enum.zip()
    |> Enum.with_index()
    |> Enum.map(fn {{unit_true_positives, fire_count}, unit} ->
      unit_true_positives = List.to_tuple(unit_true_positives)

      families
      |> Enum.flat_map(fn {key, members} ->
        prefixes(key, members, unit_true_positives, fire_count)
      end)
      |> Enum.max_by(&{&1.f1, -length(&1.values)})
      |> Map.put(:unit, unit)
    end)
  end

  @doc """
  Every unit's grade on one label, in unit order. Takes the options of
  `score/3`.
  """
  @spec grades(Nx.Tensor.t(), [map()], label(), keyword()) :: [grade()]
  def grades(activations, labels, label, options \\ []) do
    %{grades: grades} = grade_table(activations, labels, options)
    Enum.map(grades, fn unit_grades -> Enum.find(unit_grades, &(&1.label == label)) end)
  end

  @doc """
  For every unit, its most precise label: the one it is most often right
  about when it fires, among labels it catches at least `:min_recall` of
  (default 0.1). `nil` when no label qualifies.

  This is closer to how Goodfire judges a neuron, by whether its firings
  are about one thing, without asking it to catch every instance of that
  thing. Labels true on more than `:max_base_rate` of tokens (default
  0.5) are left out, since firing on a few tokens of a label that is true
  almost everywhere is precise by accident. Takes the options of
  `score/3` too.
  """
  @spec precise(Nx.Tensor.t(), [map()], keyword()) :: [grade() | nil]
  def precise(activations, labels, options \\ []) do
    %{grades: grades, label_rates: label_rates} = grade_table(activations, labels, options)
    min_recall = Keyword.get(options, :min_recall, 0.1)
    max_base_rate = Keyword.get(options, :max_base_rate, 0.5)

    Enum.map(grades, fn unit_grades ->
      unit_grades
      |> Enum.filter(&(&1.recall >= min_recall and label_rates[&1.label] <= max_base_rate))
      |> Enum.max_by(&{&1.precision, &1.recall}, fn -> nil end)
    end)
  end

  @doc """
  For every unit, the best pair of labels from different keys, graded as
  one label that is true where both are: `part_of_speech = :noun` and
  `word_number = :plural` together make "plural noun".

  A unit can have one job that no single label names, because the job is
  two labels at once. Without this check such a unit would look mixed.

  Every pair is tallied at once, `:chunk_size` units at a time (default
  16) to bound memory. Takes the options of `score/3` too.
  """
  @spec combined(Nx.Tensor.t(), [map()], keyword()) :: [combination()]
  def combined(activations, labels, options \\ []) do
    threshold_fraction = Keyword.get(options, :threshold_fraction, 0.2)
    chunk_size = Keyword.get(options, :chunk_size, 16)
    {label_names, label_matrix} = label_matrix(indicators(labels))
    names = List.to_tuple(label_names)
    label_count = tuple_size(names)

    pairs =
      for first <- 0..(label_count - 1)//1,
          second <- (first + 1)..(label_count - 1)//1,
          elem(elem(names, first), 0) != elem(elem(names, second), 0),
          do: {first, second}

    pair_index =
      Nx.tensor(Enum.map(pairs, fn {first, second} -> first * label_count + second end))

    pair_counts =
      Nx.Defn.jit(&pair_counts/2, Compiler.options()).(label_matrix, pair_index)

    fires =
      Nx.Defn.jit(&fires(&1, threshold_fraction: threshold_fraction), Compiler.options()).(
        activations
      )

    {_token_count, unit_count} = Nx.shape(activations)
    tally = Nx.Defn.jit(&pair_tallies/4, Compiler.options())
    pair_tuple = List.to_tuple(pairs)
    pair_count_tuple = pair_counts |> counts() |> List.to_tuple()

    0..(unit_count - 1)//chunk_size
    |> Enum.flat_map(fn start ->
      stop = min(start + chunk_size, unit_count) - 1
      fires_chunk = fires[[.., start..stop]]

      {best, true_positives, fire_counts} =
        tally.(fires_chunk, label_matrix, pair_index, pair_counts)

      [counts(best), counts(true_positives), counts(fire_counts)]
      |> Enum.zip_with(fn [pair, true_positive_count, fire_count] ->
        {first, second} = elem(pair_tuple, pair)
        combined_labels = [elem(names, first), elem(names, second)]

        combined_labels
        |> grade(true_positive_count, fire_count, elem(pair_count_tuple, pair))
        |> Map.delete(:label)
        |> Map.put(:labels, combined_labels)
      end)
    end)
    |> Enum.with_index()
    |> Enum.map(fn {combination, unit} -> Map.put(combination, :unit, unit) end)
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

    %{
      grades: grades,
      fire_counts: fire_counts,
      label_rates: label_rates,
      label_names: label_names,
      label_counts: label_counts,
      true_positives: true_positives
    }
  end

  # Every growing family for one key: the values the unit fires on most
  # often first, then the next, and so on.
  defp prefixes(key, members, unit_true_positives, fire_count) do
    members
    |> Enum.map(fn {{{_key, value}, label_count}, index} ->
      {value, elem(unit_true_positives, index), label_count}
    end)
    |> Enum.sort_by(fn {_value, true_positive_count, label_count} ->
      -ratio(true_positive_count, label_count)
    end)
    |> Enum.scan({[], 0, 0}, fn {value, true_positive_count, label_count},
                                {values, true_positive_total, label_total} ->
      {[value | values], true_positive_total + true_positive_count, label_total + label_count}
    end)
    |> Enum.map(fn {values, true_positive_total, label_total} ->
      {key, values}
      |> grade(true_positive_total, fire_count, label_total)
      |> Map.delete(:label)
      |> Map.merge(%{key: key, values: Enum.sort(values)})
    end)
  end

  # The heavy part, compiled: which tokens each unit fires on, and how many
  # of those carry each label. With 512 features and about 200 labels over
  # 15,000 tokens, this is billions of multiply adds.
  defnp tallies(activations, label_matrix, opts \\ []) do
    fires = fires(activations, opts)

    {Nx.dot(fires, [0], label_matrix, [0]), Nx.sum(fires, axes: [0]),
     Nx.sum(label_matrix, axes: [0])}
  end

  defnp fires(activations, opts \\ []) do
    maximum = Nx.reduce_max(activations, axes: [0], keep_axes: true)

    (activations > maximum * opts[:threshold_fraction] and maximum > 0)
    |> Nx.as_type(:f32)
  end

  defnp pair_counts(label_matrix, pair_index) do
    label_matrix
    |> Nx.dot([0], label_matrix, [0])
    |> Nx.flatten()
    |> Nx.take(pair_index)
  end

  # For each unit in the chunk, the pair with the best F1, how many of the
  # unit's firings carry both labels, and how often the unit fires. The
  # tokens × (units · labels) product is why units go a chunk at a time.
  defnp pair_tallies(fires, label_matrix, pair_index, pair_counts) do
    {token_count, unit_count} = Nx.shape(fires)
    label_count = Nx.axis_size(label_matrix, 1)

    true_positives =
      (Nx.new_axis(fires, 2) * Nx.new_axis(label_matrix, 1))
      |> Nx.reshape({token_count, unit_count * label_count})
      |> Nx.dot([0], label_matrix, [0])
      |> Nx.reshape({unit_count, label_count * label_count})
      |> Nx.take(pair_index, axis: 1)

    fire_counts = Nx.sum(fires, axes: [0])
    f1 = 2 * true_positives / Nx.max(Nx.new_axis(fire_counts, 1) + pair_counts, 1)
    best = Nx.argmax(f1, axis: 1)

    {best,
     Nx.take_along_axis(true_positives, Nx.new_axis(best, 1), axis: 1) |> Nx.squeeze(axes: [1]),
     fire_counts}
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
