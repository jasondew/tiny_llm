defmodule Interpretability.ScoringTest do
  use ExUnit.Case, async: true

  alias Interpretability.Scoring

  # Six tokens: who, that, llama, who, dogs, that.
  @labels [
    %{word: "who", part_of_speech: :relative_pronoun, tense: nil},
    %{word: "that", part_of_speech: :relative_pronoun, tense: nil},
    %{word: "llama", part_of_speech: :noun, tense: nil},
    %{word: "who", part_of_speech: :relative_pronoun, tense: nil},
    %{word: "dogs", part_of_speech: :noun, tense: nil},
    %{word: "that", part_of_speech: :relative_pronoun, tense: nil}
  ]

  test "indicators cover every non-nil value and skip labels with no positives" do
    indicators = Scoring.indicators(@labels)

    assert Nx.to_list(indicators[{:word, "who"}]) == [1, 0, 0, 1, 0, 0]
    assert Nx.to_list(indicators[{:part_of_speech, :noun}]) == [0, 0, 1, 0, 1, 0]
    refute Map.has_key?(indicators, {:tense, nil})
    refute Map.has_key?(indicators, {:part_of_speech, :verb})
  end

  test "a unit that matches a label exactly scores 1.0 on both" do
    [score] = Scoring.score(column([0.0, 0.0, 2.0, 0.0, 1.5, 0.0]), @labels)

    assert score.best == %{label: {:part_of_speech, :noun}, precision: 1.0, recall: 1.0, f1: 1.0}
    assert score.firing_rate == 2 / 6
  end

  test "firing on two equal labels halves precision against either one" do
    [score] = Scoring.score(column([1.0, 1.0, 0.0, 1.0, 0.0, 1.0]), @labels)
    who = Enum.find(score.top, &(&1.label == {:word, "who"}))

    assert score.best.label == {:part_of_speech, :relative_pronoun}
    assert who.precision == 0.5
    assert who.recall == 1.0
  end

  test "small activations below the threshold do not count as firing" do
    [score] = Scoring.score(column([0.1, 0.0, 2.0, 0.0, 2.0, 0.0]), @labels)

    assert score.best.label == {:part_of_speech, :noun}
    assert score.best.precision == 1.0
  end

  test "a unit that never fires scores zero, not NaN" do
    [score] = Scoring.score(column([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), @labels)

    assert score.firing_rate == 0.0
    assert score.best.precision == 0.0
    assert score.best.recall == 0.0
    assert score.best.f1 == 0.0
  end

  test "by_label finds the best unit for every label" do
    activations =
      Nx.tensor([
        [0.0, 1.0],
        [0.0, 1.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.5, 0.0],
        [0.0, 1.0]
      ])

    best = Scoring.by_label(activations, @labels)

    assert best[{:part_of_speech, :noun}] == %{unit: 0, precision: 1.0, recall: 1.0, f1: 1.0}
    assert best[{:word, "who"}].unit == 1
    assert best[{:word, "who"}].precision == 0.5
    assert best[{:word, "who"}].recall == 1.0
    refute Map.has_key?(best, {:tense, nil})
  end

  test "by_label can leave out labels true for most tokens" do
    activations = Nx.tensor([[1.0], [1.0], [1.0], [1.0], [0.0], [0.0]])

    labels =
      Enum.map([true, true, true, true, false, true], &%{common: &1, rare: !&1})

    assert Map.has_key?(Scoring.by_label(activations, labels), {:common, true})

    best = Scoring.by_label(activations, labels, max_base_rate: 0.5)
    refute Map.has_key?(best, {:common, true})
    assert Map.has_key?(best, {:rare, true})
  end

  test "monosemantic means both precision and recall reach the threshold" do
    assert Scoring.monosemantic?(%{best: %{precision: 0.95, recall: 0.9}})
    refute Scoring.monosemantic?(%{best: %{precision: 0.95, recall: 0.85}})
    assert Scoring.monosemantic?(%{best: %{precision: 0.85, recall: 0.85}}, 0.8)
  end

  describe "coarse/3" do
    # Six tokens: llama, sees, happy, dogs, chases, calm.
    @mixed_labels [
      %{word: "llama", part_of_speech: :noun, word_number: :singular},
      %{word: "sees", part_of_speech: :verb, word_number: :singular},
      %{word: "happy", part_of_speech: :adjective, word_number: :singular},
      %{word: "dogs", part_of_speech: :noun, word_number: :plural},
      %{word: "chases", part_of_speech: :verb, word_number: :singular},
      %{word: "calm", part_of_speech: :adjective, word_number: nil}
    ]

    test "a unit for nouns and verbs is coarse: one family, two values" do
      [single] = Scoring.score(column([1.0, 1.0, 0.0, 1.0, 1.0, 0.0]), @mixed_labels)
      [coarse] = Scoring.coarse(column([1.0, 1.0, 0.0, 1.0, 1.0, 0.0]), @mixed_labels)

      assert single.best.f1 < 0.9
      assert coarse.key == :part_of_speech
      assert coarse.values == [:noun, :verb]
      assert coarse.precision == 1.0
      assert coarse.recall == 1.0
    end

    test "unrelated words do not make a family" do
      [coarse] = Scoring.coarse(column([1.0, 0.0, 1.0, 0.0, 0.0, 0.0]), @mixed_labels)

      assert coarse.key != :word
      assert coarse.f1 < 0.9
    end

    test "a unit that never fires has an empty family" do
      [coarse] = Scoring.coarse(column([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), @mixed_labels)

      assert coarse.f1 == 0.0
    end
  end

  ## PRIVATE FUNCTIONS

  defp column(values), do: Nx.reshape(Nx.tensor(values), {length(values), 1})
end
