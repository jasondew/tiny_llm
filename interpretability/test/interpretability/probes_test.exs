defmodule Interpretability.ProbesTest do
  use ExUnit.Case, async: true

  alias Interpretability.Probes
  alias Interpretability.Trees
  alias Interpretability.Vocab

  test "one probe per present tense verb, with category and distractor flag" do
    assert Probes.from_tree(Trees.probe_sentence()) == [
             %{
               prefix: ~w(the llama who),
               number: :singular,
               category: :inside_relative,
               mismatch: false
             },
             %{
               prefix: ~w(the llama who chases the dogs),
               number: :singular,
               category: :relative_distractor,
               mismatch: true
             }
           ]
  end

  test "a relative clause on the object is its own category" do
    assert [%{category: :simple_subject}, %{category: :object_relative, number: :plural}] =
             Probes.from_tree(Trees.object_relative_sentence())
  end

  test "a past tense verb yields no probe" do
    assert Probes.from_tree(Trees.past_tense_sentence()) == []
  end

  test "accuracy groups probes by category and distractor flag" do
    always_singular = fn _prefix ->
      Enum.map(Vocab.words(), fn word -> if word == "flees", do: 1.0, else: 0.0 end)
    end

    probes = Probes.from_tree(Trees.probe_sentence())

    assert Probes.accuracy(always_singular, probes) == %{
             {:inside_relative, false} => 1.0,
             {:relative_distractor, true} => 1.0
           }

    plural_probe = %{
      prefix: ~w(the dogs),
      number: :plural,
      category: :simple_subject,
      mismatch: false
    }

    assert Probes.accuracy(always_singular, [plural_probe]) == %{{:simple_subject, false} => 0.0}
  end

  test "held out probes come only from sentences the model never trained on" do
    config = %Interpretability.Model.Config{training_corpus_size: 300}
    seen = Interpretability.Model.training_sentences(config)
    trees = Probes.held_out_trees(config, 300)

    assert length(trees) < 300
    refute Enum.any?(trees, &MapSet.member?(seen, Interpretability.Grammar.words(&1)))
    assert Probes.held_out(config, 300) == Enum.flat_map(trees, &Probes.from_tree/1)
  end
end
