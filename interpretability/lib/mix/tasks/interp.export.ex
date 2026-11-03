defmodule Mix.Tasks.Interp.Export do
  @shortdoc "Writes the model, SAE and unit data for the browser explorer"

  @moduledoc """
  Writes everything the browser explorer loads into `explorer/data/`:

  - `model.json` and `model.bin`: the llama model's weights, its
    vocabulary and its settings
  - `sae.json` and `sae.bin`: the 512 feature SAE
  - `neurons.json` and `features.json`: what each unit is about, and the
    evaluation words it fired on most strongly

  - `steering.json`: three plural directions to push along (the mean
    difference, the best neuron and the best SAE feature), the push unit,
    and the present tense verbs the agreement margin compares

  It also writes `explorer/test/fixtures.json`: the cold open and the
  first 20 evaluation sentences, with the logits, neurons and active SAE
  features Nx computes for them, and the cold open's agreement margin
  under each steering direction at pushes 0, 1 and 2. The explorer's parity test checks the
  page's JavaScript forward pass against these.

      mix interp.export

  Reads `priv/checkpoints/model.etf` and `priv/checkpoints/sae_512.etf`
  and changes neither. See `Interpretability.Export`.
  """

  use Mix.Task

  alias Interpretability.Activations
  alias Interpretability.Checkpoint
  alias Interpretability.Export
  alias Interpretability.Grammar
  alias Interpretability.Model
  alias Interpretability.Probes
  alias Interpretability.SAE
  alias Interpretability.Scoring
  alias Interpretability.Steering
  alias Interpretability.Vocab

  @model_path "priv/checkpoints/model.etf"
  @sae_path "priv/checkpoints/sae_512.etf"
  @data_directory "explorer/data"
  @fixtures_path "explorer/test/fixtures.json"
  @fixture_sentence_count 20
  @cold_open ~w(the llama who chases the dogs)
  @plural {:agreement_number, :plural}
  @singular {:agreement_number, :singular}
  @fixture_multiples [0.0, 1.0, 2.0]
  @example_count 10
  @rmsnorm_epsilon 1.0e-6

  @impl Mix.Task
  def run(_arguments) do
    Mix.Task.run("app.start")
    %{config: config, params: params} = Checkpoint.load(@model_path)
    %{config: sae_config, sae: sae} = Checkpoint.load(@sae_path)
    {_training, evaluation} = Model.datasets(config)
    activations = Activations.collect(params, evaluation)
    features = SAE.features(sae, activations.hidden, sae_config.active_count)
    File.mkdir_p!(@data_directory)

    {model_manifest, model_binary} = Export.weights(params)

    write_json(
      "model.json",
      Map.merge(model_manifest, %{
        "vocabulary" => Vocab.words(),
        "start_token" => Vocab.start_token(),
        "context_length" => Nx.axis_size(params.positions, 0),
        "rmsnorm_epsilon" => @rmsnorm_epsilon
      })
    )

    write("model.bin", model_binary)

    {sae_manifest, sae_binary} = Export.weights(sae)
    write_json("sae.json", Map.put(sae_manifest, "active_count", sae_config.active_count))
    write("sae.bin", sae_binary)

    neuron_scores = Scoring.score(activations.hidden, activations.labels)

    write_json(
      "neurons.json",
      Export.units(activations, evaluation, neuron_scores, @example_count)
    )

    feature_scores = Scoring.score(features, activations.labels)

    write_json(
      "features.json",
      Export.units(%{activations | hidden: features}, evaluation, feature_scores, @example_count)
    )

    sentences =
      [
        @cold_open
        | evaluation.trees |> Enum.take(@fixture_sentence_count) |> Enum.map(&Grammar.words/1)
      ]

    fixtures = Enum.map(sentences, &fixture(&1, params, sae, sae_config.active_count))
    {steering, margins} = steering(params, activations, features, sae)
    write_json("steering.json", steering)
    File.write!(@fixtures_path, JSON.encode!(%{"sentences" => fixtures, "margins" => margins}))
    Mix.shell().info("Wrote #{@fixtures_path}")
  end

  ## PRIVATE FUNCTIONS

  defp steering(params, activations, features, sae) do
    best_neuron = Scoring.by_label(activations.hidden, activations.labels)[@plural]
    best_feature = Scoring.by_label(features, activations.labels)[@plural]

    word_length =
      activations.hidden
      |> Nx.pow(2)
      |> Nx.sum(axes: [1])
      |> Nx.sqrt()
      |> Nx.mean()
      |> Nx.to_number()

    directions = [
      {"mean difference", "average plural subject neurons minus singular",
       Steering.mean_difference(activations, @plural, @singular)},
      {"neuron #{best_neuron.unit}", "the neuron that best matches plural agreement",
       Steering.neuron(best_neuron.unit, Nx.axis_size(activations.hidden, 1))},
      {"feature #{best_feature.unit}", "the SAE feature that best matches plural agreement",
       Steering.features(sae, [best_feature.unit])}
    ]

    {plural_verbs, singular_verbs} =
      Enum.split_with(Probes.present_verbs(), &(elem(&1, 1) == :plural))

    steering = %{
      "word_length" => word_length,
      "plural_verbs" => Enum.map(plural_verbs, &elem(&1, 0)),
      "singular_verbs" => Enum.map(singular_verbs, &elem(&1, 0)),
      "directions" =>
        Enum.map(directions, fn {name, description, direction} ->
          %{"name" => name, "description" => description, "vector" => Nx.to_list(direction)}
        end)
    }

    margins =
      for {name, _description, direction} <- directions, multiple <- @fixture_multiples do
        %{
          "direction" => name,
          "multiple" => multiple,
          "margin" => Steering.margin(params, @cold_open, direction, multiple * word_length)
        }
      end

    {steering, margins}
  end

  defp fixture(words, params, sae, active_count) do
    context_length = Nx.axis_size(params.positions, 0)
    ids = Vocab.encode([Vocab.start_token() | words])
    padding = List.duplicate(Vocab.word_to_id(Vocab.end_token()), context_length - length(ids))
    tokens = Nx.tensor([ids ++ padding], type: :s64)
    length = length(ids)
    logits = Model.forward(params, tokens)[0][0..(length - 1)]
    hidden = Model.hidden(params, tokens)[0][0..(length - 1)]

    features =
      sae
      |> SAE.features(hidden, active_count)
      |> Nx.to_list()
      |> Enum.map(fn row ->
        for {value, index} <- Enum.with_index(row), value != 0.0, do: [index, value]
      end)

    %{
      "words" => words,
      "ids" => ids,
      "logits" => Nx.to_list(logits),
      "hidden" => Nx.to_list(hidden),
      "features" => features
    }
  end

  defp write_json(name, data), do: write(name, JSON.encode!(data))

  defp write(name, contents) do
    path = Path.join(@data_directory, name)
    File.write!(path, contents)
    Mix.shell().info("Wrote #{path} (#{div(byte_size(contents), 1024)} KB)")
  end
end
