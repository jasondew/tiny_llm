defmodule Mix.Tasks.Interp.Export do
  @shortdoc "Writes the model, SAE and unit data for the browser explorer"

  @moduledoc """
  Writes everything the browser explorer loads into `explorer/data/`:

  - `model.json` and `model.bin`: the llama model's weights, its
    vocabulary and its settings
  - `sae.json` and `sae.bin`: the 512 feature SAE
  - `neurons.json` and `features.json`: what each unit is about, and the
    evaluation words it fired on most strongly

  It also writes `explorer/test/fixtures.json`: the cold open and the
  first 20 evaluation sentences, with the logits, neurons and active SAE
  features Nx computes for them. The explorer's parity test checks the
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
  alias Interpretability.SAE
  alias Interpretability.Scoring
  alias Interpretability.Vocab

  @model_path "priv/checkpoints/model.etf"
  @sae_path "priv/checkpoints/sae_512.etf"
  @data_directory "explorer/data"
  @fixtures_path "explorer/test/fixtures.json"
  @fixture_sentence_count 20
  @cold_open ~w(the llama who chases the dogs)
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
    File.write!(@fixtures_path, JSON.encode!(%{"sentences" => fixtures}))
    Mix.shell().info("Wrote #{@fixtures_path}")
  end

  ## PRIVATE FUNCTIONS

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
