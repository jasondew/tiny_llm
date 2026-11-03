defmodule Mix.Tasks.Interp.Export do
  @shortdoc "Writes the model, SAE and unit data for the browser explorer"

  @moduledoc """
  Writes everything the browser explorer loads into `explorer/data/`:

  - `model.json` and `model.bin`: the llama model's weights, its
    vocabulary and its settings
  - `sae.json` and `sae.bin`: the 512 feature SAE
  - `neurons.json` and `features.json`: what each unit is about, and the
    evaluation words it fired on most strongly

      mix interp.export

  Reads `priv/checkpoints/model.etf` and `priv/checkpoints/sae_512.etf`
  and changes neither. See `Interpretability.Export`.
  """

  use Mix.Task

  alias Interpretability.Activations
  alias Interpretability.Checkpoint
  alias Interpretability.Export
  alias Interpretability.Model
  alias Interpretability.SAE
  alias Interpretability.Scoring
  alias Interpretability.Vocab

  @model_path "priv/checkpoints/model.etf"
  @sae_path "priv/checkpoints/sae_512.etf"
  @data_directory "explorer/data"
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
  end

  ## PRIVATE FUNCTIONS

  defp write_json(name, data), do: write(name, JSON.encode!(data))

  defp write(name, contents) do
    path = Path.join(@data_directory, name)
    File.write!(path, contents)
    Mix.shell().info("Wrote #{path} (#{div(byte_size(contents), 1024)} KB)")
  end
end
