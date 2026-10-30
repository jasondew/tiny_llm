defmodule Mix.Tasks.Interp.Steer do
  @shortdoc "Pushes five candidate plural directions and measures flips against damage"

  @moduledoc """
  Which direction in the neurons best means "the subject is plural"?
  Pushes five candidates at the same sizes and measures each with
  `Interpretability.Steering`:

  1. **mean difference**: average neurons where the subject is plural,
     minus where it is singular. No interpretability needed: the yardstick.
  2. **best neuron**: the neuron that best matches plural agreement.
  3. **best SAE feature**: the 512 feature SAE's best plural feature.
  4. **SAE fragments**: every 512 SAE feature at least 90% precise for
     plural agreement (and catching at least 1% of it), added together.
  5. **narrow SAE feature**: the best plural feature of a 64 feature SAE,
     which has less room to split the concept into fragments.

      mix interp.steer

  Push sizes are multiples of the average length of the neuron vector at
  a word, so 1.0 is a push as large as everything the neurons normally
  say. The 64 feature SAE is trained once and saved to
  `priv/checkpoints/sae_64.etf`. Writes `priv/results/steering.etf`.
  """

  use Mix.Task

  alias Interpretability.Activations
  alias Interpretability.Checkpoint
  alias Interpretability.Dataset
  alias Interpretability.Model
  alias Interpretability.Probes
  alias Interpretability.SAE
  alias Interpretability.Scoring
  alias Interpretability.Steering

  @model_path "priv/checkpoints/model.etf"
  @sae_path "priv/checkpoints/sae_512.etf"
  @narrow_sae_path "priv/checkpoints/sae_64.etf"
  @results_path "priv/results/steering.etf"
  @training_sentence_count 20_000
  @plural {:agreement_number, :plural}
  @singular {:agreement_number, :singular}
  @multiples [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
  @cold_open ~w(the llama who chases the dogs)

  @impl Mix.Task
  def run(_arguments) do
    Mix.Task.run("app.start")
    %{config: config, params: params} = Checkpoint.load(@model_path)
    {training, evaluation} = Model.datasets(config)
    activations = Activations.collect(params, evaluation)
    probes = Probes.held_out(config)
    directions = directions(params, training, activations)

    word_length =
      activations.hidden
      |> Nx.pow(2)
      |> Nx.sum(axes: [1])
      |> Nx.sqrt()
      |> Nx.mean()
      |> Nx.to_number()

    info("Average neuron vector length at a word: #{format(word_length, 2)}")
    info("Singular subject probes: #{Enum.count(probes, &(&1.number == :singular))}")

    curves =
      Map.new(directions, fn {name, %{direction: direction} = candidate} ->
        info("\n#{name}: #{candidate.description}")
        info("  push    flip rate    damage (nats)    cold open margin")

        points =
          Enum.map(@multiples, fn multiple ->
            strength = multiple * word_length

            point = %{
              multiple: multiple,
              flip_rate: Steering.flip_rate(params, probes, direction, strength),
              damage: Steering.damage(params, evaluation, direction, strength),
              cold_open_margin: Steering.margin(params, @cold_open, direction, strength)
            }

            info(
              "  #{String.pad_leading(format(multiple, 2), 4)}" <>
                "    #{String.pad_leading(format(point.flip_rate, 3), 9)}" <>
                "    #{String.pad_leading(format(point.damage, 3), 13)}" <>
                "    #{String.pad_leading(format(point.cold_open_margin, 3), 16)}"
            )

            point
          end)

        {name, Map.put(candidate, :points, points)}
      end)

    Checkpoint.save(%{word_length: word_length, curves: curves}, @results_path)
    info("\nWrote #{@results_path}")
  end

  ## PRIVATE FUNCTIONS

  defp directions(params, training, activations) do
    %{config: sae_config, sae: sae} = Checkpoint.load(@sae_path)
    narrow = narrow_sae(params, training)
    features = SAE.features(sae, activations.hidden, sae_config.active_count)
    narrow_features = SAE.features(narrow.sae, activations.hidden, narrow.config.active_count)

    best_neuron = Scoring.by_label(activations.hidden, activations.labels)[@plural]
    best_feature = Scoring.by_label(features, activations.labels)[@plural]
    best_narrow = Scoring.by_label(narrow_features, activations.labels)[@plural]

    fragments =
      features
      |> Scoring.grades(activations.labels, @plural)
      |> Enum.with_index()
      |> Enum.filter(fn {grade, _feature} -> grade.precision >= 0.9 and grade.recall >= 0.01 end)
      |> Enum.map(fn {_grade, feature} -> feature end)

    [
      {"1 mean difference",
       %{
         direction: Steering.mean_difference(activations, @plural, @singular),
         description: "plural minus singular average"
       }},
      {"2 best neuron",
       %{
         direction: Steering.neuron(best_neuron.unit, Nx.axis_size(activations.hidden, 1)),
         description: "neuron #{best_neuron.unit}, #{describe(best_neuron)}"
       }},
      {"3 best SAE feature",
       %{
         direction: Steering.features(sae, [best_feature.unit]),
         description: "feature #{best_feature.unit} of 512, #{describe(best_feature)}"
       }},
      {"4 SAE fragments",
       %{
         direction: Steering.features(sae, fragments),
         description: "#{length(fragments)} features: #{Enum.join(fragments, ", ")}"
       }},
      {"5 narrow SAE feature",
       %{
         direction: Steering.features(narrow.sae, [best_narrow.unit]),
         description: "feature #{best_narrow.unit} of 64, #{describe(best_narrow)}"
       }}
    ]
  end

  defp narrow_sae(params, training) do
    if File.exists?(@narrow_sae_path) do
      Checkpoint.load(@narrow_sae_path)
    else
      info("Training the 64 feature SAE...")
      config = %SAE.Config{width: 64}

      training_activations =
        params
        |> Activations.collect(Dataset.build(Enum.take(training.trees, @training_sentence_count)))
        |> Map.fetch!(:hidden)

      {sae, _history} = SAE.train(config, training_activations)
      narrow = %{config: config, sae: sae}
      Checkpoint.save(narrow, @narrow_sae_path)
      narrow
    end
  end

  defp describe(grade),
    do: "precision #{format(grade.precision, 2)}, recall #{format(grade.recall, 2)}"

  defp info(message), do: Mix.shell().info(message)

  defp format(number, decimals), do: :erlang.float_to_binary(number * 1.0, decimals: decimals)
end
