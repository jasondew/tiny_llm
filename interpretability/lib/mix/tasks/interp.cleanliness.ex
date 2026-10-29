defmodule Mix.Tasks.Interp.Cleanliness do
  @shortdoc "Retrains the llama model one change at a time and measures its neurons"

  @moduledoc """
  Tests whether training choices make the neurons cleaner. See
  `Interpretability.Cleanliness` for the variants and what is measured.

      mix interp.cleanliness
      mix interp.cleanliness --only "SoLU,width 512"
      mix interp.cleanliness --force

  The baseline is the talk's saved model, `priv/checkpoints/model.etf`,
  and is never retrained or overwritten. Every other variant trains on
  the talk's training sentences, from scratch. Each result is cached in
  `priv/results/cleanliness/`, so a rerun only trains what is missing;
  `--force` retrains everything asked for.
  """

  use Mix.Task

  alias Interpretability.Checkpoint
  alias Interpretability.Cleanliness
  alias Interpretability.Model
  alias Interpretability.Probes

  @model_path "priv/checkpoints/model.etf"
  @results_directory "priv/results/cleanliness"

  @impl Mix.Task
  def run(arguments) do
    Mix.Task.run("app.start")

    {switches, _rest} = OptionParser.parse!(arguments, strict: [only: :string, force: :boolean])
    variants = selected(Cleanliness.variants(), switches[:only])
    base = %Model.Config{}
    {training, evaluation} = Model.datasets(base)
    probes = Probes.held_out(base)

    results =
      Enum.map(variants, fn variant ->
        path = Path.join(@results_directory, slug(variant.name) <> ".etf")

        if File.exists?(path) and not Keyword.get(switches, :force, false) do
          {variant, Checkpoint.load(path)}
        else
          info("Training #{variant.name}...")
          {microseconds, params} = :timer.tc(fn -> params(variant, training) end)
          result = Cleanliness.measure(params, evaluation, probes)
          Checkpoint.save(result, path)
          info("  done in #{format(microseconds / 1.0e6, 0)} seconds")
          {variant, result}
        end
      end)

    print_model_table(results)
    print_kind_table(results, 0.9)
    print_kind_table(results, 0.8)
    print_rubric_table(results)
  end

  ## PRIVATE FUNCTIONS

  defp selected(variants, nil), do: variants

  defp selected(variants, only) do
    names = only |> String.split(",") |> Enum.map(&String.trim/1)
    Enum.filter(variants, &(&1.name in names))
  end

  defp params(%{name: "baseline"}, _training), do: Checkpoint.load(@model_path).params

  defp params(variant, training) do
    {params, _history} = Model.train(Cleanliness.config(variant), training)
    params
  end

  defp print_model_table(results) do
    info("\nIs the model still good?")
    info(row(["variant", "eval loss", "probes", "mismatched", "median F1", "family F1"]))

    for {variant, result} <- results do
      info(
        row([
          variant.name,
          format(result.evaluation_loss, 4),
          format(result.probe_accuracy, 4),
          format(result.mismatch_accuracy, 4),
          format(result.median_f1, 2),
          format(result.median_family_f1, 2)
        ])
      )
    end
  end

  defp print_kind_table(results, bar) do
    info(
      "\nWhat kind is each neuron? (precision and recall at #{bar}; " <>
        "precise: precision 0.9 at recall 0.1 or more)"
    )

    info(row(["variant", "neurons", "clean", "coarse", "combined", "mixed", "dead", "precise"]))

    for {variant, result} <- results do
      kinds = result.kinds[bar]

      info(
        row([
          variant.name,
          Integer.to_string(result.neuron_count),
          percent(kinds.clean, result.neuron_count),
          percent(kinds.coarse, result.neuron_count),
          percent(kinds.combined, result.neuron_count),
          percent(kinds.mixed, result.neuron_count),
          percent(kinds.dead, result.neuron_count),
          percent(result.precise, result.neuron_count)
        ])
      )
    end
  end

  defp print_rubric_table(results) do
    keys = results |> hd() |> elem(1) |> Map.fetch!(:rubric) |> Map.keys() |> Enum.sort()

    info("\nClean neurons under other rubrics (fires above share of max / bar)")
    info(row(["variant" | Enum.map(keys, fn {threshold, bar} -> "#{threshold}/#{bar}" end)]))

    for {variant, result} <- results do
      info(row([variant.name | Enum.map(keys, &Integer.to_string(result.rubric[&1]))]))
    end
  end

  defp row([name | cells]),
    do: String.pad_trailing(name, 30) <> Enum.map_join(cells, &String.pad_leading(&1, 12))

  defp percent(count, total), do: "#{count} (#{format(100 * count / total, 0)}%)"

  defp slug(name), do: name |> String.downcase() |> String.replace(~r/[^a-z0-9]+/, "_")

  defp info(message), do: Mix.shell().info(message)

  defp format(number, decimals), do: :erlang.float_to_binary(number * 1.0, decimals: decimals)
end
