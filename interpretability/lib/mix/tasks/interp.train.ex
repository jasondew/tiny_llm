defmodule Mix.Tasks.Interp.Train do
  @shortdoc "Trains the llama model and prints its agreement probes"

  @moduledoc """
  Trains the llama model on the extended grammar and saves it.

      mix interp.train
      mix interp.train --steps 8000 --corpus 40000

  The training, evaluation and probe sentences all come from seeded
  grammar runs, so the same seed retrains the same model. Probe accuracy
  is measured on a separate 20,000 sentence corpus, because the cold open
  case is rare: about one sentence in forty. Probe sentences that also
  appear in training are dropped, so every probe is new to the model.

  Writes `priv/checkpoints/model.etf` as `%{config, params, history}`.
  """

  use Mix.Task

  alias Interpretability.Checkpoint
  alias Interpretability.Grammar
  alias Interpretability.Model
  alias Interpretability.Probes

  @path "priv/checkpoints/model.etf"
  @probe_corpus_size 20_000
  @cold_open ~w(the llama who chases the dogs)

  @impl Mix.Task
  def run(arguments) do
    Mix.Task.run("app.start")

    {switches, _rest} =
      OptionParser.parse!(arguments, strict: [steps: :integer, corpus: :integer])

    defaults = %Model.Config{}

    config = %{
      defaults
      | steps: Keyword.get(switches, :steps, defaults.steps),
        training_corpus_size: Keyword.get(switches, :corpus, defaults.training_corpus_size)
    }

    {training, evaluation} = Model.datasets(config)
    seen = Model.training_sentences(config)
    Grammar.seed(config.seed + 1)

    probes =
      @probe_corpus_size
      |> Grammar.trees()
      |> Enum.reject(&MapSet.member?(seen, Grammar.words(&1)))
      |> Enum.flat_map(&Probes.from_tree/1)

    {microseconds, {params, history}} = :timer.tc(fn -> Model.train(config, training) end)
    Checkpoint.save(%{config: config, params: params, history: history}, @path)

    info("Trained #{config.steps} steps in #{format(microseconds / 1.0e6, 1)} seconds")
    info("Loss curve: " <> Enum.map_join(Enum.take_every(history, 5), "  ", &format(&1.loss, 3)))

    info(
      "Evaluation loss: #{format(Nx.to_number(Model.evaluate_loss(params, evaluation)), 4)} nats"
    )

    predictor = Model.predictor(params)
    print_probe_table(Probes.accuracy(predictor, probes), probes)
    print_cold_open(predictor)
    info("Wrote #{@path}")
  end

  ## PRIVATE FUNCTIONS

  defp print_probe_table(accuracy, probes) do
    counts = Enum.frequencies_by(probes, &{&1.category, &1.mismatch})

    info("\ncategory                   nearest noun differs?  probes  accuracy")

    for {{category, mismatch} = group, value} <- Enum.sort(accuracy) do
      info(
        String.pad_trailing(to_string(category), 27) <>
          String.pad_trailing(if(mismatch, do: "yes", else: "no"), 23) <>
          String.pad_leading(Integer.to_string(counts[group]), 6) <>
          "  " <> format(value, 4)
      )
    end
  end

  defp print_cold_open(predictor) do
    scores = predictor.(@cold_open)
    flees = Enum.at(scores, Interpretability.Vocab.word_to_id("flees"))
    flee = Enum.at(scores, Interpretability.Vocab.word_to_id("flee"))

    info(
      "\n#{Enum.join(@cold_open, " ")} ... flees #{format(flees, 3)} vs flee #{format(flee, 3)}: " <>
        if(flees > flee, do: "flees wins", else: "FLEE WINS")
    )
  end

  defp info(message), do: Mix.shell().info(message)

  defp format(number, decimals), do: :erlang.float_to_binary(number * 1.0, decimals: decimals)
end
