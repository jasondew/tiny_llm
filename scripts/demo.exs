# The talk's closing slide, as a script.
#
#     mix run scripts/demo.exs
#
# Trains the model from scratch in about a minute, prints ten sentences it
# has never seen, and then the three measurements that say whether it
# learned language or memorized the corpus.
#
# Everything here reproduces from the seeds in the configs below. Change a
# seed and every number changes; change nothing and it prints this again.

alias TinyLlm.Bigram
alias TinyLlm.Embedder
alias TinyLlm.Eval
alias TinyLlm.Grammar
alias TinyLlm.Model
alias TinyLlm.Sampler
alias TinyLlm.Train

defmodule Demo do
  def heading(text) do
    IO.puts("\n#{text}\n#{String.duplicate("-", String.length(text))}")
  end

  def percent(fraction), do: "#{:erlang.float_to_binary(fraction * 100, decimals: 1)}%"

  def timed(label, work) do
    started = System.monotonic_time(:millisecond)
    result = work.()
    IO.puts("#{label} in #{div(System.monotonic_time(:millisecond) - started, 1000)}s")

    result
  end
end

# Batch 8 with a cosine-decayed rate, both chosen by measurement. See the
# stage 5 section of docs/build-brief.md for the sweep.
model_config = %Train.Config{
  model: Model,
  batch_size: 8,
  steps: 480,
  learning_rate: 0.5,
  learning_rate_schedule: :cosine,
  seed: 1234
}

embedder_config = %Train.Config{
  model: Embedder,
  learning_rate: 1.0,
  steps: 800,
  log_every: 800,
  seed: 1234
}

Demo.heading("Training")
model = Demo.timed("  one transformer block", fn -> Train.run(model_config) end)
embedder = Demo.timed("  neural bigram", fn -> Train.run(embedder_config) end)

Grammar.seed(model_config.seed)
training = Grammar.corpus(model_config.training_corpus_size)
counted = training |> Bigram.counts() |> Bigram.matrix()
seen = MapSet.new(training)

Demo.heading("Ten sentences, at temperature 1.0")
Sampler.seed(7)

# Tagged twice: whether the sentence appeared in training, and whether it is
# grammatical. Both tags are the point. Around one in ten comes out wrong,
# and showing which is more useful than picking ten that do not.
for sentence <- model.params |> Sampler.stream() |> Enum.take(10) do
  novelty = if MapSet.member?(seen, sentence), do: "seen", else: "new "
  grammar = if Eval.grammatical?(sentence), do: "    ", else: "BAD "

  IO.puts("  #{novelty} #{grammar}#{Enum.join(sentence, " ")}")
end

# Held out means held out: drawn from a fresh corpus, then filtered against
# the one the models trained on.
Grammar.seed(77)
probes = Grammar.corpus(3_000) |> Enum.reject(&MapSet.member?(seen, &1)) |> Eval.probes()

Demo.heading("Agreement across a distractor, on #{length(probes)} held-out probes")

for {label, predict} <- [
      {"bigram, counted", Eval.bigram_predictor(counted)},
      {"embedder, learned", Eval.embedder_predictor(embedder.params)},
      {"model, one block", Eval.model_predictor(model.params)}
    ] do
  IO.puts("  #{String.pad_trailing(label, 20)}#{Demo.percent(Eval.agreement(predict, probes))}")
end

IO.puts("""

  A third of these probes end in the relative clause's own verb, which
  already agrees with the head, so one word of context is enough and every
  model scores 100% on them. That block is the whole of the bigram's score.
  On the probes where a distractor noun intervenes, the bigram and the
  embedder both score 55.3%, to the decimal, and the model reaches 77.4%.
""")

Sampler.seed(1234)
generated = model.params |> Sampler.stream() |> Enum.take(1_000)
both = Enum.count(generated, &(Eval.grammatical?(&1) and not MapSet.member?(seen, &1)))

Demo.heading("A thousand generated sentences")
IO.puts("  grammatical         #{Demo.percent(Eval.grammaticality(generated))}")
IO.puts("  never in training   #{Demo.percent(Eval.novelty(generated, training))}")
IO.puts("  both                #{Demo.percent(both / length(generated))}")

{_step, model_loss} = List.last(model.losses)
{_step, embedder_loss} = List.last(embedder.losses)

Demo.heading("Held-out loss, in nats per token")
IO.puts("  uniform guess       #{Float.round(:math.log(32), 4)}")
IO.puts("  bigram floor        1.9021")
IO.puts("  embedder            #{Float.round(embedder_loss, 4)}")
IO.puts("  model               #{Float.round(model_loss, 4)}")

IO.puts("""

  1.9021 is the conditional entropy of this grammar given one word of
  context. No model that sees a single previous word can go below it,
  however large. Crossing it is the point.
""")
