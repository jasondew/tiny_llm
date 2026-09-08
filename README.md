# the llama who chases the dogs

**Attention from scratch, in Elixir.**

    the llama who chases the dogs ____

Flees or flee? You know instantly. Saying *how* you know takes a paragraph:
the word that decides it is five back, and there is a plural noun sitting
right next to the blank pointing the wrong way.

This repo builds, from nothing, a program that draws the bracket from the
blank back to `llama`. It is a complete decoder-only transformer language
model in the **pure Elixir standard library**, and it is the companion
artifact to a 60 minute conference talk.

## From nothing means from nothing

* No Nx, no Axon, no hex packages. `mix.exs` deps are empty.
* No autodiff. Every gradient is derived analytically in
  [`docs/backprop.md`](docs/backprop.md) and written out by hand.
* No tokenizer. One word is one token is one integer.
* No GPU, no binaries, no NIFs. Matrices are lists of lists of floats.

Every hand-written backward pass has a finite-difference gradient check
next to it, because a plausible-looking wrong gradient is the most likely
way a project like this fails quietly.

## The model

| | |
| --- | --- |
| vocabulary | 32 words |
| context length | 16 tokens |
| d_model | 32 |
| attention heads | 1 |
| transformer blocks | 1 |
| parameters | 15,104 |
| training time | about 30 seconds on a laptop CPU |

Training data comes from a probabilistic grammar we wrote, so the ground
truth is knowable. When the model gets subject-verb agreement right across
a distractor, we know it is not luck, because we know the rule that
generated the sentence.

## Run it

```
mix run scripts/demo.exs
```

Trains the model from scratch, prints ten sentences it has never seen, and
then the three measurements that say whether it learned language or
memorized the corpus. Everything reproduces from the seeds in the script.

For the visual version, with attention heatmaps, loss curves, a PCA scatter
of the embedding table and a temperature slider:

```
livebook server notebooks/attention_from_scratch.livemd
```

The notebook is the talk in talk order. Kino and VegaLite appear there for
presentation only and never touch the math.

```
mix test
```

264 tests, including a gradient check on every backward pass.

## The arc

Four models in a row, each one answering the question the previous one
raised.

1. **`Bigram`** counts pairs. It works until the answer is more than one
   word back.
2. **`Embedder`** is the same bigram, learned instead of counted. It shows
   that learning does not help when the *context* is the problem.
3. **`Attention`** is a single causal head. Each position looks back at
   every earlier position, scores them, and pulls in what it needs.
4. **`Transformer`** wraps that head in residuals, RMSNorm and an MLP,
   which is where the model gets a place to think.

## The numbers

Agreement accuracy on 420 held-out probes, where the verb is masked and the
model has to pick the right form:

| what sits next to the blank | share | bigram | embedder | model |
| --- | --- | --- | --- | --- |
| the relative clause's own verb | 32.4% | 100.0% | 100.0% | 100.0% |
| an adjective | 29.8% | 44.8% | 61.6% | 100.0% |
| **a distractor noun** | 37.9% | 55.3% | 55.3% | **77.4%** |
| all probes | | 66.7% | 71.7% | **91.4%** |

The aggregate averages three different problems and one of them is free, so
the third row is the one that matters. It is the `the llama who chases the
dogs ____` case. The bigram and the embedder score identically there,
55.3% to the decimal, because they see the same single word and give the
same answer.

Held-out loss, in nats per token:

| uniform guess | bigram floor | embedder | model |
| --- | --- | --- | --- |
| 3.4657 | 1.9021 | 1.9436 | **1.5842** |

1.9021 is the conditional entropy of this grammar given one word of
context. No model that sees a single previous word can go below it, however
large. Crossing it is the point.

Of 1,000 generated sentences: 90.6% grammatical, 64.0% never seen in
training, 54.6% both.

## What is not here, and what is

Not here: autodiff, a tokenizer, a GPU, multiple heads, depth,
dependencies.

Here: embeddings, learned positional embeddings, scaled dot-product
attention, a causal mask, residual connections, RMSNorm, an MLP,
cross-entropy loss, hand-written backprop with a finite-difference check,
and temperature sampling.

Every one of those is the same thing a frontier model does. Scale is the
difference. Structure is not.

## Layout

```
lib/tiny_llm/     the model, thirteen modules, no dependencies
test/             one test file per module, gradient checks included
docs/
  build-brief.md  the original spec, and what each stage measured
  backprop.md     every gradient, derived
  talk-outline.md the design for the slide deck
notebooks/        the Livebook
scripts/demo.exs  the closing slide, as a script
```

Start with `lib/tiny_llm/attention.ex`. That is the one the talk is about,
and its moduledoc carries the derivation.
