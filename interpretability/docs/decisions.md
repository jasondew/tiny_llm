# Builder decisions

Each decision the spec left to the builder, with the reason.

## Nx.Defn and Polaris instead of Axon

Steering and SAE splicing both need to replace the MLP's hidden
activations partway through the forward pass. With a hand written
`Nx.Defn` forward pass that is one function argument. In Axon it is a
hook on a layer inside a compiled graph. Polaris supplies the Adam
optimizer that Axon would otherwise have provided.

## The toy model trains with EXLA, and the deck shows a precomputed sweep

Without EXLA, one full toy run (3,000 steps of 1,024 examples) takes about
160 seconds on the M1 Max. With EXLA it takes well under a second. That is
too slow for a live slider in the deck, so, as the spec allows, the deck
will scrub through `priv/results/toy_sweep.etf` instead of training live.

## A 70 word vocabulary

The spec expected 80 to 150 words. 70 is enough to carry every concept
the grammar adds (animacy, tense, pronouns, prepositional distractors,
relative clauses on objects). More nouns or adjectives would add word
identities without adding concepts, which the spec rules out as growth
for its own sake.

## Grammar probabilities

Subject: simple 50, compound 12, relative clause 18, prepositional 12,
pronoun 8. Object: simple 80, relative clause 20 (until two clauses
deep). Subjects animate 80%, objects 70%. Tense present 70%, past 30%.
Simple sentences stay the majority so the model learns the basics first;
every distractor construction still appears in roughly one sentence in
eight or more. With seed 11, 10,000 sentences average 7.4 tokens, 249
have two relative clauses, and only 7 reach the 20 token limit, so
resampling long sentences barely skews the rates.

## Context length 20

The first talk used 16. Nested relative clauses do not fit in 16 tokens.
This changes the positional table only; `d_model` stays 32.

## Agreement labels use left context only

`:agreement_number` is the number the next verb would need given only the
words read so far. A label that knew the rest of the sentence would let a
feature score well for information the model cannot have, which would
inflate every precision and recall built on it.

## Connective words carry no role

`who`, `that`, `and`, `near` and `behind` get `role: nil`. Role belongs to
the words that make up a noun: determiners, adjectives, nouns, pronouns.

## Training run: 8,000 steps on 60,000 sentences

The plan's starting point (4,000 steps, 20,000 sentences) missed the gate:
the cold open case scored 0.933 on 448 probes. More steps alone helped
and then hurt (0.953 at 8,000, 0.935 at 16,000), and evaluation loss
rose with them (1.986, 1.994, 2.013): the model was memorising a corpus
it had seen too often. More sentences fixed it:

| Steps | Corpus | Evaluation loss | Cold open case |
| --- | --- | --- | --- |
| 4,000 | 20,000 | 1.9857 | 0.9330 |
| 8,000 | 20,000 | 1.9941 | 0.9531 |
| 16,000 | 20,000 | 2.0126 | 0.9353 |
| 8,000 | 60,000 | 1.9635 | 0.9911 |
| 16,000 | 100,000 | 1.9667 | 0.9777 |

These evaluation losses were measured before the evaluation set was
deduplicated against training (38.6% overlap); the held out loss of the
chosen run is 2.0096. The comparison between runs stands.

8,000 steps on 60,000 sentences trains in about 36 seconds on the M1 Max
CPU with EXLA. `d_model` and the MLP width were not touched. Probe
accuracy is measured on a separate 20,000 sentence corpus (seed + 1),
because the cold open case appears in only about one sentence in forty.

## One firing rule for neurons and features

A unit fires on a token when its activation exceeds 20% of that unit's
maximum over the dataset. Neurons and SAE features use the same rule. A
raw ReLU neuron is above zero on many tokens, often only slightly, so
"above zero" would make neurons look messier than they are and rig the
comparison in the SAE's favour. Monosemantic means precision and recall
both at least 0.9 for the unit's best label (highest F1).

## A tenth label: can_end

Added after the first neuron report showed neuron 64 firing on the last
word of finished sentences. True when no clause is still waiting for its
verb and the word finishes a phrase: a noun, a pronoun, an intransitive
verb, or a copula's adjective. Left context only, like every other label.

## TopK sparsity, not an L1 penalty

The SAE keeps each word's 8 strongest features and zeroes the rest. The
sparsity is then a number that fits on a slide ("at most 8 of 512 on at
once"), with no penalty strength to tune and no shrinking of the
features that stay on. The cost: "Towards Monosemanticity" (2023) used an
L1 penalty, so this differs from the paper the talk cites, and
`hard-questions.md` says so. The SAE trains on the activations of the
first 20,000 training sentences.

## Held out evaluation, no <start> rows, informative labels only

Evaluation and probe sentences skip any sentence that also appears in
training. Activations start at position 1, because `<start>` is the same
vector in every sentence. The neuron against feature comparison leaves
out labels true for more than half of all words, which any always on unit
wins and no sparse feature can.
