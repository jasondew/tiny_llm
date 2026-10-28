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
