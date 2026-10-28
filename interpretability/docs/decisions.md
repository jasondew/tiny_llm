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
