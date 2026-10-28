# CLAUDE.md — interpretability

## What this is

The code behind the second talk, **"where the llama keeps its grammar"**:
finding monosemantic features in a tiny transformer. It trains a model
like the first talk's on a larger grammar, then opens it up: raw MLP
neurons, a toy model of superposition, a sparse autoencoder (SAE), feature
scoring against the grammar's ground truth labels, and steering.

- Spec: `docs/superpowers/specs/2026-10-26-where-the-llama-keeps-its-grammar-design.md`
  in the `interpretability_talk` repo.
- Plans: `docs/superpowers/plans/` in the same repo.
- Builder decisions and their reasons: `docs/decisions.md`.
- Speaker predictions against results: `docs/findings.md`.
- Answers for the Q&A: `docs/hard-questions.md`.

## Relaxed from the parent project

- **Dependencies are allowed.** Nx and Polaris for the math, EXLA in
  `:dev` and `:test` only. The parent `mix.exs` stays at zero deps.
- **Autodiff is fine.** The first talk's point was hand written backprop.
  This talk's point is reading what the model learned.
- **Tensors, not lists of lists**, wherever Nx is the natural tool.

## Still binding

1. **Reproducibility.** All randomness flows from explicit seeds. Every
   number destined for a slide reproduces from a seed: bitwise on
   `Nx.BinaryBackend`, and to the precision shown on slides on EXLA.
2. **Moduledocs are talk material.** Every module's `@moduledoc` explains
   the concept in plain words for software engineers without a statistics
   background.
3. **Tests mirror `lib/`.** `test/interpretability/grammar_test.exs`
   covers `lib/interpretability/grammar.ex`.
4. **Clarity over performance.** Training should take a couple of minutes
   at most on a laptop CPU; if it is slower, shrink the workload.
5. **The model's size is fixed**: `d_model` 32, MLP width 128, one block,
   one head. It is not tuned to manufacture a result.
6. **Never change the parent project's files**, apart from the pointer in
   its README and `CLAUDE.md`.
7. **Library code never names EXLA at compile time.** Use
   `Interpretability.Compiler.options/0`.
8. **Results are reported honestly.** If features are not clean, say so.
