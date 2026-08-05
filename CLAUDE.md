# CLAUDE.md — tiny_llm

## What this is

A complete decoder-only transformer language model in **pure Elixir stdlib**,
built as the companion artifact to a 30-minute conference talk:
**"Attention, From Scratch — a transformer in pure Elixir, no libraries."**
Audience: Elixir engineers curious about AI. Every module may end up on a
slide and the repo link appears on every slide, so **the code IS the talk**.

The model: 32-word vocabulary, context length 16, d_model 32, one attention
head, one transformer block, ~13K parameters. Trained on sentences generated
by a probabilistic grammar we own, so ground truth is knowable. Training
must complete in seconds-to-a-minute on a laptop CPU.

The stage-by-stage specification lives in `docs/build-brief.md`. Read it
before starting any stage; it defines the acceptance criteria. Modules that
already exist are authoritative: verify them against the brief, and don't
refactor them except to add functions.

## Hard constraints — do not violate

1. **Zero dependencies.** No Nx, no Axon, no hex packages in model code.
   `mix.exs` deps stay empty. The point of the project is that nothing is
   hidden. (Sole exception: the stage-7 Livebook may use Kino/VegaLite for
   *presentation* — it visualizes JSON the model exports, never the math.)
2. **Hand-written backprop.** No autodiff, no numerical-only training.
   Gradients are derived analytically and implemented explicitly.
3. **Clarity beats performance.** Matrices are lists of lists of floats.
   No binaries, NIFs, ETS, or process-parallelism for speed. If something
   is slow, shrink the workload, not the abstraction.
4. **Reproducibility.** All randomness through `:rand` with explicit seed
   functions (`Module.seed/1` pattern, `:exsss` algorithm). Every number
   destined for a slide must reproduce from a seed.
5. **Fixed design decisions.** The vocabulary, grammar, and architecture
   sizes in the brief are final — they interlock with the talk. Do not
   "improve" them.

## Conventions

- Params are maps of named matrices, e.g. `%{e: E, wq: Wq, ...}`. Training
  is `Enum.reduce(batches, params, &step/2)` — params in, params out.
- Checkpoints: `:erlang.term_to_binary/1` → `priv/checkpoints/`.
  Visual exports: hand-rolled JSON encoder (~30 lines, in `TinyLlm.Export`;
  no JSON dep) → `priv/exports/`.
- Every module gets a `@moduledoc` explaining the *concept*, not just the
  API — moduledocs are talk material.
- Tests in `test/dayN_test.exs` matching stages. `mix test` stays green at
  every stage boundary.
- **Gradient checking is mandatory.** Every hand-written backward pass gets
  a finite-difference test: `(f(θ+ε) − f(θ−ε)) / 2ε`, ε ≈ 1.0e-4, relative
  error < 1.0e-3, run on a tiny config. Implement once in `TinyLlm.GradCheck`
  (perturb each param entry, compare to analytic gradient). This is the
  safety net for constraint 2; a plausible-looking wrong gradient is the
  most likely failure mode of this entire project.

When in doubt about scope: smaller and clearer wins. This codebase's job is
to fit in someone's head.
