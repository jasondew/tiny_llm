# Build brief — stages 1 through 7

The original specification for the whole project. Kept verbatim as the
authority on what each stage must contain and what "done" means for it.
Durable rules that outlive the build live in `CLAUDE.md`.

Stage status is tracked by what's in `lib/` and `test/`, not here.

Amended once, before any implementation existed: seven surface words were
swapped for livelier ones (`cat/cats` to `llama/llamas`, `bird/birds` to
`goose/geese`, `eats/eat` to `ignores/ignore`, `sleeps/sleep` to
`flees/flee`, `happy` to `grumpy`, `old` to `sleepy`). Counts, word
classes, and every rule are unchanged. The vocabulary is closed again now
that training is about to start.

## Stage 1a. `TinyLlm.Vocab` (~30 lines)

Exactly these 32 words, in this order (ids 0–31):

    the a
    llama llamas dog dogs goose geese fox foxes mouse mice
    sees see chases chase ignores ignore flees flee
    is are
    big small hungry grumpy fast sleepy
    and then who
    .

One word = one token = one integer; there is no tokenizer anywhere.
`word_to_id/1` / `id_to_word/1` as compile-time generated function clauses;
`encode/1`, `decode/1`, `size/0`, `words/0`. Compile-time assert size == 32.
**Accept:** round-trip test over all 32 words.

## Stage 1b. `TinyLlm.Grammar` (~90 lines)

Probabilistic context-free corpus generator:

    Sentence        -> NounPhrase VerbPhrase "."

    NounPhrase      -> Determiner AdjectivePhrase Noun (70%)
                     | NounPhrase "and" NounPhrase (15%)
                           (subject position only; result :plural)
                     | Determiner AdjectivePhrase Noun "who" VerbPhrase (15%)
                           (subject position only; verb agrees with head noun)

    AdjectivePhrase -> nothing (55%)
                     | Adjective (30%)
                     | Adjective Adjective (15%)

    VerbPhrase      -> IntransitiveVerb (35%)
                     | TransitiveVerb NounPhrase (35%)
                     | Copula Adjective (30%)

Word classes: nouns singular `llama dog goose fox mouse` / plural `llamas
dogs geese foxes mice`; verbs singular `sees chases ignores flees` /
plural `see chase ignore flee`;
transitive only `sees/chases/ignores` + plural forms (never "flees" followed
by an object noun phrase);
copula is/are; adjectives `big small hungry grumpy fast sleepy`.

Rules enforced at sampling time, so every sentence is grammatical by
construction:
- a `:singular | :plural` number flag threads from the subject noun phrase
  into its verb phrase
- a relative-clause verb agrees with the **head** noun (the distractor-object
  case `the llama who chases the dogs flees` is the talk's centerpiece)
- compound subjects (`X and Y`) are `:plural`
- `"a"` only before singular nouns (`"the"` works for both)
- compound and relative noun phrases only in subject position (depth 0);
  object noun phrases are always simple
- sentences ≤ 16 tokens (resample stragglers), always ending `"."`
- relative-clause probability 15%, compound 15% at subject position

API: `seed/1`, `sentence/0`, `corpus/1`.
**Accept:** over 10k seeded sentences — all tokens in vocab, all end `"."`,
all ≤ 16 tokens; a structural checker confirms agreement in simple clauses
and that `"a"` never precedes a plural; `who` and `and` both occur; ~70%
of 5k sentences are unique.

## Stage 1c. `TinyLlm.Tensor` (~80 lines)

The entire math library, matrices as lists of rows of floats:
`zeros/2`, `random/3` (uniform ±scale, default 0.02), `one_hot/2`,
`shape/1`, `transpose/1`, `add/2`, `sub/2`, `hadamard/2`, `scale/2`,
`map/2`, `dot/2`, `matmul/2`, row-wise `softmax/1` (max-subtracted for
stability), `argmax/1`.
**Accept:** matmul/transpose/elementwise vs hand-computed values; softmax
rows sum to 1 and survive logits of 1000.

## Stage 2. `TinyLlm.Bigram` (~40 lines)

Count-based bigram: `%{ {prev_id, next_id} => count }` built from a corpus;
row-normalize to sample. Export the 32×32 count matrix (JSON) for the talk's
heatmap slide.
**Accept:** samples end `"."`; exported rows sum to ~1.

## Stage 3. `TinyLlm.Embedder` + `TinyLlm.Train` (~150 lines)

The "neural bigram": embedding (32×32) → linear projection → softmax,
next-token from current token only. Cross-entropy loss. Backward:
`dlogits = softmax − one_hot`, chained through the projection.
Train harness (shared with stage 5 via a config struct): mini-batches from
`Grammar.corpus/1`, plain SGD, loss logged every N steps, seedable, loss
history exported.
**Accept:** gradient check passes; loss starts ≈ ln(32) ≈ 3.466 and drops
below 2.0 within seconds; embedding matrix exported.

## Stage 4. `TinyLlm.Attention` (~150 lines) — the hard one

Single causal self-attention head over the full context: learned positional
embeddings added to token embeddings; Wq/Wk/Wv/Wo; scores = QKᵀ/√d; causal
mask = −1.0e9 on future positions pre-softmax; per-position attention
weights retrievable for export.

Forward AND hand-derived backward. **Write the derivation in
`docs/backprop.md` before implementing** — it doubles as a backup slide.
Key steps: softmax Jacobian applied to upstream grads, the QKᵀ split, and
accumulating position-embedding grads.
**Accept:** gradient check passes for every parameter matrix on a 4-token,
d=8 config.

## Stage 5. `TinyLlm.Block` + `TinyLlm.Model` (~150 lines)

MLP 32→128→32 with ReLU; residual connections around attention and MLP;
RMSNorm (`x / rms(x) * g`, learned gain). `Model.forward/2` composes
embed+pos → block → project(tied or separate — your call, document it) →
softmax. Config struct shared with stage 3 so both models use `Train`.
**Accept:** full-model gradient check (tiny config); held-out loss beats the
Embedder's; end-to-end training < 60s on laptop CPU.

## Stage 6. `TinyLlm.Sampler` + `TinyLlm.Eval` (~130 lines)

Sampler: `Stream.unfold/2` autoregressive loop; temperature divides logits
pre-softmax; T=0 means argmax; stop at `"."` or max length.

Eval — three checks that produce the talk's numbers:
  a. **Agreement accuracy**: held-out `who`-clause sentences, mask the verb,
     argmax vs correct form — for Bigram, Embedder, and Model (the contrast
     slide). Ensure held-out means held out: filter against training set.
  b. **Memorization**: % of 1000 generated sentences NOT in the training
     set (MapSet lookup).
  c. **Grammaticality**: structural checker (extend the stage-1 one) over
     generated sentences.
**Accept:** Model beats Embedder and Bigram on (a) by a wide margin;
majority of generated sentences are both unseen and grammatical.

## Stage 7. `TinyLlm.Export` + `TinyLlm.PCA` + Livebook

Exports (JSON): embeddings, attention maps for probe sentences (must
include `the llama who chases the dogs`), per-step next-token distributions
for a sample generation, loss curves. PCA = power iteration on the
covariance matrix (~20 lines, stdlib, first two components).

`notebooks/attention_from_scratch.livemd` — the talk's demo vehicle, cells
in talk order: corpus gen → bigram heatmap → train with live loss chart →
embedding PCA scatter (expect POS clusters + parallel singular→plural
offsets) →
probe-sentence attention heatmap → temperature slider (Kino.Control) with
per-step probability bars.
**Accept:** Livebook runs top-to-bottom on a fresh machine with only
Livebook installed; the probe heatmap shows the verb position attending to
the subject noun (`llama`), not the distractor (`dogs`).

## Definition of done

`mix test` green including every gradient check; a `mix run` demo script
prints 10 seeded sentences and the eval table (agreement accuracy ×3,
% unseen, % grammatical); all exports written; Livebook verified.
When in doubt about scope: smaller and clearer wins — this codebase's job
is to fit in someone's head.
