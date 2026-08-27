# Build brief — stages 1 through 7

The original specification for the whole project. Kept verbatim as the
authority on what each stage must contain and what "done" means for it.
Durable rules that outlive the build live in `CLAUDE.md`.

Stage status is tracked by what's in `lib/` and `test/`, not here.

Amended twice. First, before any implementation existed, seven surface
words were swapped for livelier ones (`cat/cats` to `llama/llamas`,
`bird/birds` to `goose/geese`, `eats/eat` to `ignores/ignore`,
`sleeps/sleep` to `flees/flee`, `happy` to `grumpy`, `old` to `sleepy`).

Second, at stage 2, the unused word `then` was replaced by an explicit
`<start>` marker, so sequences begin the way real models begin them
rather than by reusing the period. `<start>` is never emitted by the
grammar; whatever consumes sentences prepends it. The period keeps its
job as the terminator, so no second marker is needed.

Both amendments preserve the 32 word count and every rule. The
vocabulary is closed.

## Stage 1a. `TinyLlm.Vocab` (~30 lines)

Exactly these 32 words, in this order (ids 0–31):

    the a
    llama llamas dog dogs goose geese fox foxes mouse mice
    sees see chases chase ignores ignore flees flee
    is are
    big small hungry grumpy fast sleepy
    and who
    <start> .

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
row-normalize to sample. The 32×32 matrix is returned as plain lists and
feeds the talk's heatmap slide directly.
**Accept:** samples end `"."`; matrix rows sum to ~1.

## Stage 3. `TinyLlm.Embedder` + `TinyLlm.Train` (~150 lines)

The "neural bigram": embedding (32×32) → linear projection → softmax,
next-token from current token only. Cross-entropy loss. Backward:
`dlogits = softmax − one_hot`, chained through the projection.
Train harness (shared with stage 5 via a config struct): mini-batches from
`Grammar.corpus/1`, plain SGD, loss logged every N steps, seedable, loss
history returned.
**Accept:** gradient check passes; loss starts ≈ ln(32) ≈ 3.466 and drops
below 2.0 within seconds.

## Stage 4. `TinyLlm.Attention` (~150 lines) — the hard one

Single causal self-attention head over the full context: learned positional
embeddings added to token embeddings; Wq/Wk/Wv/Wo; scores = QKᵀ/√d; causal
mask = −1.0e9 on future positions pre-softmax; per-position attention
weights retrievable.

Forward AND hand-derived backward. **Write the derivation in
`docs/backprop.md` before implementing** — it doubles as a backup slide.
Key steps: softmax Jacobian applied to upstream grads, the QKᵀ split, and
accumulating position-embedding grads.
**Accept:** gradient check passes for every parameter matrix on a 4-token,
d=8 config.

**Measured once built**, on the default config (1000 steps, lr 0.5, batch
64, seed 1234): held-out loss 1.5764, well under the bigram floor of
1.9021. Four findings that later stages depend on.

  * **Beating the floor and resolving agreement are not the same
    milestone.** At 300 steps the loss is already 1.6742, past the floor,
    and the model still emits a plural verb after `the llama who chases the
    dogs` whatever the subject is. Agreement is close to the last thing it
    learns. Never read a loss number as evidence of agreement; measure
    agreement directly (stage 6a).
  * **It resolves agreement through the embedded verb, not the subject.**
    Predicting the main verb of `the llama who chases the dogs`, the final
    position attends 0.30 to `chases`, 0.21 to `who`, 0.19 to itself and
    only 0.11 to `llama`. `chases` already carries the subject's number, so
    reading it off there is sufficient, and that is what the model does.
  * **The head is largely positional.** The attention matrices for `the
    llama who chases the dogs` and `the dogs who chase the llama` agree to
    roughly 0.02 entry for entry, despite sharing no content word in the
    same slot. Number rides in the value vectors, not in the attention
    pattern. One head with no MLP has little else available to it.
  * **An attention sink appears unprompted.** The first verb position puts
    0.79 on `<start>`, having nothing useful to look back at. A documented
    production-transformer behavior, reproduced in 13K parameters.

## Stage 5. `TinyLlm.Block` + `TinyLlm.Model` (~150 lines)

MLP 32→128→32 with ReLU; residual connections around attention and MLP;
RMSNorm (`x / rms(x) * g`, learned gain). `Model.forward/2` composes
embed+pos → block → project(tied or separate — your call, document it) →
softmax. Config struct shared with stage 3 so both models use `Train`.
**Accept:** full-model gradient check (tiny config); held-out loss beats the
Embedder's; end-to-end training < 60s on laptop CPU.

**Measured once built.** All three criteria met, but the time budget is the
binding one and it has almost no slack.

  * **Cost is 445ms per step** at batch 64, against the stage 4 head's
    209ms. Not a pathology: the MLP is 8192 multiplies per token where the
    head's four projections are 4096, so roughly double is the expectation.
    The consequence is that 60 seconds buys about 130 steps at batch 64,
    and `Config`'s default of 1000 steps would take 7.5 minutes.
  * **The budget is better spent on smaller batches.** At a fixed budget of
    3840 examples, roughly 27s, batch 4 and batch 8 both reach 1.597 mean
    held-out loss where batch 16 reaches 1.627 and batch 64 reaches 1.899.
  * **Cosine decay beats a constant rate everywhere tested**, by 0.02 to
    0.04 nats, and roughly halves the spread between seeds. A constant rate
    keeps taking full-size steps after it has arrived, so where it stops
    depends on which step it stopped on. Now in `Train.learning_rate/2`.
  * **The rate scales with the batch size**, linearly up to batch 8 and
    sublinearly after: batch 4 wants 0.25, batch 8 wants 0.5, batch 16
    wants 0.5 to 0.75 rather than the 1.0 the rule predicts. Batch 4's
    optimum is bracketed on both sides, 1.668 at 0.0625 rising back to
    1.696 at 1.0.
  * **Recommended stage 5 config:** batch 8, `learning_rate: 0.5`,
    `learning_rate_schedule: :cosine`. Both tables above are 8 seeds per
    cell at a fixed compute budget.

**One seed is worth about ±0.05 nats on this model, so no two
configurations may be compared on one run each.** A first single-seed sweep
picked batch 4 at lr 1.0 as the best cell; over 8 seeds that is the *worst*
cell in its row, 1.696 against 1.597. It also appeared to show the optimal
rate falling as the batch grew, the reverse of the real relationship. Every
number quoted in the talk needs its seed count stated.

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
majority of generated sentences are both unseen and grammatical. Measure
(a) at the full step count, not at the point the loss curve flattens:
stage 4 showed agreement arriving several hundred steps after the loss has
already passed the bigram floor.

**Measured once built.** Model trained 480 steps at batch 8 with cosine
decay from 0.5; Embedder 800 steps; Bigram counted over the same 2,000
sentence corpus. 420 probes drawn from a fresh corpus and filtered against
the training set.

| | bigram | embedder | model |
| --- | --- | --- | --- |
| agreement, all probes | 66.7% | 71.7% | **91.4%** |
| held-out loss | | 1.9436 | **1.5842** |

1,000 generated sentences: 64.0% unseen, 90.6% grammatical, **54.6% both**.

**The aggregate averages three different problems, and one of them is free.
Quote the split, not the total.**

| what sits next to the blank | share | bigram | embedder | model |
| --- | --- | --- | --- | --- |
| the relative clause's own verb (intransitive) | 32.4% | 100.0% | 100.0% | 100.0% |
| an adjective (copula) | 29.8% | 44.8% | 61.6% | 100.0% |
| a distractor noun (transitive) | 37.9% | 55.3% | 55.3% | **77.4%** |

A third of the probes put the relative clause's verb immediately before the
blank, and that verb already agrees with the head, so one word of context
is enough and all three models score 100%. That block is the whole of the
bigram's 66.7%.

The case the talk is about is the last row. There the bigram and the
Embedder score **identically**, 55.3%, because they see the same single
word and give the same answer; the model reaches 77.4%. The defensible
claim is not "only the model can do agreement", it is "only the model does
better than chance when a distractor intervenes".

## Stage 7. `TinyLlm.PCA` + Livebook

Everything the Livebook plots is a plain Elixir term the model already
returns: embeddings, attention maps for probe sentences (must include
`the llama who chases the dogs`), per-step next-token distributions for a
sample generation, loss curves. There is no serialization step; the
notebook shares a BEAM with the model. PCA = power iteration on the
covariance matrix (~20 lines, stdlib, first two components).

`notebooks/attention_from_scratch.livemd` — the talk's demo vehicle, cells
in talk order: corpus gen → bigram heatmap → train with live loss chart →
embedding PCA scatter →
probe-sentence attention heatmap → temperature slider (Kino.Control) with
per-step probability bars.

**What the PCA scatter actually shows, measured.** This criterion used to
predict "POS clusters + parallel singular→plural offsets". Half of that is
right.

Class centroids and radii in the 2D projection:

| class | centroid | radius |
| --- | --- | --- |
| singular nouns | (−0.64, 0.06) | 0.18 |
| plural nouns | (−0.52, −0.27) | 0.29 |
| adjectives | (0.12, 0.43) | 0.25 |
| determiners | (0.82, 0.50) | 0.25 |
| singular verbs | (0.38, −0.06) | 0.97 |
| plural verbs | (0.30, −0.31) | 0.82 |

Nouns, adjectives and determiners cluster tightly and separate cleanly, and
the first component is essentially a noun detector: every noun sits at
x ≈ −0.6 and everything else at x > 0. Verbs do not cluster at all, with
radii larger than the distance to most other centroids.

The offsets claim does not survive. Pairwise cosines between distinct
singular→plural offsets, in all 32 dimensions, against 200 random pairs:

| | pairs | mean | min | max |
| --- | --- | --- | --- | --- |
| random baseline | 200 | 0.146 | | 0.423 |
| nouns | 10 | **0.372** | 0.211 | 0.630 |
| action verbs | 6 | 0.105 | −0.001 | 0.347 |
| all ten | 45 | 0.168 | −0.290 | 0.630 |

**Nouns share a plural direction, modestly; verbs do not.** Verb offsets sit
below the random baseline. Do not measure this as cosine against the mean
offset: every vector contributes to that mean, the bias is upward, and it
made the verbs look like weak signal rather than none.

**Tested, and it holds.** A word's row in `embeddings` is how it is read;
its column in `projection` is how it is predicted. The same pairwise
measurement on both:

| | embeddings | projection |
| --- | --- | --- |
| random baseline | 0.146 | 0.146 |
| nouns | 0.372 (0.21 to 0.63) | 0.414 (0.25 to 0.62) |
| action verbs | 0.105 (−0.00 to 0.35) | **0.487** (0.31 to 0.69) |

Verb number exists **only** in the output representation: no shared
direction whatsoever going in, a strong one coming out, with every pair
above 0.31. Noun number exists in both.

That is exactly what the grammar demands. A noun's number has to be
readable, since it governs the verb that follows, and writable, since the
model must choose `llama` over `llamas`. A verb's number only ever has to
be written: nothing downstream of a verb depends on it, so there is no
pressure to encode it on the way in, and the model did not.

It also justifies the untied unembedding after the fact. Stage 5 chose
separate tables so the PCA picture would mean one thing; it turns out the
two tables carry genuinely different information, and tying them would have
forced one set of vectors to be both "what I read" and "what I predict",
which for verbs are not the same.

**The temperature slider's range is 0 to 3**, measured. Grammaticality
falls 100% / 96.5% / 89.5% / 71.5% / 59% / 29.5% at temperatures 0, 0.5, 1,
1.5, 2, 3 and is already word salad past 3, so a 0 to 2 slider would show
almost nothing and a 0 to 10 one would waste two thirds of its travel.

The pairing to put on screen is grammaticality against distinctness, since
they trade off directly: at 0.0 the model is 100% grammatical and 0.5%
distinct, saying one sentence forever; at 1.0 it is 89.5% grammatical and
90.5% distinct. Temperature is the dial between correct-and-boring and
varied-and-wrong, and it is watchable.

Two details worth a sentence each. The model's single most likely sentence,
`the fast dogs and the llamas are fast .`, does not appear in the training
corpus. And the failure modes come apart in layers rather than at random:
`a llama is dog` keeps the syntax and loses the semantics, `the sleepy fast
chase the llamas` drops the noun, `sleepy fast goose dogs flee` loses the
determiner. The outer structure goes first.
**Accept:** Livebook runs top-to-bottom on a fresh machine with only
Livebook installed; the probe heatmap is legible and the verb position's
attention is visibly structured rather than flat.

Do **not** gate on the verb position attending to `llama` over `dogs`,
which is what this criterion asked for until stage 4 measured it. The
stage 4 model attends 0.11 to `llama` and 0.19 to `dogs` and still predicts
the singular verb correctly, because it reads the subject's number off the
embedded verb `chases` (0.30).

**Re-measured on the stage 5 model, and the criterion still must not gate
on it.** The mechanism changed and got no more legible. On `the llama who
chases the dogs`, the verb position now attends 0.55 to `who`, 0.17 to the
distractor `dogs` and 0.13 to the subject `llama`.

The `who` mass carries no number information whatsoever. This is one
attention layer, so the value at position 3 is built from that position's
own input, `embedding("who") + P[3]`, which is identical in both probes.
Over half the verb position's attention goes somewhere that cannot
disambiguate singular from plural.

**What the heatmap does show is the argument for depth**, and it is a
better slide than the original criterion would have been. Look at the `who`
row: it attends 0.57 to `llama`, and 0.65 to `dogs` in the mirror probe. The
`who` position has gathered the head noun into itself, so its *output* is a
summary of the subject. The model has built half of a two-hop route, head
noun into `who`, then `who` into the verb position, and with one block it
cannot use the second half: both hops run in parallel, not in sequence. A
second block would read the first block's output at `who` and get the
subject for free. That is what depth buys, drawn by the model itself.

The attention sink from stage 4 survives: the first verb position puts 0.96
on `<start>`.

## Definition of done

`mix test` green including every gradient check; a `mix run` demo script
prints 10 seeded sentences and the eval table (agreement accuracy ×3,
% unseen, % grammatical); Livebook verified.
When in doubt about scope: smaller and clearer wins — this codebase's job
is to fit in someone's head.
