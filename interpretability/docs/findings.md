# Findings

Each experiment, the speaker's prediction before it ran, what actually
happened, and the explanation in words that could be said on stage.

## Toy model of superposition (2026-10-28)

**Prediction.** At 0% sparsity the model keeps all 5 features. At 95% it
still keeps all 5. The kept arrows are mostly orthogonal.

**Result** (`mix interp.toy`, seed 1234, 3.0 seconds on EXLA):

| Sparsity | Features kept | Loss |
| --- | --- | --- |
| 0.00 | 2 | 0.1329 |
| 0.50 | 3 | 0.1619 |
| 0.80 | 4 | 0.0504 |
| 0.90 | 5 | 0.0175 |
| 0.95 | 5 | 0.0069 |
| 0.99 | 5 | 0.0009 |

At 0% sparsity the two most important features sit at right angles (95°
and 185°). The three dropped features get a bias of about 0.5: the model
cannot store them, so it guesses their average every time. At 95% all
five arrows sit about 72° apart (28°, 102°, 177°, 248°, 317°), which is a
regular pentagon, and every bias is about −0.23.

**Explanation for the stage.** When every feature is always on, any
overlap between arrows causes errors on every input, so the model keeps
only as many features as it has dimensions, at right angles, and guesses
the rest. When features are rarely on together, overlap almost never
costs anything, so the model fits all five in, spread as evenly as
possible. The negative bias is the trick that makes it work: when one
feature is on, its neighbours read a small false signal, and the bias
plus the clip at zero erases it. The model trades a little accuracy on
rare collisions for storing more than it has room for. That trade is
superposition.

**Where the prediction missed.** "All five, always" is right only for
sparse features. "Mostly orthogonal" cannot happen in two dimensions with
more than two arrows; the pentagon is the next best thing, as far apart
as five arrows can get. A quirk worth knowing: at 0.8 the model kept
features 0, 1, 2 and 4 but dropped 3, so a less important feature
survived over a more important one. Training finds a good arrangement,
not always the best one.

## Labels (2026-10-28)

The speaker checked the key idea: in `the llama who chases the dogs`, the
next verb still needs singular at `dogs`, because `llama` is the subject
still waiting for its verb. That gap is the distractor case.

In 2,000 generated sentences, the cold open case (a relative clause whose
object has the other number, followed by a present tense main verb)
appears only 48 times, so probe accuracy is measured on a larger corpus.

## Agreement probes on the trained model (2026-10-28)

**Prediction.** The compound subject is weakest, because the model has to
track both nouns, or perhaps just notice the `and`.

**Result** (`mix interp.train`, seed 1234, 8,000 steps on 60,000 sentences,
probes from 20,000 separate sentences):

| Category | Nearest noun differs? | Probes | Accuracy |
| --- | --- | --- | --- |
| simple subject | no | 6,998 | 1.0000 |
| compound subject | no | 813 | 1.0000 |
| compound subject | yes | 860 | 1.0000 |
| pronoun subject | no | 1,111 | 1.0000 |
| prepositional distractor | no | 842 | 1.0000 |
| prepositional distractor | yes | 863 | 1.0000 |
| inside relative clause | no | 2,546 | 1.0000 |
| relative clause on the object | no | 1,124 | 0.9991 |
| relative clause distractor | no | 2,097 | 0.9986 |
| relative clause distractor | yes | 448 | **0.9911** |

The cold open case is the weakest. On the first training run (4,000 steps,
20,000 sentences) it was the only category that failed the 0.95 gate, at
0.933.

**Explanation for the stage.** The compound subject is perfect because
`and` alone gives it away: in this grammar `and` only ever joins two
subjects, and a joined subject is always plural, so "seen an `and`?
then plural" is a one word rule. The relative clause distractor is
hardest because nothing so simple works. Between the subject and its verb
sits a whole clause with its own verb and a noun of the opposite number,
and the model has to look past all of it to the noun that started the
sentence. The prepositional distractor looks similar but is easier,
because `near` and `behind` are followed by exactly one short noun
phrase, while a relative clause can be long and can nest.

**Where the prediction missed.** The instinct about `and` was right; it
just makes the compound subject the easiest case, not the hardest.

## How clean are the raw neurons? (2026-10-29)

**Prediction.** 40 of the 128 neurons are monosemantic, mostly matching
individual words.

**Result** (`mix interp.neurons`, 14,713 words from the 2,000 evaluation
sentences): **0 of 128 neurons are monosemantic.** None never fires; the
median neuron fires on 15% of words. Best F1 per neuron:

| Best F1 | Neurons |
| --- | --- |
| 0.1 to 0.2 | 1 |
| 0.2 to 0.3 | 7 |
| 0.3 to 0.4 | 13 |
| 0.4 to 0.5 | 25 |
| 0.5 to 0.6 | 25 |
| 0.6 to 0.7 | 22 |
| 0.7 to 0.8 | 27 |
| 0.8 to 0.9 | 8 |
| 0.9 to 1.0 | 0 |

The cleanest neurons match broad parts of speech, not words: neuron 88
fires on every `<start>` (recall 1.0) but a fifth of its firings are
elsewhere (precision 0.81); neurons 113, 57 and 33 catch every copula
with precision about 0.75. Their best labels by kind: part of speech 66,
after the main verb 15, word 12, agreement number 9, animacy 9, role 6,
word number 5, depth 5, tense 1.

The messiest neurons mix things. Neuron 32 fires on the verb right after
`who`, mostly when the subject is plural (`the owls who [slept]`, `the
cats who [flee]`): part depth, part agreement, part position. Neuron 72
fires on `who` after an object, and on a noun after `near`. Neuron 64
fires on the last word of a finished sentence (`... ignore a small wet
[wolf]`, `... were [small]`), a concept our labels do not have: "the
sentence could end here".

**Explanation for the stage.** A neuron is not a concept. The cleanest
ones are broad (every copula, plus some extra) and the rest are mixtures,
which is what superposition predicts: the model has more to track than
it has neurons, so it spreads concepts across them. If you went looking
for "the plural neuron" here, you would not find one.

**Where the prediction missed.** 40 clean neurons was far too
optimistic; there are none. And the best matches are grammar categories,
not single words: a neuron that fired only on `who` would waste a whole
neuron on one word out of 70.

**Open question for the speaker.** Neuron 64 suggests the labels are
missing at least one concept the model clearly uses: "a sentence could
end here". It is computable from left context. Adding it would make the
answer key more complete; leaving it out is a good honest Q&A answer
("our labels are only the concepts we thought of").

## Adding the missing label (2026-10-29)

Prompted by Goodfire's "Neural Networks Are Already Clean" (2026), which
argues that messy looking neurons usually mean a missing label, the labels
gained `:can_end` ("the words so far already make a whole sentence") and
the neurons were scored again.

**Result.** Still **0 of 128** monosemantic. But `can_end` became the best
label for **23 neurons**, more than any other single label value, so the
model spends a large share of its neurons on a concept the answer key had
missed. No one neuron owns it. The neurons split it into slices, each one
precise but partial: neuron 124 is right 99% of the time it fires but
catches 56% of the places a sentence could end; neuron 123 is right 95%
of the time and catches 64%; neuron 104 catches 98% but is right only 71%
of the time. Neuron 64, the one that prompted the label, is a narrow
slice: precision 0.74, recall 0.15.

**Explanation for the stage.** Both camps get something. Goodfire is
right that the answer key was incomplete: a whole concept was missing,
and the model cares about it a lot. But filling the gap did not make any
neuron clean. Here the concept is spread across many neurons, each
covering part of it, rather than packed into one. Whether the SAE can
gather those slices back into a single clean feature is the head to head
question.

## The first SAE, head to head with the neurons (2026-10-29)

**Prediction.** 0 of 512 features monosemantic; a few dead features; the
SAE wins the head to head; one SAE feature captures `can_end` cleanly.

**Result** (`mix interp.sae`, 512 features, 8 active, trained in 138
seconds on the activations of 20,000 training sentences, measured on the
2,000 evaluation sentences):

| Measure | Value |
| --- | --- |
| Variance explained | 98.9% |
| Model loss with its own neurons | 1.9635 |
| Model loss with the SAE spliced in | 1.9657 |
| Model loss with the neurons switched off | 2.7664 |
| Loss recovered | 99.7% |
| Dead features | 32 of 512 |
| Dense features (on for over half of all words) | 0 |
| Monosemantic at 0.9 | **33 of 512 features**, against 0 of 128 neurons |

The 33 clean features are 21 single words (`he`, `they`, `ignores`,
`near`, `behind`, ...) and 12 parts of speech (copula, conjunction, and
seven separate `<start>` features).

Head to head, best neuron against best feature for each label, a win
needing 0.05 more F1:

| Labels | Features win | Neurons win | Ties |
| --- | --- | --- | --- |
| 69 single words | 69 | 0 | 0 |
| 29 grammar labels | 9 | 14 | 6 |

The features win the closed class words outright (copula 1.00 against
0.85, preposition 0.91 against 0.45, conjunction 1.00 against 0.46). The
neurons win the abstract concepts: agreement number plural 0.65 against
0.40, subject role 0.67 against 0.52, after the main verb 0.76 against
0.49. On those, every feature is precise but partial: the best plural
agreement feature is right 100% of the time it fires and catches 25% of
cases. 20 different features are at least 90% precise for plural
agreement, and 28 for `can_end`. The SAE has split each abstract concept
into many narrow pieces (`can_end`: one feature at 0.99 precision and
0.72 recall, tied with the best neuron at F1 0.83).

**Explanation for the stage.** The SAE is faithful: splice it in and the
model barely notices (99.7% of the loss recovered), so it is not
inventing things. It finds clean units for words and grammar words,
where the neurons had none. For the ideas the talk is about, like "the
subject is plural", it does not find one clean unit. It finds twenty
precise slices, each covering a different situation, which is the
feature splitting "Towards Monosemanticity" describes. Neither camp
wins outright: the SAE cleans up what the neurons smeared, but the
deeper grammar is spread out on both sides.

**Where the prediction missed.** Zero clean features was too pessimistic:
33. "A few" dead features was 32, about 6%. "The SAE wins" is true for
words and false for grammar concepts. One clean `can_end` feature: no,
28 precise fragments, the best catching 72%.

**Risk for steering (November 20).** Steering needs a "plural subject"
direction. No single feature is one. Options: steer with the sum of the
plural agreement fragments, steer with the best neuron, or train a
narrower SAE (fewer features, less splitting). This is now the most
useful reason to run the width experiment.

## Correction after the final review (2026-10-29)

A fresh reviewer found three ways the numbers above overstated
themselves, and all three are fixed:

- **The evaluation set was not held out.** 772 of the 2,000 evaluation
  sentences (38.6%) also appeared in training, because the grammar repeats
  short sentences. Evaluation and probe sentences now skip anything seen
  in training.
- **`<start>` was counted 2,000 times.** Position 0 is the same vector in
  every sentence, which gave the SAE seven features for one fact.
  Activations now start at position 1.
- **Labels true for most words favoured the neurons.** A unit that fires
  everywhere beats both sides on `depth = 0` or `can_end = false`, and a
  sparse feature never can. The head to head now leaves out labels true
  for more than half of all words.

The model itself did not change (same training data, same cold open
scores). The corrected numbers, which supersede the ones above:

| Measure | Before | Corrected |
| --- | --- | --- |
| Evaluation loss | 1.9635 | 2.0096 (held out) |
| Cold open case accuracy | 0.9911 | 0.9911 (448 probes, all unseen) |
| Neurons monosemantic | 0 of 128 | 0 of 128 |
| SAE features monosemantic | 33 of 512 | **23 of 512**, covering 22 distinct labels |
| Loss recovered by the SAE | 99.7% | 100.0% (2.0095 spliced against 2.0096) |
| Dead features | 32 | 31 |
| Single word labels, features win | 69 of 69 | 67 of 67 |
| Grammar labels, features / neurons / ties | 9 / 14 / 6 | **7 / 12 / 6** of 25 |
| Precise fragments for plural agreement | 20 | 17 |
| Precise fragments for `can_end` | 28 | 36 |

The story is unchanged and now rests on cleaner numbers: the SAE is
faithful, it wins every single word, and on abstract grammar concepts the
best neuron still beats the best feature more often than not, because the
SAE splits each concept into many precise fragments.

## Does training make neurons clean? (2026-10-29)

Goodfire's "Neural Networks Are Already Clean" (2026) argues that
polysemantic neurons are mostly a training problem, and that a squeezed
layer makes neurons *coarse* (one broad job) rather than *mixed*. We
retrained the llama model one change at a time (`mix interp.cleanliness`)
and sorted every neuron by kind, in this order: *clean* (one label at
precision and recall 0.9), *coarse* (a family of one label's values, such
as noun or verb), *combined* (two labels together, such as plural noun),
*mixed*, or *dead*. *Precise* is a looser count closer to Goodfire's:
right 90% of the time it fires, catching at least 10% of its label. The
talk's model is the baseline and was not retrained.

Every variant kept the model as good as before: evaluation loss 2.008 to
2.023, agreement probes at least 0.997.

| Variant | Clean | Coarse | Combined | Mixed | Dead | Precise |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (seeds 1234, 1235, 1236) | 0, 1, 0 | 4, 3, 2 | 0, 0, 0 | 124, 124, 126 | 0 | 8, 18, 8 |
| weight decay 0.01 | 0 | 4 | 0 | 124 | 0 | 6 |
| weight decay 0.1 (three seeds) | 3, 3, 3 | 7, 8, 6 | 0, 0, 1 | 118, 116, 118 | 0, 1, 0 | 20, 27, 27 |
| activity penalty 0.001 | 1 | 0 | 1 | 124 | 2 | 30 |
| activity penalty 0.01 | 0 | 0 | 1 | 45 | 82 | 7 |
| SoLU | 3 | 0 | 6 | 119 | 0 | 28 |
| width 512 (of 512) | 2 | 7 | 0 | 503 | 0 | 35 |

Clean neurons in the baseline at other bars: 0 at 0.8, 1 at 0.75, 3 at
0.7, 24 at 0.6, 40 at 0.5. The speaker kept 0.9 rather than lower the bar
after seeing results.

Predictions against results:

| Speaker predicted | Result |
| --- | --- |
| No variant gets clean neurons above zero | Mostly right: the best reach 3 of 128 |
| Messy neurons are mostly coarse | Wrong: 4 coarse, 124 mixed |
| Width 512 helps a little | Wrong: no help, median F1 fell from 0.50 to 0.44 |
| More than half the mixed neurons are combined | Wrong: none are |
| 35 of 128 are precise | Too high: 8 |
| Weight decay 0.1 does not hold up on new seeds | Wrong: 3 clean on every seed, precise 20 to 27 against 8 to 18 |

What it means:

- **Our neurons really are mixed.** No single label, family of values or
  pair of labels explains 97% of them, and even the forgiving precise
  count finds only 6 to 14%. Goodfire's roughly 90% does not reproduce in
  a model this small, by their kind of yardstick or ours.
- **Weight decay is the one training change that helps, and only a
  little.** It holds across three seeds: about 3 clean, 7 coarse and 25
  precise neurons against 0, 3 and 11. It costs nothing measurable in
  loss. It moves the needle from 0% to 2%, not to 90%.
- **Squeezing makes it worse, as superposition predicts.** The strong
  activity penalty killed 64% of the neurons without hurting the loss,
  and the survivors became messier (median F1 0.11): the same work packed
  into fewer neurons.
- **Room alone does not help.** With 512 neurons for about 190 label
  values, neither camp's story shows up: no clean neurons and no coarse
  ones. Nothing in training pushes a concept onto one neuron when it does
  not have to share.
- **SoLU** gains a few clean and combined neurons, but its median F1
  collapses to 0.17, probably because its spiky activations defeat a
  firing threshold set at 20% of each neuron's maximum.

Limits: labels cover only what the grammar knows, so a neuron tracking
something unlabeled (position, "a verb comes next") counts as mixed; the
combined check tries pairs only; the non weight decay variants ran on one
seed each.

## Steering: which direction means "the subject is plural"? (2026-10-30)

Five candidate directions in the 128 neuron space, each scaled to length
one and pushed at multiples of the average neuron vector length at a word
(15.75), at every word (`mix interp.steer`). Flip rate is the share of
5,412 held out singular subject probes where the model switches to a
plural verb. Damage is the rise in loss on words whose next word is not a
verb. The cold open margin is the best plural verb's score minus the best
singular verb's after `the llama who chases the dogs` (−5.02 unpushed).

| Direction | Push for a full flip | Damage there | Cold open margin there | At push 1: flips, damage |
| --- | --- | --- | --- | --- |
| Mean difference (plural minus singular average) | 1 | 0.456 | 8.27 | 100%, 0.456 |
| Best neuron (74: precision 0.78, recall 0.54) | 2 | **0.094** | 5.44 | 68%, −0.003 |
| Best SAE feature (29 of 512: precision 0.99, recall 0.23) | 2 | 2.654 | 5.14 | 94%, 0.784 |
| Sum of 17 SAE plural fragments | 4 | 3.057 | 3.20 | 8%, 0.523 |
| Narrow SAE feature (28 of 64: precision 0.62, recall 0.33) | 2 | 0.228 | 3.57 | 50%, 0.030 |

The speaker made no prediction.

What it means:

- **The messy neuron is the cleanest knob.** Neuron 74 is a mediocre
  plural detector (wrong on 22% of its firings, missing 46% of plural
  subjects), yet pushing it flips every probe for a tenth of a nat of
  damage. Detecting a concept and controlling it are different tests,
  and a unit can pass one and fail the other.
- **The SAE's cleanest feature is the worst knob of the single
  directions.** Feature 29 is right 99% of the time it fires, but pushing
  it costs eight to thirty times the damage of the neuron for the same
  flips. Its decoder row carries the context its fragment lives in, not
  just "plural".
- **Adding the fragments up makes it worse, not better.** The 17 plural
  fragments summed need twice the push of any single direction and do
  the most damage. Fragments of a split concept do not add back up to
  the concept.
- **The narrower SAE helps.** With 64 features its plural feature is a
  worse detector (precision 0.62) but a far better knob than the 512
  SAE's (0.228 against 2.654 damage for a full flip): less splitting, a
  more general direction.
- **The dumb average is a strong yardstick.** The mean difference flips
  everything at the smallest push, with moderate damage. Only the neuron
  and the narrow SAE beat it on damage.

Caveat: equal length is not equal naturalness. Neuron 74 never exceeds
8.94 on its own, and a full flip pushes it by 31.5, three and a half
times its maximum. It stays cheap because in a one layer model a neuron
only writes one fixed vector into the residual stream, and neuron 74's
vector mostly moves verb number. One model, one seed.

An earlier run selected 55 "fragments" by precision for plural alone,
which swept in features about other things (such as the word `they`).
Fragments are now features with plural among their top three labels,
which gives the 17 counted before.
