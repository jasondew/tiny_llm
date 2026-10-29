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
