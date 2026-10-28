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
