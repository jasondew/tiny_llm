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
