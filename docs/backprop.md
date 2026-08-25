# Backpropagation, derived

Every backward pass in this project, worked out by hand. The brief requires
this for stage 4; stage 3 is here too because it is the same machinery on an
easier problem, and because the identity it produces reappears inside
attention.

Nothing here is approximate. Each result is checked against finite
differences by `TinyLlm.GradCheck`, and the test suite fails if a derivation
and its implementation ever disagree by more than a relative `1.0e-3`.

## Notation

| symbol | meaning | shape |
| --- | --- | --- |
| `v` | vocabulary size | 32 |
| `d` | d_model | 32 |
| `E` | embeddings | `v × d` |
| `W` | projection | `d × v` |
| `a` | input token id, one example | scalar |
| `t` | target token id, one example | scalar |
| `x` | the input token's embedding, row `a` of `E` | `d` |
| `z` | logits | `v` |
| `p` | softmax(z) | `v` |
| `L` | cross-entropy loss | scalar |

Indices: `i` ranges over `d`, and `j`, `k`, `m` range over `v`. `δ_ab` is the
Kronecker delta, 1 when `a = b` and 0 otherwise. Vectors are rows, so `x W`
is `(1×d)(d×v)`.

The forward pass for one example:

    x_i = E_{a,i}
    z_j = Σ_i x_i W_{ij}
    p_j = exp(z_j) / Σ_k exp(z_k)
    L   = −ln p_t

## Stage 3, the Embedder

### A. dL/dz, the softmax and cross-entropy collapse

Expand `L` before differentiating. The logarithm turns the quotient into a
difference and the quotient rule never appears:

    L = −ln( exp(z_t) / Σ_k exp(z_k) )
      = −z_t + ln( Σ_k exp(z_k) )

Differentiate with respect to a single logit `z_j`. The first term
contributes only when `j = t`:

    ∂/∂z_j (−z_t) = −δ_jt

For the second, apply the chain rule to `ln`. Only the `k = j` term of the
sum depends on `z_j`:

    ∂/∂z_j ln( Σ_k exp(z_k) ) = 1/(Σ_k exp(z_k)) · exp(z_j)
                              = exp(z_j) / Σ_k exp(z_k)
                              = p_j

Adding them:

    ∂L/∂z_j = p_j − δ_jt

which as a row is

    g ≡ ∂L/∂z = p − one_hot(t)                                        (A)

**Via the Jacobian, for stage 4.** The collapse above hides the softmax
Jacobian, and stage 4 needs it in a case where nothing cancels. The general
form is

    ∂p_m/∂z_j = p_m (δ_mj − p_j)

Since `L = −ln p_t` depends on `p` only through the `t`-th entry,
`∂L/∂p_m = −δ_mt / p_t`, and the chain rule gives

    ∂L/∂z_j = Σ_m (∂L/∂p_m)(∂p_m/∂z_j)
            = −(1/p_t) · p_t (δ_tj − p_j)
            = p_j − δ_tj

Same answer. The `1/p_t` cancels against the `p_t` the Jacobian carries, and
that cancellation is the only reason (A) is a subtraction rather than a
matrix product. Attention's softmax has no such cancellation waiting for it.

### B. dL/dW

`W_{ij}` appears in exactly one logit, `z_j`, with coefficient `x_i`:

    ∂z_m/∂W_{ij} = x_i δ_mj

Chain rule over all logits, of which one survives:

    ∂L/∂W_{ij} = Σ_m (∂L/∂z_m)(∂z_m/∂W_{ij})
               = Σ_m g_m x_i δ_mj
               = x_i g_j

Every entry is one factor from `x` times one from `g`, which is the
definition of an outer product:

    ∂L/∂W = xᵀ g                                    shape d × v         (B)

### C. dL/dx

`x_i` appears in every logit, with coefficient `W_{im}` in `z_m`:

    ∂z_m/∂x_i = W_{im}

Now nothing collapses and the sum survives:

    ∂L/∂x_i = Σ_m (∂L/∂z_m)(∂z_m/∂x_i)
            = Σ_m g_m W_{im}

That is the dot product of `g` with row `i` of `W`, and doing it for every
`i` at once is a matrix product:

    ∂L/∂x = g Wᵀ                                    shape 1 × d         (C)

**B and C differ only in how many logits each parameter touches.** `W_{ij}`
touches one, so the sum has a single term and the result is a product. `x_i`
touches all `v`, so the sum survives and the result is a contraction. That
is the whole reason one is an outer product and the other a matmul.

### D. dL/dE

`x` was copied out of row `a`:

    x_i = E_{a,i}   ⟹   ∂x_i/∂E_{r,i} = δ_ra

so

    ∂L/∂E_{r,i} = (∂L/∂x_i) δ_ra                                        (D)

Row `a` receives `∂L/∂x`; every other row receives zero. Not "approximately
zero": those entries do not appear in `L` at all for this example, so their
partial derivatives are exactly `0`.

### Batches

For a batch of `N` examples the loss is the mean,

    L_batch = (1/N) Σ_n L⁽ⁿ⁾

and differentiation is linear, so every gradient above is the mean of its
per-example value. Two consequences:

- Sum the per-example gradients, then multiply by `1/N`. Forgetting the
  `1/N` makes every gradient `N` times too large; `GradCheck` reports a
  relative error of `(N−1)/(N+1)`, which is `0.6` for `N = 4`.
- If token `a` appears in more than one example, row `a` of `E` receives a
  contribution from each and they **add**, because `∂/∂E_{a,i}` of a sum is
  the sum of the partials.

### Summary

    g = p − one_hot(t)                    v
    ∂L/∂W = xᵀ g                          d × v
    ∂L/∂x = g Wᵀ                          d
    ∂L/∂E = zeros, row a set to ∂L/∂x     v × d

## The one rule that does most of the work

Stage 3 had two shapes of gradient and they looked unrelated: an outer
product for `W`, a contraction for `x`. They are the same rule seen at
`T = 1`. State it once here and stage 4 is mostly bookkeeping.

For any matrix product `Y = F G` with `F` of shape `n × m`, `G` of shape
`m × p`, and a scalar loss downstream, write `dY ≡ ∂L/∂Y`. Then

    Y_{ab} = Σ_c F_{ac} G_{cb}

    ∂L/∂F_{ac} = Σ_{a'b} dY_{a'b} · ∂Y_{a'b}/∂F_{ac}
               = Σ_{a'b} dY_{a'b} · δ_{a'a} G_{cb}
               = Σ_b dY_{ab} G_{cb}
               = (dY Gᵀ)_{ac}

    ∂L/∂G_{cb} = Σ_{a} dY_{ab} F_{ac}
               = (Fᵀ dY)_{cb}

so

    dF = dY Gᵀ        dG = Fᵀ dY                                       (R)

Two things are worth noticing before moving on.

**Every gradient has the shape of the thing it is the gradient of**, which
is the cheapest check available: if `dWq` is not `d × d`, a transpose is in
the wrong place, and no amount of staring at the algebra finds that faster
than looking at the shape.

**(B) and (C) are (R) with one row.** `∂L/∂W = xᵀ g` is `Fᵀ dY` where `F` is
the `1 × d` matrix holding `x`, and `∂L/∂x = g Wᵀ` is `dY Gᵀ`. What looked
like two different operations in stage 3 is one operation, and the outer
product was only an outer product because the batch had a single example.
With `T` positions, `Fᵀ dY` sums over positions and forms the products at
the same time.

## Stage 4, attention

### What is being differentiated

Stage 4 is a complete model, not a loose layer: it predicts the next token
at every position at once. The head is the interesting part, but there has
to be a loss on the end of it or there is nothing to check gradients
against.

| symbol | meaning | shape |
| --- | --- | --- |
| `T` | positions in this sequence, at most `context_length` | ≤ 16 |
| `E` | token embeddings | `v × d` |
| `P` | position embeddings | `context_length × d` |
| `X` | head input, one row per position | `T × d` |
| `Wq`, `Wk`, `Wv` | query, key, value projections | `d × d` |
| `Wo` | head output projection | `d × d` |
| `Wu` | unembedding, back to vocabulary | `d × v` |
| `Q`, `K`, `V` | queries, keys, values | `T × d` |
| `U` | raw scores, `Q Kᵀ` | `T × T` |
| `S` | scaled scores, `U/√d` | `T × T` |
| `M` | masked scores | `T × T` |
| `A` | attention weights, `softmax` of each row of `M` | `T × T` |
| `C` | context, `A V` | `T × d` |
| `O` | head output, `C Wo` | `T × d` |
| `Z` | logits | `T × v` |

Indices: `t` and `s` range over positions, `i` over `d`, `j`, `m` over the
vocabulary or over positions where the context makes it obvious. `a_t` is
the input token id at position `t` and `y_t` the target.

The forward pass:

    X_t   = E[a_t] + P[t]                        row t of X
    Q     = X Wq        K = X Wk       V = X Wv
    U     = Q Kᵀ
    S     = U / √d
    M_ts  = S_ts + (s > t ? −1.0e9 : 0)
    A_t   = softmax(M_t)                         each row on its own
    C     = A V
    O     = C Wo
    Z     = O Wu
    L     = −(1/N) Σ_t ln p_{t,y_t}              p_t = softmax(Z_t)

`A_ts` reads "how much position `t` draws on position `s`". Row `t` is a
distribution over the positions `t` is allowed to see, which is why the mask
goes on before the softmax rather than after: masking after would leave rows
that no longer sum to 1.

### E. dL/dZ, one row per position

`N` is the total number of predicted positions in the batch, counted across
every sequence, not the number of sequences. Each row of `Z` is an
independent instance of stage 3's problem, so (A) applies row by row:

    dZ_{tj} = (p_{tj} − δ_{j,y_t}) / N                                  (E)

**Divide by tokens, not by sentences.** Averaging each sequence and then
averaging the averages weights a 4 token sentence's tokens four times as
heavily as a 16 token sentence's. The number that comes out is still a
loss and still falls, but it is no longer measured in the same unit as the
bigram's 1.9021 nats, and the one comparison this whole project is built
around stops meaning anything.

### F. Out through the two projections

`Z = O Wu` and `O = C Wo` are plain matrix products, so (R) does both:

    dWu = Oᵀ dZ           d × v
    dO  = dZ Wuᵀ          T × d
    dWo = Cᵀ dO           d × d
    dC  = dO Woᵀ          T × d                                         (F)

**`Wo` is redundant at stage 4, and that is worth saying out loud.** Since
`Z = C Wo Wu` and `Wo Wu` is a single `d × v` matrix, this model can express
exactly what it could express with `Wu` alone. It is the same collapse as
`E · W` in stage 3. `Wo` is carried anyway because stage 5 wraps the head in
a residual, `X + C Wo`, and a sum of two terms does not fold into a product
of one. The redundancy dies the moment the residual arrives.

### G. The context splits into weights and values

`C = A V`, so by (R):

    dA = dC Vᵀ            T × T
    dV = Aᵀ dC            T × d                                         (G)

Read `dA` as: how much would the loss change if position `t` had leaned a
little harder on position `s`. That is the gradient the softmax Jacobian is
about to receive, and unlike stage 3 it is an arbitrary row of numbers.

### H. The softmax Jacobian, which this time does not cancel

Rows of `A` are independent, so there are no cross row terms and the whole
thing is done one row at a time. The general Jacobian, the one stage 3
introduced and then dodged:

    ∂A_{tm}/∂M_{tj} = A_{tm} (δ_{mj} − A_{tj})

Chain rule over the row:

    dM_{tj} = Σ_m dA_{tm} · A_{tm} (δ_{mj} − A_{tj})
            = A_{tj} dA_{tj} − A_{tj} Σ_m dA_{tm} A_{tm}

Name the sum, which is one scalar per row:

    r_t = Σ_m dA_{tm} A_{tm} = dA_t · A_t

    dM_{tj} = A_{tj} (dA_{tj} − r_t)                                    (H)

In stage 3, `∂L/∂p` was `−1/p_t` on a single entry and zero everywhere else.
The `1/p_t` cancelled against the `p_t` the Jacobian carries, `r_t` collapsed
to `−1`, and the result was the subtraction in (A). Here `dA_t` is a full
row that arrived from `dC Vᵀ`, nothing cancels, and `r_t` survives. **(A) is
the special case. (H) is the rule.**

There is a free self test hiding in this. Adding a constant to a row of `M`
does not change `softmax(M_t)`, so the loss cannot depend on that direction,
so every row of `dM` must sum to exactly zero:

    Σ_j dM_{tj} = Σ_j A_{tj} dA_{tj} − r_t Σ_j A_{tj} = r_t − r_t = 0

`Σ_j A_{tj} = 1` is what makes the second term collapse. If a row of `dM`
does not sum to zero, (H) is wrong and no finite difference run is needed to
know it.

### I. The mask costs nothing on the way back

`M = S + mask` with `mask` constant, so `dS = dM` entry for entry. There is
no masking step in the backward pass at all, and there does not need to be:
for `s > t` the forward pass computed `exp(−1.0e9 − max)`, which **underflows
to exactly 0.0**, so `A_{ts} = 0.0` and (H) gives

    dM_{ts} = A_{ts} (dA_{ts} − r_t) = 0.0 · (anything) = 0.0

Exactly zero, not nearly zero. It is a multiplication by a float that is
literally `0.0`. The forward mask already did the work.

This is the one place the magic number matters. A "large enough looking"
`−100.0` would leave `A_{ts} ≈ 3.7e−44`, a subnormal that is not zero, and
the future would leak a whisper of gradient into the past forever. `−1.0e9`
underflows; that is the entire reason for the size of it.

### J. The scale

`S = U/√d` is a constant multiple, so

    dU = dS / √d                                                        (J)

Why `√d` at all: `U_ts` is a sum of `d` products, so if the entries of `Q`
and `K` are roughly independent with variance `σ²`, then `U_ts` has variance
`d σ²` and the scores grow like `√d`. Large scores saturate the softmax,
every row of `A` goes to a one hot, and by (H) the gradient `A_{tj}(…)` goes
to zero along with it. Dividing by `√d` holds the score variance fixed as
`d` changes, so the head still learns at `d = 32` for the same reason it
learns at `d = 8`.

### K. Splitting QKᵀ

`U = Q Kᵀ`. Apply (R) with `F = Q` and `G = Kᵀ`:

    dQ = dU (Kᵀ)ᵀ = dU K
    dKᵀ = Qᵀ dU     ⟹     dK = (Qᵀ dU)ᵀ = dUᵀ Q                        (K)

Neither result has a transpose on the matrix coming back, because `Kᵀ` was
already the transposed factor. What distinguishes them is `dUᵀ`. Every
position's query meets every position's key, so the same `T × T` block of
gradient has to be read twice: once indexed by the querying position and
once by the keyed position. The transpose is what swaps which index is
which, and getting it backwards is the single easiest mistake in this
derivation to make and the hardest to see, because `dQ` and `dK` have
identical shapes and the loss still falls.

Shapes, since they are all that will save you here: `dU` is `T × T`, `K` is
`T × d`, so `dU K` is `T × d`. `dUᵀ` is `T × T`, `Q` is `T × d`, so `dUᵀ Q`
is `T × d`. Both correct, which is exactly why the shape check does **not**
catch a swapped transpose. The gradient check does.

### L. Three branches meet at X

`Q = X Wq`, `K = X Wk`, `V = X Wv`, three independent applications of (R):

    dWq = Xᵀ dQ        dWk = Xᵀ dK        dWv = Xᵀ dV                   d × d

and then the three contributions to `X` itself, which **add**:

    dX = dQ Wqᵀ + dK Wkᵀ + dV Wvᵀ                     T × d             (L)

This is the first value in the project that gets used more than once, and
the multivariable chain rule says to sum over every path from a variable to
the loss. There are three paths out of `X` and all three come back.

Dropping one of the three terms is the classic error, and it is quiet: two
thirds of the gradient is still right, the loss still falls, the model still
trains. Its signature under the gradient check is specific and easy to read.
`Wq`, `Wk` and `Wv` all pass, because they are computed from `dQ`, `dK` and
`dV` directly and never touch `dX`. Only `E` and `P` fail.

### M. Down to the tables

Row `t` of `X` was assembled by adding two rows that were looked up:

    X_{ti} = E_{a_t,i} + P_{t,i}
    ∂X_{ti}/∂E_{r,i} = δ_{r,a_t}          ∂X_{ti}/∂P_{u,i} = δ_{ut}

so both gradients are scatter adds of the same rows of `dX` into different
tables:

    dE[a_t] += dX_t     for every position t of every sequence
    dP[t]   += dX_t     for every position t of every sequence          (M)

The difference is what indexes them, and it shows up as soon as there is
more than one sequence in the batch.

`P` is indexed by position. Within one sequence each row receives exactly
one contribution, but every sequence in a batch of `B` has a position 0, so
across the batch row 0 of `dP` receives `B` of them. Early positions
accumulate from every sequence; late positions only from sequences long
enough to reach them, so `dP`'s last rows are quieter and its unreached rows
are exactly zero.

`E` is indexed by token. It accumulates whenever a token repeats, which
`"the"` does in nearly every sentence in this grammar, and rows for tokens
absent from the batch stay exactly zero, the same as stage 3's (D).

A second free self test falls out of this. Both tables receive every row of
`dX` exactly once, just filed differently, so their column sums are equal:

    Σ_r dE_r = Σ_t dX_t = Σ_u dP_u

Two tables of different heights, `v × d` and `context_length × d`, summing
down to the identical `d` wide row. That is cheap to assert and it fails
loudly if either scatter add drops a position or double counts one.

### Summary

    dZ  = (p − onehot(y)) / N                 T × v      per position
    dWu = Oᵀ dZ                               d × v
    dO  = dZ Wuᵀ                              T × d
    dWo = Cᵀ dO                               d × d
    dC  = dO Woᵀ                              T × d
    dA  = dC Vᵀ                               T × T
    dV  = Aᵀ dC                               T × d
    dM_{tj} = A_{tj}(dA_{tj} − dA_t · A_t)    T × T      rows sum to 0
    dS  = dM                                  T × T      mask is additive
    dU  = dS / √d                             T × T
    dQ  = dU K                                T × d
    dK  = dUᵀ Q                               T × d
    dWq = Xᵀ dQ   dWk = Xᵀ dK   dWv = Xᵀ dV   d × d
    dX  = dQ Wqᵀ + dK Wkᵀ + dV Wvᵀ            T × d      three paths, added
    dE[a_t] += dX_t   dP[t] += dX_t           scatter adds

### The gradient check has a hole, and it is exactly here

`GradCheck.relative_error/2` divides by `max(|analytic| + |numeric|, guard)`
with `guard = 1.0e-8`. That guard exists so a legitimately zero gradient does
not divide by zero. It also means that **when both gradients are far below
the guard, the check reports agreement no matter what the derivation says**,
because it is comparing two numbers that are both effectively zero against a
denominator that is not.

The score path walks straight into it. `Tensor.random/3` defaults to a scale
of `0.02`, and `U = (X Wq)(X Wk)ᵀ` carries four of those factors before the
gradient starts back. Measured on the `v = 6`, `d = 8`, `T ≤ 4` config, the
largest analytic gradient in each matrix at init:

    scale   E        P        Wq       Wk       Wv       Wo       Wu
    0.02    2.8e−06  2.1e−06  8.3e−13  5.5e−13  2.7e−06  5.6e−06  3.9e−06
    0.5     4.1e−02  3.0e−02  4.8e−03  3.2e−03  4.5e−02  8.5e−02  5.7e−02

At `0.02`, `dWq` and `dWk` sit five orders of magnitude *below* the guard,
and every bug in (H), (J) and (K) passes the check. Verified by injecting
them: at the default scale, a swapped transpose, a dropped `r_t` and a
missing `√d` all report `ok` on all seven matrices. The safety net has a hole
precisely over the hardest math in the project.

A check that cannot fail is worse than no check, because it is trusted. The
test has to assert the analytic gradients clear the guard rather than assume
it.

### The same number kills training outright

This is not confined to the check. At the full `32 × 32` config, learning
rate `0.5`, batch 64, held out on 200 sentences:

    init scale   dWq at init   step 0    50      100     150     200
    0.02         1.1e−12       3.4657  3.4657  3.4657  3.4657  3.4657
    0.177        4.6e−06       3.4633  2.9313  2.7572  2.1523  2.0491
    0.5          5.7e−03       3.3470  2.2862  1.8312  1.7327  1.6833

The first row is not slow learning. It is `ln(32) = 3.4657` to four decimal
places, unchanged for 500 steps: a flat loss curve, no error raised, and a
completely correct backward pass underneath it. The gradients reaching `Wq`
are `1.1e−12`, so the parameters never move.

**Stage 3's `0.02` was not wrong, it was short.** That path was
`E → W → logits`, two factors deep, and `0.02` kept the initial logits near
zero so the first loss landed on `ln(v)`, which is exactly what stage 3
wanted. The score path is four factors deep before the gradient turns
around, and a scale that is merely small in a two factor product is
annihilating in a four factor one.

### The fix, and why it is a calculation rather than a preference

Scale each matrix by its own fan in and fan out rather than by one constant:

    scale(fan_in, fan_out) = √( 6 / (fan_in + fan_out) )

which is `0.6124` at `d = 8` and `0.3062` at `d = 32`. The `6` is what makes
a uniform draw on `−scale..scale` carry the variance `2/(fan_in + fan_out)`
that keeps activations from growing or shrinking as they pass through a
layer, which is the same argument the `1/√d` in (J) makes about scores.

Measured, with everything else unchanged:

- the gradient check at `v = 6`, `d = 8` now clears the guard by a factor
  between `1.8e6` and `1.8e7` on every matrix, and reports errors of `1e−9`
  to `4e−8`. Vacuity is no longer possible.
- at full size it crosses the bigram floor of `1.9021` between step 100 and
  step 150, and reaches `1.6742` by step 300. **301 steps and 7 held out
  evaluations take 50.3 seconds**, inside the brief's budget with room for
  stage 5 to spend some.

That crossing is the point of the whole project. It is the first time
anything here has done something a bigram cannot.

### Reading a failed gradient check

Which matrices fail localizes the bug better than any amount of rereading.
Every row below was produced by injecting that bug into a working
implementation at `v = 6`, `d = 8`, `T ≤ 4`, `N = 8`, init scale `0.5`, so
these are measurements and not predictions.

| failing | error | the bug |
| --- | --- | --- |
| all seven | `0.7778` | the `1/N` in (E), dropped. `(N−1)/(N+1) = 7/9` |
| `E`, `P` only | `1.0000` | a missing term in `dX`, (L) |
| `E`, `P`, `Wk`, but `Wq` fine | `1.0000` | the transpose in (K). Only `dK` is wrong, so `Wq` stays clean and `Wk` and everything downstream of `dX` does not |
| `E`, `P`, `Wq`, `Wk` | `1.0000` | `r_t` dropped from (H) |
| `Wq`, `Wk` only | `0.4776` | the `√d` of (J), applied forward but not back. `(√d−1)/(√d+1)` at `d = 8` |

Two patterns are worth internalizing. **`E` and `P` fail whenever anything
upstream of `dX` is wrong**, because every path to the tables runs through
it, so they are the noisiest signal and the last one to diagnose from.
**`Wq` and `Wk` are the only matrices downstream of the softmax Jacobian and
the scale**, so a failure confined to those two points at (H), (J) or (K)
and nowhere else.

Two cheaper checks come before finite differences. Rows of `dM` must sum to
zero, from (H). `dE` and `dP` must have identical column sums, from (M).
Both are one line and neither needs a perturbation loop.

Then: check shapes, because half the errors above are a transpose and shapes
find those instantly. And run on the smallest config that still exercises
every path. `T = 4` and `d = 8` is enough to distinguish `dU K` from
`dU Kᵀ`; `T = 1` is not, because at `T = 1` the mask is empty, `A` is the
scalar `1.0`, and (H) reports zero whatever you wrote.

## Stage 5, the block

Stage 4 was one sublayer with a loss bolted on. Stage 5 wraps it in the
three things that make a transformer block: a normalization before each
sublayer, a residual around each sublayer, and a position-wise MLP.

### What is being differentiated

| symbol | meaning | shape |
| --- | --- | --- |
| `h` | MLP hidden width | 128 |
| `g1`, `g2`, `g3` | RMSNorm gains | `d` each |
| `W1`, `b1` | MLP up-projection and bias | `d × h`, `h` |
| `W2`, `b2` | MLP down-projection and bias | `h × d`, `d` |
| `N1`, `N2`, `N3` | normalized activations | `T × d` |
| `O` | attention sublayer output, stage 4's `C Wo` | `T × d` |
| `R` | first residual stream, `X + O` | `T × d` |
| `Pre` | MLP hidden before ReLU | `T × h` |
| `Hid` | MLP hidden after ReLU | `T × h` |
| `Mlp` | MLP sublayer output | `T × d` |
| `Y` | second residual stream, `R + Mlp` | `T × d` |

The forward pass:

    X    = E[a_t] + P[t]                    as in stage 4
    N1   = rmsnorm(X, g1)
    O    = attention(N1)                    every line of stage 4, on N1
    R    = X + O
    N2   = rmsnorm(R, g2)
    Pre  = N2 W1 + b1
    Hid  = max(Pre, 0)
    Mlp  = Hid W2 + b2
    Y    = R + Mlp
    N3   = rmsnorm(Y, g3)
    Z    = N3 Wu
    L    = −(1/N) Σ_t ln p_{t,y_t}

Two choices are worth stating because the alternatives are defensible and
this file is the record of which was taken.

**Pre-norm, not post-norm.** The normalization sits *before* each sublayer
and the residual skips *around* both. The alternative, `R = rmsnorm(X + O)`,
was the original 2017 arrangement and trains noticeably worse without a
warmup schedule, because the residual path is no longer a clean identity
from the loss all the way back to the embeddings.

**A third norm before the unembedding.** Without it, `Y` reaches `Wu` with
whatever scale the residual stream happened to accumulate, which is neither
controlled nor stable across training.

### N. RMSNorm, forward

Per row `x` of the input, independently. `ε` guards a row of exact zeros.

    ms  = (1/d) Σ_i x_i²
    r   = √(ms + ε)
    n_i = x_i / r
    y_i = g_i n_i

Note what is missing compared to LayerNorm: no mean subtraction, and no
bias. RMSNorm rescales without recentring. That is not a simplification
made for this project, it is what the RMSNorm paper found: the recentring
does almost nothing and costs a pass over the row.

### O. RMSNorm, backward

The gain is easy, because `g_i` multiplies exactly one output:

    dg_i = Σ_over rows dy_{t,i} n_{t,i}                              (O1)

`g` is a single vector shared by every row, so its gradient accumulates over
every position in every sequence, exactly like `P` does.

For the input, write `dn_i = dy_i g_i` and let

    c = (1/d) Σ_i dn_i n_i

Then

    dx_j = (1/r) (dn_j − n_j c)                                      (O2)

Worth seeing where `c` comes from, because the shape of this result is the
same one that appeared in the softmax Jacobian and it is not a coincidence.
`n_j` depends on *every* `x_i`, through `r`. So

    ∂n_i/∂x_j = δ_ij / r + x_i ∂(1/r)/∂x_j

and since `∂r/∂x_j = x_j / (d r)`,

    ∂(1/r)/∂x_j = −x_j / (d r³)

Contracting with `dn` and substituting `x_j = n_j r` collapses the second
term to `n_j c / r`, giving (O2).

Both (H) and (O2) have the form *"the direct term, minus the output times a
scalar summarizing the whole row"*. Any operation that normalizes a row
couples every entry to every other, and the price is always one row-scalar
subtracted off. Softmax normalizes by a sum, RMSNorm by a root-mean-square,
and the algebra rhymes.

**Free check.** Take `ε = 0`, and note `Σ_j n_j² = d`. Then

    Σ_j dx_j n_j = (1/r)(Σ_j dn_j n_j − c Σ_j n_j²) = (1/r)(d c − d c) = 0

so **every row of `dx` is orthogonal to the corresponding row of `n`**. This
is the RMSNorm counterpart of "rows of `dM` sum to zero", it costs one dot
product, and it catches a dropped or mis-scaled `c` immediately.

### P. ReLU

    dPre_{ti} = dHid_{ti} · [Pre_{ti} > 0]

Strictly `>`, not `≥`. ReLU has no derivative at exactly zero and the choice
is arbitrary, but it must be *made*, and 0 is the conventional pick. It also
matters that the condition tests `Pre`, the value before the ReLU, not
`Hid` after it. They agree wherever `Pre > 0` and differ nowhere that
matters here, but reaching for the cached pre-activation is the habit that
stays correct when the nonlinearity is not ReLU.

### Q. The MLP

Two applications of (R), with the biases falling out as column sums:

    dW2 = Hidᵀ dMlp      db2 = Σ_t dMlp_t       dHid = dMlp W2ᵀ      (Q1)
    dW1 = N2ᵀ dPre       db1 = Σ_t dPre_t       dN2  = dPre W1ᵀ      (Q2)

A bias is added identically to every row, so its gradient is the sum of the
gradient over every row. That is the same scatter-add logic as `P` in stage
4, with one destination instead of `T` of them.

### R. Residuals, and why they are the easy part

`R = X + O` and `Y = R + Mlp` are additions, so each sends its incoming
gradient to both of its inputs unchanged:

    dR gets dY                dMlp gets dY
    dX gets dR                dO   gets dR

That is the whole rule, and it is why residual connections fix vanishing
gradients: there is now a path from the loss to `X` that passes through no
matrix at all. Whatever the sublayers do to their share, the identity path
delivers `dY` to the embeddings undiminished.

The consequence for implementation is that `dR` and `dX` are each a **sum of
two contributions** arriving from different places, and neither is complete
until both have arrived:

    dR = dY + dR_from_norm2                                          (R1)
    dX = dR + dX_from_norm1                                          (R2)

This is the same trap as `dX` in stage 4, one level up. Drop the second term
of (R2) and the loss still falls, most matrices still pass the check, and
only `E` and `P` fail.

### S. The whole chain, in the order it runs

    dZ                 = (p − onehot(y)) / N                          (E)
    dWu = N3ᵀ dZ       dN3 = dZ Wuᵀ                                   (F)
    dg3, dY            = rmsnorm_backward(Y, g3, dN3)                 (O)
    dMlp               = dY
    dW2, db2, dHid     = (Q1) on dMlp
    dPre               = dHid ⊙ [Pre > 0]                             (P)
    dW1, db1, dN2      = (Q2) on dPre
    dg2, dR_from_norm2 = rmsnorm_backward(R, g2, dN2)                 (O)
    dR                 = dY + dR_from_norm2                          (R1)
    dO                 = dR
    dWq…dWo, dN1       = stage 4 (F) through (L), on dO
    dg1, dX_from_norm1 = rmsnorm_backward(X, g1, dN1)                 (O)
    dX                 = dR + dX_from_norm1                          (R2)
    dE[a_t] += dX_t    dP[t] += dX_t                                  (M)

Everything from `dO` to `dN1` is stage 4 unchanged. The head does not know
it has been wrapped; it receives a gradient on its output and returns one on
its input, exactly as before. That is worth saying out loud when presenting
this: a transformer is not a new idea per block, it is the same block again.

### Wo stops being redundant here

Stage 4 noted that `Z = C Wo Wu` collapses: two matrices in series with
nothing between them are one matrix, so `Wo` was free parameters buying no
expressiveness, just as `E W` in stage 3 was.

The residual breaks that. `O = C Wo` is now *added to* `X` rather than fed
straight onward, and `X + C Wo` cannot be rewritten as `C` times anything.
The same is true of the MLP: `W1` and `W2` in series would collapse if the
ReLU were not between them. Every one of these matrices earns its place only
because something non-linear or non-composable sits next to it, which is a
reasonable one-line summary of why deep networks are built the way they are.

### What to expect from the gradient check

Nine matrices plus three gain vectors now. The new failure signatures:

| failing | the likely bug |
| --- | --- |
| `E`, `P` only | a dropped term in (R1) or (R2) |
| `g3`, and nothing else | (O1) summing the wrong factor; it is `dy · n`, not `dy · y` |
| everything from `g2` back, `W1`/`W2` clean | `c` dropped from (O2) at norm 2 |
| `W1`, `b1` fail, `W2`, `b2` clean | the ReLU mask, (P), applied to the wrong side |
| `b1`, `b2` only | summing the bias gradient down the wrong axis |

And the two free checks, both cheaper than a perturbation loop: every row of
`dx` out of `rmsnorm_backward` is orthogonal to the corresponding row of `n`,
and `dE` and `dP` still have identical column sums.
