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

## Stage 4, attention

To be derived before implementing, per the brief. The pieces that will not
collapse the way (A) did: the softmax Jacobian applied to upstream
gradients, splitting the gradient of `QKᵀ` between `Q` and `K`, the
`1/√d` scale, the causal mask contributing zero through masked positions,
and accumulating position-embedding gradients across every position that
used them.
