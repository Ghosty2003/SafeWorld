# Stage 3 feasibility report: white-box L3 on a real trained world model

**Status: assessment only, no implementation code written, per task scope.**
Investigates the three items assigned: (1) does TD-MPC2's transition head fall
in Theorem D.5's covered class, (2) confirm DreamerV3's RSSM exclusion,
(3) a concrete (not just qualitative) Theorem D.3 query-count estimate,
seeded with Stage 2's real spectral-norm tooling applied to the *actual*
loaded TD-MPC2 checkpoint rather than a hypothetical.

---

## 1. TD-MPC2: does the transition head fall in Theorem D.5's covered class?

Investigated directly from source, not assumed. Source: the upstream TD-MPC2
clone `tdmpc2_src/tdmpc2/common/world_model.py` + `layers.py` (the same
scratchpad clone `wrappers/tdmpc2_wrapper.py::TDMPC2_SRC_DEFAULT` points at
runtime), cross-checked against the actual loaded checkpoint
`models/walker-walk-3.pt`.

```python
# common/world_model.py
self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim,
                             2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))

def next(self, z, a, task):
    z = torch.cat([z, a], dim=-1)
    return self._dynamics(z)
```

**Feedforward: yes.** `_dynamics` is a plain `nn.Sequential` of 3
`NormedLinear` layers (`layers.mlp`), no recurrence, no hidden state carried
across calls — architecturally the easy half of Theorem D.5's hypothesis (i),
and the part DreamerV3's RSSM fails (§2 below).

**Deterministic, not "additive/heteroscedastic Gaussian": also yes, but not
literally either of Theorem D.5's two stated noise cases.** `next()` returns
`self._dynamics(z)` directly — **no noise term appears anywhere in the
transition.** TD-MPC2's latent dynamics are a bare point map z' = f_θ(z,a),
not a distribution. This is not what Theorem D.5(a)/(b) describe as written
(both assume a genuine noise draw ξ), but it *is* the degenerate σ→0 limit of
case (a) — a Dirac point mass is a (trivial) instance of "state-independent
additive noise." Practically this makes the post-expectation map *easier*,
not harder, than the paper's stated cases: `G_φ(z) = E_ξ[φ(f_θ(z,a)+ξ)]`
collapses to `φ(f_θ(z,a))` exactly, no expectation over transition noise to
bound at all (only over the action distribution, if the policy itself is
stochastic). This point is not mentioned in the paper and is worth recording
as a positive, non-obvious finding: TD-MPC2's determinism is not a
disqualifying feature.

**1-Lipschitz activations: no — this is the real gap, and it is not the one
the paper's own scope note anticipates.** `NormedLinear` is
`Linear -> LayerNorm -> Mish` (`layers.py`'s `NormedLinear.forward`), not
`Linear -> (an activation with Lipschitz constant exactly 1)`:

- **Mish**: `x * tanh(softplus(x))`. Its Lipschitz constant is not exactly 1
  — the known max-derivative value is ≈1.088 (attained near x≈0.6). Small,
  cleanly correctable (multiply by 1.088 per activation instead of 1) — not
  the blocking issue.
- **LayerNorm**: `γ ⊙ (x-μ)/√(σ²+ε) + β`. This is the blocking issue.
  LayerNorm's Lipschitz constant is *not* a fixed number at all — it scales
  with `‖γ‖/√(σ²+ε)`, and as the pre-normalization variance σ² of the input
  vector shrinks toward 0, the Lipschitz constant is *unbounded*. Theorem
  D.5's clean composition bound `L ≤ L_φ · Π‖W_i‖₂` assumes activations
  contribute a *fixed* multiplicative factor per layer; LayerNorm does not
  offer one without an additional, separately-argued assumption (e.g. a
  declared lower bound on the pre-normalization variance actually reachable
  by the trained network, which would need its own justification — not
  something Theorem D.5 as stated supplies).

  Measured from the actual checkpoint (`models/walker-walk-3.pt`), the
  LayerNorm `γ` scales alone are non-trivial (not ≈1, which would be the
  best case): layer 0 γ mean 2.30 (max 7.99), layer 1 γ mean 1.75 (max
  6.48), layer 2 γ mean 1.69 (max 5.26) — so even setting aside the
  unbounded-near-zero-variance issue, the *typical*-case multiplier is
  already several-fold per layer, not ≈1.

**Verdict on item 1**: TD-MPC2 is architecturally *closer* to Theorem D.5's
scope than DreamerV3 (feedforward, no recurrence, no categorical latents,
single transition head) — but it is **not a clean drop-in fit**. It needs
"additional derivation" too, specifically a LayerNorm-Lipschitz bound (a
well-studied but nontrivial problem — bounding LayerNorm's Lipschitz
constant requires either restricting to a region where pre-normalization
variance is bounded away from 0, or accepting a much looser worst-case
bound), not the harder recurrence/categorical-latent problem DreamerV3 needs.
This is a smaller gap than DreamerV3's, not a zero gap.

---

## 2. DreamerV3 RSSM exclusion — confirmed from source, not just cited

Investigated `/home/bot/CarDreamer/dreamerv3/nets.py::RSSM` (the actual
network this project's `CarDreamerWrapper` loads), not just taken from the
paper's own exclusion statement.

```python
class RSSM(nj.Module):
    def __init__(self, deter=1024, stoch=32, classes=32, ...):
        ...
    # ...
    def _gru(self, x, deter):
        x = self.get("gru", Linear, **kw)(x)
        ...
        x, deter = self._gru(x, prev_state["deter"])
```

Two independent, concrete confirmations of the paper's own exclusion
criteria:

1. **Recurrent, not feedforward**: `_gru()` explicitly feeds `deter`
   (the previous step's hidden state) back into itself every call — a
   genuine recurrent loop, the literal opposite of Theorem D.5(i)'s
   "feedforward transition head" hypothesis.
2. **Discrete/categorical stochastic component**: `stoch=32, classes=32` —
   32 categorical distributions of 32 classes each (matching the paper's own
   Remark B.1 description of DreamerV3's `z_t`). Theorem D.5's Gaussian-noise
   cases (additive or heteroscedastic) do not cover a categorical/discrete
   noise model at all — a third, independent reason this architecture is out
   of scope, on top of the recurrence issue.

**Verdict on item 2**: confirmed, not merely re-asserted. DreamerV3's RSSM
fails Theorem D.5's hypothesis on two independent grounds (recurrence,
discrete stochastic latent), either of which alone would exclude it. No
partial/degenerate-case rescue analogous to TD-MPC2's "deterministic is
secretly easy" argument applies here — recurrence and categorical latents
are structural, not limiting cases of something the theorem already covers.

---

## 3. Concrete Theorem D.3 query-count estimate, from real numbers

### 3.0 Correcting an input assumption before estimating

The starting brief referenced "TD-MPC2 是 256 維量級" — this traces to the
*paper's own* Table 7 (a GPU-pending, never-run, illustrative SafetyPointGoal1-v0
design target), not this project's actual checkpoint. Checked directly:

```
ckpt = torch.load('models/walker-walk-3.pt')['metadata']
-> {'latent_dim': 512, 'mlp_dim': 512, ...}
```

This project's real TD-MPC2 checkpoint is **d=512**, not 256. The estimate
below uses the real number. (This doesn't change the headline conclusion —
see the closing remark on why the ambient dimension isn't actually the
deciding factor here.)

### 3.1 L, computed from the real checkpoint (reusing Stage 2's `spectral_norm_bound()` unmodified)

```
core/lbsm/lipschitz.py::spectral_norm_bound() applied to sd['_dynamics.{0,1,2}.weight']:
  layer 0 (518->512): 70.86
  layer 1 (512->512): 57.98
  layer 2 (512->512): 78.20
  weights-only product:            321,230
  x Mish^2 (2 hidden activations):  380,255      (Mish L~=1.088 per layer, minor correction)
  + illustrative LayerNorm gamma factor: ~1.04e8  (NOT a rigorous bound -- gamma-scale only,
                                                     ignores the unbounded 1/sqrt(variance) term
                                                     flagged in §1; shown only to illustrate that
                                                     the true bound is at least this much larger,
                                                     not smaller)
```

For comparison, Stage 2's own trained (tiny, 32-hidden-unit, 300-epoch) toy
certificate network had `L_U ≈ 10.98` — **TD-MPC2's real, fully-trained
dynamics network's weights-only bound is already ~4 orders of magnitude
larger** than that toy example, before even accounting for LayerNorm.

### 3.2 D (latent diameter), derived from SimNorm's structure, not assumed

TD-MPC2's output layer is `SimNorm` (softmax over groups of `simnorm_dim=8`):
each group of 8 latent coordinates sums to 1 and is non-negative, i.e. each
group is a point on the 7-simplex, with L2 norm in `[1/sqrt(8), 1]`. With
512/8 = 64 such groups, `‖z‖ ∈ [sqrt(64/8), sqrt(64)] = [2.83, 8]`. Latent
space is therefore *compact* (a genuinely useful structural fact — Z is not
all of R^512), with diameter `D ≲ 16`.

### 3.3 Plugging into Theorem D.3: N = Õ(c^{d_eff} · (B̄/ε_min)²), c = Θ(D·L/ε_min)

Using `ε_min = 0.02` (Stage 2's own default training margin), `B̄ = 10`:

| L used | c = D·L/ε_min | N at d_eff=2 | N at d_eff=5 | N at d_eff=10 |
|---|---|---|---|---|
| weights-only (optimistic floor) | 2.57e8 | **1.65e22** | 2.80e47 | 3.14e89 |
| x Mish² | 3.04e8 | 2.31e22 | 6.51e47 | 1.70e90 |
| + illustrative LN factor | 8.30e10 | 1.72e27 | 9.84e59 | 3.88e114 |

**Headline result: even at the single most optimistic combination possible
— the loosest (weights-only) L estimate AND the lowest plausible d_eff (2,
matching Appendix E.2's own toy construction) — N ≈ 1.65×10²² resettable-
generative queries.** For scale, that is comparable to Avogadro's number
squared, or roughly 10⁴ times the number of grains of sand estimated to
exist on Earth. This is not a "large but conceivably reducible with
engineering effort" number; it is unconditionally outside anything a
verification pipeline could execute.

### 3.4 Why this conclusion doesn't hinge on resolving d_eff

SimNorm's simplex constraint (sum=1 per group) removes one degree of freedom
per group, so the *full-manifold* intrinsic dimension is already
64 × 7 = 448, not 512 — but that's still enormous, and says nothing about
the *reachable operative sublevel set*'s dimension, which could plausibly be
far lower if walker-walk's dynamics collapse onto a low-dimensional
attractor (a periodic gait) in latent space. This project has no measurement
of that reachable-set dimension, and it would be a substantial undertaking
to obtain one (box-counting an occupied region of a 512-dim space from
rollout data). The point of §3.3's table is precisely that **this
uncertainty doesn't matter for the feasibility verdict**: the constant `c`
alone, driven by the real trained network's Lipschitz product, is already
~9-11 orders of magnitude too large for the covering-net approach to be
viable even at the most forgiving d_eff=2. Resolving d_eff more precisely
would only make the number *larger*, never bring it into a feasible range.

---

## Overall verdict

Neither world model in this project supports a real Stage-3 L3 warrant today:

- **CarDreamer/DreamerV3**: excluded by Theorem D.5's own stated scope, on
  two independent structural grounds (recurrence, categorical latents),
  confirmed by reading the actual RSSM source, not just citing the paper.
  Not a "small derivation" gap — the paper's own words ("not covered without
  additional derivation") likely understate the distance, since the
  categorical-latent issue alone would need a materially different
  post-expectation formalism (softmax/categorical Jacobian, not a Gaussian
  reparameterization), independent of solving the recurrence problem.

- **TD-MPC2**: architecturally closer (feedforward, deterministic, single
  head) but blocked by two separate, independent issues: (a) LayerNorm's
  Lipschitz constant is not covered by Theorem D.5's clean composition bound
  and needs its own argument (a real, currently-open theoretical gap, small
  in scope but not yet closed anywhere in this project or the paper); and
  (b) even setting (a) aside entirely and using the most optimistic possible
  numbers, the trained network's actual spectral-norm product makes Theorem
  D.3's covering-net query count astronomically infeasible (≥10²²) at any
  plausible d_eff.

**Recommendation**: do not proceed with a Stage-3 implementation against
either checkpoint as they currently stand. If L3-on-a-real-model is still a
priority, the productive next questions are research-scoped, not
engineering-scoped: (i) does a Lipschitz-regularized retraining of TD-MPC2's
dynamics head (or a LayerNorm-free architecture variant) bring L down by the
~9-11 orders of magnitude needed, and (ii) is there a fundamentally different
certification strategy (e.g. exploiting problem-specific structure rather
than a generic covering net) that doesn't inherit Theorem D.3's
dimension-and-Lipschitz-driven cost at all. Both are open research
questions, not implementation tasks — consistent with why this stage
produced a report and not code.
