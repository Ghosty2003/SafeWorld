# Stage 5 feasibility report: height-projection recurrence — infeasible, do not proceed to Stage 6

**Verdict up front**: even at the single most optimistic assumption
(`d_eff=1`), the required query count is **N ≈ 2.95×10¹³**, roughly **2.95×10⁸
times over** the 10⁵ feasibility threshold. Per the task's explicit stop
condition, Stage 6 is **not** unblocked. This report explains why, and why
the shortfall does not depend on resolving `d_eff` more precisely — the
premise itself (projecting the *property* to 1-D reduces the *dynamics'*
effective complexity) does not hold for this system, which is the real
finding here, not just a numeric miss.

---

## 1. Projection direction — reused, not redesigned

`wrappers/tdmpc2_probes.py::fit_height_probe()` on the real, already-
persisted pilot dataset (`/tmp/tdmpc2_pilot_data.npz`, 7500 real `(z, h)`
pairs from 15 real 500-step MPC episodes — predates this session, not
newly generated, so no fresh CEM sampling / seed question arises for this
part at all):

```
Ridge(alpha=10.0), coef_ shape (512,), ||w||_2 = 0.5433
in-sample R^2 = 0.99905  (matches the documented CV R^2=0.9986 closely enough
                           to confirm this is the same validated probe/dataset)
```

## 2. D — from the same real data, not guessed

`h` over all 7500 real samples: min=0.0437, max=1.3663 →
**D = 1.3663 − 0.0437 = 1.3226** (meters).

## 3. L — from Stage 4's *validated* per-layer bounds, not Stage 3's illustrative path

Per the task correction, this uses Stage 4's real, multi-seed-validated
LayerNorm bounds (8.36 / 1.08 / 0.96) folded into the full 3-layer dynamics
composition bound already computed there:

```
L_dynamics_full = 3,286,776   (Stage 4: weights x LayerNorm(validated) x Mish/SimNorm, all 3 layers)
```

Projecting through the linear height probe: for a linear functional `w·(·)`
composed with an `L`-Lipschitz map, `|w·(f(z1)-f(z2))| <= ||w||_2 * L *
||z1-z2||` (Cauchy-Schwarz) — so an upper bound on the projected map's
Lipschitz constant is:

```
L_proj = ||w||_2 * L_dynamics_full = 0.5433 * 3,286,776 = 1,785,750
```

**This is the crux of the negative result, so it needs to be said plainly**:
projecting the *output* onto 1 dimension only multiplies `L` by `||w||_2 ≈
0.54` — barely a factor of 2. It does **not** shrink `L` by anything close to
what would be needed. `L_dynamics_full` is a property of how sensitive the
*whole 512-dim dynamics* is to its input; restricting attention to a 1-D
*observable* of the output does not make the underlying map itself less
sensitive. The premise "narrow the property to make the covering-net cost
tractable" conflates two different notions of dimension (the property's
dimension vs. the dynamics' effective complexity) — see §4.

## 4. d_eff — measured on real data, and it does NOT default to 1

This is the substantive finding, not a formality. Using the pilot dataset's
trajectory structure (`ep`, `step` fields — 15 episodes × 500 steps, 7485
real consecutive `(z_t, z_{t+1})` pairs, entirely from already-collected
data, no fresh sampling needed), I tested the premise a 1-D height reduction
requires: **does knowing only `h_t` (not the other 511 latent dimensions)
tightly constrain `h_{t+1}`?** If yes, a 1-D covering net over the height
axis is a meaningful reduction. If no, states that share a height but differ
elsewhere behave very differently, and the "1-D" reduction is only nominal —
the real complexity the covering net must still handle lives in the full
latent space.

Binned by `h_t` (10 real-data deciles), comparing conditional `std(h_{t+1})`
within each bin to the unconditional `std(h_{t+1}) = 0.534`:

| `h_t` range | n | std(h_{t+1}) within bin | ratio to unconditional |
|---|---|---|---|
| [0.044, 0.107] | 749 | 0.024 | 0.045 |
| [0.107, 0.168] | 748 | 0.039 | 0.073 |
| [0.168, 0.251] | 749 | 0.047 | 0.088 |
| [0.251, 0.385] | 748 | 0.054 | 0.100 |
| **[0.385, 1.241]** | **748** | **0.354** | **0.663** |
| [1.241, 1.265] | 749 | 0.018 | 0.033 |
| [1.265, 1.281] | 748 | 0.018 | 0.034 |
| [1.281, 1.293] | 749 | 0.015 | 0.028 |
| [1.293, 1.301] | 748 | 0.011 | 0.021 |
| [1.301, 1.366] | 749 | 0.019 | 0.035 |

**Sharp, physically-interpretable split**: near the nominal standing/walking
height (bins with `h_t ≳ 1.24`), height alone predicts next-height tightly
(ratio 2-3%) — the reduction looks reasonable *there*. But the wide
transitional bin `h_t ∈ [0.385, 1.241]` — the walker mid-fall or
mid-recovery — has conditional spread **66% of the unconditional spread**:
knowing the current height barely constrains the next height at all in this
regime. This makes physical sense: whether a walker at height 0.7 is falling
toward 0 or recovering toward 1.3 depends on velocity and joint configuration
— information the height scalar alone discards. In this regime the
"effective dimensionality" the covering net must handle is **not reduced
from 512 to 1** — it is closer to still needing most of the original state.

**This directly matters for the actual target property**: `h_min = 0.6`
(reused unchanged, per the "don't re-pick thresholds after the fact"
principle) sits *inside* the badly-behaved `[0.385, 1.241]` bin. A
`□◇(height ∈ [h_min, ...])`-style recurrence property is precisely a claim
about behavior in the region where the 1-D reduction is *least* valid — the
one place this shortcut cannot legitimately be applied to the property it
was meant to certify.

## 5. Query count, and why the shortfall is decisive rather than borderline

Using `eps_min = 0.02` (Stage 2/3's own convention, kept for comparability)
and `B̄ = 10` (same illustrative scale as Stage 3):

```
c = D * L_proj / eps_min = 1.3226 * 1,785,750 / 0.02 = 118,091,669
```

| d_eff | N = c^d_eff * (B̄/eps_min)² | vs. 1e5 threshold |
|---|---|---|
| 1 (most optimistic possible) | **2.95×10¹³** | 2.95×10⁸x over |
| 2 | 3.49×10²¹ | — |
| 3 | 4.12×10²⁹ | — |

Per the task's explicit instruction ("超出這個範圍要停下匯報實際數量級,不要
自行決定還是試試看"): **N is not in the executable range at any tested
d_eff, including the best case. Stopping here — Stage 6 is not approved to
proceed.**

The reason this doesn't hinge on nailing down `d_eff` more precisely (echoing
Stage 3's own closing point): `d_eff` only sets the *exponent*. The *base*,
`c ≈ 1.18×10⁸`, is already ~8 orders of magnitude too large on its own
(comparing `c` directly against the 1e5 threshold at `d_eff` effectively
≈1). No plausible refinement of `d_eff` rescues that gap; only a
fundamentally smaller `L` would, and §3 already showed the projection
strategy does not deliver one.

## 6. Seed discipline (per the Stage-5 addendum)

No fresh CEM-based sampling was needed anywhere in this analysis — the probe
fit, `D`, and the `d_eff` fiber-uniformity check all reused the existing,
already-collected `/tmp/tdmpc2_pilot_data.npz` (15 real episodes, predating
this session). This sidesteps the CEM/`cfg.seed` non-reproducibility issue
documented in `EXPERIMENT_CONFIG.md` §8.7b entirely for Stage 5's own
numbers — worth stating plainly rather than leaving it ambiguous whether
that gap affects this report (it does not, because no new sampling occurred).
If a future stage needs fresh TD-MPC2 rollouts, §8.7b's finding applies and
multi-seed cross-validation (as done there) is required before trusting a
single run.

## Bottom line

The height-projection strategy does not rescue Stage 3's infeasibility
conclusion. The reason is structural, not a matter of insufficiently
optimistic constants: reducing the *property* to a 1-D observable does not
reduce the *dynamics'* effective Lipschitz constant (§3), and empirically,
the height observable is **not even an approximately-Markovian reduction**
in the exact region (`h ∈ [0.385, 1.241]`, containing `h_min=0.6`) that the
target safety property is actually about (§4). Both of these are findings
about this specific system (TD-MPC2 walker-walk), grounded in real data, not
generic pessimism about projection strategies in the abstract.
