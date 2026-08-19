# Stage 5B feasibility report: does a 2-3D projection (height + velocity) rescue Stage 5? No.

**Verdict up front**: at `d_eff=2` (matching the projection's own dimension,
as instructed), **N ≈ 2.36×10²²** — not just still infeasible, but roughly
**9 orders of magnitude *worse*** than Stage 5's already-catastrophic 1-D
result (2.95×10¹³), and ~14 orders of magnitude past the "still >1e8, stop"
threshold. Stage B (repeat the full Stage 5→6 pipeline on this projection)
is **not entered**. A third dimension (joint angle) was **not attempted**,
per the explicit instruction to stop once 2D already blows the budget by
many orders of magnitude.

One finding along the way is genuinely mixed and worth surfacing honestly
rather than flattening into the headline "no" — see §2.

---

## 0. Data provenance (no fresh CEM sampling needed)

Everything below reuses already-persisted real data plus one small new probe
fit, per the Stage-2 discipline the task required if a new probe were
needed:

- `/tmp/tdmpc2_pilot_data.npz` — Stage 5's 7500-sample sequential dataset (15
  real episodes), reused unchanged for height and for the conditional-
  variance test.
- `/tmp/tdmpc2_anchors_with_physics_state.npz` — 200 real anchors with saved
  dm_control `physics_state` (18-dim), from earlier project work. Used here
  to reconstruct **exact, non-probe, ground-truth** observations via
  `TDMPC2Wrapper._obs_from_physics_state()` (`physics.set_state()` +
  `physics.forward()` — deterministic, lossless, **no CEM/randomness
  involved at all**: confirmed by reconstructing height this way and
  comparing to the file's already-stored `h` — max abs diff = 0.0).
- From the reconstructed observations, extracted **real torso horizontal
  velocity** = `obs[15]`, the root joint's horizontal `qvel` component
  (`dm_control/suite/walker.py::PlanarWalker.get_observation()`:
  `obs['velocity'] = physics.velocity()`, a 9-dim generalized-velocity
  vector immediately following `orientations`(14) and `height`(1) — index
  15 is the first entry, the planar walker's root-x velocity). Note this is
  the qvel-based root velocity, not identical to (but closely related to)
  the reward function's separate sensor-based `horizontal_velocity()`
  method — a minor caveat, not load-bearing for what follows.
- New **velocity probe**: `Ridge(alpha=10.0)` fit on these 200 real `(z, v)`
  pairs, seed=0 explicit, disjoint 60/40 train/calib split (mirrors Stage
  2's mandatory-disjointness discipline — `find_trajectory_overlap()`-style
  index-disjointness asserted before fitting).

## 1. The velocity probe itself is a decisive early signal

```
n_train=120  n_calib=80  HELD-OUT R^2 = 0.227   (MAE=0.216, v range [-1.26,1.01], std=0.368)
```

Compare to the height probe: **R²=0.9986**. Torso horizontal velocity is
**not** well captured by a linear projection of `z` — TD-MPC2's latent
representation, at least along this direction, does not encode velocity
anywhere near as cleanly as it encodes height. This alone is worth stating
plainly: the premise "just add a second linear probe direction" is already
on shaky ground before any dynamics question is even asked. Everything below
uses this admittedly-noisy probe (there was no better real option available
without fresh CEM-based data collection, which would reopen the §8.7b
seed-reproducibility problem) — treat the velocity-conditioned numbers below
as optimistic, not conservative, given how noisy the input is.

## 2. Conditional-variance test, extended to 2D — the mixed result

Reused Stage 5's exact method (bin by conditioning variables, compare
conditional spread of the next state to the unconditional spread), extended
from 1D (`h_t`) to 2D (`h_t`, probe-predicted `v_t`), on the same 7485 real
consecutive pairs, restricted to Stage 5's identified problem region
`h_t ∈ [0.385, 1.241]` (748 samples, a 3×3 grid over height × velocity
terciles):

| Target | Stage 5 (1D, height only) | Stage 5B (2D, height+velocity) |
|---|---|---|
| conditional std(h_{t+1}) / unconditional | **66.3%** | **15.3%** |
| conditional std(v_{t+1}) / unconditional | (not tested) | **83.8%** |

**Honest reading**: conditioning on velocity in addition to height
*substantially* tightens the prediction of next-height (66%→15%, now under
the task's own 30% stop-before-adding-more-dimensions threshold) — some
individual grid cells (n≈100-140 each, not just small-sample noise) show
`std(h_{t+1})` as low as 0.07-0.11 against an unconditional 0.53. Physically
this makes sense: knowing whether the walker is moving up or down
(velocity's sign) resolves most of the "falling vs. recovering" ambiguity
that made height-alone hopeless in Stage 5. **But** next-*velocity* itself
remains poorly constrained even by (height, velocity) jointly (84% of
unconditional spread) — velocity is not becoming self-predictable, and given
the probe estimating it only has R²=0.227 to begin with, this ratio is
likely inflated further by estimation noise on top of genuine
unpredictability.

So: if the only question were "does conditioning on (height, velocity)
statistically tighten next-height," the answer is a qualified yes. That is
not, however, the question Theorem D.3's covering-net cost depends on — see
§3.

## 3. L does not improve — and gets slightly worse, not better

The projection is now a matrix, not a vector, so the correct quantity is the
**projection matrix's spectral norm** (largest singular value), not a
per-row norm:

```
M = [height_probe.coef_ ; velocity_probe.coef_]     (2 x 512)
singular values: [0.7122, 0.5429]
||M||_2 = 0.7122
```

```
L_proj(2D) = ||M||_2 * L_dynamics_full = 0.7122 * 3,286,776 = 2,340,749
```

Compare to Stage 5's 1D `L_proj = 1,785,750`: **the 2D bound is 1.31x
*larger*, not smaller.** This is not a coincidence or a bad probe choice —
it is structural. Stacking a second (even noisy) output direction onto a
linear projection can only ever *increase or hold* the matrix's spectral
norm relative to any single row's norm; it can never decrease it, since the
spectral norm is a max over unit directions and the row space has only
grown. §2's conditional-variance improvement and this section's L are
answering genuinely different questions (empirical predictability of a
specific transition vs. worst-case sensitivity of the projection), and they
do not move together. This is the core reason the mixed finding in §2 does
not rescue feasibility.

## 4. Query count at d_eff=2 (matching the projection's own dimension, as instructed)

```
D_h = 1.3663 - 0.0437 = 1.323   (Stage 5's real height range, unchanged)
D_v = 1.0056 - (-1.2631) = 2.269   (real velocity range, from the 200 anchors)
D_2d = sqrt(D_h^2 + D_v^2) = 2.626
eps_min = 0.02, B = 10  (unchanged from Stage 3/5, for direct comparability)

c = D_2d * L_proj / eps_min = 2.626 * 2,340,749 / 0.02 = 307,349,160

d_eff=2:  N = c^2 * (B/eps_min)^2 = 2.36e+22
d_eff=3:  N = c^3 * (B/eps_min)^2 = 7.26e+30   (reported for reference only —
                                                  not reached, see below)
```

**This is worse than Stage 5's 1-D result (2.95×10¹³) by about 9 orders of
magnitude.** The reason is arithmetic, not a new finding about the dynamics:
moving from `d_eff=1` to `d_eff=2` squares `c` in the query-count formula,
and `c` itself did not shrink (§3) — it grew slightly. Adding a dimension to
the projection, even a dimension that helps predict the target observable
(§2), makes the *covering-net cost* worse, because the cost is driven by
`c^{d_eff}` and `c` is dominated by `L`, which a projection cannot reduce.

## Stopping decision

Per the explicit instruction ("查询数仍然超出上限若干个数量级(比如仍然
>10^8),直接停在这里写报告,不进阶段B"): N is ~14 orders of magnitude past
that bar. **Stage B is not entered.** A third dimension (key joint angle)
was not attempted — adding more projected dimensions can only repeat §3's
structural argument (spectral norm cannot decrease by adding rows) while
`d_eff` climbs to 3, making N worse again by roughly another factor of `c`
(~3×10⁸). There is no plausible number of added dimensions that reverses
this trend; the mechanism itself (linear projection composed with a fixed,
large, real `L_dynamics_full`) guarantees it gets worse, not better, every
time a dimension is added.

## Bottom line

Two independent, real-data-grounded findings, reported without collapsing
one into the other:

1. **(Encouraging, but not decisive)** Adding velocity as a second
   conditioning variable substantially improves *empirical* next-height
   predictability in the region that mattered in Stage 5 — a genuine,
   real-data finding, not assumed.
2. **(Decisive)** This does not translate into covering-net tractability,
   because Theorem D.3's cost is governed by worst-case Lipschitz sensitivity
   (`L`) and the covering exponent (`d_eff`), neither of which improves when
   you add a projected dimension — `L` cannot decrease (matrix spectral norms
   only grow as rows are added) and `d_eff` necessarily grows, so the net
   effect of "adding dimensions to help the property" is to make the formal
   verification cost *worse*, even in a case where it demonstrably helps the
   *statistical* question. This is the load-bearing conclusion for whether
   dimension-reduction-via-projection is a viable strategy at all for this
   system: it is not, and this is now confirmed at two different projection
   sizes (1D and 2D), not just asserted from the 1D case alone.
