# SafeDreamer L3: paper-premise candidate experiment

Entry: `experiments/l3_safedreamer_paper.py`.

This implements candidate fitting and independent **pointwise** validation
for the H1 route of Theorem 5.5 in `_ICLR_27_VeriWorld(1).pdf`. It is NOT
a completed high-dimensional SafeDreamer infinite-horizon proof backend.
The top-level `paper_L3.verdict` is fail-closed `ABSTAIN` until such a
backend exists. There is no flag to promote sampled success to a theorem.

## Fixed semantics

- Formula: `GF G[0,47](decoded_hazard_margin >= 0)`.
- Hazard margin: minimum decoded 2-D hazard distance minus 0.20.
- A 49-state counter records consecutive safe sampled states, saturated at
  48 and reset to zero by a hazard. Count 48 is accepting but NOT absorbing.
- Input: full RSSM latent (512), planner mean/std (64), planner-initialized
  bit (1), and counter/48 (1): 578 coordinates. No elapsed-time countdown.
- Model and local CCEPlanner configuration match the frozen L2 experiment.
  Planner's own cost predicate is unchanged; it need not equal the spec AP.
- Initial distribution is still restricted to decoded goal distance >=1.0.
  Rejection uses the initial state only, never future success.

## Data and computation

Default pilot: 12 training paths and 20 new validation paths; horizon 192;
32 independently randomized one-step successors per anchor; 500 CPU training
epochs for the small W/U heads. Model/planner imagination runs on JAX GPU.
This is a pilot, not a high-powered benchmark or proof-complete run.

Each path supplies one uniformly selected anchor at t in [1,H-1]. The
training role may also use its other states for normalization and shaping.
W/U are frozen before validation paths are drawn. At each query the same
RSSM AND planner carry are restored; branch RNGs differ. Successors are
not filtered on acceptance or membership in the candidate region.

W/U are bounded sigmoid networks evaluated independently at both endpoints.
The training loss penalizes expected W increase outside acceptance and U
increase everywhere. Counter-based shaping is a heuristic against collapse,
not evidence of invariance. Candidate C is `{U<=0.8}`. There is NO asserted
invariant core `{counter>=24}`; in particular U checks are not bypassed at
accepting states.

Frozen validation uses the paper's Hoeffding sampling allowance at each
anchor, with delta_i = 0.05/(2*n) for both functions. It reports raw empirical
drift and conservative pointwise upper bounds separately. It does not assume
zero interpolation error: interpolation/covering is explicitly missing.
With kappa=32 and small desired margins, these upper bounds can be very
conservative. That is insufficient evidence, not proof of unsafe dynamics.

## Output distinctions

- `validation_finite_behavior`: observed finite windows on new paths.
- `independent_point_validation`: conditions at the finite sampled anchors,
  with simultaneous confidence conditional on frozen certificates and
  independent successor queries. No claim about an entire trajectory.
- `paper_L3`: formal audit. Missing closed-loop cellwise bounds, covering,
  containment, H1 validity throughout C, initial support, initial U expectation,
  and collar-entry bound prohibit an infinite-horizon warrant.

The simulator reset RNG is not fully controlled by imagination seeds; this
limits bit-exact path regeneration and is recorded in the plan. Exact saved
paths, latent/planner state, branch outputs, seeds and hashes are retained.
`--evaluate-only` verifies artifact/code hashes, restores the frozen networks,
and recalculates the report without GPU sampling or fitting. Set
`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1` when replaying: the experiment fits
and evaluates with two Torch threads. The strict exact-report comparison can
reject tiny floating-point differences under a different reduction/thread
configuration; it does not silently replace the original report.

## Commands (from repository root)

Use the SafeDreamer environment with the previously working JAX CUDA setup:

```bash
python experiments/l3_safedreamer_paper.py --smoke \
  --output artifacts/safedreamer_l3_paper_smoke_new
python experiments/l3_safedreamer_paper.py \
  --output artifacts/safedreamer_l3_paper_pilot_new
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python experiments/l3_safedreamer_paper.py --evaluate-only \
  --output artifacts/safedreamer_l3_paper_pilot_new
```

Output directories must be new. Existing results are never overwritten by
the runner. `--smoke` uses 2 fit / 3 validation paths, horizon 60 and kappa 4
solely to test wiring. Original L2 code/results and the main branch are not
modified by these commands.
