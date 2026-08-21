# SafeDreamer L1 verification pipeline

This directory holds the L1 (bounded-STL) verification pipeline for
`wrappers/safedreamer_wrapper.py`, plus a few diagnostic scripts written
while stress-testing it against a real SafeDreamer (PKU-Alignment)
`osrp_vector` checkpoint on `SafetyPointGoal1-v0`.

## Setup

```bash
conda activate safedreamer   # must have SafeDreamer's own deps (jax, ninjax, etc.)
```

`wrappers/safedreamer_wrapper.py` expects a config dict (see `BASE_EXTRA` in
`l1_pipeline.py`) with `repo_root`, `checkpoint_path`, `method`, `task`.
CPU is forced via `config_overrides: {"jax": {"platform": "cpu"}}` — GPU
(`ptxas`) failed in this environment; adjust if yours works.

## Files

| File | What it does |
|---|---|
| `l1_pipeline.py` | Core `run_l1()` — the 3-way split (N_cal → q̂_δ, N_err → ĉ_err, N_test → held-out check) for one STL spec. |
| `l1_all_stl_n500.py` | Runs all 7 grounded STL specs at N=500, generating rollout data once and reusing it across specs. |
| `l1_repeat_coverage.py` | Repeats `run_l1()` K times with disjoint seeds to empirically check coverage of the claimed `1-δ_cp-δ_err` bound. |
| `model_belief_check.py` | Raw STL satisfaction rate of the model's own imagined rollouts (no calibration) — for comparing against real-env satisfaction. |
| `hazard_proximity_error_check.py` | Bucket-by-hazard-distance diagnostic for whether imagination error is larger near vs. far from hazards. |
| `onesided_horizon50_pilot.py` | Pilot comparing the standard two-sided `ĉ_err` (`|model-real|`) against a one-sided variant, at the spec's native horizon=50. |

## Worked example (N_test scenario)

Running `l1_pipeline.py` directly (`N_cal=N_err=N_test=20`, `stl_hazard_avoidance`,
horizon=10) prints:

```
[stl_hazard_avoidance] N_cal=20 N_err=20 N_test=20
  q_hat_delta_cp = -0.1433
  c_hat_err      = 0.7560
  rho_net        = -0.8993  -> VIOLATION
  claimed lower bound (1-delta_cp-delta_err) = 0.900
  N_test empirical real-env satisfaction     = 497/500 (99.4%)   # N=500 run
  bound holds on this test set: True
```

How to read this:

- `q_hat_delta_cp` (q̂_δ): a statistically-guaranteed lower bound on the
  model's own imagined-rollout robustness — the ⌊δ_cp·(n+1)⌋-th smallest of
  the N_cal margins, **not** an average.
- `c_hat_err` (ĉ_err): a statistically-guaranteed upper bound on how far the
  model's imagined trajectories can diverge from paired real-environment
  trajectories — the ⌈(1-δ_err)·(n+1)⌉-th smallest of the N_err per-pair
  distortions.
- `rho_net = q_hat - c_hat_err`. `rho_net > 0` → **WARRANT** (statistically
  certified transfer to the real environment, confidence ≥ 1-δ_cp-δ_err).
  `rho_net ≤ 0` → **VIOLATION** (no certificate — not necessarily unsafe in
  reality, just not provable at this sample size/spec/horizon).
- `N_test` is *never* used to compute `q_hat`/`c_hat_err`/the verdict — it is
  purely a held-out sanity check on whether the claimed bound actually held
  against fresh real-environment data.

At N=500 across all 7 grounded STL specs (`l1_all_stl_n500.py`, horizon=10),
every spec came back VIOLATION despite 3 of them showing 94–99% real-world
satisfaction on N_test. Root causes traced this session (see inline
docstrings for detail):

1. **Imagination is noisier than real physics by construction** — the RSSM
   samples a categorical latent every step; MuJoCo evolves deterministically
   given a fixed action sequence. This noise is not a training defect.
2. **STL's `G` (always) operator takes a `min` over time, and `hazard_dist`
   is itself a `min` over 8 hazards** — nesting two `min`s over noisy inputs
   is a textbook selection-bias amplifier (Jensen's inequality for concave
   functions): `E[min(noisy candidates)] ≤ min(true candidates)`, and the
   effect grows with the number of candidates (8 × horizon).
3. **`ĉ_err`'s distortion metric is symmetric (`|model-real|`)**, which for
   un-negated `G`-type safety specs penalizes the model being *too
   pessimistic* just as much as *too optimistic* — only the latter direction
   can produce a false safety claim. `onesided_horizon50_pilot.py` shows a
   one-sided variant cuts `ĉ_err` by ~40% on `stl_hazard_avoidance`, though
   not enough alone to flip the verdict at N=100/horizon=50.
4. **horizon=10 vs. the specs' native horizon=50/55 matters a lot**: at the
   correct horizon=50, `stl_hazard_avoidance`'s real-env satisfaction itself
   drops from 99.4% (horizon=10, N=500) to 69% (horizon=50, N=100) — much of
   the earlier "reality is basically always safe" impression was an artifact
   of the truncated horizon giving random actions too little time to reach
   any hazard.
5. **All rollouts in this pipeline use `action_source="random"`** —
   `sample_paired_rollouts()` does not implement CCEPlanner-driven pairing
   (would require lockstep real-env/shadow-imagination stepping with
   per-step replanning; not implemented). So none of these results say
   anything about how well imagination predicts SafeDreamer's actual
   deployed safety-aware policy — only about a randomly-acting baseline.

## Known scope limits

- Every specific number above is from **one checkpoint**
  (`20240307-010600_osrp_vector_safetygymcoor_SafetyPointGoal1-v0_0.ckpt`) on
  **one task** (`SafetyPointGoal1-v0`) — not a general claim about
  SafeDreamer or SAFEWORLD.
- `core/lppm/calibrator.py`'s `_clopper_pearson_lower()` is not a real
  Clopper-Pearson interval; irrelevant here since no spec used LPPM, but
  flagged for anyone extending this to unbounded specs.
