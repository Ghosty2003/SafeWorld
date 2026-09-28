# SafeDreamer experiment guide

## Final L2 results (2026-09-28)

These completed runs are separate from the development protocols below.
They use a fixed SafeDreamer checkpoint, `CCEPlanner.policy()`, and model-only
RSSM imagination for `SafetyPointGoal1-v0`. Initial eligibility is decoded goal
distance >= 1.0; no future-success filtering is allowed. Specification, monitor,
V, eta=0.01, thresholds, horizon and gates are frozen before 500 new calibration
paths and 1,000 independent Test1 paths. No test-driven adjustment is made.

P1 checks `V(x') <= V(x)`; P2 checks `V(x') <= V(x) - eta` when the source
automaton state is bad. The complete event ANDs all relevant P1/P2 transitions,
task completion, endpoint/sublevel conditions, sampled closure, and all enabled
region/numerical gates. Sampled checks are not support-wide verification.

| Experiment | H | Calibration certificate | CP lower (95%) | Verdict | Test1 certificate | Test1 CP lower (95%) |
|---|---:|---:|---:|---|---:|---:|
| 48-state hazard-free window | 64 | 493/500 | 0.9739 | SAFE (finite model-only) | 970/1000 | 0.9595 |
| LOW speed then HIGH speed | 300 | 446/500 | 0.8664 | ABSTAIN | 898/1000 | 0.8809 |

The warrant threshold is 0.95. Test1 is reported separately and does not choose
the verdict. Confidence bounds are individual, not jointly 95% across metrics.

Window specification: `F[1,17] G[0,47](not hazard)`. The start is in steps 1..17,
with 48 hazard-free sampled states ending no later than step 64. Frozen V:
`linear_max4`, training update 4800. Task completion was 500/500 and 1000/1000,
but full certificates were not 100%.

Motion specification: decoded speed <= 0.22450116704240514 followed at a
strictly later state by speed >= 0.7390221068978473, by step 300. The monitor
waits for LOW, then HIGH, then remains accepting. This is bounded completion,
not recurrence or remaining fast forever. The selected V reached 19/20 internal
development certificate successes, **not double-100% qualification**; the final
run was a separately authorized evaluation.

| Motion metric | Calibration (N=500) | CP lower (95%) | Test1 (N=1000) | CP lower (95%) |
|---|---:|---:|---:|---:|
| Task completion | 500/500 | 0.9940 | 1000/1000 | 0.9970 |
| P1 whole-path pass | 450/500 | 0.8751 | 903/1000 | 0.8862 |
| P2 whole-path pass | 448/500 | 0.8708 | 899/1000 | 0.8819 |
| P1 AND P2 | 448/500 | 0.8708 | 899/1000 | 0.8819 |
| Complete certificate | 446/500 | 0.8664 | 898/1000 | 0.8809 |

Task success does not imply certificate success. Absorbing monitor acceptance
means the bounded task is completed, not that future physical states stay safe.

### Entry points and audit records

| Purpose | Entry point | Local directory under `artifacts/` |
|---|---|---|
| Frozen window calibration/Test1 | [l2_clear_window_revalidation.py](l2_clear_window_revalidation.py) | `safedreamer_l2_clear_window_revalidation_v1/` |
| Frozen motion calibration/Test1 | [l2_motion_final.py](l2_motion_final.py) | `safedreamer_l2_motion_final_v1/` |
| Motion V development | [l2_motion_v_wide_search.py](l2_motion_v_wide_search.py) | `safedreamer_l2_motion_v_wide_v1/` |
| Motion V development audit | [l2_motion_v_wide_audit.py](l2_motion_v_wide_audit.py) | Same development directory |

Final directories contain `RESULT_CN.md`, reports and `audit.json`. Motion also
contains `final_table.csv`, `calibration_report.json`, `test_report.json`, and
the prospective calibration forecast `prediction.json`. Raw paths, weights
and hashes are retained locally for replay. A source-only checkout excludes
large checkpoint/weight files, `.npz` arrays and logs via `.gitignore`; obtain
the matching frozen inputs before reproduction. Inspect the entry point and
saved plan first, and use a new output directory rather than overwrite records.

### L3 / finite-recurrence development is not a final infinite proof

See [L3 spec ablation](L3_SPEC_ABLATION.md), [W/U ablation](L3_WU_ABLATION.md),
the [paper-aligned pilot](L3_SAFEDREAMER_PAPER.md), and
[finite recurrence](SAFEDREAMER_FINITE_RECURRENCE.md).
These investigate candidate learning, conditional-successor drift, causal
events and budget generalization. Finite event counts or procedural gates do
not replace expected-drift, coverage, containment and retention premises.
`finite_recurrence_motion_dev.py` is a development entry point, not a reported
completed final evaluation. No new run is implied by this documentation update.

## SafeDreamer L1 verification pipeline

## Current default: CCE policy, model-only bounded STL

`l1_pipeline.py` and `l1_all_stl_n500.py` now call `l1_policy_stl.py`.
Actions come from the checkpoint's native OSRP-Vector CCEPlanner; RSSM generates
the imagined successors. The simulator supplies reset observations only.
No random-action paired environment error budget is reused.

```bash
conda activate safedreamer
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
python experiments/l1_pipeline.py --device gpu --n-cal 100 --n-test 100 \
  --specs stl_hazard_avoidance --output artifacts/stl_cce_hazard_run1
```

For all seven grounded formulas use `l1_all_stl_n500.py` with a new `--output`
directory (defaults to 500 calibration and 500 test paths). The horizon is
computed from the **formula**, including nested intervals: hazard avoidance
needs states 0..49 (49 transitions); all seven need states 0..54. Explicit
shorter horizons are rejected, not silently truncated. `--horizon 64` is also
allowed but does **not** alter the formulas' time bounds.

Before sampling, the program saves specification, checkpoint hash, policy,
threshold and split seeds. It saves calibration predictions before generating
the independent test split; paths/actions and reports are retained. It reports
a one-sided Clopper–Pearson probability lower bound for positive STL robustness.
`MODEL_STL_WARRANT` means that bound meets the predeclared threshold (default
0.95); otherwise `NO_WARRANT`. This is **not** real-environment transfer, an
infinite-horizon proof, or a guarantee about the number of successes in one
future test batch. For multiple preselected specs, Bonferroni allocation gives
simultaneous calibration confidence `1-gamma` under independent, same-law
trajectory sampling. No selection or tuning on the final test split is allowed.

The old `run_l1()` API / `BASE_EXTRA`, `l1_repeat_coverage.py`, and
`l1_all_stl_random_legacy.py` remain explicitly **legacy random-action** tools
for historical reproducibility. Their horizon=10 results truncate longer STL
formulas and must not be interpreted as full-formula policy verification.
The older tables and command descriptions below describe those historical
transfer experiments, not the new default policy entry points. Existing frozen
L2/L3 experiments, wrapper implementation and saved results were not changed.

This directory holds the L1 (bounded-STL) verification pipeline for
`wrappers/safedreamer_wrapper.py`, plus a few diagnostic scripts written
while stress-testing it against a real SafeDreamer (PKU-Alignment)
`osrp_vector` checkpoint on `SafetyPointGoal1-v0`.

## Setup

**1. Get SafeDreamer itself** (this repo does not vendor it):
```bash
git clone https://github.com/PKU-Alignment/SafeDreamer.git
```
This wrapper was built and tested against the `osrp_vector` method on
`SafetyPointGoal1-v0` specifically — other methods/tasks are untested and
may need wrapper changes.

**2. Get a checkpoint.** Not included in this repo (training artifact, not
core code). Either train your own following SafeDreamer's own instructions,
or obtain a pretrained one separately. All numbers in this README came from
`20240307-010600_osrp_vector_safetygymcoor_SafetyPointGoal1-v0_0.ckpt`.

**3. Build the conda env** with SafeDreamer's own dependencies. The exact
versions this session's results were produced with:
```bash
conda create -n safedreamer python=3.8
conda activate safedreamer
pip install -r /path/to/SafeDreamer/requirements.txt
```
Key pinned versions (`pip freeze` from the working env — JAX in particular
is very version-sensitive, so mismatches here are a likely source of
non-reproduction even with the same checkpoint):
```
jax==0.3.25
jaxlib==0.3.25+cuda11.cudnn82
mujoco==2.3.3
gymnasium==0.28.1
gymnasium-robotics==1.2.2
tensorflow==2.12.0
tensorflow-probability==0.20.1
numpy==1.23.5
safety_gymnasium @ git+https://github.com/PKU-Alignment/safety-gymnasium.git@ae966e511b9927f06b39c727ca5c650136a4e696
```
(Follow SafeDreamer's own README for the current recommended setup if these
versions become unavailable — not duplicated in full here to avoid drifting
out of sync with upstream.)

**4. Point this repo at your SafeDreamer clone and checkpoint** via env vars
(defaults to `~/Documents/SafeDreamer` and the checkpoint path above if
unset — override for any other machine/layout):
```bash
export SAFEDREAMER_REPO_ROOT=/path/to/your/SafeDreamer
export SAFEDREAMER_CHECKPOINT_PATH=/path/to/your/checkpoint.ckpt
```

`wrappers/safedreamer_wrapper.py` expects a config dict (see `BASE_EXTRA` in
`l1_pipeline.py`) with `repo_root`, `checkpoint_path`, `method`, `task`.
CPU is forced via `config_overrides: {"jax": {"platform": "cpu"}}` — GPU
(`ptxas`) failed in this environment; adjust if yours works.

**Platform note**: JAX (SafeDreamer's core dependency) is officially
supported on Linux/macOS only — Windows users should run this under WSL2
rather than native Windows.

**Before running anything**, read the module docstring and inline comments
in `wrappers/safedreamer_wrapper.py` — it documents several non-obvious
JAX/ninjax pitfalls hit while building this (e.g. `import SafeDreamer.ninjax`
vs bare `import ninjax` being two different module objects with separate
state, why `nj.jit()` is required over raw `jax.jit()`, and why CCEPlanner —
not `task_behavior.ac.actor` — is what `osrp_vector` actually dispatches to
at eval time). This README doesn't restate those; the wrapper's comments are
the source of truth.

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

## L2 training and calibration diagnostics

`l2_safedreamer.py` collects separate training, calibration, and paired-error
splits. It requires PyTorch to train a NeuralLPPM. For hazard avoidance, run:

```bash
OMP_NUM_THREADS=1 python -u experiments/l2_safedreamer.py \
  --spec ltl_hazard_avoidance --horizon 50 \
  --n-train 60 --n-cal 40 --n-err 40 --epochs 300 --seed 1 \
  --diagnose-after-violation
```

Normally, `verify()` returns immediately when it finds an invariant
counterexample. The opt-in `--diagnose-after-violation` flag continues
NeuralLPPM fitting and held-out calibration for unbounded Safety specs after
that return. It reports the training backend/loss, certificate-event success
count, P1/P2 violation counts, `p_hat_gamma`, and sampled `Z_free` closure.
The original safety verdict and witness are retained. A completed diagnostic
run does not imply a successful certificate; sampled closure is not a
support-wide closure proof. The existing L1 calibration and horizon issues
are not resolved by this mode.

`l2_safedreamer_v_sweep.py` collects one fixed train/calibration split and
reuses it to compare constructed constant candidates with several NeuralLPPM
widths and initialization seeds. It also reports the function-independent
certificate ceiling imposed by calibration trajectories that end in the
absorbing odd Safety trap:

```bash
MUJOCO_GL=egl python experiments/l2_safedreamer_v_sweep.py

# Repeat only the function fitting/evaluation on the saved identical rollouts.
MUJOCO_GL=egl python experiments/l2_safedreamer_v_sweep.py --reuse-rollouts
```

The cache and machine-readable report are written under
`artifacts/safedreamer_l2_diagnostics/`. This is a diagnostic sweep only; it
does not allow a fitted candidate to override a concrete model/environment
Safety counterexample.

## Full-latent reachability trial

`l2_safedreamer_latent_trial.py` is the corrected experimental path for
`ltl_goal_reach` (`F(goal_dist < 0)`). `sample_latent_rollouts()` retains
51 actual RSSM states for 50 imagined transitions, including `deter`,
`stoch`, distribution parameters, decoded observations, and grounded APs.
The product state at t has consumed the label at t. The last observation
therefore affects acceptance without adding an artificial dynamics edge.
The older AP-only experiment and its synthetic-terminal logic are legacy;
this entry point uses `core/lppm/latent_reachability.py` instead.

```bash
# Fresh data and experiment (use a new --out directory for another run).
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/safeworld-mpl OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=2 python experiments/l2_safedreamer_latent_trial.py

# Resume only frozen-candidate evaluation; does not refit or reselect.
OPENBLAS_NUM_THREADS=1 python experiments/l2_safedreamer_latent_trial.py --phase evaluate
```

Splits are 60 training, 40 validation, 100 calibration, and 100 test paths,
with new imagination RNG seeds. The action distribution remains random;
initial states come from simulator resets, with no environment trajectory
comparison. Input normalization is fitted on training only and stored with
each model. The waiting branch is positive and the accepting branch is zero.
MLP and Gaussian-kernel candidates are selected using validation after a
zero-training-violation preference. `frozen_selection.json` is saved before
calibration/test evaluation. The kernel's auxiliary targets are finite
training-graph ranks, not true arrival-time labels for censored paths.

Results are saved under `artifacts/safedreamer_l2_latent_trial/`. The report
distinguishes goal-reaching from the full P1/P2-plus-terminal event and
reports a one-sided 95% Clopper-Pearson lower bound. Event-rate error is not
classification accuracy. Training interpolation and finite-sample closure
do not establish a support-wide deductive LPPM.

### Additional V search with fresh confirmation

`l2_latent_v_search.py` compares continued tanh and wider SiLU networks,
using either the complete latent or its deterministic component plus the
three grounded APs. It optimizes transition inequalities directly; it does
not regress on time-to-truncation labels or provide time as an input.
Forty checkpoints are assessed on the existing train/validation splits.
The fixed selection rule prefers <=1% training P2 violations, then validation
certificate successes excluding initially satisfied goals, then validation
P2 pass rate. Any residual training violations remain in the report.

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/safeworld-mpl OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=2 python experiments/l2_latent_v_search.py --phase all
```

The selected checkpoint is frozen and hashed before collecting 100 fresh
confirmation paths (imagination RNG seed 9307). Both it and the old Gaussian
baseline are evaluated on the same fresh paths. Earlier test/calibration
splits are explicitly exploratory reuse; inferential confidence bounds are
reported only for fresh confirmation. Outputs are under
`artifacts/safedreamer_l2_v_search/`. Use `--phase evaluate` to reload the
frozen model and reproduce the report without refitting or resampling.

### Coverage expansion and pathwise refinement

`l2_latent_v_refine.py` adds 120 fresh training and 60 validation paths to
the original 60/40 split; prior calibration/test paths never enter fitting
or selection. It starts from the previously frozen SiLU model and compares
path-maximum violation loss, a larger *training* margin, and top-violation
loss. The evaluation margin remains eta=0.01. All sampled training paths
are retained, including those that do not reach goal. Input preprocessing
remains frozen from the original training split to preserve the pretrained
model's feature coordinates.

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/safeworld-mpl OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=2 python -u experiments/l2_latent_v_refine.py --phase all
```

Selection uses the expanded validation split, after preferring candidates
with at most 1% training P2 violations. Checkpoints and the previous model
are preserved. One hundred fresh confirmation paths are collected after
freezing; both models are evaluated on those same paths. Results are under
`artifacts/safedreamer_l2_v_refine/`. Rates on reused test paths are
exploratory; only the fresh confirmation receives probability bounds.

### Prospective prediction check

`l2_latent_prediction_check.py` freezes the refined candidate, collects
100 new calibration imagination paths, and writes `prediction.json` before
collecting the next 100 test paths. It compares the estimated and observed
rates of the same events: goal reaching, P1/P2 path success, and the full
terminal certificate. It does not compare to environment trajectories or
refit/select any function. Each path has 50 actual model transitions.

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/safeworld-mpl OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=2 python -u experiments/l2_latent_prediction_check.py
```

The comparison reports expected and actual counts, percentage-point errors,
Clopper-Pearson intervals, and a descriptive two-sample Fisher test.
A probability lower confidence bound is not a guaranteed success count for
the next batch. Outputs are in `artifacts/safedreamer_l2_prediction_check/`;
`--evaluate-only` reproduces the comparison using the saved frozen prediction.

### Scale and ensemble V search

`l2_latent_v_ensemble.py` compares continued training, positive scales
1/2/4/8, and mean/min/max combinations on the existing 180/100 train/validation
split. The threshold remains eta=0.01: scaling increases positive descent
margins but cannot turn an increase into a decrease. All candidate waiting
values stay positive and the accepting automaton branch is fixed at zero.
Neither time nor future trajectory information is an inference input.

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/safeworld-mpl OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=2 python -u experiments/l2_latent_v_ensemble.py --phase all
```

The selected bundle is frozen before drawing 100 fresh confirmation paths
(seed 12307); the previous refined network is evaluated on the same paths.
Old confirmation/prediction test sets are reported only as exploratory
reuse. Saved bundles include all component networks, normalization, the
combination rule, and scale. Outputs are under
`artifacts/safedreamer_l2_v_ensemble/`; `--phase evaluate` reloads that exact
bundle. Lower P2 violation rates alone do not imply a higher full-path event
rate or an L2 warrant.

### Validation-only representation search (no testing)

`l2_latent_v_validation_search.py` fits state-only candidates on the existing
180 training paths and repeatedly selects checkpoints/hyperparameters on the
100 development validation paths. It tries train-fitted PCA, multi-step
descent losses on real training edges, input-noise regularization, and
ensembles. Eta remains 0.01. Validation paths never enter gradients or PCA;
there is no calibration/test-loading or rollout-collection phase.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python -u experiments/l2_latent_v_validation_search.py
```

Outputs are under `artifacts/safedreamer_l2_validation_search/`. Existing
outputs are not overwritten. Selection prefers <=1% training P2 violations,
then minimizes validation P2 violations; it retains the previous baseline
unless improved. Saved models include preprocessing and combination rules.
Rates on repeatedly selected validation data are development results, not
independent generalization estimates; confidence bounds are omitted. A zero
validation count, if achieved, would not itself establish an L2 warrant.

### Frozen latest-V prospective check

`l2_latest_v_prediction_check.py` copies the exact selected validation-search
bundle (including its preprocessing) into a new artifact directory, then
collects 100 fresh calibration and 100 fresh test imagination paths. The
calibration forecast is written and hashed before test collection starts.
No model fitting, checkpoint selection, or threshold adjustment is performed.
The task remains decoded goal radius 0.30, 50 actual transitions, and random
actions. The world-model checkpoint is unchanged; new RNG seeds are recorded.

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/safeworld-mpl OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=2 python -u experiments/l2_latest_v_prediction_check.py
```

Artifacts are in `artifacts/safedreamer_l2_latest_prediction_check/`. Existing
runs are not overwritten. `--evaluate-only` checks the saved hashes and
reproduces the report without drawing paths. Certificate events and direct
goal visits get separate statistics. The configured 80% acceptance check is
separate from statistical validity; this script does not perform a complete
warrant audit or establish support-wide deductive validity. Its Z_free is the
structurally absorbing goal-done product branch, not a learned physical zone.

### Paper-style learned sublevel, finite-prefix experiment

`l2_paper_sublevel_finite.py` is a separate experiment implementing the LPPM
residuals and latent gradient penalty of paper Eq. (10)-(12). A nonnegative
network takes all 512 latent coordinates and the waiting/done automaton
state. BOTH branches are learned: no done-to-zero assignment, waiting-value
floor, terminal constraint, time feature, or goal-region shaping loss.
Membership is exactly `V(z,q) < eta` (Eq. 3). P1 is checked on all transitions;
P2 is additionally checked on bad-source transitions. These denominators
differ from the older mutually exclusive P1/P2 report.

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/safeworld-mpl OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=2 python -u experiments/l2_paper_sublevel_finite.py
```

The existing 180/100 train/development split is reused. Three predefined
configurations are compared by development transition residual before
freezing. Then 100 fresh calibration and 100 fresh test paths are collected,
with the forecast saved before test collection. All paths contain 50 genuine
transitions; no artificial terminal self-loop is introduced. Results go to
`artifacts/safedreamer_l2_paper_sublevel_finite/`, preserving earlier runs.
Use `--phase evaluate` to reproduce saved metrics without training/collection.

This tests FINITE observed prefixes, not the paper's infinite-horizon theorem.
In particular, low-valued pending states are not silently filtered out:
candidate C, C intersected with actual finite goal success, pending sublevel
members, observed exits, and source-transition checks are reported separately.
Entry only at the endpoint has no observed successor and provides no closure
evidence. No observed source states yields an unassessable closure rate, not
100%. Statistical bounds do not establish support-wide premises, and no 80%
threshold is used as a substitute for a warrant audit.

Eq. (12) does not explicitly require a nonempty sublevel: adding a positive
constant to V preserves its P1/P2 differences and latent gradients but can
remove all sampled sublevel members. An empty observed sublevel is therefore
a possible, explicitly reported result, not permission to force acceptance
values to zero after fitting.

### 64-step achievement-spec comparison

`l2_achievement64.py` collects new 120/60/100/100 train/development/calibration/test
paths with 64 actual transitions and 65 aligned states. It compares decoded
goal-vector radii 0.30 (original task), 0.60 and 1.00 (different, relaxed approach
tasks). Initial satisfaction and subsequent arrivals are reported separately.
The existing OSRP-Vector checkpoint and random-action distribution stay fixed.
64 matches the configured replay sequence length, NOT the 15-step actor-critic
imagination horizon, and does not guarantee model fidelity at that horizon.

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/safeworld-mpl OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=2 python -u experiments/l2_achievement64.py
```

Each radius gets two candidates: the original paper loss/architecture, and an
explicit training extension with a learned constant accepting branch and a
low-value anchor. The latter exploits F(goal)'s absorbing completion state;
constancy after completion is an architectural choice, NOT evidence that the
network discovered a physical invariant region. Its initial accepting value
is above eta. No output is forced to zero or floored; sublevel membership remains
exactly V<eta, and low-valued pending states are reported as counterevidence.
This is an architecture-plus-objective comparison, not a loss-only ablation.

The selection rule, seeds and hyperparameters are saved before collection.
Only the development-selected candidate/spec is calibrated and tested. Its
forecast is saved before collecting test paths; no further fitting is done.
All metrics are finite-prefix, model-only diagnostics, not a support-wide or
infinite-horizon warrant. Individual 95% CP bounds concern event probabilities,
not correctness of V. Artifacts are preserved separately in
`artifacts/safedreamer_l2_achievement64/`; `--phase evaluate` checks hashes and
reproduces the report without additional data or training.

### Original 0.30 goal, multiple PointGoal1 checkpoints (historical random baseline)

`l2_goal03_checkpoint64.py` fixes the requested predicate to decoded distance
strictly below 0.30, with 64 real model transitions per path. It compares the
PointGoal1 OSRP-Vector checkpoint seeds 0, 5 and 10; CarGoal is not substituted.
Each model receives fresh 120/60/100/100 training/development/calibration/test
paths and a separately fitted/scaled V. Both original-loss and anchored-scalar
variants are retained, with selection performed only on each development set.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python -u experiments/l2_goal03_checkpoint64.py
```

The driver uses three CPU-affinity-limited child processes with separate logs
and artifacts in `artifacts/safedreamer_l2_goal03_checkpoint64/seed{0,5,10}/`.
All three results are reported; no holdout-based winner is certified. In addition
to individual 95% CP bounds, comparison.json supplies Bonferroni lower bounds
for the six primary calibration probabilities (goal and C&goal for each model).
Matching sampling rules/seeds does not assert exactly matched reset states or
actions. Random-action imagination still differs from deploying the learned
SafeDreamer planner. Goal radius agrees with the environment; strict `<` differs
from its `<=` only at the boundary.

`--compare-only` reconstructs the comparison. For a full per-model artifact
check, use `l2_achievement64.py --phase evaluate --output <seed directory>`.
The latter now accepts `--checkpoint`, `--radii`, `--seed-base` and `--output`
for isolated new runs, restoring the recorded checkpoint/spec/seeds on resume.
Existing runs cannot be overwritten.

### Current default: fixed-policy model verification

#### V-only refinement with fresh holdouts

`l2_policy_v_refine.py` copies only the completed GPU run's 120 training and
60 development paths, keeping the checkpoint, CCEPlanner, strict goal radius
0.30, horizon 64, eta 0.01 and 577-dimensional inputs unchanged. It compares
the original V with four warm/cold MLP candidates. Its explicitly extended
loss emphasizes the worst transition per path; a stronger **training** margin
does not change the exact P1/P2 evaluation inequalities. No unsuccessful path
is discarded, and no time index is supplied as a shortcut.

Selection maximizes development **whole-path P1/P2** success, then the full
certificate event and violation tie-breakers. After freezing, fresh seeds
22503/22504 generate 100 calibration and 100 test paths. Prediction is saved
before test collection; old holdouts are never loaded for fitting or selection.
The optional final paired comparison evaluates the original and frozen new V
on identical fresh paths without using the result to select another model.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -u experiments/l2_policy_v_refine.py
# Run with a GPU-visible environment and the safedreamer env/bin on PATH:
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=2 python -u experiments/l2_achievement64.py --phase holdout \
  --output artifacts/safedreamer_l2_policy64_gpu_vrefine
python experiments/l2_policy_v_refine.py --compare-only
```

Outputs are isolated in `artifacts/safedreamer_l2_policy64_gpu_vrefine`.
The default fixed seeds are for this one experiment, not repeated independent
trials. Development gains alone are not evidence of improved generalization;
fresh holdout results may be worse. Finite sampled evidence is not a global
or infinite-horizon warrant.

The achievement and checkpoint-comparison entry points now default to
`--action-source policy`, radius 0.30, and 64 genuine outer transitions. The
SafeDreamer runtime JSON also defaults to policy/64. Historical random runs
remain intact and can be re-evaluated with an explicit `--output` pointing to
their directory; their recorded action source is restored automatically.

```bash
# Single checkpoint, default PointGoal1 seed 0:
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/safeworld-mpl OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=2 python -u experiments/l2_achievement64.py --action-source policy

# All three checkpoints, separately fitted/calibrated/tested:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python -u experiments/l2_goal03_checkpoint64.py --action-source policy
```

New output directories: `safedreamer_l2_policy64` (single run) and
`safedreamer_l2_goal03_policy64` (comparison), under artifacts. Random control
requires `--action-source random` and a distinct `--output` directory.
`experiments/l2_safedreamer.py` remains the legacy random-action paired-transfer
baseline and now warns explicitly; policy/environment paired transfer is not
silently substituted or claimed to be implemented.

Policy means the local reconstructed OSRP-Vector CCEPlanner, matching Agent's
evaluation dispatch rather than sampling the bare actor. Each outer transition
replans from the current imagined latent and carries the previous plan forward.
The first step uses the planner's empty initial carry, then a compiled Ninjax
scan carries both latent and planner state. Planner budget is unchanged:
500 candidates, 50 elites, 6 iterations, 15-step lookahead, actor mixture 0.05.
Policy JIT compilation is cached across splits within a loaded wrapper.

Actual actions and H+1 planner-state snapshots are saved. V now receives
512 latent coordinates + 32 plan-mean values + 32 plan-std values + an initialized
flag (577 continuous inputs), plus q. The same feature pipeline is used for
training, selection, calibration and evaluation; missing planner memory fails
rather than reverting to latent-only inputs. A new policy dataset/V is required;
old random-policy latent-only artifacts cannot be reused as calibration.

The goal remains strict `<0.30` (the real task uses `<=0.30`). Policy actions are
chosen from imagined latent states, not repeatedly corrected by real future
observations: these are MODEL-SCOPE policy results, not real deployment rates.
The checkpoint lacks its original training config. In particular the local
`cost_limit=0` and planner's strict `cost < cost_limit` with nonnegative decoded
cost force the minimum-cost branch. That behavior is reported, not silently
fixed, and no improvement in goal success is promised.

`smoke_safedreamer_policy.py --horizon 2` (or 64) runs a single real-checkpoint
integration check with the FULL planner budget. It checks actual actions equal
each returned plan's first action, planner-memory alignment, and the 577-input
schema. It is not certificate calibration. CPU policy collection is much more
expensive than random imagination; a full statistical rerun is a separate run.

### Conditional approach-goal V training (radius 1.0)

`l2_approach1_outside_v.py` is a separate, training-only experiment. It changes
the specification to `F_[0,64](decoded goal distance < 1.0)` and retains only
paths whose initial decoded distance is at least 1.0. It does not filter by
future success: the retained 81 training paths include 17 non-arrivals, and
the 47 development paths include 12 non-arrivals. These are model-decoded
distances, not verified simulator distances. Source data are the previous
GPU policy run's train/development splits only.

Four freshly initialized MLP candidates keep the same model, policy, eta=.01,
577 continuous inputs and genuine 64 transitions. The primary development
criterion is whole-path P1/P2 coverage among goal-reaching paths, followed by
the complete event including endpoint sublevel membership. These distinct
metrics are both reported. Scaling is fitted only on retained training states.
The accepting scalar is learned; no forced zero or time-index shortcut is used.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python -u experiments/l2_approach1_outside_v.py
```

Artifacts go to `artifacts/safedreamer_l2_approach1_outside_v`; existing output
is never overwritten. `selected.pt` is frozen with hashes and a complete
development report. This command does not collect or inspect calibration/test
paths and issues no warrant or confidence lower bound. A formal follow-up must
use fresh paths sampled under the same **initial decoded distance >= 1.0**
condition, with the frozen radius and V. Earlier exploratory spec-screening
holdouts must not be reused as independent confirmatory data.

### Alternative V functions on the same conditional approach data

`l2_approach1_function_search.py` keeps the radius-1.0 / initial-distance>=1.0
population and the same 81/47 train/development paths. It compares the frozen
previous MLP with 12 freshly fitted candidates: linear, low-rank quadratic,
RBF plus linear, Tanh MLP, residual MLP, and fixed Fourier features plus linear,
each with mean-only and worst-edge-augmented training objectives and different
initializations. Nonnegativity is enforced with softplus; the accepting scalar
remains learned. RBF centers are sampled from normalized training waiting states
only, and Fourier frequencies are data-independent random fixed buffers. All
families retain the same state inputs and exact eta=.01 evaluation criteria.

Selection is development-only, prioritizing P1/P2 coverage on goal-reaching
paths, then the full event including endpoint sublevel membership. The old V
is retained if no candidate improves this recorded ordering. All eligible
training paths, including unsuccessful paths, contribute to the loss.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python -u experiments/l2_approach1_function_search.py
python experiments/l2_approach1_function_search.py --evaluate-only
```

Outputs are isolated in `artifacts/safedreamer_l2_approach1_function_search`.
Use `experiments.l2_approach1_function_search.predict(model,z,done)` to evaluate
the selected checkpoint: it supports both the old MLP and the new family schema.
The `--evaluate-only` path verifies frozen hashes and exactly reproduces the
saved training/development reports. No calibration/test data are read and no
confidence bound or warrant is issued. Repeated development selection does not
constitute independent generalization evidence.

### Fresh conditional calibration and prospective test

`l2_approach1_holdout.py` uses the frozen function-search winner without fitting
anything. It collects 100 eligible calibration paths, persists the forecast,
then collects 100 eligible test paths. Eligibility is exclusively decoded
initial distance >=1.0. Full 64-step paths are drawn in batches of 20; the first
100 eligible draws are retained in chronological order, regardless of future
success. Rejected initial-inside paths and unused final-batch draws are saved
with indices and hashes, so conditional selection can be replayed exactly.
The per-split draw limit is 1000; failure to fill the quota aborts rather than
changing the sample size or confidence claim.

```bash
# Requires GPU visibility and the safedreamer env/bin on PATH for ptxas.
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=2 python -u experiments/l2_approach1_holdout.py
python experiments/l2_approach1_holdout.py --evaluate-only
```

Outputs: `artifacts/safedreamer_l2_approach1_linear_holdout`. Separate batch seed
ranges start at 25503000 for calibration and 25504000 for test. Simulator reset
RNG is not fully specified by those imagination seeds; inference assumes iid
same-distribution conditional draws. Runtime planner settings and source hashes
must match the training run. No real-environment trajectories are evaluated.

The predeclared primary event is whole-path P1/P2 AND endpoint V<eta AND actual
decoded goal arrival within the horizon. Its individual one-sided 95% CP lower
bound is compared to the predeclared probability threshold .95. The result is
`NO_WARRANT` or `FINITE_EVENT_THRESHOLD_MET`, never an infinite-horizon warrant.
Other event bounds are individual, not simultaneous. Test results are not used
to adjust weights, thresholds or the saved calibration decision.

### Larger training set and wider function search

`l2_approach1_more_data.py` expands the existing 81 training / 47 development
paths to 300 / 100 by collecting 219 / 53 new eligible imagination paths on
GPU. Initial decoded distance >=1.0 is the only eligibility condition; all
future failures are retained. Raw draws, accepted indices, hashes and the old /
new components are stored separately. Old calibration and test trajectories
are not pooled into fitting; only identity metadata are read to detect reuse.
New seed ranges begin at 26501000 (training) and 26502000 (development).

The fixed original V is evaluated on the same enlarged development set as
20 new fits: the earlier six families plus ELU MLP and gated affine-expert
mixtures, with additional width-256 residual/ELU candidates. New models use
normalization fitted on the enlarged training set only. The old reference
retains its original weights and normalization. Pending-state low-value
penalty is increased from 1 to 10 in training; exact P1/P2, eta, sublevel
membership, world model and policy are unchanged. No hard output floor is used.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=2 python -u experiments/l2_approach1_more_data.py
python experiments/l2_approach1_more_data.py --phase evaluate
```

Output: `artifacts/safedreamer_l2_approach1_more_data`. GPU visibility and the
environment's `ptxas` on PATH are required for collection. The small V networks
train on CPU. `--phase train` can start fitting after fully completed data
collection, but refuses to overwrite a frozen model. No new calibration/test
or confidence claim is included; development data remain model-selection data.
Any later confirmation of this expanded-data model needs new holdout seed
ranges as well as a new output directory. Do not replay the earlier holdout
script's fixed 25503000/25504000 ranges and call those draws a fresh test.

### Finite clear-window specification and V search

`l2_spec_screen64.py` screens saved training/development paths only. The selected
alternative is `F[1,17] G[0,47](decoded hazard_margin >= 0)`: at least one run of
48 consecutive clear states beginning at a time from 1 through 17. This allows
hazards outside the chosen window and does not mean full-horizon safety or goal
arrival. Hazard APs are decoded imagination observations, not environment costs.

`l2_clear_window_v.py` builds causal monitor labels (ignore t0, count consecutive
clear states, reset on hazard, accept after 48, absorbing acceptance). Every
saved training/development path is retained. V receives the original 577
latent/planner features plus the causal run counter, with a learned accepting
constant. It does not receive future labels, path identifiers, or elapsed or
remaining time. P1 is checked on all genuine transitions, P2 on every source
state not yet accepting, with exact eta=0.01. No artificial terminal edge is
added. This is a finite experimental construction, not an infinite co-Buchi
support-wide certificate. The full finite certificate event also requires
monitor completion and endpoint V<eta.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python -u experiments/l2_clear_window_v.py
python experiments/l2_clear_window_v.py --evaluate-only
# Separate extended linear search; preserves the earlier fit directory.
python experiments/l2_clear_window_v.py --families linear --steps 6000 \
  --output artifacts/safedreamer_l2_clear_window_v_linear6000
```

The initial population remains decoded goal distance >=1.0, inherited from the
source data even though the new specification is about hazards. All reported
selection metrics are development metrics; no calibration/test or probability
confidence claim is produced. New specifications and function selection must
be frozen before collecting fresh confirmatory data.

### Frozen clear-window calibration and test

`l2_clear_window_holdout.py` freezes the selected `linear_max4` (update 4800)
from `safedreamer_l2_clear_window_v_linear6000`. It collects 100 new eligible
calibration paths, saves the forecast, then collects 100 new eligible test
paths. Seed ranges are 28503000 / 28504000; do not reuse them for later claims
of fresh confirmation. The initial condition remains decoded goal distance
>=1.0 even though the property now concerns hazards. No future outcome filters
the samples; raw batches and all initial exclusions are retained.

The primary event is whole-path P1/P2 AND endpoint V<0.01 AND clear-window
completion by t64. The predeclared criterion is an individual one-sided 95%
Clopper-Pearson lower bound >=0.95, plus clean observed candidate-region
diagnostics on calibration data. This is a finite model-scope statistical
claim, not an infinite-horizon or support-wide proof. Weights, code hashes,
policy metadata, prior trajectory identities and the prospective forecast are
checked on reevaluation. No fitting or selection uses either new split.

```bash
PATH=/home/sunyhg/miniconda3/envs/safedreamer/bin:$PATH \
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=2 python -u experiments/l2_clear_window_holdout.py
python experiments/l2_clear_window_holdout.py --evaluate-only
```

GPU visibility and the environment's CUDA compiler are required for collection.
Output: `artifacts/safedreamer_l2_clear_window_holdout`.

### Goal radius 0.30, 100-step imagination screening (no V)

`goal03_policy100.py` generates 100 new eligible paths under the same local
CCEPlanner/checkpoint as the clear-window holdout. Only the OUTER rollout
length changes to 100 transitions (101 states); planner lookahead remains15.
The initial distribution stays decoded goal distance>=1.0. Every future
failure is retained; selection uses only the initial condition and draw order.
The strict goal predicate is decoded distance<0.30 (stored goal AP<0).

The report compares arrival by64 and by100 on the SAME paths, counts new
arrivals during65..100, and records first-arrival times. No V is loaded,
trained or evaluated; no P1/P2, calibration bound or warrant is computed.
This is exploratory model-only task screening, not environment accuracy.

```bash
PATH=/home/sunyhg/miniconda3/envs/safedreamer/bin:$PATH \
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=2 python -u experiments/goal03_policy100.py
python experiments/goal03_policy100.py --evaluate-only
```

Output: `artifacts/safedreamer_goal03_policy100`. Seed range begins29505000.
Raw batches, initial exclusions, complete states/actions and accepted paths
are saved. Reevaluation checks selection, data hashes and AP/decoder agreement.

### SafeDreamer L3: paper H1 candidate and independent point checks

`l3_safedreamer_paper.py` trains bounded state functions W/U for
`GF G[0,47](decoded_hazard_margin>=0)`. It checks U at accepting states too,
without assuming the counter defines an invariant core. The default pilot
uses 12 fresh training and 20 fresh validation paths, horizon 192, and 32
resettable successor queries per path's random anchor. W/U are frozen before
validation sampling. Original L2 artifacts are unchanged.

This is NOT a completed global verifier: output `paper_L3` explicitly abstains
because SafeDreamer's certified envelopes, coverage, containment, retention,
initial-distribution and collar premises are not established. Finite behavior
and conservative pointwise drift results are reported separately, never as
infinite recurrence probability. See [protocol](L3_SAFEDREAMER_PAPER.md).

```bash
PATH=/home/sunyhg/miniconda3/envs/safedreamer/bin:$PATH \
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=2 python -u experiments/l3_safedreamer_paper.py \
  --output artifacts/safedreamer_l3_paper_pilot192_new
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python experiments/l3_safedreamer_paper.py --evaluate-only \
  --output artifacts/safedreamer_l3_paper_pilot192_new
```
