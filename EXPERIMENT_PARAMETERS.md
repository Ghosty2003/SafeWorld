# SAFEWORLD V2 — Experiment Configuration

**Scope of this document: configuration only.** Every value here is a
parameter that was *set before running* something (a checkpoint path, a
hyperparameter, a split fraction, a CV threshold) — not a value that *came out
of* running something. No ρ\*, p̂_γ, R², TPR/FPR, SAT/VIOLATION counts, verdicts,
or confusion-matrix entries appear anywhere below, not even as placeholders.
For results and their provenance, see `experimental_setup.md` (narrative +
placeholder ledger) and `EXPERIMENT_CONFIG.md` (per-number ledger with
Tier/caveat judgments).

**Document date:** 2026-08-04. **Source-of-truth:** `/home/bot/SafeWorld` (SAFEWORLD V2)

---

## 1. World models

| Model | Checkpoint path | Task | Wrapper |
|---|---|---|---|
| CarDreamer (DreamerV3) | `/home/bot/CarDreamer/logdir/carla_four_lane/checkpoint.ckpt` | `carla_four_lane` | `wrappers/cardreamer_wrapper.py::CarDreamerWrapper` |
| CarDreamer (DreamerV3) | `/home/bot/CarDreamer/logdir/carla_roundabout/checkpoint.ckpt` (frozen copy: `/home/bot/SafeWorld/verified_checkpoints/carla_roundabout_20260713/checkpoint.ckpt` — same underlying artifact, use the frozen path for any citation since the live logdir path can be overwritten by future training) | `carla_roundabout` | `wrappers/cardreamer_wrapper.py::CarDreamerWrapper` |
| TD-MPC2 | `/home/bot/SafeWorld/models/walker-walk-3.pt` (sibling checkpoints on disk, not wired into any dataset: `walker-run-3.pt`, `walker-stand-3.pt`, `walker-walk-backwards-3.pt`, `walker-run-backwards-3.pt`) | `walker-walk` (dm_control) | `wrappers/tdmpc2_wrapper.py::TDMPC2Wrapper` |

**Action mechanism per model (config choice, not a result):**

| Model | Primary action_source | Alternative(s) | Notes |
|---|---|---|---|
| CarDreamer | `"actor"` (trained deployment policy) | `"random"` (coverage only) | set via `RolloutConfig(action_source=...)` |
| TD-MPC2 | `"mpc_plan"` (real CEM/MPPI, via `plan_from_z()`) | `"pi_prior"` (bare policy, no planning) | required kwarg, no default, in `TDMPC2Wrapper.sample_rollouts()`/`sample_latent_rollouts()`; `pi_prior` trajectories are tagged `_approx_pi_prior: 1.0` |

**TD-MPC2 CEM/MPPI planner hyperparameters** (`tdmpc2_src/config.yaml`, used verbatim by `plan_from_z()`):

| Param | Value |
|---|---|
| `mpc` | `true` |
| `horizon` | 3 — **planning-horizon**: the CEM/MPPI planner's internal lookahead, in agent decision steps. Not to be confused with rollout-horizon (§4) or episode-horizon (below); see the "three horizons" note under §1 |
| `iterations` | 6 |
| `num_samples` | 512 |
| `num_elites` | 64 |
| `num_pi_trajs` | 24 |
| `min_std` / `max_std` | 0.05 / 2 |
| `temperature` | 0.5 |
| `log_std_min` / `log_std_max` | -10 / 2 |

**dm_control environment wrapping:** `Timeout(max_episode_steps=500)`; `DMControlWrapper.step()` applies 2x action-repeat per agent step (`for _ in range(2): env.step(action)`, rewards summed).

**"Horizon" is used at three different levels in this project — do not conflate them:**

| Term | Unit | walker-walk value | carla value |
|---|---|---|---|
| **planning-horizon** | agent decision steps, inside the CEM/MPPI planner only | 3 | n/a (DreamerV3 actor, no MPC) |
| **rollout-horizon** | agent decision steps = latent transitions (one `model.next()` / one `wm.imagine` step each) | 100 (`ltl_height_safety` dataset) | 50 (four_lane/roundabout imagine); 80 (roundabout L3) |
| **episode-horizon** | agent decision steps, real-environment episode cap | 500 (`Timeout`) | 500 (CarDreamer `Timeout` equivalent) |

**walker-walk unit conversion (verified against `sample_latent_rollouts_from_states()`'s loop, which does one `self.next(z, a)` per iteration):** 1 agent step = 1 latent transition = 2 dm_control physics steps (action-repeat) = 2 × 0.025 s = **0.05 s sim time**. So: rollout-horizon 100 = 200 physics steps = 5.0 s; anchor spacing "every 20 steps" = 20 **agent** steps (the collection loop counts one `t` per wrapper `env.step()`) = 1.0 s; episode cap 500 agent steps = 25 s = dm_control walker's `_DEFAULT_TIME_LIMIT` exactly — the numbers are mutually consistent.

**carla unit conversion:** 1 agent step = 1 CARLA tick = 0.1 s (`fixed_delta_seconds: 0.1`, synchronous mode, no action-repeat — verified in `car_dreamer/configs/common.yaml` and `carla_base_env.py`). Rollout-horizon 50 = 50 latent RSSM transitions ≙ 5.0 s of imagined sim time; the latent transitions were trained on tick-level real transitions, so no repeat-factor ambiguity exists on this side.

---

## 2. Spec registry

- Location: `specs/ltl_specs.py` (16 LTL specs), `specs/stl_specs.py` (14 STL specs) — 30 total.
- **Version anchor:** both files are modified-uncommitted on top of git HEAD `3e2922e` (last commit 2026-07-14); the working-tree versions (mtime 2026-07-29 02:32, both files) are the ones containing the 30-spec registry described here. Snapshot as of 2026-08-05 — commit these before citing spec definitions in any released artifact.
- Each spec dict has: `id`, `level`, `name`, `mp_class` (Manna–Pnueli class), `formula` (parse tree), `horizon` (STL only), `aps` (required AP key list), `description`.
- Public accessors: `get_ltl_spec_by_id(id)`, `get_stl_spec_by_id(id)`, `get_all_ltl_specs()`, `get_all_stl_specs()`.
- Automaton construction: `core/lppm/automaton.py::build_parity_automaton(spec)` — tries Spot translation first (`_build_spot_parity_automaton`, requires the `spot` package, **not installed in any project conda env** — `dyno`, `cardreamer`, `base` all checked), falls back to hand-written template automata (Safety/Guarantee/Obligation/Recurrence patterns) when Spot is unavailable or fails.

---

## 3. AP extraction configuration (per model)

### 3.1 CarDreamer — CV thresholds on decoded `birdeye_wpt` (128×128×3)

Provenance column added 2026-08-05. All constants live in a single squash
commit (`3e2922e`, 2026-07-14) — git blame carries no earlier incremental
history, so per-constant provenance below comes from upstream-config
cross-referencing and code-comment inspection, marked honestly where nothing
is recorded.

| Constant | Value | Meaning | Provenance |
|---|---|---|---|
| `IMG_SIZE` | 128 | decoded image side length (px) | inherited from CarDreamer rendering config: `car_dreamer/configs/common.yaml` `birdeye_wpt: shape: [128, 128, 3]` |
| `OBS_RANGE_M` | 32.0 | physical extent of the birdeye view (m) | inherited: same block, `obs_range: 32` |
| `PIXELS_PER_METER` | 4.0 | `IMG_SIZE / OBS_RANGE_M` | derived (128/32) |
| `EGO_PIXEL_X`, `EGO_PIXEL_Y` | 64, 80 | ego origin in image coordinates | derived from upstream: X = center = 128/2; Y = 64 + (obs_range/2 − ego_offset)·ppm = 64 + (16−12)·4 = 80, matching `birdeye_handler.py:21`'s `pixels_ahead_vehicle` formula with `ego_offset: 12` from the same config block — not a magic number |
| `VEHICLE_COLOR_TOL` | 40 | L-inf color-match tolerance, vehicle detection (target RGB `(0,255,0)`) | target colors verified against actual CarDreamer birdeye rendering (user-confirmed; the GREEN-only choice is also documented in `configs/settings/cardreamer.json` — `_C_BUTTER`/`_C_ALUMIN` were removed after causing false detections on lane markings/pavement); tolerance absorbs decoder reconstruction noise. **The value 40 itself: empirically chosen, no sweep recorded** — no sweep script or comparison log exists in the repo |
| `WAYPOINT_COLOR_TOL` | 40 | same, for ego waypoints (target RGB `(0,0,255)`) | same as above (shared value, same status: empirically chosen, no sweep recorded) |
| `NEAR_OBS_MARGIN_M` | 5.0 | `near_obstacle` safety margin | empirically chosen, rationale not recorded (no code comment or git history explains 5.0 specifically) |
| `MIN_CLUSTER_PX` | 5 | minimum detected pixel cluster to count as a real detection | empirically chosen, rationale not recorded |
| `EGO_HALF_LEN_M` / `EGO_HALF_WID_M` | 2.3 / 0.95 | ego body half-extents (m), for body-to-body gap computation | approximates a standard CARLA sedan footprint (~4.6m × 1.9m full extents). CarDreamer's configs do **not** pin a specific ego blueprint, and no measurement script or CARLA-blueprint reference is recorded for these two numbers — treat as empirical estimates of the rendered ego footprint, not a blueprint-derived fact |
| `COLLISION_TOL_M` | 0.25 | one-pixel discretization slack added to the collision boundary | **derived, not magic**: 1 px = 1/`PIXELS_PER_METER` = 1/4.0 = 0.25 m — exactly one pixel of quantization at the birdeye resolution; the code comment "one pixel of discretization slack (4 px/m)" checks out |
| `UNCERTAIN_SENTINEL` | -999.0 | sentinel value for `velocity` (always) and any AP with zero detected pixels | arbitrary sentinel, any impossible value works; -999.0 chosen for visibility in logs |
| `bbox_inflate_px` | 0 (constructor default) | vehicle-detection bbox inflation | **all reported hazard_avoidance numbers (both four_lane and roundabout) used 0** — confirmed by grep: every call site passing it explicitly (`calibrate_cerr.py:96`, `validate_real.py:65`, `diag_frame_inspect.py:97,210`) passes 0, and `configs/settings/cardreamer.json` sets 0; `eval_ltl_hazard_real_lppm.py` uses the constructor default (also 0). Sweep protocol (0–8 FVR/FSR curve) is a results-side plan — see `experimental_setup.md` |
| `bgr_observations` | `False` (constructor default) | channel-order flag | **four_lane: verified RGB** (`bgr_observations=False`) via `verify_channel_order_multi()` multi-frame voting — recorded in `VERIFICATION_CARDREAMER.md` §A ("通道顺序判定 RGB,多帧投票一致"). **roundabout: NOT explicitly verified** — no channel-order check is recorded anywhere in §N or elsewhere for the roundabout checkpoint; all roundabout CV-based results assume the default `False`. Since both checkpoints come from the same CarDreamer rendering pipeline the assumption is plausible, but per project policy ("do NOT rely on 'numbers look plausible'"), this is an open verification gap |

`ap_keys()` (class method, **same return value regardless of which checkpoint is loaded**): `["hazard_dist", "near_obstacle", "goal_dist", "velocity"]`. `velocity` is always emitted as `UNCERTAIN_SENTINEL` — the reward head's speed component is a tent function peaking at desired speed, making it an anti-conservative proxy (see `experimental_setup.md` §2 for the reasoning, not repeated here as it's an explanation, not a config value).

### 3.2 CarDreamer roundabout — position/goal probe (`cardreamer/probe_common.py`)

| Constant | Value | Meaning |
|---|---|---|
| `GOAL_XY` | `(4.2, -43.2)` | fixed destination point (CARLA world coords), from `tasks.yaml`'s `ego_path` terminus |
| `RING_C` | `(-6.4, -6.3)` | circulation-ring center (chord-fit through three mid-route points) |
| `RING_R` | 20.9 | circulation-ring radius (m) |
| `in_ring(p, slack=4.0)` | membership test | `zone_a`: `‖p - RING_C‖ < RING_R + slack` |
| `near_goal(p, radius=8.0)` | membership test | `zone_b`: `‖p - GOAL_XY‖ < radius` |
| Probe model | `sklearn.linear_model.Ridge`, `alpha=10.0` | latent `h` → `(x, y)`, fit via `fit_position_probe()`. **alpha provenance:** no alpha sweep or `RidgeCV` exists anywhere in the repo (grep-confirmed); 10.0 is an unswept empirical choice. Note a *third* value exists elsewhere: `collect_probe_data.py:228` (the four_lane progress probe, a different probe) uses `alpha=1.0` — so 10.0 is not even a repo-wide convention, just this probe family's value |

This AP set (`hazard_dist`/`near_obstacle` reused from the four_lane CV pipeline, plus `goal_dist`/`zone_a`/`zone_b` from this probe) is **not** exposed through `CarDreamerWrapper.ap_keys()` — it is computed ad hoc inside `cardreamer/collect_goal_probe.py`, `cardreamer/eval_l2_roundabout.py`, and `cardreamer/offline_checks.py`, not through the shared wrapper interface.

### 3.3 TD-MPC2 walker-walk — height probe (`wrappers/tdmpc2_probes.py`)

| Component | Value |
|---|---|
| Probe model | `sklearn.linear_model.Ridge`, `alpha=10.0` (default, overridable). **alpha provenance: copied from the roundabout position-probe convention** (`probe_common.py`) when `tdmpc2_probes.py` was written — not independently selected for the height probe, and no cross-validation sweep was run for either probe (grep-confirmed: no `RidgeCV`/alpha-sweep anywhere in the repo) |
| Feature source | posterior latent `z` (encoder output, memoryless — no burn-in required) |
| Target | real torso height, `obs[14]` (dm_control walker obs layout: orientations(14) + height + velocity) |
| Fit function | `fit_height_probe(npz_path, alpha=10.0)` — `npz_path` is a required argument, not defaulted to any specific dataset |
| Attachment | `TDMPC2Wrapper.set_ap_extractor(make_height_ap_extractor(probe), keys=["height"])` — the wrapper itself contains no probe-specific logic |
| Anchor-restoration mechanism | `physics.get_state()` / `physics.set_state()` + `physics.forward()` (dm_control MuJoCo physics), exact state save/replay — no `burn_in` step-count approximation used for any formal calibration set |
| `sample_latent_rollouts()`'s approximate fallback | `burn_in` param, default range `(0, 480)` — documented as approximate, not used for the formal `ltl_height_safety` calibration set (which uses `sample_latent_rollouts_from_states()` with exact saved physics states instead) |

---

## 4. Dataset construction parameters

Parameters only — no dataset outcomes. All three sets already enforce
`fit_lppm()`/`calibrate_lppm()` train/calibration disjointness.

| Parameter | `ltl_height_safety` (TD-MPC2 walker-walk) | `stl_hazard_avoidance` (carla_four_lane) | roundabout position/goal-probe sets |
|---|---|---|---|
| Anchor source | 200 physics states, 8×500-agent-step MPC episodes, every 20 agent steps (=1.0 s spacing), exact `physics.set_state()` replay | replay-buffer real (obs, action) sequences, `use_replay_start=true` | real-episode posterior (h, ego_x/y), recorded every step |
| action_source | `mpc_plan` | `actor` | `actor` (implicit — episodes collected during normal deployment rollout) |
| Rollout-horizon (agent steps; see §1's three-horizons note) | 100 (=200 physics steps, 5.0 s) | 50 (=5.0 s at 0.1 s/tick) | 50 (goal-probe imagination); 80 (L3 zone-sequence formal verdict, per `VERIFICATION_CARDREAMER.md` §N.3 — exact regenerating command not logged, see gap list) |
| N (total) | 200 | 100 | 20 real episodes (goal-probe collection); 717 stratified anchors (window-reachability); 697 anchors (L3 zone-sequence, offline reuse of goal-probe anchors, no new collection); 100 rollouts (hazard/collision dataset, same pipeline as four_lane) |
| Train/calib split | 120 / 80 (60/40) | 60 / 40 (60/40) | not yet wired into `fit_lppm`/`calibrate_lppm` — no LPPM split exists for roundabout yet (STL/monitor-path and offline-consistency scripts only) |
| Split seed | `np.random.default_rng(0)` | `np.random.default_rng(0)` | n/a |
| Task/env seed | dm_control task seed=0 | n/a (real replay buffer) | n/a (real replay buffer) |
| Stratification | none (natural every-20-step sample) | none (natural replay-buffer sample) | none for the raw collection; the 717-anchor table is stratified post hoc by start-distance-to-goal bucket for analysis, not by construction |

**Collection scripts:**

| Dataset | Script |
|---|---|
| TD-MPC2 height_safety anchors (with physics state) | `tdmpc2_recollect_with_physics_state.py` (scratchpad) |
| carla_four_lane hazard calibration set | `cardreamer/eval_ltl_hazard_real_lppm.py` (collects internally via `CarDreamerWrapper`) |
| carla_roundabout goal/position probe | `cardreamer/collect_goal_probe.py --episodes 20` |
| carla_roundabout L2 (`stl_safe_goal_reach`) | `cardreamer/eval_l2_roundabout.py` |
| carla_roundabout L3/L5 offline checks | `cardreamer/offline_checks.py {l3,l5,l5sem}` |

---

## 5. LPPM training configuration (`core/lppm/model.py`, `core/lppm/trainer.py`)

| Component | Value |
|---|---|
| Architecture | `NeuralLPPM`: `Embedding(n_states, q_embed_dim)` for automaton state → concat with latent `z` → `Linear(latent_dim+q_embed_dim, hidden_dim) → ReLU → Linear(hidden_dim, hidden_dim) → ReLU → Linear(hidden_dim, n_heads)` → `softplus` |
| `hidden_dim` | 128 (default) |
| `q_embed_dim` | 16 (default) |
| `n_heads` | `max(#odd DPA priorities, 1)` |
| Loss | `p1_loss + p2_loss + lambda_reg * smoothness_penalty(v, z)` |
| `lambda_reg` | 0.01 (default) |
| Optimizer | Adam, `lr=1e-3` (default) |
| `n_epochs` | 300 (function default). Per-artifact actual values (verified in scripts): **four_lane hazard → `N_EPOCHS = 1500`** (`cardreamer/eval_ltl_hazard_real_lppm.py:35`); **walker-walk height → `N_EPOCHS = 1500`** (`tdmpc2/eval_ltl_height_safety_lppm.py:58`). Both with early-stop once loss < the script's threshold. These are the reproduction values for the corresponding placeholder-ledger numbers |
| LPPM init seed | **Asymmetric — verified by grep, this is a real reproducibility difference between the two pipelines:** the four_lane script trains its network *inline* (not via `fit_lppm()`) and calls `torch.manual_seed(0)` before construction (`eval_ltl_hazard_real_lppm.py:103`) → seeded, reproducible weights. The walker-walk script calls `fit_lppm()`, and **neither it nor `core/lppm/trainer.py` seeds torch anywhere** → network initialization is unseeded; rerunning the same configuration will produce different LPPM weights and possibly a different p̂_γ purely from init randomness. This init variance is a separate source from calibration-set sampling variance (the N-vs-gap analysis) — do not conflate the two. Fixing this (add seeding to `fit_lppm()` or the script) is a prerequisite for the planned four_lane N→200 rescale and for any multi-seed sweep |
| `eta` (P2 descent margin) | 0.01 (default) |
| `p1_tol` (P1 numeric tolerance) | 0.0 (default — strict, preserves pre-existing behavior). Added this session; when non-zero, should be derived from the calibration run's own V-distribution noise scale, not a hardcoded constant |
| Fallback | if `torch`/`NeuralLPPM`/`F` unavailable, `fit_lppm()` uses `_fit_lppm_heuristic_fallback` (heuristic epoch-loss trainer) instead |

---

## 6. Conformal calibration configuration (`core/lppm/calibrator.py`)

| Component | Value |
|---|---|
| Formula | exact one-sided Clopper–Pearson lower bound, $\hat p_\gamma = \mathrm{Beta}^{-1}(\gamma; k, n-k+1)$ (0 when $k=0$) |
| `gamma` | 0.05 (default) |
| `warrant_threshold` | 0.80 (default) |
| `eta` | 0.01 (default, shared with LPPM training) |
| `p1_tol` | 0.0 (default, threaded through to `check_pathwise_conditions`) |

**`support_level` enum** (`utils/spec_analysis.py`, added/fixed this session): `SUPPORT_DEDUCTIVE` ("deductive"), `SUPPORT_CALIBRATED` ("calibrated"), `SUPPORT_APPROXIMATE` ("approximate"). Bounded STL specs default to `SUPPORT_CALIBRATED`; unbounded specs default to `SUPPORT_APPROXIMATE`. No code path in this project may set `SUPPORT_DEDUCTIVE` (enforced by an assertion in `main.py`'s `verify()`) since no deductive/support-wide NN-verification branch exists anywhere in this codebase.

---

## 7. Code entry points

| Purpose | Entry point |
|---|---|
| General verification pipeline (STL monitor + transfer calibrator + LPPM, all specs) | `main.py` — flags: `--model`, `--spec`/`--task-config`/`--benchmark`, `--auto-paired`, `--method {stl,cegar,oneshot,soft_buchi}` |
| CarDreamer hazard_avoidance LPPM (disjoint split, liveness check) | `cardreamer/eval_ltl_hazard_real_lppm.py` |
| TD-MPC2 height_safety LPPM (disjoint split, liveness check, P1/P2 breakdown) | `tdmpc2/eval_ltl_height_safety_lppm.py` |
| CarDreamer channel-order verification (mandatory before trusting CV APs) | `verify_channel_order_multi()` in `wrappers/cardreamer_wrapper.py` |
| CEGAR / one-shot verification | `core/cegar::run_cegar()` / `run_oneshot()` |

---

## 8. Config-level structural notes (not results — properties of the code/config itself)

- **No shared `WorldModel` Protocol exists.** `wrappers/base.py` is an abstract base class; each model has its own hand-written wrapper with a materially different method surface (e.g. `TDMPC2Wrapper.sample_latent_rollouts_from_states()` has no CarDreamer equivalent).
- **`CarDreamerWrapper.ap_keys()` is one class-level method, not checkpoint-aware.** It returns the same four keys regardless of which checkpoint (`four_lane` vs `roundabout`) is loaded — roundabout's real `zone_a`/`zone_b` availability lives outside this method entirely (§3.2). Any code that gates on `ap_keys()` alone will misjudge roundabout's zone-based spec eligibility.
- **No persisted `training_info`/LPPM-weights sidecar exists for any (checkpoint, spec) pair.** Rerunning a fit script retrains from scratch; there is no cached model artifact on disk to load instead.
- **Only one seed per checkpoint exists today** — no multi-seed sweep configuration has been run for any (checkpoint, spec) pair.
- **`spot` is not installed** in any of this project's conda environments (`dyno`, `cardreamer`, `base`) — `build_parity_automaton()` always falls back to template automata in practice today, regardless of what the code would do if Spot were present.
- **Spec `aps` fields are hand-maintained with no automated consistency check** — nothing verifies that a spec's declared `"aps"` list actually matches the AP keys its formula tree references. The static AP-intersection coverage table (`experimental_setup.md` §4.0) is only as trustworthy as this field's accuracy; a mis-declared `aps` list would silently produce a wrong RUNNABLE/BLOCKED cell.
- **LPPM network initialization is unseeded on the `fit_lppm()` path** (see §5's per-artifact seed row): identical configuration reruns produce different weights. Any statement of the form "reran with the same config and got X" must account for init variance separately from calibration-sampling variance — they are independent noise sources.

---

## 9. Library & upstream version lock (recorded 2026-08-05)

Installed versions per conda environment (the two environments that actually run experiments):

| Package | `dyno` (TD-MPC2 / walker-walk pipeline) | `cardreamer` (CarDreamer / CARLA pipeline) |
|---|---|---|
| dm-control | 1.0.44 | not installed |
| gymnasium | 0.29.1 | not installed |
| scikit-learn | 1.5.1 | 1.7.2 |
| torch | 2.12.1+cu130 | 2.13.0+cpu |
| numpy | 1.26.3 | 2.2.6 |
| scipy | 1.14.0 | 1.15.3 |
| tensordict | 0.8.3 (**pinned deliberately** — TD-MPC2 checkpoint loading breaks under 0.13.x, see `wrappers/tdmpc2_wrapper.py` docstring) | not installed |
| jax | 0.6.2 | 0.6.2 |
| mujoco | 3.11.0 | not installed |

Note the two environments intentionally differ (sklearn 1.5.1 vs 1.7.2, numpy 1.x vs 2.x): each probe is fit and used within one environment, so no cross-environment pickle compatibility is currently relied upon — but do not move a fitted probe artifact between environments without re-checking.

**Upstream code versions:**

| Upstream | Location | Commit |
|---|---|---|
| TD-MPC2 (nicklashansen/tdmpc2) | scratchpad clone `tdmpc2_src/` (shallow) | `e9f5932` (clone date 2026-07-13) — **note this clone lives in the ephemeral scratchpad; relocate into the repo or record the commit in release artifacts before it evaporates** |
| CarDreamer | `/home/bot/CarDreamer` (git repo) | `1601324` (2025-12-11) |

**`obs[14]` height-index dependency:** the walker height probe's target index (`_real_height(obs) = obs[14]`) depends on dm_control walker's observation layout — `orientations` (14) + `height` (1) + `velocity` (9) — as confirmed in **dm-control 1.0.44**'s `suite/walker.py::PlanarWalker.get_observation()`. If dm-control is upgraded, re-verify this index against the new `get_observation()` ordering before trusting any probe output; a silent layout change would corrupt the probe target without any error.
