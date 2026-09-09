# SafeWorld current project status

Snapshot date: **2026-09-01**

Repository branch: **`main`**

Purpose: one self-contained source for what is implemented, what was actually
run, what may be claimed, what remains incompatible with the paper, and how to
reproduce the current results.

## 1. Executive summary

The repository now contains working L1/STL monitoring, a sampled L2/LPPM path,
an exact analytic L3/LBSM construction, a trained two-dimensional L3 test path,
and a new real-checkpoint distribution-scoped L3 alternative. The central
scientific boundary is:

- the paper's analytic Appendix E.2 L3 construction is implemented and matches
  its reported warrant of `0.92`;
- the paper's global real-model L3 route is computationally infeasible for the
  current deep recurrent checkpoints because its Lipschitz covering bounds
  require at least `10^13` queries even after a one-dimensional projection and
  around `10^22` in two dimensions;
- the new distributional L3 route avoids the dynamics Lipschitz constant, but
  proves only a marginal statement under a declared anchor distribution. It is
  not Theorem 5.6, not a Ville/pathwise recurrence warrant, and its first real
  CarDreamer experiment was inconclusive;
- both real L2 NeuralLPPM experiments produced degenerate progress measures.
  Their replacement results are explicitly weaker trajectory-level statistical
  safety rates, not Theorem 5.4 warrants;
- the TD-MPC2 `forward_speed` posterior probe is validated, but 100-step
  TD-MPC2-to-MuJoCo transfer fails after a reliable prefix of about 15 steps.

The current regression suite passes: **87 passed, 1 expected xfail**.

## 2. Guarantee vocabulary

These labels must remain distinct in code, tables, and paper prose.

| Label | Meaning | What it does not mean |
|---|---|---|
| `ENVIRONMENT_VIOLATION` | The exact replayed environment trace contains a concrete invariant counterexample. | It is not merely model pessimism. |
| `MODEL_VIOLATION` | A sampled model trace contains a concrete invariant counterexample. | It does not prove the real environment violates unless paired replay confirms it. |
| `L2_CALIBRATED_MODEL_SCOPE` | A trained LPPM and the required held-out sampled checks pass on the declared model distribution. | It is not support-wide deductive safety or an infinite-horizon environment proof. |
| trajectory-level safety rate | One-sided Clopper-Pearson lower bound on the fraction of complete sampled trajectories with no violation. | It is not a Theorem 5.4 progress-measure warrant. |
| `L3_DISTRIBUTIONAL_WARRANT` | Lower bound on the mass of sampled-distribution states satisfying calibrated drift conditions. | It is not Theorem 5.6 and not a pathwise recurrence probability. |
| paper L3 warrant | Global/certified-region drift premises plus containment, retention, collar/initial-risk terms, and Ville's inequality. | It cannot be inferred from a distributional drift-mass estimate. |

`VerificationResult.is_safe()` now returns true only for a future genuine
support-wide deductive proof. A finite positive STL margin or a sampled L2
number is no longer silently treated as deductive safety.

## 3. Completed implementation

### 3.1 Counterexample-first evidence layer

`core/safety_evidence.py` provides formula-driven invariant witness extraction,
stable witness/action/trajectory hashes, checkpoint and anchor provenance, and
one shared precedence rule:

1. environment counterexample;
2. model counterexample;
3. deductive certificate;
4. calibrated model-scope evidence;
5. inconclusive.

`main.py`, the CarDreamer hazard evaluator, and the TD-MPC2 height evaluator use
this rule. Exact TD-MPC2 action sequences and physics-anchor hashes are retained
for paired replay. A sampled calibration statistic can no longer override an
observed strict-invariant violation, including equality at a strict boundary.

### 3.2 L2/LPPM correctness work

The current changes include:

- terminal transitions are evaluated instead of silently dropped;
- independent train/calibration split enforcement;
- safety/co-Büchi automaton corrections;
- sequential-guarantee transition corrections;
- obligation/disjunctive scope guards;
- Lemma E.6 exclusion checks;
- explicit `Z_free`-source closure diagnostics;
- a trained-backend requirement before sampled L2 evidence can be accepted;
- reusable paired model/environment counterexample handling;
- a separate trajectory-level statistical fallback in
  `core/trajectory_calibration.py`.

### 3.3 Strict paper L3/LBSM path

The exact Appendix E.2 implementation checks:

- `{U <= ell} subseteq C_cert`;
- positive `W` drift margin on the required region outside `F`;
- companion `U` retention drift;
- `F` strictly inside `I`;
- the Appendix E.24 exit precondition;
- `alpha = E_mu0[U(x0)] / ell`;
- collar handling and the final Ville warrant.

The analytic test reproduces **warrant `0.92`**. The trained sample-based path
also implements the complete
`eps_eff = eps - eta_conc - (L + L_phi) Delta` correction, but its strict
covering verifier currently applies only to the understood two-dimensional
analytic kernel, not to a real RSSM.

### 3.4 Distribution-scoped L3 alternative

The alternative implementation consists of:

- bounded learned `W/U` certificates with non-degenerate core shaping;
- learned post-expectation maps `H_W/H_U`;
- one-sided split-conformal residual calibration;
- per-anchor Monte Carlo Hoeffding correction;
- Clopper-Pearson lower confidence bound on drift-condition mass;
- four disjoint data roles: certificate training, operator training, residual
  calibration, and warrant evaluation;
- content fingerprints that reject cross-split anchor leakage;
- verdicts restricted to `L3_DISTRIBUTIONAL_WARRANT` or `INCONCLUSIVE`.

It deliberately never calls the global Theorem 5.6 verifier or Ville's
inequality.

### 3.5 CarDreamer distributional-L3 adapter

The adapter uses the full Markov product representation rather than a lossy
projection:

- RSSM deterministic state: 512 dimensions;
- flattened categorical stochastic state: 1,024 dimensions;
- current automaton acceptance bit: 1 dimension;
- normalized position probe output: 2 dimensions;
- total: **1,539 dimensions**.

The experiment uses the genuine recurrence formula `GF(zone_a)`, stochastic
actor/RSSM one-step successors, `F` strictly inside the retention set `I`, and
independent CARLA episodes for statistical calibration and warrant evaluation.

### 3.6 TD-MPC2 forward-speed probe

`forward_speed` is defined as dm_control walker's signed centre-of-mass
horizontal velocity in metres/second:

```text
Physics.horizontal_velocity()
```

This is the signal used by walker-walk's reward. It is intentionally not
`obs[15]`, which is the root generalized velocity.

The fitted artifact is a standardized Ridge model with `alpha=10` and the
numerically stable `lsqr` solver. Probe fitting and AP extraction live in
`wrappers/tdmpc2_probes.py`; the complete validation runner lives in
`tdmpc2/validate_forward_speed_probe.py`.

## 4. Current empirical evidence ledger

### 4.1 Real L2 results

| Model/spec | Progress-measure result | Diagnosis | Current citable fallback |
|---|---:|---|---:|
| CarDreamer `carla_four_lane` / `ltl_hazard_avoidance` | reproducible `p_hat_gamma=0.1227`, `k=9/40`, below 0.80 | `V` collapses near zero; only 2/60 training trajectories visit the trap; reweighting, oversampling existing examples, and removing smoothness did not repair it | trajectory safety rate `p_hat_safety=0.9243`, `k=97/100`, 95% one-sided CP lower bound |
| TD-MPC2 `walker-walk` / `ltl_height_safety` | `p_hat_gamma=0.0000`, `k=0/80`; `n_zfree=0` | `V` is near constant; calibration has no trap-visiting rollout and no observable `Z_free` closure | trajectory safety rate `p_hat_safety=0.9765`, `k=199/200`, 95% one-sided CP lower bound |

The fallback rates concern complete finite trajectories under their sampled
deployment distributions. They carry no progress-measure, absorption-witness,
or infinite-horizon guarantee.

### 4.2 Paper L3 feasibility

| Case | Status |
|---|---|
| Appendix E.2 analytic construction | implemented; warrant `0.92` |
| trained W/U on the two-dimensional analytic kernel | implemented with full Theorem D.3 correction |
| real deep recurrent checkpoint, global latent space | infeasible: covering complexity at least around `10^22` |
| most optimistic one-dimensional linear projection | `N approximately 2.95 x 10^13`, still infeasible |
| two-dimensional projection | `N approximately 2.36 x 10^22`, worse than 1-D |

Projection does not remove the recurrent network's internal Lipschitz product;
it only adds an `O(1)` readout norm after that product.

### 4.3 First real CarDreamer distributional L3 experiment

Configuration and data:

- checkpoint: `carla_roundabout_20260713`;
- formula: `GF(zone_a)`;
- 5 position-probe episodes;
- 5 certificate episodes, 184 anchors;
- 5 operator episodes, 214 anchors;
- 20 residual-calibration episodes, one anchor each;
- 40 warrant episodes, one anchor each;
- 458 total L3 anchors;
- 128 stochastic one-step RSSM successors per anchor, 58,624 successors total.

Result:

| Quantity | Value |
|---|---:|
| product-state dimension | 1,539 |
| held-out position-probe RMSE | 3.9979 m |
| held-out position-probe p90 | 6.3831 m |
| held-out zone accuracy | 0.8426, below required 0.95 |
| calibrated drift conditions met | 17/40 |
| empirical condition rate | 0.425 |
| Clopper-Pearson lower bound | 0.29185 |
| joint operator-error budget | 0.12000 |
| drift-valid mass lower bound | 0.17185 |
| required distributional threshold | 0.80 |
| core verdict | `INCONCLUSIVE` |
| report verdict | `INCONCLUSIVE_AP_PROBE` |

This run used no RSSM dynamics Lipschitz constant. The result is not evidence
that the system is unsafe; it says the collected evidence cannot support the
declared distributional threshold. The original result JSON was written to
`/tmp` and has since expired, so these values are retained from the completed
run log. The script is durable, but a new formal run is required to create a
durable result artifact.

### 4.4 TD-MPC2 forward-speed validation

Scope:

- checkpoint: `models/walker-walk-3.pt`;
- checkpoint SHA-256:
  `dba5a0545862ef5aa92b334ae43daae76294a4dedaee3d804f5f87134e8670d3`;
- upstream TD-MPC2 commit:
  `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`;
- deployment action mechanism: `mpc_plan`.

C0 uses nested episode-grouped cross-validation over 15 real episodes
(8 MPC + 7 random, 7,500 posterior states):

| Metric | Overall | MPC subset |
|---|---:|---:|
| R2 | 0.9867 | 0.9383 |
| MAE | 0.0617 m/s | 0.0339 m/s |
| p90 absolute error | 0.1471 m/s | 0.0647 m/s |
| 1 m/s threshold agreement | 99.43% | 99.65% |

C0 passes all fixed criteria. C1 uses 200 exact physics-state anchors from
8 additional MPC episodes. Model-generated MPC actions are recorded and then
replayed open loop from the identical MuJoCo state:

| Depth | p90 absolute error | 1 m/s threshold agreement |
|---:|---:|---:|
| 1 | 0.0774 m/s | 99.5% |
| 5 | 0.1279 m/s | 99.5% |
| 10 | 0.2280 m/s | 98.5% |
| 25 | 1.5142 m/s | 59.0% |
| 50 | 2.4531 m/s | 6.5% |
| 100 | 2.3679 m/s | 1.0% |

The fixed C1 criteria hold through depth 15 and fail starting at depth 16.
Final verdict:

```text
C0_VALID_C1_MODEL_TRANSFER_INCONCLUSIVE
```

The probe is validated for real posterior decoding and short-horizon
diagnostics. It must not be used for a 100-step L2 claim without a transfer
error treatment. At long horizon the model continues predicting walking speed
while the same open-loop actions cause the real walker to slow or fall.

## 5. Current AP/spec coverage

### TD-MPC2 walker-walk

- `height`: implemented and previously C0/C1 validated for the existing height
  pipeline;
- `forward_speed`: posterior probe validated; C1 supports only a 15-step prefix;
- all other catalog APs remain unavailable;
- no registered benchmark spec currently consumes the new `forward_speed`
  key.

Therefore the existing 100-step `ltl_height_safety`/`stl_height_safety` paths
remain the only directly runnable registered walker specs. A walker-native
obligation such as
`G(height > 0.6) and F(forward_speed > 1.0)` can now be represented, but only a
short bounded version is empirically supported by the current transfer test.

### CarDreamer

- `carla_four_lane`: `hazard_dist` is the only reliably evaluatable model-side
  AP for the formal LPPM path; velocity remains unavailable from decoded BEV;
- `carla_roundabout`: coordinate/zone probes and hazard CV paths exist, but
  probe validity and property base-rate caveats must be reported per spec;
- a positive monitor result is not automatically an L2/L3 formal warrant.

## 6. What still does not match the paper

| Paper requirement or claim | Current state | Required resolution |
|---|---|---|
| Global real-model L3 drift over a certified continuous region | Only analytic 2-D strict verification; real deep checkpoints are infeasible under existing bounds | Retrain a contractive/Lipschitz-controlled model, derive a new tight structured bound, or report strict real L3 as infeasible |
| Theorem 5.6 pathwise recurrence probability | Distributional L3 only lower-bounds drift-condition mass under an anchor distribution | Keep as a separately named extension and supply a formal new theorem; never relabel it Theorem 5.6 |
| Ville initial-risk term `E_mu0[U]/ell` | Absent from distributional L3 | Define `mu0`, `ell`, containment, and a valid initial-risk estimate if pursuing paper L3 |
| Certified containment and collar | Distributional core shaping is empirical, not containment proof | Add support-wide containment/collar verification or retain weaker wording |
| Genuine repeated-return/exit premise | Real distributional run checks `F subset I` but no uniform exit condition | Establish the appropriate exit/recurrence premise on the real product process |
| Guarantee for the real environment | Most successors come from learned world-model imagination | Add a formal world-model-to-environment transfer bound or paired real-transition validation |
| Reliable AP semantics | CarDreamer zone accuracy was 84.26%; TD-MPC speed fails long-horizon transfer | Improve and independently validate AP heads, then price AP/model errors in the theorem |
| Real L2 Theorem 5.4 warrant | Both current NeuralLPPMs collapse; `Z_free` closure is absent/unverifiable | Collect genuinely diverse bad-set-adjacent training data or redesign the objective; do not promote fallback trajectory rates |
| Paper's 23-spec statement | Current registry/document audit counts 30 registered specs after extensions | Freeze one canonical catalog and update every table consistently |
| Durable, fully reproducible artifacts | Some historical scripts/data were in scratch or `/tmp`; CarDreamer distributional result expired | Move runners and final artifacts into durable paths and rerun missing experiments |

Adding more samples can tighten the distributional results but does not turn
them into the original global theorem. That requires resolving the missing
structural premises, not only increasing `N`.

## 7. Reproducibility and artifacts

### Versioned in this repository

- source code and tests;
- experiment/audit Markdown documents;
- `artifacts/tdmpc2_forward_speed/forward_speed_probe.joblib`;
- `artifacts/tdmpc2_forward_speed/validation.json` and its README.

### Local but ignored by Git

- all model checkpoints (`*.pt`, `*.pth`, and `*.ckpt`), including the five
  TD-MPC2 walker checkpoints under `models/`;
- `*.npz`, including the 14 MB forward-speed C0 dataset, exact C1 physics
  anchors, and paired predictions;
- `verified_checkpoints/`, including the approximately 371 MB CarDreamer
  roundabout checkpoint;
- the previously generated CarDreamer distributional-L3 `/tmp` artifact is no
  longer present.

### External source dependencies

The forward-speed result used official TD-MPC2 commit `e9f5932`:

```bash
git clone https://github.com/nicklashansen/tdmpc2.git /tmp/tdmpc2_src
git -C /tmp/tdmpc2_src checkout e9f5932
```

CarDreamer evaluation also depends on the external `/home/bot/CarDreamer`
runtime/configuration and CARLA 0.9.15 installation; these are not vendored in
this repository.

## 8. Reproduction commands

Run the unit/regression suite:

```bash
conda run --no-capture-output -n dyno python -m pytest -q
```

Re-evaluate the cached forward-speed artifacts:

```bash
conda run --no-capture-output -n dyno python \
  tdmpc2/validate_forward_speed_probe.py --resume
```

Run a fresh forward-speed collection and validation by omitting `--resume`.

Run the real CarDreamer distributional L3 experiment after starting CARLA on
port 2100:

```bash
conda run --no-capture-output -n cardreamer python -u \
  cardreamer/eval_l3_distributional.py --port 2100
```

The CarDreamer script sets deterministic XLA flags and persists an anchor cache
under `/tmp`; for publication, override its cache/output arguments to a durable
experiment directory.

## 9. Immediate next priorities

1. Decide the paper framing: retain Theorem 5.6 as analytic/strict with real
   checkpoint infeasibility, and present distributional L3 as a separate
   extension.
2. Write and review the formal theorem for the distributional operator,
   conformal, Monte Carlo, and CP composition.
3. Rerun CarDreamer distributional L3 to a durable result path and preserve
   checkpoint/config hashes.
4. Improve CarDreamer zone AP accuracy without tuning on the warrant split.
5. If a TD-MPC2 speed obligation is desired, register a walker-native bounded
   spec with horizon at most 15; do not reuse the 100-step L2 wording.
6. Resolve L2 progress-measure collapse with genuinely new bad-set-adjacent
   training data or a non-degenerate objective.
7. Freeze the canonical spec count and synchronize the paper, README, experiment
   tables, and code registry.

## 10. Code map

| Area | Main files |
|---|---|
| shared verification entry point | `main.py` |
| safety evidence and precedence | `core/safety_evidence.py` |
| L2/LPPM | `core/lppm/` |
| trajectory-level fallback | `core/trajectory_calibration.py` |
| strict and analytic L3 | `core/lbsm/verifier.py`, `core/lbsm/trainer.py` |
| distributional L3 | `core/lbsm/distributional_*.py`, `core/lbsm/operator_*.py` |
| CarDreamer L3 adapter/run | `cardreamer/lbsm_adapter.py`, `cardreamer/eval_l3_distributional.py` |
| TD-MPC2 wrapper/probes | `wrappers/tdmpc2_wrapper.py`, `wrappers/tdmpc2_probes.py` |
| forward-speed validation | `tdmpc2/validate_forward_speed_probe.py`, `artifacts/tdmpc2_forward_speed/` |
| detailed audit/provenance | `EXPERIMENT_CONFIG.md`, `EXPERIMENT_PARAMETERS.md`, `EXPERIMENTS_TEX_AUDIT.md`, `experimental_setup.md` |
