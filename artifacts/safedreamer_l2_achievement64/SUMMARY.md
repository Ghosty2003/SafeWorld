# 64-step achievement comparison

This is a model-imagination-only experiment with the existing OSRP-Vector
SafetyPointGoal1 `_0.ckpt`, random actions, and independent encoded environment
resets as initial states. There are no future real-environment comparisons.
All 380 paths contain 64 genuine transitions / 65 states. The old 50-step
experiments and default specifications are unchanged.

## Protocol

- Fresh train/development/calibration/test counts: 120 / 60 / 100 / 100.
- Spec: F_[0,64](norm(decoded goal vector) < radius), radius 0.30, 0.60, 1.00.
- 0.30 is the original task; 0.60/1.00 are relaxed approach tasks, NOT success
  at the original goal threshold. Initial satisfaction counts as satisfaction
  and is reported separately rather than silently excluded.
- V inputs: all 512 latent coordinates plus automaton completion information.
  Completion is historical (absorbing), not current physical occupancy.
- Eta = 0.01. Membership is exactly V < eta; no post-hoc threshold change,
  forced-zero output, waiting-value floor, time input, or invented final edge.
- Two variants per radius, each trained 1,500 updates and evaluated every 300:
  original paper Eq.12 / ProductValue, and an explicitly extended architecture
  and loss with a learned constant accepting branch and low-value anchor.
- In the extension, accepting value starts at softplus(-4) = 0.01815 > eta.
  The anchor trains it toward eta/4; it is NOT initialized inside Z_free.
  Waiting states get an additional ReLU(2*eta - V) penalty, not a hard floor.
- Accepting-branch constancy is a structural design choice exploiting F(goal),
  not a learned physical invariant. Original-loss versus extension therefore
  is NOT a loss-only ablation.
- Selection rule declared in plan.json before collection. Spec, architecture
  and V frozen before collecting calibration. Forecast saved before test.
- Only the selected candidate is evaluated for certificates on holdout data.
  No fitting/selection uses calibration or test; no 80% acceptance gate.

## Development comparison (60 paths)

| Radius | Reached / initially satisfied | Original-loss Z_free entries | Extended Z_free entries | Extended full C & goal |
|---|---|---|---|---|
| 0.30 | 10 / 1 | 0 | 10 | 6 |
| 0.60 | 22 / 6 | 0 | 22 | 16 |
| 1.00 | 35 / 18 | 0 | 35 | 30 |

The selected candidate is radius 1.00, `anchored_accepting_scalar`, update 900.
Its accepting value is 0.00816008 < eta. Development full C includes 18 initial
successes and 12 subsequent arrivals; selection prioritized noninitial C.
Original-loss candidates had no observed sublevel members at all 15 inspected
checkpoints. This does not prove no useful original-loss solution exists.

## Fresh prospective prediction versus test

| Metric | Calibration point estimate | Calibration one-sided 95% CP lower | Test |
|---|---|---|---|
| Reach relaxed target / enter Z_free | 71/100 = 71% | 62.63% | 58/100 = 58% |
| Full finite certificate event AND goal | 55/100 = 55% | 46.29% | 42/100 = 42% |
| Initially satisfied | 28/100 | — | 20/100 |
| Arrive later, among initially pending paths | 43/72 = 59.72% | 49.35% | 38/80 = 47.50% |
| Noninitial full certificate & goal (unconditional) | 27/100 | 19.79% | 22/100 |
| P1 violating transitions, all sources | 697/6400 | — | 890/6400 |
| P2 violating transitions, waiting sources | 863/2635 | — | 1070/3451 |

Both primary point estimates overpredict the realized test proportion by 13
percentage points. Test proportions are below the corresponding calibration
population-probability lower bounds. This is not a logical contradiction:
those bounds are NOT guarantees on the realized fraction in the next 100
paths. However, this run does not justify claiming high prediction accuracy
or validating 95% coverage from one batch. Two-sided test intervals are
[47.71%, 67.80%] for arrival and [32.20%, 52.29%] for full C & goal.

## Region interpretation

- Test: 58 paths enter Z_free; 20 are already there initially, 38 enter later.
- No pending low-value states or certificate events without goal were observed.
- All 2,949 test transitions sourced in Z_free have no P1/P2 violations or exits.
- 58/58 eligible observed suffixes pass the closure checks (conditional
  one-sided 95% CP lower 94.97%). Calibration: 70/70 (lower 95.81%); one other
  calibration path enters only at the endpoint and gives no successor evidence.
- Test waiting values: [0.07302, 6.54034]; accepting value is constant 0.00816008.
- The region is meaningful as a completion-memory product-state region for
  the RELAXED task, not a physical area that the agent must remain inside.
- This is a nonempty candidate sublevel with no violations on its observed
  sources. Its global boundary has NOT been verified: unseen waiting states
  could still have low values. Full support-wide P1/P2 are NOT established,
  and many observed waiting-source constraints fail.
- Report status: SAMPLED_DIAGNOSTICS_ONLY; region status:
  NO_VIOLATION_ON_OBSERVED_SOURCES. No support-wide/infinite-horizon warrant,
  and no claim about success for the original 0.30 task or deployment policy.

Individual CP bounds assume iid paths from the same fixed distribution and
are not simultaneous confidence guarantees or confidence in the neural V.
Imagination seeds are recorded; reset RNG is not fully controlled by them.
Trajectory fingerprints are unique and disjoint across the four splits.

## Reproduction and checks

Run `python experiments/l2_achievement64.py --phase evaluate` to verify model,
forecast and data hashes and exactly reproduce the saved report, without
training or drawing more paths. Definitions and choices are in plan.json;
all development candidates are in progress.json and frozen.json. Checkpoint
weights and all raw latent/decoded/RSSM/AP data are saved alongside report.json.

The process completed with exit code 0. MuJoCo/EGL emitted destructor warnings
at shutdown; saved-data/report re-evaluation completed successfully afterward.
