# V-only refinement: finite model-imagination experiment

## Fixed scope

- Same PointGoal1 seed-0 SafeDreamer checkpoint and local CCEPlanner as the preceding GPU run.
- Policy acts on imagined latent states and retains planner memory. Environment resets supply initial observations only; subsequent 64 transitions are model imagination.
- Goal: decoded goal-vector norm strictly below 0.30 within states t=0,...,64.
- Eta remains 0.01. P1 checks all actual transitions; P2 additionally checks waiting-state sources.
- Candidate region remains exactly V < eta. No synthetic terminal edge, time-index input, unsafe-path filtering, or relaxed evaluation tolerances.
- Local policy cost_limit is 0.0. Historical checkpoint training configuration is not guaranteed identical to this reconstruction.

## Selection and data separation

Reused only 120 training and 60 development paths. Compared the original V and
four predeclared warm/cold MLP candidates, with 1600 updates per candidate and
development evaluation every 200 updates. Training uses an explicitly extended
mean-edge plus worst-edge-per-path loss, stronger training margin 2*eta,
accepting-scalar anchor and pending-sublevel penalty. These are not the original
paper objective alone. All training trajectories contribute.

Selected cold128_max4 at update 1200 using complete-path P1/P2 as the primary
development metric. Original/new development success: 13/60 versus 17/60.
Original/new training success: 73/120 versus 58/120; training improvement was
not the selection criterion and is not claimed.

The model was frozen before fresh calibration (seed 22503, 100 paths), then
the prediction was saved before fresh test collection (seed 22504, 100 paths).
No post-test selection or refitting occurred. Saved report reproduces exactly.
New holdouts have no full-trajectory fingerprint overlap with the prior run
or each other. This check does not prove statistical independence; confidence
statements remain conditional on iid same-distribution sampling. Simulator
reset randomness is not fully controlled by the imagination seeds.

## Results

| Event for refined V | Calibration | One-sided 95% calibration lower bound | Fresh test |
|---|---:|---:|---:|
| Complete certificate AND finite goal | 35/100 | 27.08% | 29/100 |
| Finite goal reached / candidate region entered | 62/100 | 53.32% | 59/100 |

Certificate point-estimate gap: 6 percentage points. Goal gap: 3 points.
These bounds concern population event probabilities, not guaranteed counts in
the next 100 paths and not a 95% probability that V is globally valid.

## Paired comparison on identical fresh paths

| Metric | Original V | Refined V |
|---|---:|---:|
| Calibration complete certificate | 30/100 | 35/100 |
| Test complete certificate | 22/100 | 29/100 |
| Test P1 violations | 788/6400 | 891/6400 |
| Test P2 violations | 946/4030 | 1953/4030 |

On test, seven additional paths pass, none of the original 22 passing paths
fail under refined V. This is improved observed whole-path coverage, not a
uniform improvement in transition constraints or a proof of generalization.

Test candidate-region sources: 2370 observed transitions, zero P1/P2 violations
or exits. Eligible closure paths: 59/59; individual one-sided 95% lower bound
95.05%. P2's bad-state premise does not apply after goal completion. The
accepting branch is an explicitly trained latent-independent scalar, so this
is a completion-memory sublevel, not an independently discovered physical
region in which the robot must remain near the goal forever.

Final status: SAMPLED_DIAGNOSTICS_ONLY. Support-wide validity and
infinite-horizon warrant remain NOT_ESTABLISHED. No arbitrary probability
threshold was introduced to issue a positive warrant.

See report.json, paired_v_comparison.json, frozen.json, prediction.json,
plan.json and the per-split provenance records for the saved evidence.
