# Paper-style learned sublevel: finite 50-step rerun

This is a new experiment; previous artifacts are preserved. The implementation
uses paper Eq. (3), (4), and (10)-(12), not an infinite-horizon proof procedure.
Both waiting and accepting branches of V are trainable. No acceptance clamp,
waiting floor, time feature, goal-shaping loss, or artificial terminal edge
is used. Z_free membership is solely V(z,q) < 0.01.

## Protocol

- Existing training: 180 paths; development validation: 100 paths.
- Full 512-dimensional latent state plus automaton state as inputs.
- Three configurations, 2400 updates each, 24 checkpoint evaluations.
- Checkpoint selection: minimum validation mean P1/P2 residual, then train
  residual. This selects `paper_w128_s101`, update 300, not the checkpoint
  with minimum training violation count. Every evaluated checkpoint had
  zero validation paths entering the sublevel.
- Independent calibration: 100 paths, imagination RNG seed 15301.
- Forecast saved before independent test collection: 100 paths, seed 15302.
- Fixed SafeDreamer checkpoint, random actions, decoded goal radius 0.30.
- Each path has 51 states and 50 genuine transitions. Initial goal counts
  are included in the headline goal event (calibration 3, test 2).

## Results

| Metric | Calibration | Independent test |
|---|---:|---:|
| Goal reached by step 50 | 18/100 | 10/100 |
| Candidate Z_free entered | 0/100 | 0/100 |
| Entire-path P1/P2 pass | 0/100 | 0/100 |
| Candidate certificate event, Eq. (4) | 0/100 | 0/100 |
| P1 violating transitions (ALL sources) | 872/5000 | 886/5000 |
| P2 violating transitions (waiting sources) | 1637/4236 | 1805/4602 |
| Minimum observed V | 0.217389 | 0.208137 |

The calibration one-sided 95% CP lower bound is 0 for candidate certificate
events and 0.119691 for finite goal success. These are distinct events.
The latter bound concerns the population probability, not a guaranteed
success count for the next 100 paths. Statistical interpretation requires
the usual independent same-distribution path assumptions.

## Interpretation

No useful sampled Z_free was obtained in this run. All observed V values
exceed eta; the goal-done values were not artificially set to zero. No
sublevel source transitions were observed, so closure is UNASSESSABLE,
not passed. This does not prove that the sublevel is globally empty or that
a better certificate cannot exist. P1/P2 also have many sampled violations.

One limitation of the objective is that a positive constant offset preserves
P1/P2 differences and latent gradients, but can remove sublevel membership.
The paper loss alone does not require a nonempty observed sublevel. We did
not shift V or change eta after observing calibration/test results.

The zero forecast matching zero observed certificate successes does not
demonstrate a useful certificate. No support-wide or infinite-horizon
warrant is established; no 80% acceptance threshold is used to label this
experiment's result.

## Verification

Eight new learned-sublevel unit checks and five existing finite-transition
checks pass. Saved-model predictions and the report reproduce exactly.
Model/forecast/data hashes and disjoint full-path fingerprints were checked.
The forecast timestamp precedes test collection.

Reproduce saved metrics without training or new sampling:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python experiments/l2_paper_sublevel_finite.py --phase evaluate
```
