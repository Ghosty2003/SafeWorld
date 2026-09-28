# Frozen clear-window V: fresh calibration and independent test

## Fixed scope

- Specification: `F[1,17] G[0,47](decoded hazard_margin>=0)`.
- Same PointGoal1 seed0 checkpoint and local CCEPlanner as development.
- Initial distribution: decoded goal distance >=1.0. Future outcomes never filter paths.
- 64 model transitions /65 states; no real-environment future rollout comparison.
- Frozen V: `linear_max4`, update4800, from `safedreamer_l2_clear_window_v_linear6000`.
- Eta=.01; required probability=.95; individual one-sided confidence=.95.
- No fitting or model/spec selection used these new calibration/test paths.

The 48-state property permits hazard outside the chosen window. Its accepting
monitor state is absorbing; Z_free is a completion set in model/monitor product
state, not a promise of physical hazard avoidance for all future time.

## Results

| Metric | New calibration | Calibration one-sided 95% lower bound | Independent test |
|---|---:|---:|---:|
| Whole-path P1/P2 | 98/100 | 93.8381% | 97/100 |
| Full certificate AND specification completion | 98/100 | 93.8381% | 97/100 |
| Specification completion | 100/100 | 97.0487% | 100/100 |
| Entered V<eta | 100/100 | 97.0487% | 100/100 |

Full-event point forecast=98%; test frequency=97%; absolute difference=1
percentage point. This is descriptive agreement, not proof of population
accuracy or a guarantee about future finite batches. The bound concerns the
underlying same-distribution event probability under the sampling assumptions.

**Decision: NO_WARRANT for the predeclared full-certificate probability target.**
Its calibration lower bound .9383807996 is below .95. Candidate region checks
were clean. This is not a finding that the specification was violated.

The separately measured specification-completion probability has a lower bound
above .95, but this different event does not replace the full-certificate event
in the predeclared decision. No pooled calibration+test decision or post-hoc
threshold change was made.

## Transition and region diagnostics

| Metric | Calibration | Test |
|---|---:|---:|
| P1 violations /all transitions | 1/6400 | 2/6400 |
| P2 violations /waiting-source transitions | 2/4831 | 3/4815 |
| Completed but full certificate failed | 2 | 3 |
| Pending states with V<eta | 0 | 0 |
| Z_free source transitions | 1569 | 1585 |
| Region-source P1/P2 violations/exits | 0/0/0 | 0/0/0 |
| First entry step range | 48–54 | 48–55 |

Accepting value=.0025024132; minimum waiting V=.19316435 on calibration and
.19667695 on test. No incomplete path was incorrectly certified in these
samples, but all paths completed the specification: this experiment contains
no failing-specification examples with which to estimate false-positive
performance. Constant accepting output and absorbing acceptance explain region
closure; clean observed closure does not establish global validity of all
unobserved low-value states.

## Sampling and audit

Calibration seeds28503000..28503007:160raw draws,54excluded by initial
condition,6eligible surplus; first100eligible retained. Test seeds
28504000..28504007:160raw draws,48initial exclusions,12eligible surplus;
first100eligible retained. No future-success filtering or repeated sampling
until a passing statistical decision. Both splits stopped at the planned100.

Weights/spec/code/policy hashes and prior identities were frozen before draws.
The calibration forecast was written before test generation. Re-evaluation
replayed the initial selector on every saved raw batch, checked identity
disjointness and hashes, and exactly reproduced the forecast/report. An
independent explicit window scan agreed with causal monitor labels on both
splits; checkpoint hash unchanged. 50 related unit tests passed.

Duration from frozen collection plan to last saved split:8.2523minutes. Process
exit0. All model imagination ran on GPU. No calibration/test is still running.

Assumptions: same-distribution independent draws conditional on eligibility;
imagination seeds explicit but simulator-reset RNG not fully specified by
them. Bounds are individual, not simultaneous across reported metrics. These
are finite learned-model results, not a real-world or infinite-horizon proof.

Reproduce:

```bash
python experiments/l2_clear_window_holdout.py --evaluate-only
```

Files: `selected.pt`, `plan.json`, `frozen.json`, `prediction.json`,
`test_started.json`, `report.json`, both NPZ splits and raw batch provenance.
Log: `../safedreamer_l2_clear_window_holdout.log`.
