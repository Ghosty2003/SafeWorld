# Goal distance <0.30: 100-step imagination screen (no V)

Same SafeDreamer PointGoal1 seed0 checkpoint and local CCEPlanner as the recent
experiments. Planner lookahead remains15. Only the outer rollout length is100
transitions (101 states). Initial decoded goal distance>=1.0. Goal predicate is
strict decoded distance<0.30, equivalently stored goal AP<0. No V was loaded,
fitted or evaluated. No P1/P2, calibration bound or warrant was calculated.

## Matched-path results

| Metric | Result |
|---|---:|
| Accepted new paths | 100 |
| Already in goal at t0 | 0 |
| Ever arrived by step64 | 50/100 (50%) |
| Ever arrived by step100 | 54/100 (54%) |
| Newly arrived during65..100 | 4/100 (4 percentage points) |
| Never arrived by100 | 46/100 |
| In goal at exactly step100 | 1/100 |
| Median first arrival among54successful paths | 27.5steps |

The extra arrivals first reached at steps70,71,88,93 (zero-based path indices
34,73,37,90 respectively). Both horizon rates use the SAME100paths, not different
samples. Once reached, the eventual-goal property remains satisfied even if the
agent subsequently leaves the radius; final-step occupancy is a different metric.

Extending64to100 increased this batch's arrival rate modestly; it did not raise it
near98–99%. This is a descriptive model-only result, not a measured real-environment
success rate, model accuracy estimate or infinite-horizon claim. It does not change
or replace the earlier clear-window specification or its frozen certificate.

## Sampling and reproducibility

Seven batches, seeds29505000..29505006, yielded140raw paths.39were rejected solely
because decoded initial distance<1.0;1eligible surplus path was beyond the quota.
First100eligible paths in draw order were retained, including all future failures.
Runtime about5.01minutes; GPU collection; process exit0. Explicit imagination seeds
do not completely specify simulator-reset randomness.

Saved: `plan.json`, `policy_config.json`, `rollouts.npz`, `raw/`, `provenance.json`,
`report.json`. The source checkpoint and policy were checked before collection;
runtime policy matched the stored configuration. Reevaluation checked raw/accepted
hashes, replayed the initial-only selector, compared all accepted arrays, and
cross-checked goal APs against decoded goal vectors.11related tests passed.

```bash
cd /home/sunyhg/Documents/SafeWorld
/home/sunyhg/miniconda3/envs/safedreamer/bin/python experiments/goal03_policy100.py --evaluate-only
```

Log: `../safedreamer_goal03_policy100.log`.
