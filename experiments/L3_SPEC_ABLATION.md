# L3 specification comparison on shared SafeDreamer dynamics

`l3_spec_ablation.py` compares five monitors:

- A: GF S48, S48 = 48 consecutive states with decoded hazard margin >= 0.
- B: GF S16, GF S24, GF S32, with the same predicate.
- C: GF p, p = decoded hazard margin >= 0.2 (nearest decoded hazard center
  distance >= 0.4). This threshold was screened on TRAINING paths only;
  11/12 training paths exhibited an exit. It is not a claimed invariant set.

Reuse the 12 fit and 8 development paths from the data/kappa timing study.
Nothing is sampled in the real environment. Each path is already a 192-step
RSSM + CCEPlanner rollout from the same initial distribution.

For every path, select one accepting and one nonaccepting time for each
monitor when available. All five monitors use the same UNION of physical
anchor times. Query the RSSM/planner only once per anchor, then reconstruct
each monitor's counter on the same decoded successors. No absorbing accepting
state: a violated predicate resets the counter, including after acceptance.
Full latent and planner carry are retained. Counter is normalized by its own
window length; all other 577 product coordinates are shared unchanged.

The architecture, optimizer step budget, random initialization seeds, W
epsilon 0.01, ell 0.8 and structural bound 1 are fixed. The heuristic W target
is .48*(1-normalized progress) in every group; U uses identical sublevel
shaping. Thus this is a controlled practical comparison with this fixed
training recipe, not a proof that one specification inherently lacks a
certificate. Fit queries have kappa=32; fresh development queries have
kappa=256 after ALL five model pairs and anchors are frozen. The sampling
allowance allocates .05 across both functions, all five pairs and every
common anchor. No cellwise correction/global proof is implemented.

The primary empirical behavior proxy is at least two NONOVERLAPPING complete
windows within the finite path. Nonoverlapping does not mean statistically
independent. Report late-window and actual exit/return counts separately.
Always-safe paths satisfy GF S_L; recurrence never requires a hazard between
windows. These proxies are not infinite recurrence satisfaction rates.

Report raw and sampling-only corrected W/U counts with denominators and C
coverage. Since accepting sets differ, W-required counts differ even on the
same physical anchors. `l3_spec_ablation_audit.py` additionally reports W on
the common nonaccepting subset for the four window specs, with common C
membership. This is descriptive, not a new independent validation set.

```bash
conda activate safedreamer
XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
python experiments/l3_spec_ablation.py --kappa 256 --steps 400 \
  --output artifacts/l3_spec_comparison_new

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
python experiments/l3_spec_ablation_audit.py artifacts/l3_spec_comparison_new
```

Outputs: fixed plan and code hashes, raw shared query chunks, all 10 weight
files, freeze manifest, report.json, comparison.csv and audit.json. Existing
paths/pilot artifacts are read-only. This is development/spec screening;
it does not touch final validation, issue a recurrence probability, or make
the missing global proof obligations disappear.
