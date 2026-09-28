# Fixed-architecture W/U data × successor experiment

Use `l3_wu_data_ablation.py`; this does not modify the saved L2/L3 pilot or
candidate-family search. Keep the same CCEPlanner, RSSM, initial eligibility
(`decoded initial goal distance >= 1`), `GF G[0,47] safe`, horizon 192, W epsilon
0.01, U nonincrease, ell 0.8 and structural range [0,1]. W is the 128-wide,
3-hidden-layer MLP; U is the 64-wide, 2-hidden-layer shaped MLP. Seeds W=73,
U=41 and optimizer update budget are fixed across fit sizes. This compares
nested training-data prefixes, not independent replications of each size.

## Data roles

- Fit: up to 200 new eligible paths; each supplies one uniformly selected
  nonaccepting anchor and one accepting anchor when available. Each fit anchor
  gets 32 fresh conditional successors. All actual path transitions also enter
  the training auxiliary loss, with no synthetic terminal step.
- Development: normally 100 new paths, aiming for at least 100 W anchors and
  about 200 U anchors. Selection is stratified by automaton state, not by W/U
  outcome. Report actual counts, including missing strata.
- Final validation: **not collected or used** by this program.

All compared weights and exact development anchor identities are frozen before
development successor queries. Each query resets the complete RSSM and planner
carry to the same saved anchor. Independent PRNG streams supply chunks of 32;
kappa=8,16,32,64,128,256 uses prefixes of the SAME 256 successors. Successors
are never filtered for safety, acceptance, C membership or drift.

The paired anchors in a path are correlated; these are not independent rollout
success trials, and no Clopper–Pearson recurrence bound is computed. The
Hoeffding/union allowance is `sqrt(log(1/delta_each)/(2*kappa))`, with
`delta_each = .05/(2 * number_of_fits * number_of_anchors * number_of_kappas)`.
The simultaneous bound is conditional on the frozen models and anchors and
requires conditionally independent successor sampling. It is not a confidence
statement about infinite recurrence. Correlated prefixes do not invalidate the
union bound. Future adaptive changes require new validation or an appropriate
additional statistical design.

U is checked including accepting states (H1). Each row reports candidate-C
coverage and all-stratum raw counts so shrinking C cannot silently improve
the denominator. There is NO H2 accepting-region exemption. There is NO
certified cell Lipschitz/envelope correction, region cover/containment, initial
support proof or collar bound. `sampling_corrected` must not be called full
paper L3 correction, and global status remains ABSTAIN.

## Commands

Activate the safedreamer environment so `ptxas` is on PATH. Use a new directory.

```bash
conda activate safedreamer
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1

# Timing run first (not the requested large-sample ablation):
python experiments/l3_wu_data_ablation.py \
  --fit-counts 12 --dev-paths 8 --kappas 8 16 32 64 128 256 \
  --output artifacts/l3_wu_timing_new

# Full grid, potentially substantially longer than 20 minutes:
python experiments/l3_wu_data_ablation.py \
  --fit-counts 12 50 100 200 --dev-paths 100 \
  --kappas 8 16 32 64 128 256 --seed-base 58500000 \
  --output artifacts/l3_wu_full_new

python experiments/plot_l3_wu_ablation.py artifacts/l3_wu_full_new
```

To resume, repeat the exact same command with `--resume`. Configuration/code
hashes must match. Completed paths, query chunks and trained weights are reused;
an interrupted write without its committed JSON manifest requires manual
inspection, not silent overwrite. Simulator reset randomness is not fully
controlled by recorded imagination seeds; resumption is auditable but is not
claimed to reproduce an uninterrupted simulator reset sequence bit-for-bit.

Inspect `report.json`, `grid.csv`, per-path/query timings and frozen identities.
At fixed B, large union budgets can leave Hoeffding corrections much larger
than 0.01 even at kappa=256. Low raw pass rates implicate more than sampling
slack but do not prove architecture failure or nonexistence of valid W/U.
One training seed pair and one development pool do not establish statistical
significance of an improvement. Hard-state mining is intentionally deferred:
it changes training coverage and should be a separately labeled experiment.
