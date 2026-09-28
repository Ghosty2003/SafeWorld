# Policy-verification implementation check

Real checkpoint: OSRP-Vector SafetyPointGoal1 seed 0. A full 64-transition
model-imagination path was generated using the local configured CCEPlanner,
not uniform random actions or the bare actor. Every recorded action exactly
matches the first action of the plan returned on that step. The warm-start
planner state was retained, with 65 aligned latent/planner snapshots.

The planner settings were not reduced: horizon 15, 500 candidates, 50 elites,
6 iterations, actor-mixture coefficient 0.05, momentum 0.1, initial std 1.0.
Sampling plus first compilation took 105.13 seconds on CPU; this is not a
measurement of steady-state per-path runtime. A separate 2-step check passed.

The original <0.30 goal and 64-step finite horizon are now defaults in the
achievement experiment. Policy and random results use different directories.
Training, development, calibration and testing all select the same action
source. Policy configuration/source hashes are checked across splits.
Old random-action datasets and latent-only V models are not used for new
policy calibration. Their four historical reports and comparison reproduced
exactly after the implementation change.

V's continuous input is now 577 dimensions: latent 512 + plan mean 32 + plan
std 32 + initialized flag 1, with q supplied separately. This retains controller
memory needed to describe the closed-loop process. Missing planner state is an
error. Rollouts also save actual actions for audit. The outer Ninjax scan avoids
Python-unrolling 64 nested planners, with JIT reuse within a loaded wrapper.

23 relevant unit/regression tests passed; git diff --check passed.
This is an integration test ONLY, not a new success-rate estimate, certificate
calibration or warrant. The full 120/60/100/100 policy experiment has NOT run.

Important limitation: the local reconstructed config uses cost_limit=0, while
CCEPlanner uses strict cost<cost_limit and nonnegative decoded hazard costs.
Thus its elite selection takes the minimum-cost branch instead of the
reward-ranking branch. This behavior was not silently modified. The checkpoint
has no accompanying original-run config, so exact historical deployed behavior
is not established. Model-latent feedback is also not real observation feedback.

From SafeWorld, a full single-checkpoint policy experiment can be started with:

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/safeworld-mpl OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=2 /home/sunyhg/miniconda3/envs/safedreamer/bin/python -u \
  experiments/l2_achievement64.py --action-source policy
```

This will create artifacts/safedreamer_l2_policy64 and refuse overwriting an
existing run. It uses the explicit anchored-scalar extension as one training
candidate alongside the original loss; policy dispatch does not change that
experimental distinction.
