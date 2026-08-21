"""
experiments/hazard_proximity_error_check.py

Diagnostic: does the world model's per-step prediction error (imagined vs
real hazard_dist, holding the SAME action sequence fixed via paired
rollouts) grow specifically in regions close to a hazard?

Hypothesis being tested: SafeDreamer's RSSM was trained on replay data
collected by an increasingly hazard-avoidant policy, so it saw few
close-to-hazard transitions during training -> its dynamics predictions
should be systematically less accurate (larger |imagined - real| error)
in states where the real trajectory is close to a hazard, compared to
states where it is far away.

Uses sample_paired_rollouts() (random actions, same action sequence
applied to both real env and shadow imagination) so any per-step
difference is attributable to world-model prediction error, not to
different actions being taken.
"""

import sys
import time

import numpy as np

sys.path.insert(0, "/home/sunyhg/Documents/SafeWorld")

from configs.settings import RolloutConfig
from wrappers.safedreamer_wrapper import SafeDreamerWrapper
from experiments.l1_pipeline import BASE_EXTRA

N = 200
HORIZON = 10
CLOSE_THRESHOLD = 0.5   # real hazard_dist < this => "close" bucket

w = SafeDreamerWrapper(RolloutConfig(horizon=HORIZON, n_rollouts=N, seed=0, extra=dict(BASE_EXTRA)))
print(">>> loading...")
w.load()

t0 = time.perf_counter()
print(f">>> generating {N} PAIRED rollouts (seed=42)...")
pairs = w.sample_paired_rollouts(RolloutConfig(horizon=HORIZON, n_rollouts=N, seed=42, extra=dict(BASE_EXTRA)))
print(f"    done at {time.perf_counter()-t0:.0f}s")

real_dists = []
errors = []       # imagined - real, signed
abs_errors = []

for model_traj, env_traj in pairs:
    for m_step, e_step in zip(model_traj, env_traj):
        rd = e_step.get("hazard_dist")
        md = m_step.get("hazard_dist")
        if rd is None or md is None:
            continue
        real_dists.append(rd)
        err = md - rd
        errors.append(err)
        abs_errors.append(abs(err))

real_dists = np.array(real_dists)
errors = np.array(errors)
abs_errors = np.array(abs_errors)

print(f"\n>>> total (rollout, timestep) pairs with valid hazard_dist: {len(real_dists)}")

close_mask = real_dists < CLOSE_THRESHOLD
far_mask = ~close_mask

print(f"\n=== bucketed by real hazard_dist < {CLOSE_THRESHOLD} (\"close\") vs >= {CLOSE_THRESHOLD} (\"far\") ===")
for name, mask in [("close", close_mask), ("far", far_mask)]:
    n = mask.sum()
    if n == 0:
        print(f"  {name}: n=0, skipped")
        continue
    print(f"  {name:5s}: n={n:5d}  mean|error|={abs_errors[mask].mean():.4f}  "
          f"std|error|={abs_errors[mask].std():.4f}  mean(signed error)={errors[mask].mean():+.4f}")

# Pearson correlation between real_hazard_dist and abs_error (no scipy available)
if len(real_dists) > 1:
    corr = np.corrcoef(real_dists, abs_errors)[0, 1]
    print(f"\n>>> Pearson correlation(real_hazard_dist, |imagined - real|) = {corr:+.4f}")
    print("    (negative => error grows as the robot gets closer to a hazard, "
          "consistent with the OOD/under-trained-near-hazard hypothesis)")

print(f"\n>>> total wall time: {time.perf_counter()-t0:.0f}s ({(time.perf_counter()-t0)/60:.1f} min)")
