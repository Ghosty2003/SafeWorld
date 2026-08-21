"""
experiments/model_belief_check.py

Answers a narrower question than the full L1 pipeline: out of SafeDreamer's
OWN imagined rollouts (model-only, no real env involved), how many does the
model itself "believe" satisfy each spec (raw STL satisfaction, margin >= 0,
no conformal calibration/shrinkage applied at all)?

This is compared against the N_test real-environment satisfaction rate we
already measured in l1_all_stl_n500.py / l1_pipeline.py, to answer: does the
model's own imagination underestimate how often the spec actually holds in
reality?
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.settings import RolloutConfig
from wrappers.safedreamer_wrapper import SafeDreamerWrapper
from core.stl_monitor import monitor_rollouts
from specs import get_spec_by_id
from experiments.l1_pipeline import BASE_EXTRA

N = 500
HORIZON = 10

STL_SPEC_IDS = [
    "stl_speed_limit", "stl_safe_goal_reach", "stl_obstacle_response",
    "stl_gap_recovery", "stl_safe_flow_patrol", "stl_gap_response",
    "stl_hazard_avoidance",
]

w = SafeDreamerWrapper(RolloutConfig(horizon=HORIZON, n_rollouts=N, seed=0, extra=dict(BASE_EXTRA)))
print(">>> loading...")
w.load()

t0 = time.perf_counter()
print(f">>> generating {N} MODEL-side-only rollouts (seed=1, same seed as N_cal in prior runs)...")
trajs_cal = w.sample_rollouts(RolloutConfig(horizon=HORIZON, n_rollouts=N, seed=1, extra=dict(BASE_EXTRA)))
print(f"    done at {time.perf_counter()-t0:.0f}s")

print()
print("=" * 90)
print(f"{'spec_id':<24}{'model believes satisfied':>28}{'mean_margin':>14}{'std_margin':>13}")
print("=" * 90)

for spec_id in STL_SPEC_IDS:
    spec = get_spec_by_id(spec_id)
    mon = monitor_rollouts(spec["formula"], trajs_cal)
    rate = mon.n_satisfied / N
    print(f"{spec_id:<24}{mon.n_satisfied:>10}/{N} ({100*rate:>5.1f}%){mon.mean_margin:>14.4f}{mon.std_margin:>13.4f}")

print()
print(f">>> total wall time: {time.perf_counter()-t0:.0f}s ({(time.perf_counter()-t0)/60:.1f} min)")
