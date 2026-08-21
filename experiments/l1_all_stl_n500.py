"""
experiments/l1_all_stl_n500.py

Runs the L1 pipeline (N_cal=N_err=N_test=500) for ALL grounded STL specs,
generating the underlying rollout data ONCE and reusing it across specs
(the raw trajectories don't depend on which spec is being checked -- only
which AP values / formula get read out of them).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.settings import RolloutConfig
from wrappers.safedreamer_wrapper import SafeDreamerWrapper
from core.stl_monitor import monitor_rollouts
from core.transfer_calibrator import fit_conformal_error_budget, calibrate_robustness_quantile
from specs import get_spec_by_id
from experiments.l1_pipeline import BASE_EXTRA

N = 500
HORIZON = 10
DELTA_CP = DELTA_ERR = 0.05

STL_SPEC_IDS = [
    "stl_speed_limit", "stl_safe_goal_reach", "stl_obstacle_response",
    "stl_gap_recovery", "stl_safe_flow_patrol", "stl_gap_response",
]

w = SafeDreamerWrapper(RolloutConfig(horizon=HORIZON, n_rollouts=N, seed=0, extra=dict(BASE_EXTRA)))
print(">>> loading...")
w.load()

t0 = time.perf_counter()
print(f">>> [N_cal] generating {N} MODEL-side rollouts (seed=1)...")
trajs_cal = w.sample_rollouts(RolloutConfig(horizon=HORIZON, n_rollouts=N, seed=1, extra=dict(BASE_EXTRA)))
print(f"    done at {time.perf_counter()-t0:.0f}s")

print(f">>> [N_err] generating {N} PAIRED rollouts (seed=2)...")
pairs_err = w.sample_paired_rollouts(RolloutConfig(horizon=HORIZON, n_rollouts=N, seed=2, extra=dict(BASE_EXTRA)))
print(f"    done at {time.perf_counter()-t0:.0f}s")

print(f">>> [N_test] generating {N} PAIRED rollouts (seed=3)...")
pairs_test = w.sample_paired_rollouts(RolloutConfig(horizon=HORIZON, n_rollouts=N, seed=3, extra=dict(BASE_EXTRA)))
env_trajs_test = [e for _, e in pairs_test]
print(f"    done at {time.perf_counter()-t0:.0f}s")

print()
print("=" * 100)
print(f"{'spec_id':<24}{'q_hat':>10}{'c_hat_err':>11}{'rho_net':>10}  {'verdict':<10}"
      f"{'test_sat':>12}{'bound':>8}  held")
print("=" * 100)

results = []
for spec_id in STL_SPEC_IDS:
    spec = get_spec_by_id(spec_id)
    aps = spec["aps"]

    mon_cal = monitor_rollouts(spec["formula"], trajs_cal)
    q_hat = calibrate_robustness_quantile(mon_cal.margins, delta_cp=DELTA_CP)

    c_hat_err = fit_conformal_error_budget(paired_rollouts=pairs_err, formula_aps=aps, delta_err=DELTA_ERR)

    rho_net = q_hat - c_hat_err
    verdict = "WARRANT" if rho_net > 0 else "VIOLATION"

    mon_test = monitor_rollouts(spec["formula"], env_trajs_test)
    test_sat_rate = mon_test.n_satisfied / N
    claimed_bound = 1.0 - DELTA_CP - DELTA_ERR
    held = test_sat_rate >= claimed_bound

    print(f"{spec_id:<24}{q_hat:>+10.4f}{c_hat_err:>11.4f}{rho_net:>+10.4f}  {verdict:<10}"
          f"{mon_test.n_satisfied:>5}/{N} ({100*test_sat_rate:.1f}%){claimed_bound:>8.2f}  {held}")

    results.append(dict(spec_id=spec_id, q_hat=q_hat, c_hat_err=c_hat_err, rho_net=rho_net,
                         verdict=verdict, test_sat=mon_test.n_satisfied, held=held))

print()
print(f">>> total wall time: {time.perf_counter()-t0:.0f}s ({(time.perf_counter()-t0)/60:.1f} min)")
