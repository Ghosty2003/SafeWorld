"""
experiments/onesided_horizon50_pilot.py

Pilot comparison for stl_hazard_avoidance:
  - horizon: 50 (matches the spec's native design and the paper), not 10
  - N_cal = N_err = N_test = 100 (pilot scale, not full 500)
  - TWO ways of computing c_hat_err on the SAME N_err data:
      (a) "old" -- core.transfer_calibrator.fit_conformal_error_budget(),
          unmodified, two-sided |model - real| distortion (as used all session)
      (b) "new" -- one-sided distortion, only penalizing the direction where
          the model UNDERSTATES danger (model_hazard_dist > real_hazard_dist,
          i.e. model thinks the robot is farther from the hazard than it
          really is). The opposite direction (model MORE pessimistic than
          real) is not penalized, since it can't cause a false safety claim.

This is a narrow, spec-specific patch living entirely in this experiment
script -- core/transfer_calibrator.py is NOT modified (per this session's
standing instruction not to touch existing SafeWorld core files).

Only valid for stl_hazard_avoidance's exact formula (G[0,49](hazard_dist>0),
no negation, single atom) where the "bad" direction is unambiguous. This
one-sided trick is NOT generally correct for arbitrary STL formulas (see
this session's discussion of sign propagation through negation/implies).
"""

import math
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

SPEC_ID = "stl_hazard_avoidance"
N = 100
HORIZON = 50
DELTA_CP = DELTA_ERR = 0.05


def one_sided_distortion(model_traj, env_traj, ap_key="hazard_dist"):
    """
    max_t max(0, model_hazard_dist(t) - real_hazard_dist(t))
    Only penalizes the model claiming MORE distance (more safety) than
    reality -- the direction that could cause a false WARRANT.
    """
    T = min(len(model_traj), len(env_traj))
    worst = 0.0
    for t in range(T):
        m = model_traj[t].get(ap_key, 0.0)
        e = env_traj[t].get(ap_key, 0.0)
        d = m - e
        if d > worst:
            worst = d
    return worst


def fit_one_sided_error_budget(paired_rollouts, ap_key="hazard_dist", delta_err=0.05):
    scores = [one_sided_distortion(m, e, ap_key) for m, e in paired_rollouts]
    n = len(scores)
    sorted_scores = sorted(scores)
    idx = math.ceil((1.0 - delta_err) * (n + 1)) - 1
    idx = max(0, min(idx, n - 1))
    return sorted_scores[idx], scores


w = SafeDreamerWrapper(RolloutConfig(horizon=HORIZON, n_rollouts=N, seed=0, extra=dict(BASE_EXTRA)))
print(">>> loading...")
w.load()

spec = get_spec_by_id(SPEC_ID)
aps = spec["aps"]
assert aps == ["hazard_dist"], aps

t0 = time.perf_counter()
print(f">>> [N_cal] generating {N} MODEL-side rollouts (seed=1, horizon={HORIZON})...")
trajs_cal = w.sample_rollouts(RolloutConfig(horizon=HORIZON, n_rollouts=N, seed=1, extra=dict(BASE_EXTRA)))
print(f"    done at {time.perf_counter()-t0:.0f}s")

mon_cal = monitor_rollouts(spec["formula"], trajs_cal)
q_hat = calibrate_robustness_quantile(mon_cal.margins, delta_cp=DELTA_CP)
print(f"    q_hat_delta_cp = {q_hat:+.4f}   (model raw satisfaction: {mon_cal.n_satisfied}/{N})")

print(f">>> [N_err] generating {N} PAIRED rollouts (seed=2, horizon={HORIZON})...")
pairs_err = w.sample_paired_rollouts(RolloutConfig(horizon=HORIZON, n_rollouts=N, seed=2, extra=dict(BASE_EXTRA)))
print(f"    done at {time.perf_counter()-t0:.0f}s")

c_hat_err_old = fit_conformal_error_budget(paired_rollouts=pairs_err, formula_aps=aps, delta_err=DELTA_ERR)
c_hat_err_new, raw_scores_new = fit_one_sided_error_budget(pairs_err, ap_key="hazard_dist", delta_err=DELTA_ERR)

rho_net_old = q_hat - c_hat_err_old
rho_net_new = q_hat - c_hat_err_new
verdict_old = "WARRANT" if rho_net_old > 0 else "VIOLATION"
verdict_new = "WARRANT" if rho_net_new > 0 else "VIOLATION"

print(f">>> [N_test] generating {N} PAIRED rollouts (seed=3, horizon={HORIZON})...")
pairs_test = w.sample_paired_rollouts(RolloutConfig(horizon=HORIZON, n_rollouts=N, seed=3, extra=dict(BASE_EXTRA)))
env_trajs_test = [e for _, e in pairs_test]
print(f"    done at {time.perf_counter()-t0:.0f}s")

mon_test = monitor_rollouts(spec["formula"], env_trajs_test)
test_sat_rate = mon_test.n_satisfied / N
claimed_bound = 1.0 - DELTA_CP - DELTA_ERR

print()
print("=" * 90)
print(f"stl_hazard_avoidance -- horizon={HORIZON}, N_cal=N_err=N_test={N}")
print("=" * 90)
print(f"q_hat_delta_cp = {q_hat:+.4f}")
print()
print(f"{'method':<12}{'c_hat_err':>12}{'rho_net':>12}  {'verdict':<10}")
print(f"{'OLD(2-sided)':<12}{c_hat_err_old:>12.4f}{rho_net_old:>+12.4f}  {verdict_old:<10}")
print(f"{'NEW(1-sided)':<12}{c_hat_err_new:>12.4f}{rho_net_new:>+12.4f}  {verdict_new:<10}")
print()
print(f"N_test real-env satisfaction: {mon_test.n_satisfied}/{N} ({100*test_sat_rate:.1f}%)   "
      f"claimed bound: {claimed_bound:.2f}")
print()
print(">>> for reference, previous horizon=10, N=500, OLD method result was:")
print("    q_hat=-0.1433  c_hat_err=0.7560  rho_net=-0.8993  VIOLATION  "
      "real=497/500 (99.4%)")
print()
print(f">>> total wall time: {time.perf_counter()-t0:.0f}s ({(time.perf_counter()-t0)/60:.1f} min)")
