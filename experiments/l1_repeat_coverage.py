"""
experiments/l1_repeat_coverage.py

Repeats the L1 pipeline (experiments/l1_pipeline.py::run_l1) K times with
disjoint seeds each time, and checks how often the claimed lower bound
(1 - delta_cp - delta_err) actually held against that run's independent
N_test real-environment data.

This is the "held-out coverage validity" check, generalized from the
one-shot version done earlier this session (that one-shot check found 80%
empirical coverage against a 95% claim -- this repeats it K times to see
whether that was a fluke or a real pattern).
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/sunyhg/Documents/SafeWorld")

from configs.settings import RolloutConfig
from wrappers.safedreamer_wrapper import SafeDreamerWrapper
from experiments.l1_pipeline import run_l1, format_result, BASE_EXTRA

SPEC_ID = "stl_hazard_avoidance"
K = 5
N_CAL = N_ERR = N_TEST = 20
HORIZON = 10

w = SafeDreamerWrapper(RolloutConfig(horizon=HORIZON, n_rollouts=N_CAL, seed=0, extra=dict(BASE_EXTRA)))
print(">>> loading...")
w.load()

results = []
t_start = time.perf_counter()
for k in range(K):
    t0 = time.perf_counter()
    # Disjoint seed block per repeat: repeat k uses seeds (10k+1, 10k+2, 10k+3)
    # so no repeat's N_cal/N_err/N_test rollouts overlap with any other repeat's.
    r = run_l1(
        w, SPEC_ID, horizon=HORIZON, n_cal=N_CAL, n_err=N_ERR, n_test=N_TEST,
        seed_cal=10 * k + 1, seed_err=10 * k + 2, seed_test=10 * k + 3,
    )
    dt = time.perf_counter() - t0
    results.append(r)
    print(f">>> repeat {k+1}/{K} done in {dt:.1f}s -- rho_net={r.rho_net:+.4f} verdict={r.verdict} "
          f"test_sat={r.test_n_satisfied}/{r.n_test} held={r.held}")

total_dt = time.perf_counter() - t_start
n_held = sum(1 for r in results if r.held)
n_violated_bound = K - n_held

print()
print("=" * 70)
print(f"K={K} 次重复结果汇总（spec={SPEC_ID}, N_cal=N_err=N_test={N_CAL}）")
print("=" * 70)
print(f"  总耗时: {total_dt:.1f}s  (平均每次 {total_dt/K:.1f}s)")
print(f"  {K}次里，声称下界成立(held=True)的次数: {n_held}/{K}")
print(f"  {K}次里，声称下界被违反(held=False)的次数: {n_violated_bound}/{K}")
print(f"  违反比例: {100*n_violated_bound/K:.1f}%  (理论上应该 <= delta_cp+delta_err = 10%)")
