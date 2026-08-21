"""
experiments/l1_pipeline.py

L1 (bounded STL) verification pipeline matching the new paper draft's
Algorithm 1 / Table 1 exactly:

  Require: model M (resettable-generative), labeling L, spec phi,
           disjoint fit/calibration splits
  1. if phi bounded STL:
       calibrate rho_net = q_hat_delta_cp - c_hat_err
       return WARRANT if rho_net > 0 else VIOLATION

Table 1's L1 row: "exchangeable calibration; paired env. rollouts for c_hat_err"
-- i.e. TWO separate, non-overlapping data splits are load-bearing premises:
  - N_cal : MODEL-side only rollouts -> q_hat_delta_cp (Eq. 1: finite-sample
            lower quantile of {rho(phi, tau_model_i, 0)})
  - N_err : PAIRED (model, env) rollouts -> c_hat_err (split-conformal upper
            quantile of per-pair distortion)
  - N_test: held out entirely, used only to empirically check whether the
            claimed lower bound (1 - delta_cp - delta_err) actually holds
            against real environment outcomes.

All three splits are drawn as INDEPENDENT calls (different seeds) rather than
literally partitioning one big batch -- functionally equivalent for i.i.d.
sampled rollouts, and simpler to implement given SafeDreamerWrapper's API.

IMPORTANT (this session's hard-won lesson): every rollout, whether used for
N_cal, N_err, or N_test, MUST start from a real-env-encoded latent (the
"encoder bridge" in safedreamer_wrapper.py's _imagine()/sample_paired_rollouts())
-- NOT from the model's fixed wm.rssm.initial() point. Confirmed by trial
earlier this session: using the fixed point for N_cal-type rollouts produced
a 7/20 (35%) satisfaction rate on the SAME checkpoint that showed 84/100 (84%)
via the encoder-bridge path -- a large, spurious gap purely from a mismatched
starting-point distribution between calibration and prediction data. Both
sample_rollouts() and sample_paired_rollouts() already use the encoder bridge
as of this session's fix, so calling either gives exchangeable data by
construction; this comment exists so nobody "simplifies" it back out.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

sys.path.insert(0, "/home/sunyhg/Documents/SafeWorld")

from configs.settings import RolloutConfig
from wrappers.safedreamer_wrapper import SafeDreamerWrapper
from core.stl_monitor import monitor_rollouts
from core.transfer_calibrator import fit_conformal_error_budget, calibrate_robustness_quantile
from specs import get_spec_by_id

BASE_EXTRA = {
    "repo_root": "/home/sunyhg/Documents/SafeDreamer",
    "checkpoint_path": "/home/sunyhg/Documents/SafeDreamer/checkpoint/"
                        "20240307-010600_osrp_vector_safetygymcoor_"
                        "SafetyPointGoal1-v0_0.ckpt",
    "method": "osrp_vector",
    "task": "safetygymcoor_SafetyPointGoal1-v0",
    "action_source": "random",
    "config_overrides": {"jax": {"platform": "cpu"}},
}


@dataclass
class L1Result:
    spec_id: str
    n_cal: int
    n_err: int
    n_test: int
    q_hat_delta_cp: float
    c_hat_err: float
    rho_net: float
    verdict: str                    # "WARRANT" | "VIOLATION"
    claimed_lower_bound: float       # 1 - delta_cp - delta_err
    test_empirical_sat_rate: float   # real env satisfaction rate on N_test
    test_n_satisfied: int
    held: bool                       # test_empirical_sat_rate >= claimed_lower_bound


def run_l1(
    w: SafeDreamerWrapper,
    spec_id: str,
    horizon: int = 10,
    n_cal: int = 100,
    n_err: int = 100,
    n_test: int = 100,
    delta_cp: float = 0.05,
    delta_err: float = 0.05,
    seed_cal: int = 1,
    seed_err: int = 2,
    seed_test: int = 3,
) -> L1Result:
    spec = get_spec_by_id(spec_id)

    # --- N_cal: MODEL-side only rollouts -> q_hat_delta_cp ---
    cfg_cal = RolloutConfig(horizon=horizon, n_rollouts=n_cal, seed=seed_cal, extra=dict(BASE_EXTRA))
    trajs_cal = w.sample_rollouts(cfg_cal)
    mon_cal = monitor_rollouts(spec["formula"], trajs_cal)
    q_hat = calibrate_robustness_quantile(mon_cal.margins, delta_cp=delta_cp)

    # --- N_err: PAIRED rollouts -> c_hat_err ---
    cfg_err = RolloutConfig(horizon=horizon, n_rollouts=n_err, seed=seed_err, extra=dict(BASE_EXTRA))
    pairs_err = w.sample_paired_rollouts(cfg_err)
    c_hat_err = fit_conformal_error_budget(
        paired_rollouts=pairs_err, formula_aps=spec["aps"], delta_err=delta_err,
    )

    rho_net = q_hat - c_hat_err
    verdict = "WARRANT" if rho_net > 0 else "VIOLATION"

    # --- N_test: held out, REAL env only, ground-truth check ---
    cfg_test = RolloutConfig(horizon=horizon, n_rollouts=n_test, seed=seed_test, extra=dict(BASE_EXTRA))
    pairs_test = w.sample_paired_rollouts(cfg_test)
    env_trajs_test = [e for _, e in pairs_test]
    mon_test = monitor_rollouts(spec["formula"], env_trajs_test)
    test_sat_rate = mon_test.n_satisfied / n_test

    claimed_lower_bound = 1.0 - delta_cp - delta_err

    return L1Result(
        spec_id=spec_id, n_cal=n_cal, n_err=n_err, n_test=n_test,
        q_hat_delta_cp=q_hat, c_hat_err=c_hat_err, rho_net=rho_net, verdict=verdict,
        claimed_lower_bound=claimed_lower_bound,
        test_empirical_sat_rate=test_sat_rate, test_n_satisfied=mon_test.n_satisfied,
        held=(test_sat_rate >= claimed_lower_bound),
    )


def format_result(r: L1Result) -> str:
    return (
        f"[{r.spec_id}] N_cal={r.n_cal} N_err={r.n_err} N_test={r.n_test}\n"
        f"  q_hat_delta_cp = {r.q_hat_delta_cp:+.4f}\n"
        f"  c_hat_err      = {r.c_hat_err:.4f}\n"
        f"  rho_net        = {r.rho_net:+.4f}  -> {r.verdict}\n"
        f"  claimed lower bound (1-delta_cp-delta_err) = {r.claimed_lower_bound:.3f}\n"
        f"  N_test empirical real-env satisfaction     = {r.test_n_satisfied}/{r.n_test} "
        f"({100*r.test_empirical_sat_rate:.1f}%)\n"
        f"  bound holds on this test set: {r.held}"
    )


if __name__ == "__main__":
    w = SafeDreamerWrapper(RolloutConfig(horizon=10, n_rollouts=20, seed=0, extra=dict(BASE_EXTRA)))
    print(">>> loading...")
    w.load()
    print(">>> running L1 pipeline for stl_hazard_avoidance (small scale: 20/20/20)...")
    result = run_l1(w, "stl_hazard_avoidance", horizon=10, n_cal=20, n_err=20, n_test=20)
    print()
    print(format_result(result))
