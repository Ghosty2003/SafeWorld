"""One-shot held-out confirmation of the pre-registered finite-horizon V.

The candidate and all statistical settings are frozen in
artifacts/safedreamer_l2_diagnostics/finite_v_preregistration.json.  This
runner deliberately exposes no CLI flags for changing them.  The collected
split must not be reused to tune the candidate after seeing the result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.settings import RolloutConfig
from experiments.l2_safedreamer import build_extra, collect_model_rollouts
from experiments.l2_safedreamer_finite_v import evaluate
from specs import get_spec_by_id
from utils.spec_analysis import analyze_spec_structure
from wrappers.safedreamer_wrapper import SafeDreamerWrapper


ROOT = Path(__file__).resolve().parent.parent
PREREGISTRATION = ROOT / "artifacts/safedreamer_l2_diagnostics/finite_v_preregistration.json"
ROLLOUT_OUTPUT = ROOT / "artifacts/safedreamer_l2_diagnostics/finite_v_confirm_rollouts.json"
REPORT_OUTPUT = ROOT / "artifacts/safedreamer_l2_diagnostics/finite_v_confirm_report.json"


def main(args):
    prereg = json.loads(PREREGISTRATION.read_text())
    horizon = int(prereg["finite_horizon"])
    n_cal = int(prereg["calibration_rollouts"])
    seed = int(prereg["calibration_seed"])
    eta = float(prereg["eta_required"])
    eta_step = float(prereg["eta_step"])
    gamma = float(prereg["gamma"])
    threshold = float(prereg["warrant_threshold"])

    extra_args = argparse.Namespace(repo_root=args.repo_root, checkpoint=args.checkpoint)
    extra = build_extra(extra_args)
    wrapper_config = RolloutConfig(
        horizon=horizon,
        n_rollouts=1,
        seed=seed,
        action_source="random",
        extra=dict(extra),
    )
    print("Frozen confirmatory configuration")
    print(json.dumps(prereg, indent=2))
    print(f"Loading checkpoint: {extra['checkpoint_path']}")
    with SafeDreamerWrapper(wrapper_config) as wrapper:
        wrapper.load()
        print(f"Collecting {n_cal} fresh held-out rollouts (seed={seed}, H={horizon})...", flush=True)
        calibration = collect_model_rollouts(
            wrapper, horizon=horizon, count=n_cal, seed=seed, extra=extra,
        )

    ROLLOUT_OUTPUT.write_text(json.dumps({
        "preregistration": prereg,
        "checkpoint": extra["checkpoint_path"],
        "calibration": calibration,
    }))

    spec = get_spec_by_id(prereg["spec"])
    if spec is None:
        raise ValueError(f"Unknown specification: {prereg['spec']}")
    spec = dict(spec)
    spec["analysis"] = analyze_spec_structure(spec)
    result = evaluate(
        calibration,
        spec,
        horizon=horizon,
        eta=eta,
        eta_step=eta_step,
        gamma=gamma,
    )
    result["preregistration"] = prereg
    result["checkpoint"] = extra["checkpoint_path"]
    result["confirmatory_split_reusable_for_tuning"] = False
    result["warrant_threshold"] = threshold
    result["model_scope_warrant"] = result["full_certificate"]["p_hat_gamma"] >= threshold
    REPORT_OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False))

    p12 = result["p1_p2"]
    terminal = result["terminal"]
    full = result["full_certificate"]
    print("\nConfirmatory result")
    print(f"  P1 violations: {p12['p1_violations']}/{p12['p1_checked']}")
    print(f"  P2 violations: {p12['p2_violations']}/{p12['p2_checked']}")
    print(f"  P1/P2 successes: {p12['trajectory_successes']}/{p12['n']}")
    print(f"  P1/P2 CP lower (95%): {p12['cp_lower']:.6f}")
    print(f"  accepting/even terminals: {terminal['accepting_even']}/{terminal['n']}")
    print(f"  odd-trap terminals: {terminal['odd_trap']}/{terminal['n']}")
    print(f"  full certificate successes: {full['successes']}/{full['n']}")
    print(f"  full p_hat_gamma (95% lower): {full['p_hat_gamma']:.6f}")
    print(f"  model-scope result: {'FINITE_HORIZON_WARRANT' if result['model_scope_warrant'] else 'NO_WARRANT'}")
    print(f"Saved rollouts: {ROLLOUT_OUTPUT}")
    print(f"Saved report: {REPORT_OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    main(parser.parse_args())
