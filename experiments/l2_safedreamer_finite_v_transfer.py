"""One-shot paired model/environment comparison for the frozen finite V."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.settings import RolloutConfig
from core.transfer_calibrator import (
    compute_atomic_distortion,
    fit_conformal_error_budget,
)
from experiments.l2_safedreamer import build_extra, collect_pairs
from experiments.l2_safedreamer_finite_v import evaluate
from specs import get_spec_by_id
from utils.spec_analysis import analyze_spec_structure
from wrappers.safedreamer_wrapper import SafeDreamerWrapper


ROOT = Path(__file__).resolve().parent.parent
PREREGISTRATION = ROOT / "artifacts/safedreamer_l2_diagnostics/finite_v_transfer_preregistration.json"
ROLLOUT_OUTPUT = ROOT / "artifacts/safedreamer_l2_diagnostics/finite_v_transfer_pairs.json"
REPORT_OUTPUT = ROOT / "artifacts/safedreamer_l2_diagnostics/finite_v_transfer_report.json"


def _percentiles(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def main(args):
    prereg = json.loads(PREREGISTRATION.read_text())
    horizon = int(prereg["finite_horizon"])
    n_pairs = int(prereg["paired_rollouts"])
    seed = int(prereg["paired_seed"])
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
    print("Frozen paired-comparison configuration")
    print(json.dumps(prereg, indent=2))
    print(f"Loading checkpoint: {extra['checkpoint_path']}")
    with SafeDreamerWrapper(wrapper_config) as wrapper:
        wrapper.load()
        print(f"Collecting {n_pairs} fresh paired rollouts (seed={seed}, H={horizon})...", flush=True)
        raw_pairs = collect_pairs(
            wrapper, horizon=horizon, count=n_pairs, seed=seed, extra=extra,
        )

    # The paired API returns T+1 states: initial plus all T action successors.
    # The finite formula G[0,49] consumes exactly the first T=50 states, matching
    # sample_rollouts() and the confirmatory model-only experiment.
    pairs = [(model[:horizon], environment[:horizon]) for model, environment in raw_pairs]
    model_trajectories = [model for model, _ in pairs]
    environment_trajectories = [environment for _, environment in pairs]
    ROLLOUT_OUTPUT.write_text(json.dumps({
        "preregistration": prereg,
        "checkpoint": extra["checkpoint_path"],
        "pairs": pairs,
    }))

    spec = get_spec_by_id(prereg["spec"])
    if spec is None:
        raise ValueError(f"Unknown specification: {prereg['spec']}")
    spec = dict(spec)
    spec["analysis"] = analyze_spec_structure(spec)
    model_result = evaluate(
        model_trajectories, spec,
        horizon=horizon, eta=eta, eta_step=eta_step, gamma=gamma,
    )
    environment_result = evaluate(
        environment_trajectories, spec,
        horizon=horizon, eta=eta, eta_step=eta_step, gamma=gamma,
    )

    model_flags = [item["certificate_satisfied"] for item in model_result["per_trajectory"]]
    environment_flags = [item["certificate_satisfied"] for item in environment_result["per_trajectory"]]
    both_safe = sum(m and e for m, e in zip(model_flags, environment_flags))
    model_safe_env_unsafe = sum(m and not e for m, e in zip(model_flags, environment_flags))
    model_unsafe_env_safe = sum(not m and e for m, e in zip(model_flags, environment_flags))
    both_unsafe = sum(not m and not e for m, e in zip(model_flags, environment_flags))

    distortions = [
        compute_atomic_distortion(model, environment, ["hazard_dist"])
        for model, environment in pairs
    ]
    model_min_margins = [min(step["hazard_dist"] for step in trajectory) for trajectory in model_trajectories]
    environment_min_margins = [min(step["hazard_dist"] for step in trajectory) for trajectory in environment_trajectories]
    paired_min_margin_errors = [
        model_margin - environment_margin
        for model_margin, environment_margin in zip(model_min_margins, environment_min_margins)
    ]
    c_hat_err = fit_conformal_error_budget(pairs, ["hazard_dist"], delta_err=gamma)

    report = {
        "preregistration": prereg,
        "checkpoint": extra["checkpoint_path"],
        "model": model_result,
        "environment": environment_result,
        "paired_safety_confusion": {
            "both_safe": both_safe,
            "model_safe_environment_unsafe": model_safe_env_unsafe,
            "model_unsafe_environment_safe": model_unsafe_env_safe,
            "both_unsafe": both_unsafe,
            "agreement_rate": (both_safe + both_unsafe) / n_pairs,
        },
        "hazard_distance_error": {
            "per_pair_max_abs_error": _percentiles(distortions),
            "conformal_c_hat_err_95": c_hat_err,
            "model_minus_environment_min_margin": _percentiles(paired_min_margin_errors),
        },
        "model_scope_warrant": model_result["full_certificate"]["p_hat_gamma"] >= threshold,
        "environment_scope_warrant": environment_result["full_certificate"]["p_hat_gamma"] >= threshold,
        "paired_split_reusable_for_tuning": False,
    }
    REPORT_OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False))

    model_full = model_result["full_certificate"]
    environment_full = environment_result["full_certificate"]
    confusion = report["paired_safety_confusion"]
    error = report["hazard_distance_error"]["per_pair_max_abs_error"]
    print("\nPaired model/environment result")
    print(f"  model finite-safe: {model_full['successes']}/{model_full['n']}, "
          f"CP lower={model_full['p_hat_gamma']:.6f}")
    print(f"  environment finite-safe: {environment_full['successes']}/{environment_full['n']}, "
          f"CP lower={environment_full['p_hat_gamma']:.6f}")
    print(f"  both safe: {confusion['both_safe']}")
    print(f"  model safe / environment unsafe: {confusion['model_safe_environment_unsafe']}")
    print(f"  model unsafe / environment safe: {confusion['model_unsafe_environment_safe']}")
    print(f"  both unsafe: {confusion['both_unsafe']}")
    print(f"  paired agreement: {confusion['agreement_rate']:.3f}")
    print(f"  hazard_dist max-abs error: p50={error['p50']:.6f}, "
          f"p90={error['p90']:.6f}, p95={error['p95']:.6f}, max={error['max']:.6f}")
    print(f"  conformal c_hat_err (95%): {c_hat_err:.6f}")
    print(f"  model-scope result: {'WARRANT' if report['model_scope_warrant'] else 'NO_WARRANT'}")
    print(f"  direct environment-scope result: "
          f"{'WARRANT' if report['environment_scope_warrant'] else 'NO_WARRANT'}")
    print(f"Saved pairs: {ROLLOUT_OUTPUT}")
    print(f"Saved report: {REPORT_OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    main(parser.parse_args())
