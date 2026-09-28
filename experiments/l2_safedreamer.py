"""Run the sampled L2/LPPM pipeline on a SafeDreamer checkpoint.

The train and calibration rollouts are collected by separate wrapper calls so
the fitted LPPM is never calibrated on the same trajectories it saw in
training.  A third, independent paired split estimates model/environment
transfer error.

Use --diagnose-after-violation to finish Safety-spec LPPM diagnostics when
verification stops at a counterexample. The original verdict is retained.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.settings import RolloutConfig
from core.lppm import build_parity_automaton, calibrate_lppm, fit_lppm
from core.lppm.verifier import iter_transitions, run_product_trajectory
from core.safety_evidence import ENVIRONMENT_VIOLATION, MODEL_VIOLATION
from main import VerifyConfig, verify
from specs import get_spec_by_id
from utils.spec_analysis import analyze_spec_structure
from wrappers.safedreamer_wrapper import SafeDreamerWrapper


DEFAULT_CHECKPOINT_NAME = (
    "20240307-010600_osrp_vector_safetygymcoor_"
    "SafetyPointGoal1-v0_0.ckpt"
)


def build_extra(args: argparse.Namespace) -> dict:
    repo_root = args.repo_root or os.environ.get(
        "SAFEDREAMER_REPO_ROOT",
        str(Path.home() / "Documents" / "SafeDreamer"),
    )
    checkpoint = args.checkpoint or os.environ.get(
        "SAFEDREAMER_CHECKPOINT_PATH",
        str(Path(repo_root) / "checkpoint" / DEFAULT_CHECKPOINT_NAME),
    )
    return {
        "repo_root": repo_root,
        "checkpoint_path": checkpoint,
        "method": "osrp_vector",
        "task": "safetygymcoor_SafetyPointGoal1-v0",
        "action_source": "random",
        "config_overrides": {"jax": {"platform": "cpu"}},
    }


def collect_model_rollouts(wrapper, *, horizon: int, count: int, seed: int, extra: dict):
    config = RolloutConfig(
        horizon=horizon,
        n_rollouts=count,
        seed=seed,
        action_source="random",
        extra=dict(extra),
    )
    return wrapper.sample_rollouts(config)


def collect_pairs(wrapper, *, horizon: int, count: int, seed: int, extra: dict):
    config = RolloutConfig(
        horizon=horizon,
        n_rollouts=count,
        seed=seed,
        action_source="random",
        extra=dict(extra),
    )
    return wrapper.sample_paired_rollouts(config)


def diagnose_after_violation(result, train_trajectories, calibration_trajectories, spec, cfg):
    """Fit/calibrate separately after an invariant counterexample; never amend result."""
    if result.lppm is not None or result.safety_verdict.verdict not in {
        ENVIRONMENT_VIOLATION, MODEL_VIOLATION,
    }:
        return None
    analysis = analyze_spec_structure(spec)
    if analysis["bounded"] or analysis["mp_class"] != "Safety":
        raise ValueError("Post-violation diagnostics currently support unbounded Safety specs only.")

    print("\n[L2 DIAGNOSTIC ONLY] Continuing after the recorded counterexample.")
    dpa = build_parity_automaton(spec)
    print(f"Training LPPM ({cfg.lppm_epochs} epochs) on the separate training split...")
    training = fit_lppm(
        train_trajectories, dpa, spec, eta=cfg.eta,
        n_epochs=cfg.lppm_epochs, calib_trajectories=calibration_trajectories,
    )
    if training.get("backend") != "torch_mlp":
        raise RuntimeError(f"NeuralLPPM training unavailable: {training.get('reason')}")
    print(f"Training backend: {training['backend']}")
    print(f"Training loss: {training['final_loss']:.6g}")
    diagnostics = calibrate_lppm(
        calibration_trajectories, dpa, spec, gamma=cfg.gamma, eta=cfg.eta,
        warrant_threshold=cfg.warrant_threshold, lppm_params=training,
        train_trajectories=train_trajectories,
    )
    diagnostics.training_info = training
    paths = diagnostics.pathwise

    # Count the exact transition populations against which P1 and P2 are
    # checked. With multiple odd priorities, one transition can contribute
    # one check per applicable odd-priority head, matching the verifier loop.
    odd_priorities = dpa.odd_priorities
    p1_checked = 0
    bad_transitions = 0
    for trajectory in calibration_trajectories:
        product_path = run_product_trajectory(trajectory, dpa, spec)
        for curr, _ in iter_transitions(product_path, dpa):
            for odd_priority in odd_priorities:
                if curr.priority == odd_priority:
                    bad_transitions += 1
                elif odd_priority > curr.priority:
                    p1_checked += 1

    certificate_successes = sum(p.satisfied for p in paths)
    p1_violations = sum(p.p1_violations for p in paths)
    p2_violations = sum(p.p2_violations for p in paths)
    closure = diagnostics.zfree_closure
    closure_passes = (
        closure is not None
        and closure.verifiable
        and closure.p_hat_closure is not None
        and closure.p_hat_closure >= cfg.warrant_threshold
    )
    calibrated_ready = diagnostics.is_warranted() and closure_passes

    print("L2 certificate:")
    print(f"  eta: {cfg.eta:.6f}")
    print(f"  P1 violating transitions: {p1_violations} / {p1_checked} p1_checked_transitions")
    print(f"  P2 violating transitions: {p2_violations} / {bad_transitions} bad_transitions")
    print(f"  certificate_event_successes: {certificate_successes}/{len(paths)}")
    print(f"  p_hat_gamma: {diagnostics.p_hat_gamma:.6f}")
    print(f"  z_free_sampled_closure: {'PASS' if closure_passes else 'FAIL'}")
    if closure is not None and closure.verifiable:
        print(f"  z_free_details: n={closure.n_zfree}, k={closure.k_zfree}, "
              f"p_hat_closure={closure.p_hat_closure:.6f}")
    else:
        print("  z_free_details: UNVERIFIABLE (no observed source in Z_free)")
    print("  deductive_status: NOT_ESTABLISHED")
    print(f"  result: {'CALIBRATED_MODEL_SCOPE' if calibrated_ready else 'NO_WARRANT'}")
    print(f"Final safety verdict retained: {result.safety_verdict.verdict}")
    print("These diagnostics do not establish a safety warrant or support-wide closure.")
    return diagnostics


def main(args: argparse.Namespace) -> None:
    print('LEGACY RANDOM-ACTION TRANSFER BASELINE: this entry point does not verify the deployed policy. '
          'For model-only policy L2 use experiments/l2_achievement64.py --action-source policy.')
    spec = get_spec_by_id(args.spec)
    if spec is None:
        raise ValueError(f"Unknown specification: {args.spec}")
    if not args.spec.startswith("ltl_"):
        raise ValueError("The L2/LPPM pipeline requires an LTL specification.")
    if args.diagnose_after_violation:
        analysis = analyze_spec_structure(spec)
        if analysis["bounded"] or analysis["mp_class"] != "Safety":
            raise ValueError("--diagnose-after-violation currently supports unbounded Safety specs only.")

    extra = build_extra(args)
    print(f"Loading SafeDreamer checkpoint: {extra['checkpoint_path']}")
    print(f"L2 specification: {args.spec}")
    print(
        "Independent splits: "
        f"train={args.n_train}, calibration={args.n_cal}, error={args.n_err}"
    )

    wrapper_config = RolloutConfig(
        horizon=args.horizon,
        n_rollouts=1,
        seed=args.seed,
        action_source="random",
        extra=dict(extra),
    )

    with SafeDreamerWrapper(wrapper_config) as wrapper:
        wrapper.load()

        print("Collecting LPPM training rollouts...")
        train_trajectories = collect_model_rollouts(
            wrapper,
            horizon=args.horizon,
            count=args.n_train,
            seed=args.seed,
            extra=extra,
        )

        print("Collecting held-out LPPM calibration rollouts...")
        calibration_trajectories = collect_model_rollouts(
            wrapper,
            horizon=args.horizon,
            count=args.n_cal,
            seed=args.seed + 1,
            extra=extra,
        )

        print("Collecting independent paired transfer rollouts...")
        paired_rollouts = collect_pairs(
            wrapper,
            horizon=args.horizon,
            count=args.n_err,
            seed=args.seed + 2,
            extra=extra,
        )

        verification_config = VerifyConfig(
            paired_rollouts=paired_rollouts,
            fit_lppm_params=True,
            lppm_train_trajectories=train_trajectories,
            lppm_epochs=args.epochs,
            gamma=args.gamma,
            warrant_threshold=args.warrant_threshold,
            method="stl",
            checkpoint_id=Path(extra["checkpoint_path"]).name,
        )
        result = verify(calibration_trajectories, spec, verification_config)

    print(result.summary())
    if args.diagnose_after_violation:
        diagnose_after_violation(
            result, train_trajectories, calibration_trajectories, spec, verification_config,
        )
    print("\nL2 interpretation:")
    print("  - lppm_training.backend must be 'torch_mlp'.")
    print("  - p_hat_gamma and Z_free closure must both meet the threshold.")
    print("  - This is sampled model-scope evidence, not a support-wide deductive proof.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="ltl_hazard_avoidance")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--n-train", type=int, default=60)
    parser.add_argument("--n-cal", type=int, default=40)
    parser.add_argument("--n-err", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--warrant-threshold", type=float, default=0.80)
    parser.add_argument(
        "--diagnose-after-violation", action="store_true",
        help="Continue Safety-spec LPPM training/calibration after a counterexample, for diagnostics only.",
    )
    main(parser.parse_args())
