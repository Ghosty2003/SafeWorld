"""Run the sampled L2/LPPM pipeline on a SafeDreamer checkpoint.

The train and calibration rollouts are collected by separate wrapper calls so
the fitted LPPM is never calibrated on the same trajectories it saw in
training.  A third, independent paired split estimates model/environment
transfer error.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.settings import RolloutConfig
from main import VerifyConfig, verify
from specs import get_spec_by_id
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


def main(args: argparse.Namespace) -> None:
    spec = get_spec_by_id(args.spec)
    if spec is None:
        raise ValueError(f"Unknown specification: {args.spec}")
    if not args.spec.startswith("ltl_"):
        raise ValueError("The L2/LPPM pipeline requires an LTL specification.")

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

        result = verify(
            calibration_trajectories,
            spec,
            VerifyConfig(
                paired_rollouts=paired_rollouts,
                fit_lppm_params=True,
                lppm_train_trajectories=train_trajectories,
                lppm_epochs=args.epochs,
                gamma=args.gamma,
                warrant_threshold=args.warrant_threshold,
                method="stl",
                checkpoint_id=Path(extra["checkpoint_path"]).name,
            ),
        )

    print(result.summary())
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
    main(parser.parse_args())
