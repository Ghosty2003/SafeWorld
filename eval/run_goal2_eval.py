"""
Trajectory-level SAFEWORLD evaluation on Goal2 oracle episodes.

This script compares method predictions against oracle STL robustness labels and
reports the metrics in eval/metrics.py.  OursMethod uses the eval protocol from
eval/protocol.py: cached spec rollouts, then 100 stochastic predictions per
oracle trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.settings import RolloutConfig
from eval.metrics import PredictionResult, compute_all_metrics
from eval.method_wrappers import OursMethod, SafeDreamerMethod, ShieldingMethod
from eval.oracle_episodes import TASK_TO_SPEC_IDS, extract_ap_trace, load_task_episodes, oracle_true_safe
from eval.protocol import predict_for_metrics
from eval.semantics import apply_dataset_semantics_to_spec
from main import VerifyConfig
from specs import get_spec_by_id
from specs.spec_calibrator import load_env_config
from wrappers import Goal2WorldModelWrapper


DEFAULT_CHECKPOINT = "/Users/ghost/Downloads/dreamv3-learned2/ckpt_0500000.pt"
DEFAULT_MODEL_DIR = "/Users/ghost/Downloads/SafeWorld-Benchmark-main/training/dreamer_world_model"
DEFAULT_EPISODES_DIR = "/Users/ghost/Downloads/SafeWorld-Benchmark-main/datasets/goal2_master/episodes"
DEFAULT_ENV_CONFIG = "configs/environments/goal2.json"

METHODS = ("ours", "safedreamer", "shielding")
UNSUPPORTED_OURS_DIMS = {"zone_a", "zone_b", "zone_c", "carrying"}


def _collect_dims(node: dict) -> set[str]:
    if not isinstance(node, dict):
        return set()
    if node.get("type") == "atom":
        return {node["dim"]}
    dims: set[str] = set()
    for key in ("child", "left", "right"):
        child = node.get(key)
        if isinstance(child, dict):
            dims |= _collect_dims(child)
    return dims


def _is_ours_supported(spec: dict) -> bool:
    return not (_collect_dims(spec.get("formula", {})) & UNSUPPORTED_OURS_DIMS)


def _jsonable_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    return {
        key: (None if isinstance(value, float) and math.isnan(value) else value)
        for key, value in metrics.items()
    }


def _make_methods(args: argparse.Namespace):
    methods: dict[str, Any] = {}

    if "ours" in args.methods:
        env_config = load_env_config(args.env_config)
        roll_cfg = RolloutConfig(
            horizon=args.horizon,
            n_rollouts=args.n,
            seed=args.seed,
            extra={
                "checkpoint_path": args.checkpoint,
                "model_dir": args.model_dir,
                "device": args.device,
                "action_source": "oracle",
                "oracle_episodes_dir": args.episodes_dir,
            },
        )
        wrapper = Goal2WorldModelWrapper(roll_cfg)
        print(f"Loading Goal2 world model for OursMethod (device={args.device}) ...", flush=True)
        wrapper.load(env_config=env_config)
        ver_cfg = VerifyConfig(
            delta_cp=args.delta_cp,
            delta_err=args.delta_err,
            model_error_budget=args.c_hat,
            verbose=False,
        )
        methods["ours"] = OursMethod(wrapper, roll_cfg, ver_cfg)

    if "safedreamer" in args.methods:
        methods["safedreamer"] = SafeDreamerMethod()

    if "shielding" in args.methods:
        methods["shielding"] = ShieldingMethod()

    return methods


def _close_methods(methods: dict[str, Any]) -> None:
    ours = methods.get("ours")
    if ours is not None:
        ours._wrapper.close()


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    methods = _make_methods(args)
    results: dict[str, list[PredictionResult]] = {name: [] for name in methods}
    per_spec_results: dict[str, dict[str, list[PredictionResult]]] = {
        name: {} for name in methods
    }
    skipped: list[dict[str, str]] = []

    try:
        for task_id, spec_ids in TASK_TO_SPEC_IDS.items():
            if args.task and task_id != args.task:
                continue

            episodes = load_task_episodes(
                args.episodes_dir,
                task_id,
                max_per_bucket=args.max_per_bucket,
            )
            print(f"\nTask {task_id}: {len(episodes)} episodes", flush=True)

            traces = [(ep, extract_ap_trace(ep)) for ep in episodes]

            for spec_id in spec_ids:
                if args.spec and spec_id != args.spec:
                    continue
                raw_spec = get_spec_by_id(spec_id)
                if raw_spec is None:
                    skipped.append({"spec_id": spec_id, "reason": "spec not found"})
                    continue
                spec = apply_dataset_semantics_to_spec(raw_spec)
                oracle_labels = [oracle_true_safe(trace, spec) for _, trace in traces]
                safe_rate = sum(oracle_labels) / len(oracle_labels) if oracle_labels else 0.0
                print(f"  Spec {spec_id}: oracle_safe_rate={safe_rate:.3f}", flush=True)

                for method_name, method in methods.items():
                    if method_name == "ours" and not _is_ours_supported(spec):
                        skipped.append({
                            "method": method_name,
                            "spec_id": spec_id,
                            "reason": "requires unsupported zone/carrying AP",
                        })
                        continue

                    spec_bucket = per_spec_results[method_name].setdefault(spec_id, [])
                    t0 = time.perf_counter()
                    for (_, trace), true_safe in zip(traces, oracle_labels):
                        pred = predict_for_metrics(
                            method,
                            spec_id,
                            spec,
                            trace,
                            true_safe,
                        )
                        results[method_name].append(pred)
                        spec_bucket.append(pred)
                    elapsed = time.perf_counter() - t0
                    metrics = compute_all_metrics(spec_bucket)
                    raw_note = ""
                    if method_name == "ours":
                        raw = method.raw_result(spec_id)
                        if raw is not None:
                            raw_note = (
                                f" raw={raw.verdict}"
                                f" rho*={raw.monitor.rho_star:+.3f}"
                                f" rho_net={raw.transfer.rho_net:+.3f}"
                                f" conf={raw.confidence:.3f}"
                            )
                    print(
                        f"    {method_name:11s} "
                        f"n={len(spec_bucket):4d} "
                        f"success={metrics['success_rate']:.3f} "
                        f"warrant={metrics['warrant_rate']:.3f} "
                        f"detect={metrics['detection_rate']:.3f} "
                        f"false_safe={metrics['false_safe_rate']:.3f} "
                        f"({elapsed:.1f}s)"
                        f"{raw_note}",
                        flush=True,
                    )

    finally:
        _close_methods(methods)

    summary = {
        "overall": {
            method_name: _jsonable_metrics(compute_all_metrics(method_results))
            for method_name, method_results in results.items()
        },
        "per_spec": {
            method_name: {
                spec_id: _jsonable_metrics(compute_all_metrics(spec_results))
                for spec_id, spec_results in specs.items()
            }
            for method_name, specs in per_spec_results.items()
        },
        "counts": {
            method_name: len(method_results)
            for method_name, method_results in results.items()
        },
        "skipped": skipped,
    }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\n=== Overall Metrics ===")
    headers = [
        "method", "confidence", "success_rate", "calibration_error",
        "warrant_rate", "detection_rate", "false_safe_rate",
    ]
    print("  " + "  ".join(f"{h:>18s}" for h in headers))
    for method_name, metrics in summary["overall"].items():
        values = [method_name]
        for key in headers[1:]:
            value = metrics[key]
            values.append("nan" if value is None else f"{value:.4f}")
        print("  " + "  ".join(f"{v:>18s}" for v in values))

    if summary["skipped"]:
        print("\nSkipped:")
        for item in summary["skipped"]:
            print(f"  {item}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Goal2 trajectory-level SAFEWORLD metrics")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--episodes-dir", default=DEFAULT_EPISODES_DIR)
    parser.add_argument("--env-config", default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--task", default=None, choices=sorted(TASK_TO_SPEC_IDS))
    parser.add_argument("--spec", default=None)
    parser.add_argument("--max-per-bucket", type=int, default=None)
    parser.add_argument("--n", type=int, default=30, help="Ours rollouts per spec")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--delta-cp", type=float, default=0.05)
    parser.add_argument("--delta-err", type=float, default=0.05)
    parser.add_argument("--c-hat", type=float, default=0.08)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_eval(args)
    print_summary(summary)
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2))
        print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
