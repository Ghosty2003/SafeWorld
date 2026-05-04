"""
run_goal2_benchmark.py

Run all 23 SAFEWORLD specifications against the Goal2 DreamerV3 wrapper
and print a structured result table organised by benchmark level (L1–L8).

Usage
─────
  cd SafeWord_V2
  python run_goal2_benchmark.py
  python run_goal2_benchmark.py --n 50 --horizon 50
  python run_goal2_benchmark.py --spec stl_hazard_avoidance   # single spec
  python run_goal2_benchmark.py --output results_goal2.json   # save JSON

Applicability rules for Goal2 wrapper
──────────────────────────────────────
  SUPPORTED   all required AP dims are real (non-zero) model outputs
  PARTIAL     near_human is used — predicted by aux head but threshold
              has no ground-truth (near_human was a training placeholder)
  N/A         requires zone_a/b/c or carrying — always 0.0 in this model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from configs.settings import RolloutConfig
from eval.semantics import apply_dataset_semantics_to_spec
from specs import ALL_SPECS, get_spec_by_id
from specs.spec_calibrator import load_env_config
from main import verify, VerifyConfig
from wrappers import Goal2WorldModelWrapper

# ─── constants ────────────────────────────────────────────────────────────────

CHECKPOINT = (
    "/home/chenmg93@netid.washington.edu/.cache/huggingface/hub/"
    "models--FanZhangg--dreamv3-learned2/snapshots/"
    "397ff8a28b9a0369defb46a6ad1808ba4ef18b9e/ckpt_0500000.pt"
)
MODEL_DIR = (
    "/home/chenmg93@netid.washington.edu/dreamerv3/"
    "SafeWorld-Benchmark-main/SafeWorld-Benchmark-main/"
    "training/dreamer_world_model"
)
ORACLE_EPISODES_DIR = (
    "/home/chenmg93@netid.washington.edu/Downloads/SafeWorld-Benchmark-main/datasets/goal2_master/episodes"
)
ENV_CONFIG_PATH = "configs/environments/goal2.json"

# APs this wrapper provides with real (non-trivial) values
_REAL_APS = {"hazard_dist", "goal_dist", "velocity", "near_obstacle", "model_cost"}
# Predicted but no ground-truth threshold — flagged as PARTIAL
_PARTIAL_APS = {"near_human"}
# Always zero — makes specs using them degenerate
_ZERO_APS = {"zone_a", "zone_b", "zone_c", "carrying"}


# ─── spec → benchmark level mapping ──────────────────────────────────────────

LEVEL_SPECS: dict[int, list[str]] = {
    1: [
        "stl_hazard_avoidance",
        "ltl_hazard_avoidance",
        "stl_speed_limit",
        "ltl_speed_limit",
    ],
    2: [
        "stl_safe_goal_reach",
        "ltl_safe_goal",
        "ltl_safe_slow_goal",
    ],
    3: [
        "stl_sequential_zones",
        "ltl_sequential_goals",
        "ltl_three_stage",
    ],
    4: [
        "stl_obstacle_response",
        "ltl_hazard_response",
    ],
    5: [
        "stl_bounded_patrol",
        "ltl_patrol",
        "stl_safe_dual_patrol",
        "ltl_dual_patrol",
    ],
    6: [
        "ltl_safe_reactive_goal",
        "ltl_safe_patrol",
    ],
    7: [
        "ltl_human_caution",
        "ltl_conditional_speed",
        "ltl_conditional_proximity",
    ],
    8: [
        "ltl_full_mission",
        "stl_full_mission",
    ],
}

LEVEL_DESCRIPTIONS = {
    1: "Hazard Avoidance / Speed Limit",
    2: "Safe Goal Reach",
    3: "Sequential Zone Visiting",
    4: "Obstacle / Hazard Response",
    5: "Patrol (bounded recurrence)",
    6: "Safe Reactive Navigation",
    7: "Conditional Constraints",
    8: "Full Mission",
}


# ─── helpers ─────────────────────────────────────────────────────────────────

def _collect_dims(node: dict) -> set[str]:
    """Recursively collect all AP dimension names from a formula tree."""
    if not isinstance(node, dict):
        return set()
    if node.get("type") == "atom":
        return {node["dim"]}
    dims: set[str] = set()
    for v in node.values():
        if isinstance(v, dict):
            dims |= _collect_dims(v)
    return dims


def _applicability(spec: dict) -> tuple[str, list[str]]:
    """
    Return (status, reason_list) where status is one of:
        "SUPPORTED"  — all dims are real model outputs
        "PARTIAL"    — some dims are near_human (no ground-truth threshold)
        "N/A"        — some dims are always 0.0 (zone_a/b/c, carrying)
    """
    dims = _collect_dims(spec.get("formula", {}))
    zero_used    = sorted(dims & _ZERO_APS)
    partial_used = sorted(dims & _PARTIAL_APS)

    if zero_used:
        return "N/A", [f"requires {d} (always 0.0 in this model)" for d in zero_used]
    if partial_used:
        return "PARTIAL", [f"uses {d} (no ground-truth threshold; human_distance head is a training placeholder)" for d in partial_used]
    return "SUPPORTED", []


def _verdict_emoji(verdict: str) -> str:
    return {"WARRANT": "✓", "STL_MARGIN": "~", "VIOLATION": "✗"}.get(verdict, "?")


# ─── main benchmark ───────────────────────────────────────────────────────────

def run_benchmark(
    n_rollouts: int = 30,
    horizon:    int = 50,
    seed:       int = 42,
    device:     str = "cpu",
    delta_cp:   float = 0.05,
    delta_err:  float = 0.05,
    c_hat:      float = 0.08,
    spec_filter: str | None = None,
    checkpoint_path: str = CHECKPOINT,
    model_dir: str = MODEL_DIR,
    episodes_dir: str = ORACLE_EPISODES_DIR,
    verbose:    bool = True,
) -> list[dict[str, Any]]:

    env_config = load_env_config(ENV_CONFIG_PATH)

    roll_cfg = RolloutConfig(
        horizon=horizon, n_rollouts=n_rollouts, seed=seed,
        extra={
            "checkpoint_path":    checkpoint_path,
            "model_dir":          model_dir,
            "device":             device,
            "action_source":      "oracle",
            "oracle_episodes_dir": episodes_dir,
        },
    )

    wrapper = Goal2WorldModelWrapper(roll_cfg)
    if verbose:
        print(f"Loading Goal2 world model (device={device}) …", flush=True)
    wrapper.load(env_config=env_config)
    if verbose:
        print("Model loaded.\n")

    ver_cfg = VerifyConfig(
        delta_cp=delta_cp,
        delta_err=delta_err,
        model_error_budget=c_hat,
        verbose=False,
    )

    all_results: list[dict[str, Any]] = []

    # level → dataset directory name mapping
    LEVEL_DIR: dict[int, str] = {
        1: "L1", 2: "L2", 3: "L3", 4: "L4",
        5: "L5", 6: "L6", 7: "L7", 8: "L8",
    }

    for level, spec_ids in sorted(LEVEL_SPECS.items()):
        if verbose:
            print(f"{'─'*70}")
            print(f"  L{level}  {LEVEL_DESCRIPTIONS[level]}")
            print(f"{'─'*70}")

        # Build a per-level RolloutConfig that filters oracle episodes to this level
        level_roll_cfg = RolloutConfig(
            horizon=horizon, n_rollouts=n_rollouts, seed=seed,
            extra={
                "checkpoint_path":     checkpoint_path,
                "model_dir":           model_dir,
                "device":              device,
                "action_source":       "oracle",
                "oracle_episodes_dir": episodes_dir,
                "oracle_level_filter": LEVEL_DIR.get(level, f"L{level}"),
            },
        )

        for spec_id in spec_ids:
            if spec_filter and spec_id != spec_filter:
                continue

            spec = get_spec_by_id(spec_id)
            if spec is None:
                continue
            spec = apply_dataset_semantics_to_spec(spec)

            status, reasons = _applicability(spec)

            row: dict[str, Any] = {
                "level":       level,
                "spec_id":     spec_id,
                "status":      status,
                "reasons":     reasons,
                "verdict":     None,
                "rho_star":    None,
                "rho_net":     None,
                "guarantee":   None,
                "confidence":  None,
                "runtime_s":   None,
                "error":       None,
            }

            if status == "N/A":
                if verbose:
                    print(f"  {'N/A':8s}  {spec_id}")
                    for r in reasons:
                        print(f"           ↳ {r}")
                row["verdict"] = "N/A"
                all_results.append(row)
                continue

            try:
                t0 = time.perf_counter()
                trajs = wrapper.sample_rollouts(level_roll_cfg)
                result = verify(trajs, spec, ver_cfg)
                elapsed = time.perf_counter() - t0

                row.update({
                    "verdict":    result.verdict,
                    "rho_star":   round(result.monitor.rho_star, 4),
                    "rho_net":    round(result.transfer.rho_net, 4),
                    "guarantee":  result.guarantee_type,
                    "confidence": round(result.confidence, 3),
                    "runtime_s":  round(elapsed, 1),
                })

                if verbose:
                    flag = "⚠ PARTIAL" if status == "PARTIAL" else ""
                    print(
                        f"  {_verdict_emoji(result.verdict)} {result.verdict:12s}"
                        f"  {spec_id:42s}"
                        f"  ρ*={result.monitor.rho_star:+.3f}"
                        f"  ρ_net={result.transfer.rho_net:+.3f}"
                        f"  [{result.guarantee_type}, {result.confidence:.0%}]"
                        f"  {flag}"
                    )
                    for r in reasons:
                        print(f"             ↳ ⚠  {r}")

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                row["error"]      = str(exc)
                row["verdict"]    = "ERROR"
                row["runtime_s"]  = round(elapsed, 1)
                if verbose:
                    print(f"  {'ERROR':8s}  {spec_id}  →  {exc}")

            all_results.append(row)

        if verbose:
            print()

    wrapper.close()
    return all_results


# ─── summary table ────────────────────────────────────────────────────────────

def print_summary(results: list[dict[str, Any]]) -> None:
    print("\n" + "═" * 80)
    print("  GOAL2 DREAMER  ·  SAFEWORLD BENCHMARK SUMMARY")
    print("═" * 80)
    print(f"  {'L':>2}  {'Spec':42s}  {'Verdict':12s}  {'ρ*':>7}  {'ρ_net':>7}  {'Conf':>5}  {'Status'}")
    print(f"  {'─'*2}  {'─'*42}  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*10}")

    counts = {"WARRANT": 0, "STL_MARGIN": 0, "VIOLATION": 0, "N/A": 0, "ERROR": 0}

    for r in results:
        verdict_str = r["verdict"] or "—"
        rho_s = f"{r['rho_star']:+.3f}" if r["rho_star"] is not None else "  —  "
        rho_n = f"{r['rho_net']:+.3f}"  if r["rho_net"]  is not None else "  —  "
        conf  = f"{r['confidence']:.0%}" if r["confidence"] is not None else "  —"
        flag  = "⚠ PARTIAL" if r["status"] == "PARTIAL" else r["status"]
        print(
            f"  {r['level']:>2}  {r['spec_id']:42s}  {verdict_str:12s}"
            f"  {rho_s:>7}  {rho_n:>7}  {conf:>5}  {flag}"
        )
        counts[verdict_str] = counts.get(verdict_str, 0) + 1

    print("═" * 80)
    ran = len(results) - counts.get("N/A", 0) - counts.get("ERROR", 0)
    print(
        f"  Ran {ran} specs  |"
        f"  WARRANT {counts['WARRANT']}  STL_MARGIN {counts['STL_MARGIN']}"
        f"  VIOLATION {counts['VIOLATION']}  |"
        f"  N/A {counts['N/A']}  ERROR {counts.get('ERROR', 0)}"
    )
    print()
    print("  N/A legend: spec requires APs this model cannot provide")
    print("    zone_a / zone_b / zone_c — model has no zone-membership head")
    print("    carrying                 — model has no object-holding head")
    print("  ⚠ PARTIAL: near_human used but its threshold has no ground truth")
    print("═" * 80)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Goal2 DreamerV3 — SAFEWORLD full benchmark")
    parser.add_argument("--n",        type=int,   default=30,    help="rollouts per spec (default 30)")
    parser.add_argument("--horizon",  type=int,   default=50,    help="steps per rollout (default 50)")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--c-hat",    type=float, default=0.08,  help="model error budget ĉ_err")
    parser.add_argument("--spec",     default=None,              help="run a single spec only")
    parser.add_argument("--output",   default=None,              help="save JSON results to this path")
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="DreamerV3 checkpoint .pt path")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="dreamer_world_model package directory")
    parser.add_argument("--episodes-dir", default=ORACLE_EPISODES_DIR, help="Goal2 oracle episodes directory")
    args = parser.parse_args()

    results = run_benchmark(
        n_rollouts  = args.n,
        horizon     = args.horizon,
        seed        = args.seed,
        device      = args.device,
        c_hat       = args.c_hat,
        spec_filter = args.spec,
        checkpoint_path = args.checkpoint,
        model_dir       = args.model_dir,
        episodes_dir    = args.episodes_dir,
        verbose     = True,
    )

    print_summary(results)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
