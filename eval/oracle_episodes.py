"""
eval/oracle_episodes.py

Load SafeWorld-Benchmark oracle episodes and extract SAFEWORLD-compatible
AP traces.  Each episode JSON already has:
  - satisfied (bool) : ground-truth label for the episode's own task spec
  - steps[t]['speed'], ['goal_distance'], ['nearest_hazard_distance'],
    ['nearest_vase_distance'], ['aps'] (task-specific boolean APs)

AP-extraction convention (consistent with Goal2WorldModelWrapper):
  hazard_dist    = nearest_hazard_distance − hazard_safe_dist    (+ve = safe)
  goal_dist      = goal_distance − goal_reach_radius             (−ve = reached)
  near_obstacle  = nearest_vase_distance − obstacle_safe_dist    (+ve = safe)
  velocity       = step['speed']                                 (m/s)
  zone_a/b/c     = 1.0 if step['aps']['A'/'B'/'C'] else 0.0

Ground-truth evaluation: we re-run the SAFEWORLD STL monitor on the oracle AP
trace so thresholds are exactly consistent with the SAFEWORLD spec definitions
(the episode's own 'satisfied' field may use different thresholds).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from eval.semantics import apply_dataset_semantics_to_spec

# ─── AP-extraction constants (Goal2 environment, matches goal2.json) ──────────

HAZARD_SAFE_DIST  = 0.20   # hazard_dist   = dist − 0.20
GOAL_REACH_RADIUS = 0.30   # goal_dist     = dist − 0.30   (−ve when reached)
OBSTACLE_SAFE_DIST = 0.30  # near_obstacle = dist − 0.30

# ─── task → SAFEWORLD spec mapping ───────────────────────────────────────────

TASK_TO_SPEC_IDS: dict[str, list[str]] = {
    "E2_L1_SpeedLimit":          ["stl_speed_limit",       "ltl_speed_limit"],
    "E2_L2_SafeSlowGoal":        ["ltl_safe_slow_goal",    "stl_safe_goal_reach", "ltl_safe_goal"],
    "E2_L3_ThreeStageABC":       ["ltl_three_stage"],
    "E2_L4_HazardResponseDense": ["stl_obstacle_response", "ltl_hazard_response"],
}

BUCKETS = ("success", "failure_or_recovery", "near_success")


# ─── AP extraction ────────────────────────────────────────────────────────────

def extract_ap_trace(episode: dict) -> list[dict[str, float]]:
    """
    Convert episode steps → list of SAFEWORLD AP dicts.

    goal_dist uses the goal-boolean backup at collection steps (where
    goal_distance has already been reset to a new far position).
    """
    trace: list[dict[str, float]] = []
    for step in episode["steps"]:
        aps    = step.get("aps", {})
        d_haz  = float(step.get("nearest_hazard_distance", 1.0))
        d_vase = float(step.get("nearest_vase_distance",   1.0))
        d_goal = float(step.get("goal_distance",           1.0))
        speed  = float(step.get("speed",                   0.0))

        # Goal boolean fires at the collection step, but goal_distance has
        # already reset.  Use 1.0 (clearly negative goal_dist < −0.2) there.
        if aps.get("goal", False):
            goal_dist = -1.0
        else:
            goal_dist = d_goal - GOAL_REACH_RADIUS   # −ve when inside goal

        trace.append({
            "hazard_dist":   d_haz  - HAZARD_SAFE_DIST,
            "goal_dist":     goal_dist,
            "velocity":      speed,
            "near_obstacle": d_vase - OBSTACLE_SAFE_DIST,
            "near_human":    0.0,
            "zone_a":  1.0 if aps.get("A", False) else 0.0,
            "zone_b":  1.0 if aps.get("B", False) else 0.0,
            "zone_c":  1.0 if aps.get("C", False) else 0.0,
            "carrying": 0.0,
            "model_cost": 0.0,
        })
    return trace


# ─── oracle ground truth ──────────────────────────────────────────────────────

def oracle_true_safe(
    trace: list[dict[str, float]],
    spec:  dict,
) -> bool:
    """
    Re-evaluate spec on the oracle AP trace using SAFEWORLD's STL monitor.
    Returns True if robustness > 0 (spec satisfied).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.stl_monitor import compute_robustness
    aligned_spec = apply_dataset_semantics_to_spec(spec)
    rho = compute_robustness(aligned_spec["formula"], trace)
    return rho > 0.0


# ─── episode loading ──────────────────────────────────────────────────────────

def load_task_episodes(
    episodes_root: str | Path,
    task_id:       str,
    max_per_bucket: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load all JSON episode files for one task across all buckets.
    Returns a list of episode dicts (with added 'ap_trace_safeworld' key).
    """
    root = Path(episodes_root)
    task_dir = root / task_id
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    episodes: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        bucket_dir = task_dir / bucket
        if not bucket_dir.exists():
            continue
        files = sorted(bucket_dir.glob("*.json"))
        if max_per_bucket is not None:
            files = files[:max_per_bucket]
        for fp in files:
            with open(fp) as f:
                ep = json.load(f)
            ep["_file"]   = str(fp)
            ep["_bucket"] = bucket
            episodes.append(ep)

    return episodes


def load_all_tasks(
    episodes_root:  str | Path,
    max_per_bucket: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return {task_id: [episode, ...]} for all four oracle tasks."""
    result: dict[str, list[dict[str, Any]]] = {}
    for task_id in TASK_TO_SPEC_IDS:
        result[task_id] = load_task_episodes(
            episodes_root, task_id, max_per_bucket=max_per_bucket
        )
    return result
