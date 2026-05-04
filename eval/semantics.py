"""
Dataset-aligned AP semantics used by evaluation baselines.

These constants mirror SafeWorld-Benchmark Goal2 annotations.  Keep them out
of spec formula definitions so evaluation rules and AP extraction stay aligned
without rewriting the benchmark specs.
"""

from __future__ import annotations

import copy

SPEED_SAFE_THRESHOLD = 0.35
GOAL_REACHED_THRESHOLD = -0.2
NEAR_OBSTACLE_UNSAFE_THRESHOLD = 0.0

OURS_EVAL_TRIALS = 100

DATASET_THRESHOLD_OVERRIDES = {
    "velocity": SPEED_SAFE_THRESHOLD,
    "goal_dist": GOAL_REACHED_THRESHOLD,
    "near_obstacle": NEAR_OBSTACLE_UNSAFE_THRESHOLD,
}


def apply_dataset_semantics_to_spec(spec: dict) -> dict:
    """
    Return a copy of spec with atom thresholds aligned to dataset semantics.

    Source spec definitions stay untouched; evaluation uses this calibrated copy
    so SAFEWORLD verify(), AP extraction, and baselines agree on thresholds.
    """
    aligned = copy.deepcopy(spec)
    aligned["formula"] = _patch_formula_thresholds(aligned["formula"])
    aligned.setdefault("threshold_source", "dataset_semantics")
    return aligned


def _patch_formula_thresholds(formula: dict) -> dict:
    node_type = formula.get("type")

    if node_type == "atom":
        dim = formula.get("dim")
        if dim in DATASET_THRESHOLD_OVERRIDES:
            formula["threshold"] = DATASET_THRESHOLD_OVERRIDES[dim]
        return formula

    for key in ("child", "left", "right"):
        child = formula.get(key)
        if isinstance(child, dict):
            formula[key] = _patch_formula_thresholds(child)

    return formula
