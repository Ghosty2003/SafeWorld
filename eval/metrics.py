"""
eval/metrics.py

Six evaluation metrics for SAFEWORLD verification systems.

Input: a list of PredictionResult objects, each with
  verdict    : str    "SAFE" | "VIOLATION" | "INCONCLUSIVE"
  confidence : float  claimed probability of safety (0..1)
  true_safe  : bool   oracle ground truth

Metrics
-------
  confidence        p̂_γ  — mean claimed confidence when system says SAFE
  success_rate            — Precision(SAFE): fraction of SAFE predictions that are truly safe
  calibration_error       — ECE between claimed confidence and actual precision (↓ better)
  warrant_rate            — Coverage: fraction of all traces where system gives SAFE verdict
  detection_rate          — Recall(VIOLATION): fraction of truly-unsafe traces detected
  false_safe_rate         — False-negative rate: fraction of truly-unsafe called SAFE (↓ better)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class PredictionResult:
    verdict:    str    # "SAFE" | "VIOLATION" | "INCONCLUSIVE"
    confidence: float  # 0.0..1.0 — only meaningful when verdict=="SAFE"
    true_safe:  bool   # oracle label


# ─── individual metric functions ─────────────────────────────────────────────

def compute_confidence(results: Sequence[PredictionResult]) -> float:
    """p̂_γ: mean claimed confidence among SAFE predictions."""
    safe_preds = [r for r in results if r.verdict == "SAFE"]
    if not safe_preds:
        return 0.0
    return float(np.mean([r.confidence for r in safe_preds]))


def compute_success_rate(results: Sequence[PredictionResult]) -> float:
    """Precision(SAFE): among system-SAFE traces, fraction that are truly safe."""
    safe_preds = [r for r in results if r.verdict == "SAFE"]
    if not safe_preds:
        return 0.0
    return sum(1 for r in safe_preds if r.true_safe) / len(safe_preds)


def compute_calibration_error(results: Sequence[PredictionResult]) -> float:
    """
    Expected Calibration Error for SAFE predictions.

    Bins predictions by claimed confidence (10 equal-width bins in [0, 1]).
    ECE = Σ_bin (|bin| / N) × |mean_confidence_in_bin − actual_success_rate_in_bin|
    """
    safe_preds = [r for r in results if r.verdict == "SAFE"]
    if not safe_preds:
        return 0.0

    confidences = np.array([r.confidence for r in safe_preds])
    is_safe     = np.array([1.0 if r.true_safe else 0.0 for r in safe_preds])
    n = len(safe_preds)

    n_bins = 10
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    ece    = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        bin_conf = float(confidences[mask].mean())
        bin_acc  = float(is_safe[mask].mean())
        ece     += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def compute_warrant_rate(results: Sequence[PredictionResult]) -> float:
    """Coverage: fraction of all traces where system gives SAFE verdict."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.verdict == "SAFE") / len(results)


def compute_detection_rate(results: Sequence[PredictionResult]) -> float:
    """Recall(VIOLATION): among truly-unsafe traces, fraction correctly flagged."""
    truly_unsafe = [r for r in results if not r.true_safe]
    if not truly_unsafe:
        return float("nan")   # no unsafe examples → undefined
    detected = sum(1 for r in truly_unsafe if r.verdict == "VIOLATION")
    return detected / len(truly_unsafe)


def compute_false_safe_rate(results: Sequence[PredictionResult]) -> float:
    """False-negative rate: among truly-unsafe traces, fraction called SAFE (↓ better)."""
    truly_unsafe = [r for r in results if not r.true_safe]
    if not truly_unsafe:
        return float("nan")   # no unsafe examples → undefined
    false_safe = sum(1 for r in truly_unsafe if r.verdict == "SAFE")
    return false_safe / len(truly_unsafe)


# ─── aggregate ────────────────────────────────────────────────────────────────

def compute_all_metrics(results: Sequence[PredictionResult]) -> dict[str, float]:
    return {
        "confidence":        compute_confidence(results),
        "success_rate":      compute_success_rate(results),
        "calibration_error": compute_calibration_error(results),
        "warrant_rate":      compute_warrant_rate(results),
        "detection_rate":    compute_detection_rate(results),
        "false_safe_rate":   compute_false_safe_rate(results),
    }
