"""
tests/test_trajectory_calibration.py

Minimal correctness check for core/trajectory_calibration.py -- the
trajectory-level statistical safety rate fallback path (independent of
core/lppm/), adopted for ltl_hazard_avoidance and ltl_height_safety after
their L2 progress-measure certificate path was confirmed structurally
infeasible (EXPERIMENT_CONFIG.md §8.17-§8.20).
"""

from __future__ import annotations

import pytest

from core.lppm.calibrator import _clopper_pearson_lower
from core.trajectory_calibration import compute_trajectory_safety_rate


def test_all_trajectories_safe_gives_k_equals_n():
    trajectories = [[{"x": 1.0}, {"x": 2.0}] for _ in range(5)]
    result = compute_trajectory_safety_rate(trajectories, lambda z: z["x"] > 0, gamma=0.05)
    assert result.n == 5
    assert result.k == 5


def test_single_violating_step_fails_the_whole_trajectory():
    """C(tau) requires the predicate to hold at EVERY step -- one bad step
    anywhere in the trajectory must count that whole trajectory as unsafe."""
    safe = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
    one_bad_step = [{"x": 1.0}, {"x": -1.0}, {"x": 3.0}]
    result = compute_trajectory_safety_rate([safe, one_bad_step], lambda z: z["x"] > 0, gamma=0.05)
    assert result.n == 2
    assert result.k == 1


def test_reuses_clopper_pearson_not_reimplemented():
    trajectories = [[{"x": 1.0}], [{"x": 1.0}], [{"x": -1.0}]]
    result = compute_trajectory_safety_rate(trajectories, lambda z: z["x"] > 0, gamma=0.05)
    expected = _clopper_pearson_lower(result.k, result.n, 0.05)
    assert result.p_hat_safety == pytest.approx(expected)


def test_summary_never_uses_the_word_warrant_as_a_verdict():
    """Guarantee-strength wording discipline: this result must never read
    like a Theorem-5.4 warrant."""
    trajectories = [[{"x": 1.0}]]
    result = compute_trajectory_safety_rate(trajectories, lambda z: z["x"] > 0, gamma=0.05)
    summary = result.summary()
    assert "NOT a Theorem-5.4 warrant" in summary


def test_zero_trajectories_edge_case():
    result = compute_trajectory_safety_rate([], lambda z: z["x"] > 0, gamma=0.05)
    assert result.n == 0
    assert result.k == 0
    assert result.p_hat_safety == 0.0
