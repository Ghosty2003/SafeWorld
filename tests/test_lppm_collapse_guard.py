"""Regression tests for rejecting a collapsed LPPM candidate."""

from __future__ import annotations

from core.lppm import build_parity_automaton, calibrate_lppm
from specs import get_spec_by_id


def test_collapsed_candidate_cannot_issue_warrant_even_with_perfect_raw_rate():
    spec = get_spec_by_id("ltl_hazard_avoidance")
    dpa = build_parity_automaton(spec)
    trajectories = [[{"hazard_dist": 1.0}] for _ in range(20)]
    collapsed = {
        "collapse_detected": True,
        "failure_reason": "synthetic collapsed candidate",
    }

    result = calibrate_lppm(
        trajectories,
        dpa,
        spec,
        gamma=0.05,
        eta=0.01,
        warrant_threshold=0.50,
        lppm_params=collapsed,
    )

    assert result.p_hat_gamma > 0.50
    assert result.candidate_rejected is True
    assert result.candidate_rejection_reason == "synthetic collapsed candidate"
    assert result.is_warranted() is False
    assert "REJECTED" in result.summary()


def test_noncollapsed_candidate_keeps_existing_calibration_behavior():
    spec = get_spec_by_id("ltl_hazard_avoidance")
    dpa = build_parity_automaton(spec)
    trajectories = [[{"hazard_dist": 1.0}] for _ in range(20)]

    result = calibrate_lppm(
        trajectories,
        dpa,
        spec,
        gamma=0.05,
        eta=0.01,
        warrant_threshold=0.50,
        lppm_params={"collapse_detected": False},
    )

    assert result.candidate_rejected is False
    assert result.is_warranted() is True
