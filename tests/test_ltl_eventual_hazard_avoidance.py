"""Checks for the canonical Persistence L2 specification."""

from __future__ import annotations

from core.lppm import build_parity_automaton
from core.lppm.verifier import run_product_trajectory
from specs import get_spec_by_id
from utils.spec_analysis import analyze_spec_structure


def test_eventual_hazard_avoidance_is_persistence():
    spec = get_spec_by_id("ltl_eventual_hazard_avoidance")
    analysis = analyze_spec_structure(spec)
    assert analysis["bounded"] is False
    assert analysis["mp_class"] == "Persistence"


def test_persistence_automaton_can_recover_after_hazard():
    spec = get_spec_by_id("ltl_eventual_hazard_avoidance")
    dpa = build_parity_automaton(spec)
    path = run_product_trajectory(
        [
            {"hazard_dist": 0.5},
            {"hazard_dist": -0.1},
            {"hazard_dist": 0.5},
        ],
        dpa,
        spec,
    )

    assert dpa.priority["pre"] % 2 == 1
    assert dpa.priority["absorbed"] % 2 == 0
    assert [state.q_next for state in path] == ["absorbed", "pre", "absorbed"]
