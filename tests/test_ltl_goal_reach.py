"""Regression checks for the canonical pure reachability specification."""

from core.lppm import build_parity_automaton
from core.lppm.verifier import run_product_trajectory
from specs import get_spec_by_id
from utils.spec_analysis import analyze_spec_structure


def test_goal_reach_is_unbounded_guarantee():
    spec = get_spec_by_id("ltl_goal_reach")
    assert spec is not None
    assert spec["ltl_str"] == "F(goal)"
    assert spec["mp_class"] == "Guarantee"
    assert spec["aps"] == ["goal_dist"]

    analysis = analyze_spec_structure(spec)
    assert analysis["bounded"] is False
    assert analysis["mp_class"] == "Guarantee"


def test_goal_reach_automaton_accepts_after_entering_goal():
    spec = get_spec_by_id("ltl_goal_reach")
    dpa = build_parity_automaton(spec)

    path = run_product_trajectory(
        [
            {"goal_dist": 1.0},
            {"goal_dist": 0.1},
            {"goal_dist": -0.01},
            {"goal_dist": 0.5},
        ],
        dpa,
        spec,
    )

    # F(goal) is fulfilled once. Leaving the goal later cannot undo it.
    assert path[0].priority % 2 == 1
    assert path[-1].q != path[0].q
    assert path[-1].priority % 2 == 0
