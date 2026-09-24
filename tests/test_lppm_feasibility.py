from __future__ import annotations

from core.lppm.automaton import build_parity_automaton
from core.lppm.feasibility import analyze_lppm_feasibility, absorbing_odd_states
from core.lppm.verifier import run_product_trajectory
from specs.ltl_specs import get_ltl_spec_by_id
from utils.spec_analysis import analyze_spec_structure


def _prepared(spec_id: str):
    spec = get_ltl_spec_by_id(spec_id)
    assert spec is not None
    spec = dict(spec)
    spec["analysis"] = analyze_spec_structure(spec)
    return spec, build_parity_automaton(spec)


def test_strict_safety_absorbing_trap_is_rejected():
    spec, dpa = _prepared("ltl_hazard_avoidance")
    assert absorbing_odd_states(dpa) == ("trap",)
    result = analyze_lppm_feasibility(
        dpa, mp_class=spec["analysis"]["mp_class"], eta=0.01
    )
    assert result.eligible is False
    assert "P2" in result.reason


def test_eventual_stability_has_recoverable_odd_state_and_is_eligible():
    spec, dpa = _prepared("ltl_eventual_hazard_stability")
    assert spec["analysis"]["mp_class"] == "Persistence"
    assert absorbing_odd_states(dpa) == ()
    result = analyze_lppm_feasibility(
        dpa, mp_class=spec["analysis"]["mp_class"], eta=0.01
    )
    assert result.eligible is True

    # Appendix C.7: either state moves to not_p on a violation and to p on a
    # safe label. Neither state is absorbing, and the initial state is even p.
    assert dpa.initial == "p"
    assert dpa.priority == {"p": 0, "not_p": 1}
    path = run_product_trajectory(
        [
            {"hazard_dist": -0.1},
            {"hazard_dist": 1.0},
            {"hazard_dist": 2.0},
        ],
        dpa,
        spec,
    )
    assert path[0].q_next == "not_p"
    assert path[1].q_next == "p"
    assert path[2].q_next == "p"
