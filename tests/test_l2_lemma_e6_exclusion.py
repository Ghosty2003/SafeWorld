"""
tests/test_l2_lemma_e6_exclusion.py

External-review finding #2: main.py::verify()'s Step-3 gate (added in
Batch 2 of the L2 audit, see tests/test_l2_automaton_fixes.py) only refuses
UNCLASSIFIED_MP_CLASS / classification_uncertain -- it never refused
mp_class=="Recurrence" or "Reactivity", which per Lemma E.6 must NOT be
routed into the co-Büchi (P1/P2, Theorem 5.4) LPPM pipeline at all (that
pipeline is specifically a co-Büchi construction; Recurrence/Reactivity are
NOT co-Büchi-expressible). Confirmed: ltl_patrol/ltl_dual_patrol/
ltl_safe_patrol (mp_class="Recurrence", verification_mode="infinite_parity")
currently sail straight through build_parity_automaton() (which DOES have a
Recurrence branch -- it can construct SOME automaton, just not one Theorem
5.4's statistical LPPM certificate is valid for) and the entire LPPM
pipeline, fabricating an unsupported guarantee.
"""

from __future__ import annotations

from specs.ltl_specs import get_ltl_spec_by_id


def _minimal_trajectory(spec: dict, n: int = 3) -> list[dict]:
    """
    All APs held "active" (value 1.0) throughout -- needed so
    main.py::verify()'s STL-monitor-based early VIOLATION exit
    (rho_star<0) does NOT fire before Step 3 is ever reached; this test is
    specifically about the Step-3 Lemma E.6 gate, not the STL monitor.
    """
    return [{ap: 1.0 for ap in spec["aps"]} for _ in range(n)]


def test_recurrence_spec_is_routed_to_inconclusive_not_lppm():
    """
    ltl_patrol (mp_class="Recurrence", a real registered spec) must be
    routed to INCONCLUSIVE by main.py::verify()'s Step-3 gate, same as an
    Unclassified spec -- not proceed into build_parity_automaton()/the LPPM
    pipeline, which fabricates a Theorem-5.4 guarantee this spec's class is
    not eligible for per Lemma E.6.
    """
    from main import INCONCLUSIVE, VerifyConfig, verify

    spec = get_ltl_spec_by_id("ltl_patrol")
    trajectory = _minimal_trajectory(spec)
    result = verify([trajectory, trajectory], spec, VerifyConfig(verbose=False))

    assert result.verdict == INCONCLUSIVE, (
        f"expected ltl_patrol (mp_class=Recurrence) to be routed to INCONCLUSIVE "
        f"per Lemma E.6, got verdict={result.verdict!r}"
    )
    assert result.lppm is None
    assert result.mp_class == "Recurrence"


def test_recurrence_inconclusive_note_distinguishes_from_unclassified():
    """
    The support_note for a Recurrence/Reactivity exclusion must say WHY --
    "co-Büchi not expressible, needs L3 (currently infeasible)" -- not the
    generic "classification uncertain" wording used for the Unclassified
    case (this spec's classification is NOT uncertain; it's confidently
    Recurrence, and confidently excluded for a different, structural reason).
    """
    from main import VerifyConfig, verify

    spec = get_ltl_spec_by_id("ltl_dual_patrol")
    trajectory = _minimal_trajectory(spec)
    result = verify([trajectory, trajectory], spec, VerifyConfig(verbose=False))

    note = result.support_note.lower()
    assert "uncertain" not in note, (
        f"Recurrence exclusion must not reuse the classification-uncertain wording "
        f"(this spec IS confidently classified) -- got: {result.support_note!r}"
    )
    assert ("co-b" in note or "cobüchi" in note or "co-buchi" in note or "büchi" in note.replace("ü", "u")), (
        f"support_note should explain the co-Büchi/Lemma E.6 structural reason -- "
        f"got: {result.support_note!r}"
    )
    assert "l3" in note, f"support_note should mention L3 -- got: {result.support_note!r}"


def test_pure_safety_is_routed_away_from_strict_descent_lppm():
    """G(safe)'s absorbing odd trap cannot sustain positive P2 descent."""
    from main import INCONCLUSIVE, VerifyConfig, verify

    spec = get_ltl_spec_by_id("ltl_hazard_avoidance")
    trajectory = [{"hazard_dist": 5.0}, {"hazard_dist": 4.0}, {"hazard_dist": 3.0}]
    result = verify([trajectory, trajectory], spec, VerifyConfig(verbose=False))
    assert result.verdict == INCONCLUSIVE
    assert result.lppm is None
    assert result.mp_class == "Safety"
    assert "absorbing" in result.support_note.lower()
    assert result.verification_mode == "direct_invariant_only"
