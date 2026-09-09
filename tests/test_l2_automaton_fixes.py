"""
tests/test_l2_automaton_fixes.py

L2 correctness-audit fixes, batch by batch. Each bug gets a test that is
verified to FAIL against the pre-fix code first (see the batch reports for
the confirmation), then the fix is applied and the test re-run to confirm it
passes. Do not conflate this with L3's tests/test_lbsm_*.py files -- this is
core/lppm/ (L2, co-Büchi), unrelated to core/lbsm/ (L3).
"""

from __future__ import annotations

import pytest

from core.lppm.automaton import ProductState, build_parity_automaton
from core.lppm.model import compute_lppm_value
from core.lppm.verifier import check_pathwise_conditions, run_product_trajectory
from utils.spec_analysis import UNCLASSIFIED_MP_CLASS, analyze_spec_structure


def _atom(dim: str, threshold: float = 0.5, op: str = ">") -> dict:
    return {"type": "atom", "dim": dim, "threshold": threshold, "op": op}


def _obligation_spec(p_dim: str = "p", q_dim: str = "q") -> dict:
    """Box(p) Or Diamond(q) -- the paper's own minimal Obligation example
    (Appendix C.7), single safety atom p and single guarantee atom q."""
    formula = {
        "type": "or",
        "left": {"type": "always", "a": 0, "b": 100000, "child": _atom(p_dim)},
        "right": {"type": "eventually", "a": 0, "b": 100000, "child": _atom(q_dim)},
    }
    return {
        "id": "test_obligation",
        "level": 2,
        "name": "test obligation",
        "formula": formula,
        "aps": [p_dim, q_dim],
    }


def _step(p: bool, q: bool, p_dim="p", q_dim="q") -> dict:
    # atom(dim, 0.5, ">"): active iff value > 0.5. Use 1.0/0.0 for true/false.
    return {p_dim: 1.0 if p else 0.0, q_dim: 1.0 if q else 0.0}


# ─── Batch 1: Obligation debt state must be able to reach accept ───────────

@pytest.mark.xfail(
    strict=False,
    reason=(
        "Batch-1 finding (see report): this formula shape (top-level OR of an "
        "always-clause and an eventually-clause, i.e. paper-canonical Box(p) Or "
        "Diamond(q)) is NOT routed to core/lppm/automaton.py's 'Obligation' "
        "branch at all. utils/spec_analysis.py::_flatten_conjunction() only "
        "splits top-level 'and', so this 'or' formula becomes a single "
        "unrecognized clause, objectives['safety']/['guarantee'] both stay "
        "empty, and infer_mp_class() falls through to mp_class='Safety' -- "
        "which then (via the 'objectives[\"safety\"] or spec.get(\"aps\", [])' "
        "fallback in the Safety branch) incorrectly treats BOTH p and q as "
        "atoms that must never be violated. This is a real bug, but it is a "
        "misclassification bug (belongs with Batch 2's silent-fallback "
        "problem), not the originally-hypothesized Obligation-automaton "
        "transition-table bug: no real registered spec has this disjunctive "
        "shape (all 3 real mp_class='Obligation' specs -- ltl_safe_goal, "
        "ltl_safe_slow_goal, stl_safe_goal_reach -- are conjunctive "
        "F(goal) & G(!hazard), for which the existing trap-is-absorbing "
        "automaton is semantically CORRECT, not buggy). Left xfail (not "
        "fixed) pending a decision on whether/how to extend the classifier -- "
        "see the batch report. If this ever starts passing on its own "
        "(XPASS), that means a Batch-2 fix incidentally covered it -- update "
        "this marker then, don't leave it stale."
    ),
)
def test_violate_then_later_achieve_goal_is_accepted():
    """
    t=0..2: p holds (safe). t=3: p violated. t=4+: q holds.
    Box(p) Or Diamond(q) must accept this trajectory -- the Diamond(q)
    disjunct is satisfied at t=4 regardless of the earlier p violation.
    """
    spec = _obligation_spec()
    dpa = build_parity_automaton(spec)

    trajectory = [
        _step(p=True, q=False),   # t=0
        _step(p=True, q=False),   # t=1
        _step(p=True, q=False),   # t=2
        _step(p=False, q=False),  # t=3: p violated
        _step(p=False, q=True),   # t=4: q now holds
    ]
    path = run_product_trajectory(trajectory, dpa, spec)
    final_state = path[-1].q_next

    assert dpa.priority.get(final_state, 1) % 2 == 0, (
        f"final automaton state {final_state!r} (priority "
        f"{dpa.priority.get(final_state)}) is not in the accepting region -- "
        "a trajectory that violates p at t=3 but achieves q at t=4 must be "
        "accepted (Box(p) Or Diamond(q)), not permanently trapped."
    )


def _conjunctive_safe_goal_spec(hazard_dim: str = "hazard", goal_dim: str = "goal") -> dict:
    """The ACTUAL shape every real mp_class='Obligation' spec in this repo
    uses: F(goal) & G(!hazard) -- see specs/ltl_specs.py::ltl_safe_goal.
    Conjunctive, not disjunctive."""
    formula = {
        "type": "and",
        "left": {"type": "always", "a": 0, "b": 100000,
                  "child": _atom(hazard_dim, threshold=0.5, op="<")},  # "always NOT hazard"
        "right": {"type": "eventually", "a": 0, "b": 100000, "child": _atom(goal_dim)},
    }
    return {
        "id": "test_conjunctive_safe_goal",
        "level": 2,
        "name": "test conjunctive safe goal",
        "formula": formula,
        "aps": [hazard_dim, goal_dim],
    }


def test_real_obligation_shape_correctly_classified_and_hazard_then_goal_is_rejected():
    """
    Positive regression test locking in current (correct) behavior for the
    formula shape every real Obligation spec actually uses: F(goal) & G(!hazard).
    A trajectory that violates safety (hazard becomes true) and only THEN
    reaches the goal must be REJECTED -- the conjunction requires safety to
    hold EVERYWHERE, so one violation is permanently fatal regardless of the
    goal. This confirms mp_class is correctly inferred as "Obligation" for
    this shape (not misrouted like the disjunctive case above), and that the
    trap-is-absorbing automaton is the semantically correct construction for
    it -- there is no bug to fix here for any real registered spec.
    """
    spec = _conjunctive_safe_goal_spec()
    analysis = analyze_spec_structure(spec)
    assert analysis["mp_class"] == "Obligation", (
        f"expected mp_class='Obligation' for F(goal) & G(!hazard), got {analysis['mp_class']!r}"
    )

    dpa = build_parity_automaton(spec)
    # atom(hazard, 0.5, "<"): "not hazard" holds iff hazard_value < 0.5.
    trajectory = [
        {"hazard": 0.0, "goal": 0.0},  # t=0: hazard_value=0.0 < 0.5 -> safe, goal not reached
        {"hazard": 1.0, "goal": 0.0},  # t=1: hazard_value=1.0 >= 0.5 -> VIOLATION
        {"hazard": 1.0, "goal": 1.0},  # t=2: goal reached, too late
    ]
    path = run_product_trajectory(trajectory, dpa, spec)
    final_state = path[-1].q_next
    assert dpa.priority.get(final_state, 0) % 2 == 1, (
        f"final state {final_state!r} should be in the (correctly) permanently-rejecting "
        "region -- safety violated before goal was reached, so F(goal) & G(!hazard) is false."
    )


# ═════════════════════════════════════════════════════════════════════════
# Batch 2: unrecognized AST shapes must fail loud (Unclassified), not
# silently default to Safety/Obligation. Option B, per the approved plan.
# ═════════════════════════════════════════════════════════════════════════

def _hazard_response_like_spec() -> dict:
    """
    The exact shape specs/ltl_specs.py::ltl_hazard_response uses:
    G(near_obstacle -> F(!high_velocity)), written as
    G(lor(atom_meaning_NOT_near_obstacle, F(atom))) -- negation encoded via
    the atom's own comparison polarity, not an explicit not(...) wrapper.
    _extract_response() only recognizes or(not(atom), F(atom)); this shape
    (or(atom, F(atom))) falls through to objectives["other"].
    """
    formula = {
        "type": "always", "a": 0, "b": 100000,
        "child": {
            "type": "or",
            "left": _atom("near_obstacle", threshold=-0.3, op="<"),  # means "NOT near"
            "right": {"type": "eventually", "a": 0, "b": 100000, "child": _atom("velocity", 0.5, "<")},
        },
    }
    return {
        "id": "test_hazard_response_like", "level": 4, "name": "test hazard response",
        "formula": formula, "aps": ["near_obstacle", "velocity"],
    }


def test_unrecognized_clause_sets_unclassified_not_default_safety():
    """
    Before the fix: objectives["safety"] and objectives["guarantee"] both
    stay empty (the or(...) clause isn't recognized by any _is_* check), and
    infer_mp_class() falls through to the "Safety" default -- silently
    misclassifying a Recurrence-class formula. After the fix: mp_class must
    be the UNCLASSIFIED sentinel, and classification_uncertain must be True,
    not a guessed default.
    """
    spec = _hazard_response_like_spec()
    analysis = analyze_spec_structure(spec)
    assert analysis["mp_class"] == UNCLASSIFIED_MP_CLASS, (
        f"expected mp_class={UNCLASSIFIED_MP_CLASS!r} for an unrecognized clause, "
        f"got {analysis['mp_class']!r} -- a silent wrong-default, not a fail-loud refusal."
    )
    assert analysis["classification_uncertain"] is True


def test_recognized_specs_are_not_marked_uncertain():
    """Regression guard: the fix must not make classification_uncertain=True
    for formulas that ARE correctly recognized (e.g. the real, working
    ltl_safe_goal-style conjunctive Obligation spec)."""
    spec = _conjunctive_safe_goal_spec()
    analysis = analyze_spec_structure(spec)
    assert analysis["mp_class"] == "Obligation"
    assert analysis["classification_uncertain"] is False


def test_build_parity_automaton_refuses_unclassified_spec():
    """
    core/lppm/automaton.py::build_parity_automaton()'s own fallback (for any
    mp_class it doesn't have a branch for) must raise, not silently return
    the trivial single-state "does nothing" automaton
    (ParityAutomaton(states=["q0"], ...)). Tested independently of the
    classifier fix above, by directly injecting a bogus/unrecognized
    mp_class via spec["analysis"] -- this is the SECOND layer the task
    explicitly asked not to leave open, since build_parity_automaton() could
    be called with a spec whose "analysis" was constructed by something
    other than analyze_spec_structure() (or by a future mp_class value
    analyze_spec_structure() itself doesn't know about yet).
    """
    spec = _hazard_response_like_spec()
    spec["analysis"] = {
        "mp_class": "ThisIsNotARealMpClass",
        "objectives": {"safety": [], "guarantee": [], "recurrence": [], "persistence": [], "responses": [], "other": []},
    }
    with pytest.raises(ValueError, match="[Uu]nrecognized|[Uu]nknown|[Nn]o.*branch|classification"):
        build_parity_automaton(spec)


def test_verify_returns_inconclusive_for_unclassified_spec():
    """
    End-to-end: main.py::verify() must route an unclassified spec to
    INCONCLUSIVE (with lppm=None) BEFORE calling build_parity_automaton(),
    not let build_parity_automaton()'s own refusal propagate as an uncaught
    exception out of verify(). Real STL monitor/transfer results still
    populate (those run unconditionally in Steps 1-2, before the mp_class
    check), only the LPPM/L2 step is skipped.
    """
    from main import INCONCLUSIVE, VerifyConfig, verify

    spec = _hazard_response_like_spec()
    trajectory = [
        {"near_obstacle": 1.0, "velocity": 0.0},
        {"near_obstacle": 1.0, "velocity": 1.0},
        {"near_obstacle": 0.0, "velocity": 0.0},
    ]
    result = verify([trajectory, trajectory], spec, VerifyConfig(verbose=False))

    assert result.verdict == INCONCLUSIVE
    assert result.lppm is None
    assert result.mp_class == UNCLASSIFIED_MP_CLASS
    assert "classification" in result.support_note.lower() or "uncertain" in result.support_note.lower()


# ═════════════════════════════════════════════════════════════════════════
# Batch 3a: G(a v b) -- pure disjunction of atoms under an invariant -- was
# either silently misclassified (pre-Batch-2) or safely-but-uselessly
# routed to Unclassified/INCONCLUSIVE (post-Batch-2). Fix: recognize the
# shape and build the correct disjunctive trap condition (trap iff ALL
# disjuncts violated simultaneously), not a conjunction of independent traps.
# ═════════════════════════════════════════════════════════════════════════

def _disjunctive_safety_spec(a_dim: str = "a", b_dim: str = "b") -> dict:
    """G(a v b): matches specs/ltl_specs.py::ltl_conditional_speed's actual
    shape (G(lor(atom, atom))), with plain ">" atoms for test simplicity."""
    formula = {
        "type": "always", "a": 0, "b": 100000,
        "child": {"type": "or", "left": _atom(a_dim), "right": _atom(b_dim)},
    }
    return {
        "id": "test_disjunctive_safety", "level": 7, "name": "test G(a or b)",
        "formula": formula, "aps": [a_dim, b_dim],
    }


def test_disjunctive_safety_body_correctly_classified():
    spec = _disjunctive_safety_spec()
    analysis = analyze_spec_structure(spec)
    assert analysis["mp_class"] == "Safety", (
        f"G(a or b) must classify as Safety, got {analysis['mp_class']!r}"
    )
    assert analysis["classification_uncertain"] is False


def test_disjunctive_safety_trajectory_a_false_b_true_is_accepted():
    """
    a=False, b=True throughout: G(a or b) is satisfied at every step (b
    always holds). Before the fix: either silently treated as G(a and b)
    (WRONG: would reject, since a is always false) if pre-Batch-2 behavior
    were still in effect, or routed to Unclassified/raises (post-Batch-2,
    safe but not actually verifying the spec). After the fix: must be
    accepted via the correct disjunctive trap condition.
    """
    spec = _disjunctive_safety_spec()
    dpa = build_parity_automaton(spec)
    trajectory = [
        {"a": 0.0, "b": 1.0},  # a=False (atom a>0.5 is false), b=True
        {"a": 0.0, "b": 1.0},
        {"a": 0.0, "b": 1.0},
    ]
    path = run_product_trajectory(trajectory, dpa, spec)
    final_state = path[-1].q_next
    assert dpa.priority.get(final_state, 1) % 2 == 0, (
        f"final state {final_state!r} not accepting -- G(a or b) with b always "
        "true should never trap, regardless of a's value."
    )


def test_disjunctive_safety_trajectory_both_false_is_rejected():
    """Both a and b false at some step: G(a or b) is violated there (neither
    disjunct holds) -- must trap, confirming the fix didn't overcorrect into
    accepting everything."""
    spec = _disjunctive_safety_spec()
    dpa = build_parity_automaton(spec)
    trajectory = [
        {"a": 1.0, "b": 0.0},  # a=True: fine
        {"a": 0.0, "b": 0.0},  # BOTH false: violation
        {"a": 1.0, "b": 1.0},  # recovers, but co-Büchi trap is permanent
    ]
    path = run_product_trajectory(trajectory, dpa, spec)
    final_state = path[-1].q_next
    assert dpa.priority.get(final_state, 0) % 2 == 1, (
        f"final state {final_state!r} should be rejecting -- both disjuncts "
        "were simultaneously false at t=1, violating G(a or b)."
    )


# ═════════════════════════════════════════════════════════════════════════
# Batch 3b: F(A and F(B)) -- nested sequential reachability -- loses order.
# The "remaining goals" set-tracking treats {A,B} as an unordered set, so
# seeing B before A still empties "remaining" and (wrongly) accepts.
# ═════════════════════════════════════════════════════════════════════════

def _sequential_goals_spec(a_dim: str = "A", b_dim: str = "B") -> dict:
    """F(A and F(B)) -- matches specs/ltl_specs.py::ltl_sequential_goals's
    exact shape."""
    formula = {
        "type": "eventually", "a": 0, "b": 100000,
        "child": {
            "type": "and",
            "left": _atom(a_dim),
            "right": {"type": "eventually", "a": 0, "b": 100000, "child": _atom(b_dim)},
        },
    }
    return {
        "id": "test_sequential_goals", "level": 3, "name": "test A then B",
        "formula": formula, "aps": [a_dim, b_dim],
    }


def test_sequential_goals_correct_order_is_accepted():
    spec = _sequential_goals_spec()
    dpa = build_parity_automaton(spec)
    trajectory = [
        {"A": 1.0, "B": 0.0},  # A first
        {"A": 0.0, "B": 1.0},  # then B
    ]
    path = run_product_trajectory(trajectory, dpa, spec)
    final_state = path[-1].q_next
    assert dpa.priority.get(final_state, 1) % 2 == 0, (
        f"final state {final_state!r} not accepting -- A then B in correct order "
        "must satisfy F(A and F(B))."
    )


def test_sequential_goals_reversed_order_is_rejected():
    """
    B seen BEFORE A: F(A and F(B)) requires A to happen, and ONLY THEN B.
    Seeing B first (with A never having happened yet) does not satisfy the
    formula even if A happens afterward -- order matters. Before the fix:
    the unordered "remaining={A,B}" set-tracking removes whichever goal is
    seen first, so this trajectory is (incorrectly) accepted.
    """
    spec = _sequential_goals_spec()
    dpa = build_parity_automaton(spec)
    trajectory = [
        {"A": 0.0, "B": 1.0},  # B first (too early -- A hasn't happened)
        {"A": 1.0, "B": 0.0},  # A happens after, but that's not F(A and F(B))
    ]
    path = run_product_trajectory(trajectory, dpa, spec)
    final_state = path[-1].q_next
    assert dpa.priority.get(final_state, 0) % 2 == 1, (
        f"final state {final_state!r} should be REJECTING -- B occurred before A, "
        "violating the required order in F(A and F(B)), regardless of the "
        "unordered unrecognized-remaining-set unless correctly tracked as a sequence."
    )


def test_ltl_full_mission_still_unclassified_after_batch_3a():
    """
    Scope-boundary check (per the batch-3 discussion): ltl_full_mission's
    blocking clause is G(near_obstacle or F(!high_velocity)) -- a MIXED
    disjunction (one branch is a plain atom, the other is F(atom)), not a
    pure atom-disjunction. Batch 3a's fix (pure G(a v b)) must NOT
    accidentally "fix" this by misrecognizing it -- it should remain
    Unclassified until the separate _extract_response() gap (same root
    cause as ltl_hazard_response/ltl_human_caution) is addressed.
    """
    from specs.ltl_specs import get_all_ltl_specs

    spec = next(s for s in get_all_ltl_specs() if s["id"] == "ltl_full_mission")
    analysis = analyze_spec_structure(dict(spec))
    assert analysis["mp_class"] == UNCLASSIFIED_MP_CLASS, (
        "ltl_full_mission should still be Unclassified after Batch 3a alone -- "
        f"its blocker is a mixed atom/F(atom) disjunction, got {analysis['mp_class']!r}"
    )


# ═════════════════════════════════════════════════════════════════════════
# Batch 4.1: P1 train/verify tolerance asymmetry. Training (core/lppm/loss.py
# ::p1_loss) is EXACTLY strict: ReLU(v_next - v_curr), zero tolerance, any
# positive gap is penalized. Verification (check_pathwise_conditions) used
# `max(1e-6, p1_tol)`, so p1_tol=0.0 (the default, and the only value any
# real caller currently passes) never actually meant zero -- a 1e-6 floor
# was hardcoded in regardless. Fix: p1_tol=0.0 must mean exactly 0, matching
# training; the floor is removed, not just documented differently.
# ═════════════════════════════════════════════════════════════════════════

def _safety_spec_single_atom(ap: str = "hazard") -> dict:
    return {
        "id": "test_p1_tolerance", "formula": {
            "type": "always", "a": 0, "b": 100000, "child": _atom(ap, threshold=0.0, op=">")
        },
        "aps": [ap],
    }


def test_default_p1_tolerance_flags_a_subminuscule_violation():
    """
    Construct an EXACT, precisely-computed gap of 5e-7 between v_curr and
    v_next (well below the old hardcoded 1e-6 floor, but strictly > 0 --
    training's p1_loss would penalize this). With the default p1_tol=0.0,
    this must be flagged as a P1 violation -- before the fix, the implicit
    max(1e-6, 0.0)=1e-6 floor silently absorbed it (0 violations reported).
    """
    spec = _safety_spec_single_atom()
    dpa = build_parity_automaton(spec)
    T = 2

    z0 = {"hazard": 0.0}
    v0 = compute_lppm_value(z0, "ok", 1, spec, 0, T, dpa=dpa)
    tiny_gap = 5e-7
    target_v1 = v0 + tiny_gap
    # V(z,"ok") = rem * (1 + max(0, margin)), rem=(T-1)/T=0.5 at t=1
    m1 = target_v1 / 0.5 - 1
    z1 = {"hazard": m1}
    v1 = compute_lppm_value(z1, "ok", 1, spec, 1, T, dpa=dpa)
    assert v1 - v0 == pytest.approx(tiny_gap, abs=1e-12), "test setup: gap must be exact"
    assert 0 < (v1 - v0) < 1e-6, "test setup: gap must sit strictly between 0 and the old floor"

    product_path = [
        ProductState(t=0, z=z0, q="ok", q_next="ok", priority=0),
        ProductState(t=1, z=z1, q="ok", q_next="ok", priority=0),
    ]
    result = check_pathwise_conditions(product_path, dpa, spec, eta=0.01, p1_tol=0.0)
    assert result.p1_violations == 1, (
        f"expected the sub-1e-6 gap to be flagged as a P1 violation under the default "
        f"(zero) tolerance, got p1_violations={result.p1_violations} -- the old implicit "
        "1e-6 floor is still absorbing it."
    )


def test_explicit_p1_tolerance_still_available_for_derived_slack():
    """Regression guard: an explicitly-passed larger p1_tol (the documented
    "derive from this run's own V distribution" use case) must still work --
    the fix removes the implicit floor, not the parameter itself."""
    spec = _safety_spec_single_atom()
    dpa = build_parity_automaton(spec)
    T = 2
    z0 = {"hazard": 0.0}
    v0 = compute_lppm_value(z0, "ok", 1, spec, 0, T, dpa=dpa)
    tiny_gap = 5e-7
    m1 = (v0 + tiny_gap) / 0.5 - 1
    z1 = {"hazard": m1}

    product_path = [
        ProductState(t=0, z=z0, q="ok", q_next="ok", priority=0),
        ProductState(t=1, z=z1, q="ok", q_next="ok", priority=0),
    ]
    result = check_pathwise_conditions(product_path, dpa, spec, eta=0.01, p1_tol=1e-5)
    assert result.p1_violations == 0, "an explicit tolerance larger than the gap must still absorb it"


# ═════════════════════════════════════════════════════════════════════════
# Batch 4.2: eta had 5 independent hardcoded defaults (3 core functions + 2
# eval scripts). Single source of truth: core/lppm/config.py::DEFAULT_LPPM_CONFIG.
# ═════════════════════════════════════════════════════════════════════════

def test_eta_defaults_trace_to_single_config_source():
    """
    Value-equality alone is a weak check here (all 5 places happened to
    already agree on 0.01 before any fix -- that coincidence is exactly the
    risk this batch exists to close). The load-bearing assertion is the
    SOURCE-level one below (each file's source must actually import and
    reference DEFAULT_LPPM_CONFIG, not just happen to type 0.01 in the same
    place) -- this value check is a secondary sanity confirmation, not proof
    of a real shared source by itself.
    """
    import inspect

    from core.lppm.calibrator import calibrate_lppm
    from core.lppm.config import DEFAULT_LPPM_CONFIG
    from core.lppm.trainer import fit_lppm

    expected = DEFAULT_LPPM_CONFIG.eta
    for fn in (fit_lppm, calibrate_lppm, check_pathwise_conditions):
        actual = inspect.signature(fn).parameters["eta"].default
        assert actual == expected

    from main import VerifyConfig
    assert VerifyConfig().eta == expected


def test_eta_defaults_are_sourced_not_independently_hardcoded():
    """
    Source-level check: each file that defines an eta/ETA default must
    actually reference core.lppm.config's DEFAULT_LPPM_CONFIG (or
    LPPMConfig) in its own source text, not merely happen to default to the
    same numeral 0.01 as config.py does. This is what actually distinguishes
    "single source of truth" from "five independent typos that currently
    agree" -- the value-equality test above cannot tell those apart.
    """
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    files = [
        repo_root / "core" / "lppm" / "trainer.py",
        repo_root / "core" / "lppm" / "calibrator.py",
        repo_root / "core" / "lppm" / "verifier.py",
        repo_root / "main.py",
        repo_root / "cardreamer" / "eval_ltl_hazard_real_lppm.py",
        repo_root / "tdmpc2" / "eval_ltl_height_safety_lppm.py",
    ]
    for f in files:
        text = f.read_text()
        assert "DEFAULT_LPPM_CONFIG" in text or "LPPMConfig" in text, (
            f"{f.relative_to(repo_root)} defines an eta/ETA default but its source does not "
            "reference core.lppm.config's shared LPPMConfig -- looks like an independently "
            "hardcoded value, not one sourced from the single config."
        )


# ═════════════════════════════════════════════════════════════════════════
# Batch 4.3: V >= 0 non-negativity is not structurally guaranteed on the
# heuristic (non-torch) fallback path in core/lppm/model.py::compute_lppm_value().
# The Guarantee/Obligation/Recurrence/Reactivity branches subtract an
# unclamped z-feature value (progress/zone_val/margin) from a small constant
# -- a large raw feature value drives V negative, which is meaningless for a
# quantity the paper's Z_free = {(z,q): V_phi(z,q) < eta} definition and the
# P2 descent check both assume is a nonnegative "distance to acceptance".
# ═════════════════════════════════════════════════════════════════════════

def _single_goal_guarantee_spec(goal_dim: str = "goal") -> dict:
    """F(goal), unbounded, single atom -- classifies as mp_class='Guarantee'
    with objectives['guarantee'] == ['goal'] (see extract_objectives())."""
    formula = {"type": "eventually", "a": 0, "b": 100000, "child": _atom(goal_dim)}
    return {
        "id": "test_guarantee_v_nonneg", "level": 2, "name": "test guarantee v>=0",
        "formula": formula, "aps": [goal_dim],
    }


def test_heuristic_guarantee_value_is_never_negative():
    """
    compute_lppm_value()'s Guarantee branch: rem * (1 + len(remaining) -
    max(0, progress)). progress is clamped only from BELOW (max(0, ...)),
    not from above -- a large raw feature value (e.g. a distance-style AP
    that happens to read 100.0) drives the whole expression negative. Before
    the fix, this is unguarded; after, compute_lppm_value() must clamp its
    return value to >= 0 for every mp_class branch.
    """
    spec = _single_goal_guarantee_spec()
    dpa = build_parity_automaton(spec)
    T = 10
    # initial state's remaining_goals == ["goal"] (nothing achieved yet)
    z = {"goal": 100.0}
    v = compute_lppm_value(z, dpa.initial, 1, spec, 0, T, dpa=dpa)
    assert v >= 0.0, (
        f"compute_lppm_value() returned {v!r} < 0 for the Guarantee heuristic branch -- "
        "V is supposed to be a nonnegative certificate value (Z_free = {(z,q): V<eta} "
        "and the P2 descent check both assume V >= 0), but a large raw 'progress' "
        "feature value drove it negative unclamped."
    )


def test_check_pathwise_conditions_raises_on_negative_lppm_value():
    """
    Defense in depth, independent of whether model.py's own clamp is
    airtight: check_pathwise_conditions() must itself refuse to silently
    proceed if compute_lppm_value() ever hands it a negative V (e.g. a
    future branch, a bug in a learned/lppm_params path, or numeric noise)
    -- fail loud with a clear error, not silently compute a meaningless
    P1/P2 result on top of a broken certificate value.
    """
    import core.lppm.verifier as verifier_module

    spec = _safety_spec_single_atom()
    dpa = build_parity_automaton(spec)
    product_path = [
        ProductState(t=0, z={"hazard": 0.0}, q="ok", q_next="ok", priority=0),
        ProductState(t=1, z={"hazard": 0.0}, q="ok", q_next="ok", priority=0),
    ]

    original = verifier_module.compute_lppm_value
    try:
        verifier_module.compute_lppm_value = lambda *a, **k: -1.0
        with pytest.raises(ValueError, match="(?i)negative"):
            check_pathwise_conditions(product_path, dpa, spec, eta=0.01, p1_tol=0.0)
    finally:
        verifier_module.compute_lppm_value = original
