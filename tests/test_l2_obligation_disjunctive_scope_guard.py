"""
tests/test_l2_obligation_disjunctive_scope_guard.py

External-review finding #5: core/lppm/automaton.py's Obligation branch only
ever reads objectives["safety"] (the plain, conjunctive safety-atom bucket)
-- it never reads objectives["safety_disjunctions"] (Batch 3a's G(a v b)
recognition) or objectives["guarantee_sequences"] (Batch 3b's ordered-chain
recognition). Since Batch 3a taught utils/spec_analysis.py::infer_mp_class()
to classify a spec as "Obligation" whenever has_safety (now TRUE for either
safety OR safety_disjunctions) AND has_guarantee both hold, a spec shaped
like G(a v b) & F(goal) now gets correctly CLASSIFIED as Obligation, but
then silently mis-CONSTRUCTED: build_parity_automaton()'s Obligation branch
reads safe_aps = objectives["safety"] = [] (empty, since this spec's safety
is entirely disjunctive) and produces zero trap-entry transitions --
completely dropping the safety requirement from the automaton. Confirmed by
direct construction (see the conversation): a trajectory violating the
disjunction (a=b=false) while achieving the guarantee goal reaches the
ACCEPTING state, when it should be rejected.

Decision (explicit, this round): NOT fixed. Confirmed via
test_no_registered_obligation_spec_uses_disjunctive_safety_or_sequences below
that none of the 30 currently-registered specs trigger this path -- all 3
real mp_class="Obligation" specs (ltl_safe_goal, ltl_safe_slow_goal,
stl_safe_goal_reach) use plain conjunctive safety and plain unordered
guarantee. Documented as a known limitation (EXPERIMENT_CONFIG.md) rather
than fixed now, per explicit scoping decision. This test is a SCOPE GUARD:
if a future spec addition ever makes it fail, that is the signal the
Obligation automaton construction must be extended (to also handle
safety_disjunctions / guarantee_sequences, mirroring the Safety/Guarantee
branches' own handling) BEFORE that spec can be trusted through this
pipeline -- do not silently let this test start failing and ship anyway.
"""

from __future__ import annotations

from specs.ltl_specs import get_all_ltl_specs
from specs.stl_specs import get_all_stl_specs
from utils.spec_analysis import analyze_spec_structure

from core.lppm.automaton import build_parity_automaton
from core.lppm.verifier import run_product_trajectory


def test_no_registered_obligation_spec_uses_disjunctive_safety_or_sequences():
    """
    Scope guard: confirms the known Obligation/disjunctive-safety gap is
    currently unreachable by any of the 30 registered specs. If this starts
    failing, a new spec has been added that WOULD hit the silently-broken
    automaton construction path -- fix core/lppm/automaton.py's Obligation
    branch (see this module's docstring) before registering that spec for
    real use, do not just update this test to tolerate it.
    """
    for spec in get_all_ltl_specs() + get_all_stl_specs():
        spec = dict(spec)
        spec.pop("analysis", None)
        analysis = analyze_spec_structure(spec)
        if analysis["mp_class"] != "Obligation":
            continue
        objectives = analysis["objectives"]
        assert not objectives["safety_disjunctions"], (
            f"{spec['id']!r} is mp_class=Obligation with a non-empty "
            "safety_disjunctions -- this WOULD hit the known-broken Obligation "
            "automaton construction path (see this module's docstring). Fix "
            "core/lppm/automaton.py's Obligation branch before shipping this spec."
        )
        assert not objectives["guarantee_sequences"], (
            f"{spec['id']!r} is mp_class=Obligation with a non-empty "
            "guarantee_sequences -- same known-broken path, sequence side."
        )


def test_disjunctive_obligation_automaton_construction_is_confirmed_broken():
    """
    Documents (does not fix) the concrete failure mode: a synthetic
    G(a v b) & F(goal) spec, correctly classified as Obligation, produces an
    automaton that silently drops the safety requirement -- a trajectory
    violating the disjunction (a=b=false) while achieving the goal reaches
    an ACCEPTING state. This test is expected to keep passing (i.e. the bug
    stays reproduced) until someone deliberately fixes the automaton
    construction -- it is not a "should eventually pass differently" xfail,
    it's a precise record of the current (wrong) behavior for whoever picks
    this up.
    """
    spec = {
        "id": "test_disjunctive_obligation_scope_guard",
        "formula": {
            "type": "and",
            "left": {
                "type": "always", "a": 0, "b": 100000,
                "child": {
                    "type": "or",
                    "left": {"type": "atom", "dim": "a", "threshold": 0.5, "op": ">"},
                    "right": {"type": "atom", "dim": "b", "threshold": 0.5, "op": ">"},
                },
            },
            "right": {
                "type": "eventually", "a": 0, "b": 100000,
                "child": {"type": "atom", "dim": "goal", "threshold": 0.5, "op": ">"},
            },
        },
        "aps": ["a", "b", "goal"],
    }
    analysis = analyze_spec_structure(spec)
    assert analysis["mp_class"] == "Obligation"
    assert analysis["objectives"]["safety_disjunctions"] == [["a", "b"]]

    dpa = build_parity_automaton(spec)
    # a=b=false violates the safety disjunction; goal is achieved anyway.
    trajectory = [{"a": 0.0, "b": 0.0, "goal": 1.0}]
    path = run_product_trajectory(trajectory, dpa, spec)
    final_priority = dpa.priority.get(path[-1].q_next, 0)
    assert final_priority % 2 == 0, (
        "KNOWN BUG (not fixed this round, see module docstring): the safety "
        "disjunction is silently dropped by the Obligation automaton "
        "construction, so a trajectory that violates it still reaches an "
        f"accepting (even-priority) state. Got final_priority={final_priority}. "
        "If this assertion ever fails, the underlying bug may have been fixed "
        "already -- update this test's framing, don't just delete it."
    )
