"""
tests/test_l2_sequential_guarantee_fixes.py

External-review finding #6: two independent bugs in the F(A and F(B and
F(C...))) sequential-Guarantee construction (Batch 3b of the earlier L2
audit).

1. core/lppm/automaton.py's advance_progress() (Guarantee branch, the
   guarantee_sequences path) only advances a sequence's progress index by
   ONE step per tick even when MULTIPLE consecutive required atoms are
   active SIMULTANEOUSLY in the same observation -- e.g. F(A and F(B)) with
   A and B both true at the same instant t IS satisfied (F(B) only needs B
   true at some time >= t, including t itself), but the pre-fix automaton
   only consumes A at t, leaving the automaton waiting for B at some FUTURE
   step -- wrongly rejecting a trajectory where B was only ever true
   simultaneously with A and never again.

2. core/lppm/model.py::compute_lppm_value()'s Guarantee branch reads
   meta["remaining_goals"] (the UNORDERED plain-goal bucket,
   objectives["guarantee"]) even for automaton states built from
   guarantee_sequences -- for a PURELY sequential spec (guarantee_sequences
   non-empty, guarantee EMPTY, e.g. ltl_sequential_goals), remaining_goals
   is the empty list for EVERY state, so the heuristic V value is
   unconditionally 0.0 everywhere, completely blind to how far the sequence
   has actually progressed (state_meta's real progress signal,
   "sequence_progress", is populated correctly but never read).
"""

from __future__ import annotations

from specs.ltl_specs import get_ltl_spec_by_id

from core.lppm.automaton import build_parity_automaton
from core.lppm.model import compute_lppm_value
from core.lppm.verifier import run_product_trajectory


def test_simultaneous_satisfaction_completes_sequence_in_one_step():
    """
    F(A and F(B)): A and B both active at the SAME timestep must complete
    the whole sequence in that one step (reach the accepting "done" state),
    not advance only one position and wait for a future observation of B.
    """
    spec = get_ltl_spec_by_id("ltl_sequential_goals")
    dpa = build_parity_automaton(spec)
    trajectory = [{"zone_a": 1.0, "zone_b": 1.0}]
    path = run_product_trajectory(trajectory, dpa, spec)
    final_state = path[-1].q_next
    final_priority = dpa.priority.get(final_state, 0)
    assert final_priority % 2 == 0, (
        f"A and B both true at the same instant must complete F(A and F(B)) "
        f"immediately -- final state {final_state!r} has priority "
        f"{final_priority} (odd = still rejecting/incomplete)"
    )


def test_simultaneous_satisfaction_of_three_stage_sequence():
    """Same bug, three-atom chain (F(A and F(B and F(C)))) -- all three
    active at once must complete the entire chain in one step."""
    spec = get_ltl_spec_by_id("ltl_three_stage")
    dpa = build_parity_automaton(spec)
    aps = spec["aps"]
    trajectory = [{ap: 1.0 for ap in aps}]
    path = run_product_trajectory(trajectory, dpa, spec)
    final_state = path[-1].q_next
    final_priority = dpa.priority.get(final_state, 0)
    assert final_priority % 2 == 0, (
        f"all three atoms true at the same instant must complete the chain "
        f"immediately -- final state {final_state!r} has priority {final_priority}"
    )


def test_sequential_order_still_enforced_after_simultaneous_fix():
    """Regression guard: the fix must not turn this into an unordered
    check -- B active WITHOUT A having ever held must still not complete
    the sequence (order still matters, only genuine simultaneity fast-forwards)."""
    spec = get_ltl_spec_by_id("ltl_sequential_goals")
    dpa = build_parity_automaton(spec)
    trajectory = [{"zone_a": 0.0, "zone_b": 1.0}]  # B true, A never true
    path = run_product_trajectory(trajectory, dpa, spec)
    final_state = path[-1].q_next
    final_priority = dpa.priority.get(final_state, 0)
    assert final_priority % 2 == 1, (
        f"B alone (A never true) must NOT complete F(A and F(B)) -- "
        f"final state {final_state!r} has priority {final_priority} (should be odd)"
    )


def test_heuristic_v_reflects_sequence_progress_not_frozen_at_zero():
    """
    compute_lppm_value()'s Guarantee branch for a purely-sequential spec
    (ltl_sequential_goals: guarantee_sequences non-empty, guarantee EMPTY)
    must distinguish states at different sequence_progress -- pre-fix, it
    reads meta["remaining_goals"] which is [] for EVERY state in this
    automaton (there are no plain unordered goals), so V collapses to the
    unconditional 0.0 branch everywhere, indifferent to real progress.
    """
    spec = get_ltl_spec_by_id("ltl_sequential_goals")
    dpa = build_parity_automaton(spec)
    T = 10

    # Confirm the state_meta setup this test relies on.
    assert dpa.state_meta["wait:|0"]["remaining_goals"] == []
    assert dpa.state_meta["wait:|0"]["sequence_progress"] == [0]
    assert dpa.state_meta["wait:|1"]["sequence_progress"] == [1]

    z = {"zone_a": 0.0, "zone_b": 0.0}
    v_no_progress = compute_lppm_value(z, "wait:|0", 1, spec, 0, T, dpa=dpa)
    v_one_step_in = compute_lppm_value(z, "wait:|1", 1, spec, 0, T, dpa=dpa)
    assert v_one_step_in != v_no_progress, (
        f"V must differ between sequence_progress=0 and sequence_progress=1 "
        f"states -- got v_no_progress={v_no_progress}, v_one_step_in={v_one_step_in} "
        "(both 0.0 would mean V is frozen, blind to real progress)"
    )
    assert v_one_step_in < v_no_progress, (
        "V should DECREASE as sequence progress advances (closer to satisfaction) -- "
        f"got v_no_progress={v_no_progress}, v_one_step_in={v_one_step_in}"
    )
