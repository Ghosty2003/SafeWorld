"""
tests/test_l2_terminal_transition_fix.py

External-review finding (highest priority): run_product_trajectory()'s
consumers (check_pathwise_conditions(), verify_zfree_closure(), and
trainer.py::fit_lppm()'s all_transitions construction) all built their
transition list by zipping ADJACENT product_path entries
(`(path[i], path[i+1])` for i in range(T-1)) -- this only covers T-1 of the
T real automaton transitions in a T-observation trajectory. The FINAL
transition (on z[T-1], from product_path[T-1].q to product_path[T-1].q_next
-- the true state after the WHOLE trajectory) was never included anywhere,
because there is no product_path[T] entry to pair it with. Confirmed via
direct execution before writing this fix (see conversation): a Safety G(p)
trajectory that violates p only on the LAST observation, and a Guarantee
F(goal) trajectory that achieves goal only on the LAST observation, were
BOTH mishandled by the pre-fix code (the Safety case happened to still
reject -- via an unrelated numeric accident, not because the check was
correct -- and the Guarantee case was wrongly REJECTED despite genuinely
satisfying F(goal)).

Also confirmed: naively swapping `last.q` for `last.q_next` in the terminal
Z_free check ALONE is insufficient and actively WRONG for Safety-class specs
-- core/lppm/model.py's Safety branch returns V=0.0 unconditionally for
q=="trap", so a trajectory that ends inside "trap" would trivially satisfy
Z_free (0.0 < eta) regardless of how it got there, and entering trap is a
V-DECREASE (never flagged by P1) sourced from a state whose priority doesn't
match r (never routed to P2 either). Fixed together: the terminal check
must treat landing in an ODD-priority state as an automatic violation
(matching the parity automaton's own acceptance condition), not defer to
V<eta.
"""

from __future__ import annotations

import pytest

from core.lppm.automaton import build_parity_automaton
from core.lppm.verifier import (
    check_pathwise_conditions,
    run_product_trajectory,
    verify_zfree_closure,
)


def _safety_gp_spec(ap: str = "p") -> dict:
    return {
        "id": "test_terminal_safety", "formula": {
            "type": "always", "a": 0, "b": 100000,
            "child": {"type": "atom", "dim": ap, "threshold": 0.5, "op": ">"},
        },
        "aps": [ap],
    }


def _guarantee_fgoal_spec(ap: str = "goal") -> dict:
    return {
        "id": "test_terminal_guarantee", "formula": {
            "type": "eventually", "a": 0, "b": 100000,
            "child": {"type": "atom", "dim": ap, "threshold": 0.5, "op": ">"},
        },
        "aps": [ap],
    }


def test_safety_violation_on_the_very_last_observation_is_rejected():
    """
    G(p), 2-step trajectory: t0 safe (p=1.0), t1 VIOLATES (p=0.0) -- the
    violation is the LAST thing that happens. Must be rejected. Pre-fix,
    this happened to already reject (via an unrelated numeric accident: the
    terminal check used the stale pre-transition "ok" state with the
    violating z1, and the clamped margin still produced V=0.5 >> eta) -- NOT
    because the check was actually examining the true final state. This
    test locks in the CORRECT reason (final state is really "trap", an odd
    "priority -- automatic rejection), which matters because the accidental
    path breaks under the natural "use q_next" fix (trap gives V=0.0
    unconditionally, which would flip this to an incorrect ACCEPT if the
    odd-priority auto-reject rule weren't also applied).
    """
    spec = _safety_gp_spec()
    dpa = build_parity_automaton(spec)
    trajectory = [{"p": 1.0}, {"p": 0.0}]
    path = run_product_trajectory(trajectory, dpa, spec)
    result = check_pathwise_conditions(path, dpa, spec, eta=0.01, p1_tol=0.0)
    assert result.satisfied is False, (
        "G(p) violated on the last observation must be rejected -- "
        f"got satisfied={result.satisfied}"
    )


def test_safety_violation_mid_trajectory_with_recovery_is_still_rejected():
    """
    G(p), 3-step trajectory: t0 safe, t1 VIOLATES, t2 safe again ("recovers"
    -- but G(p) requires p to hold at EVERY step, so one violation is fatal
    regardless of what happens after). Must be rejected. This exercises the
    newly-added final-transition check via a genuine P2 (trap self-loop,
    margin=0<eta) violation, independently of the terminal auto-reject rule
    -- a second, complementary path to the same correct verdict.
    """
    spec = _safety_gp_spec()
    dpa = build_parity_automaton(spec)
    trajectory = [{"p": 1.0}, {"p": 0.0}, {"p": 1.0}]
    path = run_product_trajectory(trajectory, dpa, spec)
    result = check_pathwise_conditions(path, dpa, spec, eta=0.01, p1_tol=0.0)
    assert result.satisfied is False


def test_guarantee_goal_achieved_only_on_the_very_last_observation_is_accepted():
    """
    F(goal), 2-step trajectory: t0 goal not yet achieved, t1 goal achieved
    -- the achievement is the LAST thing that happens. Must be ACCEPTED
    (F(goal) is satisfied). Pre-fix: the terminal check (and the loop's
    v_next for the only transition) used the stale pre-transition
    "wait:goal" state paired with the achieving z1, which never resolves to
    the true "wait_done" state -- systematically rejecting every trajectory
    that only achieves its goal on the final observation.
    """
    spec = _guarantee_fgoal_spec()
    dpa = build_parity_automaton(spec)
    trajectory = [{"goal": 0.0}, {"goal": 1.0}]
    path = run_product_trajectory(trajectory, dpa, spec)
    result = check_pathwise_conditions(path, dpa, spec, eta=0.01, p1_tol=0.0)
    assert result.satisfied is True, (
        "F(goal) achieved on the last observation must be accepted -- "
        f"got satisfied={result.satisfied}, p1={result.p1_violations}, p2={result.p2_violations}"
    )


def test_guarantee_goal_never_achieved_is_still_rejected():
    """Regression guard: the fix must not make everything vacuously True --
    a trajectory that never achieves the goal must still be rejected."""
    spec = _guarantee_fgoal_spec()
    dpa = build_parity_automaton(spec)
    trajectory = [{"goal": 0.0}, {"goal": 0.0}]
    path = run_product_trajectory(trajectory, dpa, spec)
    result = check_pathwise_conditions(path, dpa, spec, eta=0.01, p1_tol=0.0)
    assert result.satisfied is False


def test_verify_zfree_closure_sees_the_final_transition_too():
    """
    verify_zfree_closure() has the identical adjacent-pairing bug --
    construct a trajectory long enough that the SOURCE of the final
    transition itself lies in Z_free (small V via a large horizon), with a
    genuine P1 violation (V INCREASE, not merely "ends up in a bad state" --
    that narrower "entering trap via a V-decrease" case is legitimately P1-
    satisfied per the literal non-increase definition, and is deliberately
    NOT what this test exercises) on that exact final transition. Pre-fix,
    the final transition is invisible to this function's loop entirely, so
    this violation could never be detected no matter how the rest of the
    trajectory looks.
    """
    spec = _safety_gp_spec()
    dpa = build_parity_automaton(spec)
    T = 1000
    # Stays safely within "ok" throughout (p=1.0, small margin) so V stays
    # tiny (in Z_free) due to the long horizon; the LAST observation spikes
    # the margin hugely (still "safe" by the threshold -- stays in "ok",
    # not "trap") which drives V sharply UP -- an unambiguous P1 violation
    # on the previously-invisible final transition.
    trajectory = [{"p": 1.0} for _ in range(T - 1)] + [{"p": 500.0}]
    result = verify_zfree_closure([trajectory], dpa, spec, eta=0.01)
    assert result.n_zfree >= 1, f"expected at least one Z_free-source transition, got {result.n_zfree}"
    assert result.k_zfree < result.n_zfree, (
        "the P1 violation (V spike) on the final transition must be counted as a "
        f"Z_free-closure violation -- got k_zfree={result.k_zfree} == n_zfree={result.n_zfree}"
    )
