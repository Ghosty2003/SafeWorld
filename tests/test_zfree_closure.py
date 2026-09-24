"""
tests/test_zfree_closure.py

L2 Theorem-5.4-alignment task: calibrate_lppm() currently calibrates C(tau)
= "every transition satisfies (P1)-(P2) AND the terminal state is in
Z_free" -- strictly stronger than Theorem 5.4's actual minimal premise,
which only requires (P1)-(P2) on transitions whose SOURCE already lies in
Z_free = {(z,q): V_phi(z,q) < eta}. This file drives the addition of a
SEPARATE, independent diagnostic -- core/lppm/verifier.py::
verify_zfree_closure() -- that isolates exactly that Z_free-restricted
subset and reports its own Clopper-Pearson lower bound, p_hat_closure.

Confirmed before writing these tests (see batch report): grepping the
codebase for "zfree"/"z_free" turns up no function that isolates
source-in-Z_free transitions and checks (P1)-(P2) on that subset alone --
check_pathwise_conditions() only checks terminal-state Z_free membership
(once, at the end of the trajectory) and evaluates (P1)-(P2) on every
transition regardless of whether its source was in Z_free. No existing
check to avoid duplicating.
"""

from __future__ import annotations

import pytest

from core.lppm.automaton import ProductState, build_parity_automaton
from core.lppm.verifier import verify_zfree_closure


def _safety_spec_single_atom(ap: str = "hazard") -> dict:
    return {
        "id": "test_zfree_closure", "formula": {
            "type": "always", "a": 0, "b": 100000, "child": _atom(ap, threshold=0.0, op=">")
        },
        "aps": [ap],
    }


def _atom(dim: str, threshold: float = 0.5, op: str = ">") -> dict:
    return {"type": "atom", "dim": dim, "threshold": threshold, "op": op}


# For mp_class="Safety", core/lppm/model.py::compute_lppm_value() computes
# V(z, "ok") = rem * (1 + max(0, margin)), rem = (T-t)/T, margin = z[ap]
# (single-atom "safety" objective). At q="trap", V=0 always. eta is fixed
# to a value small enough that we can precisely control which transitions
# land inside Z_free by choosing margin.
ETA = 0.01


def test_zfree_closure_isolates_only_source_in_zfree_transitions():
    """
    3-step trajectory, T=3, all in state "ok" (never trapped):
      t=0: margin=100.0 -> V(t=0) = (3/3)*(1+100) = 101.0  (>> eta, NOT in Z_free)
           -> t=1: margin=0.0 -> V(t=1) = (2/3)*(1+0) = 0.667 (this transition's
              source V=101.0 is NOT in Z_free -- must be excluded from n_zfree
              even though V briefly INCREASES relative to... wait margin drops,
              V decreases here, so this transition is P1-fine regardless)
      t=1: margin=0.0 -> V(t=1) = 0.667 (still >> eta, NOT in Z_free)
           -> t=2: margin=0.0 -> V(t=2) = (1/3)*(1+0) = 0.333
      To get an actual source-in-Z_free transition we need a state whose V
      is already < eta=0.01. Use a 2-trajectory batch: a short one that
      never enters Z_free (all V >> eta), and a second one engineered so
      its single transition's source V is just under eta and P1 holds
      (V does not increase).
    """
    spec = _safety_spec_single_atom()
    dpa = build_parity_automaton(spec)

    # Trajectory A: T=2, margin large throughout -> V(t=0) >> eta, never in Z_free.
    traj_a = [{"hazard": 50.0}, {"hazard": 50.0}]

    # Trajectory B: T=2. Choose margin at t=0 so V(t=0) = rem*(1+margin) < eta.
    # rem at t=0, T=2 is (2-0)/2 = 1.0, so need 1+margin < eta -> margin < eta-1,
    # i.e. margin must be very negative (below the safety atom's own threshold,
    # meaning it's already technically a violation region) -- use max(0,margin)
    # clamp in the Safety formula: V = rem*(1+max(0,margin)), so with margin<0,
    # max(0,margin)=0, giving V=rem*1.0=1.0 regardless of how negative margin
    # is. rem=1.0 at t=0 can never be < eta=0.01 this way (rem itself is 1.0).
    # Instead use t close to T: T=100, t=98 -> rem=(100-98)/100=0.02, still
    # >eta. Push T large enough that rem alone is sub-eta: T=1000, t=999 ->
    # rem=(1000-999)/1000=0.001 < eta=0.01, giving V=0.001*(1+0)=0.001 < eta.
    traj_b = None  # constructed inline below via direct product-state injection

    product_path_b = [
        ProductState(t=999, z={"hazard": 0.0}, q="ok", q_next="ok", priority=0),
        ProductState(t=1000, z={"hazard": 0.0}, q="ok", q_next="ok", priority=0),
    ]
    # Sanity: confirm this construction actually lands source in Z_free with T=1000.
    from core.lppm.model import compute_lppm_value
    v_src = compute_lppm_value({"hazard": 0.0}, "ok", 1, spec, 999, 1000, dpa=dpa)
    assert v_src < ETA, f"test setup: expected source V < eta, got {v_src}"

    result = verify_zfree_closure([traj_a], dpa, spec, eta=ETA)
    assert result.n_zfree == 0, (
        "trajectory A never enters Z_free (V stays >> eta throughout) -- "
        f"expected n_zfree=0, got {result.n_zfree}"
    )


def test_zfree_closure_flags_p1_violation_among_source_in_zfree_transitions():
    """
    The concrete counterexample from the task spec: a transition whose
    SOURCE is inside Z_free (V_curr < eta) but whose destination violates
    P1 (V_next > V_curr) -- this must be counted as a Z_free-closure
    violation (excluded from k_zfree), distinctly from the aggregate
    check_pathwise_conditions() result, which only tells you the whole
    trajectory failed C(tau), not whether the failure was a genuine
    Z_free-internal violation or simply "never reached Z_free".

    Construct via a long horizon so rem alone drives V below eta at t=T-1,
    then force the FINAL step's margin to spike, driving V back up (P1
    violation) instead of continuing to decay to 0.
    """
    spec = _safety_spec_single_atom()
    dpa = build_parity_automaton(spec)
    T = 1000

    trajectory = [{"hazard": 0.0} for _ in range(T - 1)] + [{"hazard": 500.0}]
    # t=T-2: rem=(1000-998)/1000=0.002, margin=0 -> V=0.002 (< eta=0.01, in Z_free)
    # t=T-1: rem=(1000-999)/1000=0.001, margin=500.0 -> V=0.001*(1+500)=0.501
    #   (V_next=0.501 > V_curr=0.002 -- P1 violated on a Z_free-source transition)
    from core.lppm.model import compute_lppm_value
    v_curr = compute_lppm_value({"hazard": 0.0}, "ok", 1, spec, T - 2, T, dpa=dpa)
    v_next = compute_lppm_value({"hazard": 500.0}, "ok", 1, spec, T - 1, T, dpa=dpa)
    assert v_curr < ETA, f"test setup: expected v_curr < eta, got {v_curr}"
    assert v_next > v_curr, f"test setup: expected v_next > v_curr (P1 violation), got {v_next} <= {v_curr}"

    result = verify_zfree_closure([trajectory], dpa, spec, eta=ETA)
    assert result.n_zfree >= 1, (
        f"expected at least one source-in-Z_free transition, got n_zfree={result.n_zfree}"
    )
    assert result.k_zfree < result.n_zfree, (
        "the engineered P1 violation on a Z_free-source transition must NOT be counted "
        f"as satisfied -- got k_zfree={result.k_zfree} == n_zfree={result.n_zfree}"
    )


def test_zfree_closure_reports_unverifiable_when_no_transition_enters_zfree():
    """
    Phase-2 requirement: when the calibration set contains zero transitions
    whose source lies in Z_free (a real, known-degenerate case -- e.g.
    height_safety's k=0 diagnosis), verify_zfree_closure() must not return
    a number that looks statistically meaningful (0.0 or 1.0) -- it must
    explicitly mark the result as unverifiable.
    """
    spec = _safety_spec_single_atom()
    dpa = build_parity_automaton(spec)
    # Short trajectory, large margin throughout -> V stays >> eta always.
    trajectory = [{"hazard": 50.0}, {"hazard": 50.0}, {"hazard": 50.0}]

    result = verify_zfree_closure([trajectory], dpa, spec, eta=ETA)
    assert result.n_zfree == 0
    assert result.verifiable is False, (
        "with zero Z_free-source transitions, the result must be marked unverifiable, "
        "not silently assigned a default p_hat_closure."
    )
    assert result.p_hat_closure is None, (
        f"expected p_hat_closure=None when unverifiable, got {result.p_hat_closure!r} -- "
        "returning 0.0 or 1.0 here would look like a real statistical bound but isn't one."
    )


def test_zfree_closure_counts_trajectories_not_individual_transitions():
    """
    External-review finding #4: transitions within the SAME trajectory are
    not independent Bernoulli samples (they share the same underlying V_phi
    realization and autocorrelated dynamics), so counting them individually
    violates the exchangeability premise Clopper-Pearson requires. Fixed:
    each TRAJECTORY contributes exactly one (n,k) unit -- a binary indicator
    of whether ALL of its source-in-Z_free transitions satisfied P1/P2
    (mirroring PathwiseResult.satisfied's whole-trajectory C(tau)
    construction), not one unit per qualifying transition.

    Two trajectories: A stays safe throughout (multiple late transitions
    land in Z_free, ALL pass) -- contributes exactly 1 to n_zfree and 1 to
    k_zfree. B stays safe until a P1 violation on its FINAL transition only
    (several earlier Z_free-source transitions in B still pass) -- under
    the old per-transition counting this violation would be diluted by B's
    many passing transitions; under the correct per-trajectory indicator, a
    SINGLE violation anywhere in B's Z_free-source subset zeroes out B's
    entire contribution to k_zfree. Batch [A, B] must show n_zfree=2 (two
    trajectories, not the much larger raw transition count), k_zfree=1 (only
    A's indicator is fully satisfied).
    """
    spec = _safety_spec_single_atom()
    dpa = build_parity_automaton(spec)
    T = 1000
    # hazard=1.0 (> the 0.0 threshold) is genuinely SAFE -- hazard=0.0 would
    # already violate ">0.0" and trap immediately, which is not what this
    # test wants (both trajectories must stay in "ok" until their final step).
    traj_a = [{"hazard": 1.0} for _ in range(T)]
    traj_b = [{"hazard": 1.0} for _ in range(T - 1)] + [{"hazard": 500.0}]

    result = verify_zfree_closure([traj_a, traj_b], dpa, spec, eta=ETA)
    assert result.n_zfree == 2, (
        f"expected n_zfree=2 (one unit per trajectory), got {result.n_zfree} -- "
        "looks like transitions are still being counted individually."
    )
    assert result.k_zfree == 1, (
        f"expected k_zfree=1 (only trajectory A's all-satisfied indicator), got "
        f"{result.k_zfree} -- trajectory B's single final-transition violation must "
        "zero out its ENTIRE contribution, not be diluted by its other passing "
        "transitions."
    )


def test_calibrate_lppm_attaches_zfree_closure_to_result():
    """
    External-review finding #4 (second half): verify_zfree_closure()'s
    diagnostic was previously computed only if a caller separately went out
    of their way to call it -- calibrate_lppm() never surfaced it, so it was
    "run but nobody could see it" for any real caller. Fixed:
    calibrate_lppm() now computes it internally and attaches it to the
    returned LPPMResult as its own field (NOT mixed into p_hat_gamma), and
    LPPMResult.summary() reports it.
    """
    from core.lppm.calibrator import calibrate_lppm
    from core.lppm.verifier import ZFreeClosureResult

    spec = _safety_spec_single_atom()
    dpa = build_parity_automaton(spec)
    T = 1000
    traj_a = [{"hazard": 1.0} for _ in range(T)]
    traj_b = [{"hazard": 1.0} for _ in range(T - 1)] + [{"hazard": 500.0}]

    result = calibrate_lppm([traj_a, traj_b], dpa, spec, gamma=0.05, eta=ETA)
    assert isinstance(result.zfree_closure, ZFreeClosureResult), (
        f"expected LPPMResult.zfree_closure to be populated, got {result.zfree_closure!r}"
    )
    assert result.zfree_closure.n_zfree == 2
    assert result.zfree_closure.k_zfree == 1
    # p_hat_gamma (the main C(tau)-based bound, computed from PathwiseResult's
    # whole-trajectory satisfied flags) and p_hat_closure (computed from the
    # narrower Z_free-source-only indicator) must be independently derived
    # fields, not one overwriting or feeding into the other -- verified by
    # checking each traces back to its own distinct source data, not by
    # requiring them to differ numerically (they can coincide by chance, as
    # they happen to here since both reduce to the same k=1,n=2 in this
    # particular example).
    from core.lppm.calibrator import _clopper_pearson_lower

    k_pathwise = sum(1 for pw in result.pathwise if pw.satisfied)
    assert result.p_hat_gamma == pytest.approx(
        _clopper_pearson_lower(k_pathwise, len(result.pathwise), 0.05)
    )

    summary = result.summary()
    assert "closure" in summary.lower() or "z_free" in summary.lower(), (
        f"LPPMResult.summary() should surface the Z_free closure result, not just "
        f"compute it silently -- got: {summary!r}"
    )


def test_zfree_closure_reuses_clopper_pearson_not_reimplemented():
    """
    The task explicitly requires reusing calibrator.py's existing
    Clopper-Pearson lower-bound function, not reimplementing the formula.
    Cross-check: for a hand-picked (k_zfree, n_zfree, gamma), p_hat_closure
    must exactly match core.lppm.calibrator._clopper_pearson_lower's output.
    """
    from core.lppm.calibrator import _clopper_pearson_lower

    spec = _safety_spec_single_atom()
    dpa = build_parity_automaton(spec)
    T = 1000
    trajectory = [{"hazard": 0.0} for _ in range(T - 1)] + [{"hazard": 500.0}]

    result = verify_zfree_closure([trajectory], dpa, spec, eta=ETA, gamma=0.05)
    expected = _clopper_pearson_lower(result.k_zfree, result.n_zfree, 0.05)
    assert result.p_hat_closure == pytest.approx(expected)


def test_zfree_precondition_hard_gate_forces_abstain_despite_high_p_hat_gamma():
    """
    Permanent fix: issuing a CALIBRATED/SAFE verdict must be blocked outright
    whenever ANY sampled Z_free-source transition fails (P1)/(P2) -- i.e.
    Theorem 5.4's own minimal premise is violated on this sample -- no matter
    how high p_hat_gamma (or even p_hat_closure itself) happens to be. A
    single genuine violation diluted across many otherwise-passing Z_free
    trajectories must not be laundered into a warrant.
    """
    from core.lppm.calibrator import calibrate_lppm

    spec = _safety_spec_single_atom()
    dpa = build_parity_automaton(spec)
    T = 1000
    # 9 clean trajectories (never violate) + 1 with a single Z_free-source
    # violation at the end -- p_hat_gamma/p_hat_closure will both still be
    # comfortably high (9/10 or 10/10 depending on how the violating
    # trajectory's own C(tau) is scored), which is exactly the "diluted"
    # scenario the hard gate exists to catch.
    clean = [{"hazard": 1.0} for _ in range(T)]
    violating = [{"hazard": 1.0} for _ in range(T - 1)] + [{"hazard": 500.0}]
    trajectories = [clean] * 9 + [violating]

    result = calibrate_lppm(trajectories, dpa, spec, gamma=0.05, eta=ETA)

    assert result.zfree_closure.n_zfree >= 1, "fixture must actually exercise a Z_free-source transition"
    assert result.zfree_closure.k_zfree < result.zfree_closure.n_zfree, (
        "fixture must contain a genuine Z_free-source violation"
    )
    assert result.zfree_precondition_violated() is True
    assert result.n_zfree_violations() >= 1
    assert result.is_warranted() is False, (
        "is_warranted() must hard-reject when the Z_free precondition is violated, "
        "regardless of p_hat_gamma's own value"
    )
    assert "ABSTAIN" in result.summary()


def test_zfree_precondition_gate_does_not_trigger_on_a_clean_sample():
    """Converse of the hard-gate test: zero Z_free-source violations must
    leave is_warranted() governed by p_hat_gamma/threshold as before -- the
    gate must not become a blanket always-reject."""
    from core.lppm.calibrator import calibrate_lppm

    spec = _safety_spec_single_atom()
    dpa = build_parity_automaton(spec)
    T = 1000
    clean = [{"hazard": 1.0} for _ in range(T)]
    trajectories = [clean] * 10

    result = calibrate_lppm(trajectories, dpa, spec, gamma=0.05, eta=ETA, warrant_threshold=0.5)

    assert result.zfree_precondition_violated() is False
    assert result.n_zfree_violations() == 0
    assert result.is_warranted() == (result.p_hat_gamma >= 0.5)
