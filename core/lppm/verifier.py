from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .automaton import ParityAutomaton, ProductState, extract_active_aps
from .config import DEFAULT_LPPM_CONFIG
from .model import compute_lppm_value


def find_trajectory_overlap(a: Sequence[Any], b: Sequence[Any]) -> int:
    """
    Count elements present in both `a` and `b`, by object identity.

    Identity, not content equality: two independently-sampled trajectories (or
    L3 anchors) can be content-identical (e.g. a degenerate constant-height
    rollout -- see the height_safety k=0 diagnosis) without being the same
    held-out/training draw, so a content-based check would false-positive
    exactly on the datasets most worth checking. This only catches the case
    that actually matters: the caller passing the same objects to both a
    fit/train call and a calibrate/validate call.

    Generic over any list of identity-distinguishable objects -- despite
    living under core/lppm/, this is shared infrastructure: L2 uses it on
    trajectory lists (list[list[dict[str,float]]]) via
    core/lppm/trainer.py::fit_lppm() / calibrator.py::calibrate_lppm(); L3
    reuses it as-is on anchor lists (list[np.ndarray]) via
    core/lbsm/trainer.py::fit_lbsm(), where the check is mandatory rather
    than opt-in (see that module's docstring for why).
    """
    ids_a = {id(t) for t in a}
    return sum(1 for t in b if id(t) in ids_a)


def run_product_trajectory(
    trajectory: list[dict[str, float]],
    dpa: ParityAutomaton,
    spec: dict,
) -> list[ProductState]:
    q = dpa.initial
    product_path: list[ProductState] = []
    for t, z in enumerate(trajectory):
        active_aps = extract_active_aps(z, spec)
        q_next, priority = dpa.step_with_priority(q, active_aps)
        product_path.append(ProductState(t=t, z=z, q=q, q_next=q_next, priority=priority))
        q = q_next
    return product_path


@dataclass
class PathwiseResult:
    satisfied: bool
    p1_violations: int
    p2_violations: int
    total_transitions: int
    min_descent_margin: float
    conformity_score: float
    # Recorded for auditability (the closest equivalent to a "sidecar" for a
    # plain function return): the P1 numeric tolerance actually used. Default
    # 0.0 preserves old strict behavior for callers that don't pass one.
    p1_tolerance: float = 0.0


def iter_transitions(product_path: list[ProductState], dpa: ParityAutomaton) -> list[tuple[ProductState, ProductState]]:
    """
    Yields (curr, nxt) pairs for ALL T transitions in a T-observation
    trajectory (product_path has T entries, one per observation).

    External-review finding: every caller previously built this list by
    zipping ADJACENT product_path entries directly (`(path[i], path[i+1])`
    for i in range(T-1)). product_path[t].q always equals
    product_path[t-1].q_next (the state entering time t is the state that
    resulted from consuming z[t-1]), so consecutive entries correctly
    represent transitions 0..T-2 this way -- but the FINAL transition (on
    z[T-1], from product_path[T-1].q to product_path[T-1].q_next -- the true
    state after the ENTIRE trajectory) was never included anywhere, because
    there is no product_path[T] entry to pair it with. This silently made
    every check (P1/P2 pathwise conditions, Z_free-source detection, LPPM
    training examples) blind to whatever happened on the last observed AP
    of every trajectory.

    Fix: synthesize a terminal ProductState for the T-th (final) pair,
    representing "the true state after consuming the entire trajectory"
    (t=T, z reused from the last real observation since there is no z[T],
    q=q_next=the final automaton state, priority=that final state's OWN
    priority -- not the source state's priority used for every other
    entry). len(product_path) itself is untouched by this -- the synthetic
    entry only appears inside the returned pairs, never inside
    product_path -- so rem=(T-t)/T computations elsewhere are unaffected.
    """
    T = len(product_path)
    pairs = [(product_path[i], product_path[i + 1]) for i in range(T - 1)]
    final = product_path[T - 1]
    terminal = ProductState(
        t=T, z=final.z, q=final.q_next, q_next=final.q_next,
        priority=dpa.priority.get(final.q_next, 0),
    )
    pairs.append((final, terminal))
    return pairs


def _require_nonneg(v: float) -> float:
    """
    Batch-4 audit fix: defense in depth, independent of whether
    compute_lppm_value()'s own clamp (see core/lppm/model.py::_nonneg) is
    airtight for every call path (heuristic fallback, a learned/lppm_params
    model, or a future branch). V is required to be nonnegative by both the
    P2 descent check (curr.priority == r branch, min_descent margin) and the
    terminal Z_free = {(z,q): V_phi(z,q) < eta} membership test below -- a
    negative V would silently satisfy Z_free membership regardless of eta,
    fabricating a certificate the paper's definition does not support. Fail
    loud instead of computing a meaningless result on top of it.
    """
    if v < 0:
        raise ValueError(
            f"check_pathwise_conditions(): compute_lppm_value() returned a negative "
            f"V={v!r}. V is required to be nonnegative (Z_free = {{(z,q): V<eta}} and "
            "the P2 descent check both assume V >= 0) -- refusing to compute a "
            "pathwise result on top of an invalid certificate value."
        )
    return v


def check_pathwise_conditions(
    product_path: list[ProductState],
    dpa: ParityAutomaton,
    spec: dict,
    eta: float = DEFAULT_LPPM_CONFIG.eta,
    lppm_params: dict | None = None,
    p1_tol: float = 0.0,
) -> PathwiseResult:
    """
    p1_tol : numeric tolerance for the P1 (non-increase) check: a transition is
      only flagged as a P1 violation if v_next > v_curr + p1_tol. Default 0.0
      means EXACTLY zero tolerance -- matching core/lppm/loss.py::p1_loss's
      training-side check (ReLU(v_next - v_curr), also zero tolerance)
      exactly. Batch-4 audit fix: this function previously applied
      max(1e-6, p1_tol) regardless of what was passed, so p1_tol=0.0 (the
      only value any real caller currently passes) never actually meant
      zero -- a hardcoded 1e-6 floor was silently absorbing any P1 gap
      smaller than that, which training-side would still have penalized.
      That asymmetry is gone; p1_tol=0.0 is now truly strict.

      When V has collapsed to a near-constant output (e.g. a trap-sparse
      training set starves the P2 loss of gradient signal), float-level
      noise around that constant can trip a strict non-increase check on
      ~40% of transitions even though the "violations" are not meaningful
      non-monotonicity -- see tdmpc2/eval_ltl_height_safety_lppm.py's k=0
      diagnosis. Callers deriving p1_tol from data should compute it from
      THIS calibration run's own V distribution (e.g. a multiple of its
      std), not a hardcoded constant, and report the value used alongside
      any p_hat_gamma it affected. This is still legitimate — the default
      just no longer applies it implicitly.
    """
    odd_prios = dpa.odd_priorities
    T = len(product_path)
    p1_viols = p2_viols = 0
    min_descent = math.inf

    # External-review fix: use iter_transitions() (see its docstring) so the
    # FINAL transition (on the last observed AP) is included -- previously
    # zipping adjacent product_path entries directly silently dropped it.
    transitions = iter_transitions(product_path, dpa)
    for curr, nxt in transitions:
        for r in odd_prios:
            v_curr = _require_nonneg(compute_lppm_value(curr.z, curr.q, r, spec, curr.t, T, lppm_params, dpa))
            v_next = _require_nonneg(compute_lppm_value(nxt.z, nxt.q, r, spec, nxt.t, T, lppm_params, dpa))
            if curr.priority == r:
                margin = v_curr - v_next
                min_descent = min(min_descent, margin)
                if margin < eta:
                    p2_viols += 1
            elif r > curr.priority and v_next > v_curr + p1_tol:
                p1_viols += 1

    total = max(len(transitions), 1)
    # C(tau) per Theorem 5.4 (statistical/calibrated L2 branch -- corrected from a
    # stale "Theorem 5.5" citation, which per the paper's current numbering is the
    # L3 idealized-martingale result, a different layer entirely):
    # (P1)-(P2) hold on every transition AND the terminal
    # product state lies in Z_free = {(z,q): V_phi(z,q) < eta} on every odd head.
    # (Previously missing: `satisfied` ignored the terminal-Z_free conjunct entirely,
    # so C(tau) was systematically looser than the paper's definition.)
    #
    # External-review fix: the terminal state must be the TRUE final state
    # (product_path[-1].q_next, via the synthetic pair's own .q -- NOT
    # product_path[-1].q, which is one observation "behind"). Also: landing
    # in an ODD-priority state (e.g. Safety's "trap") is treated as an
    # automatic violation, NOT deferred to a V<eta check -- core/lppm/model.py
    # returns V=0.0 unconditionally for "trap" (and similar absorbing bad
    # states), which would otherwise trivially satisfy Z_free membership
    # regardless of how the trajectory actually ended, silently defeating
    # the very safety violations this check exists to catch.
    final_state = transitions[-1][1].q
    final_priority = dpa.priority.get(final_state, 0)
    if final_priority % 2 == 1:
        in_z_free = False
    else:
        final_z = transitions[-1][1].z
        in_z_free = all(
            _require_nonneg(compute_lppm_value(final_z, final_state, r, spec, T, T, lppm_params, dpa)) < eta
            for r in odd_prios
        )
    satisfied = (p1_viols == 0) and (p2_viols == 0) and in_z_free
    return PathwiseResult(
        satisfied=satisfied,
        p1_violations=p1_viols,
        p2_violations=p2_viols,
        total_transitions=total,
        min_descent_margin=min_descent if min_descent != math.inf else 0.0,
        conformity_score=1.0 if satisfied else 0.0,
        p1_tolerance=p1_tol,
    )


@dataclass
class ZFreeClosureResult:
    """
    Theorem 5.4's actual minimal premise: (P1)-(P2) hold on transitions whose
    SOURCE already lies in Z_free = {(z,q): V_phi(z,q) < eta} -- a strictly
    weaker (more precisely targeted) event than check_pathwise_conditions()'s
    C(tau), which requires (P1)-(P2) on EVERY transition. p_hat_gamma values
    computed against C(tau) remain valid lower bounds for the weaker event
    (a stronger event's probability lower-bounds a weaker one's), but citing
    them as satisfying Theorem 5.4's minimal premise specifically requires
    this separate, Z_free-restricted check.

    n_zfree/k_zfree count TRAJECTORIES, not individual transitions (External-
    review fix -- see verify_zfree_closure()'s docstring for why: transitions
    within one trajectory are not independent Bernoulli samples).
    """
    n_zfree: int
    k_zfree: int
    p_hat_closure: float | None
    verifiable: bool
    gamma: float = 0.05
    eta: float = DEFAULT_LPPM_CONFIG.eta


def verify_zfree_closure(
    trajectories: list[list[dict[str, float]]],
    dpa: ParityAutomaton,
    spec: dict,
    gamma: float = 0.05,
    eta: float = DEFAULT_LPPM_CONFIG.eta,
    lppm_params: dict | None = None,
    p1_tol: float = 0.0,
) -> ZFreeClosureResult:
    """
    Independent of check_pathwise_conditions() by design -- deliberately not
    sharing its loop or being folded into it via a parameter, since the two
    verify different events (whole-trajectory C(tau) vs. the Theorem-5.4-
    exact Z_free-restricted event) and conflating them into one
    parameter-branched function would make it easy to accidentally couple
    their behavior in a future edit.

    External-review fix: the statistical UNIT here is the TRAJECTORY, not
    the individual (transition, head) pair. Clopper-Pearson's exact interval
    assumes the k successes among n trials are i.i.d. Bernoulli (or at least
    exchangeable) draws -- multiple transitions from the SAME trajectory
    share one underlying V_phi realization and autocorrelated dynamics, so
    treating each one as an independent sample (the previous implementation)
    understates the true uncertainty and can badly overstate p_hat_closure
    (a single genuine violation, diluted across many passing transitions
    from the same trajectory, barely moves the per-transition rate).

    For each trajectory: find every (transition, odd-priority head r) pair
    with r >= the source state's priority (the only pairs
    check_pathwise_conditions() itself ever evaluates a pathwise condition
    for) whose SOURCE value V(curr.z, curr.q, r) < eta (source in Z_free for
    that head). If this trajectory has at least one such pair, it
    contributes exactly one trial to n_zfree, and exactly one success to
    k_zfree IFF *every* one of its Z_free-source pairs satisfies P1/P2 --
    mirroring PathwiseResult.satisfied's whole-trajectory C(tau)
    construction (an AND across all checked conditions), just restricted to
    the Z_free-source subset. A trajectory with ZERO Z_free-source
    transitions contributes nothing to either n_zfree or k_zfree (it cannot
    be counted as "passing" a condition it never triggered).

    Reuses core.lppm.calibrator._clopper_pearson_lower (imported locally to
    avoid a circular import -- calibrator.py imports from this module at
    module scope) rather than reimplementing the Clopper-Pearson formula.

    n_zfree == 0 (no trajectory in the calibration set ever had a source
    enter Z_free -- a real, previously-observed degenerate case, see
    tdmpc2/eval_ltl_height_safety_lppm.py's k=0 diagnosis) is reported as
    verifiable=False, p_hat_closure=None -- NOT as 0.0 or 1.0, either of
    which would look like a real statistical bound but carries no
    statistical meaning with zero samples.
    """
    from .calibrator import _clopper_pearson_lower

    n_zfree = 0
    k_zfree = 0
    for traj in trajectories:
        path = run_product_trajectory(traj, dpa, spec)
        T = len(path)
        had_zfree_transition = False
        all_satisfied = True
        # External-review fix: see iter_transitions()'s docstring -- this
        # previously zipped adjacent path entries directly, silently
        # dropping the FINAL transition (on the last observed AP) from
        # Z_free-source detection entirely.
        for curr, nxt in iter_transitions(path, dpa):
            for r in dpa.odd_priorities:
                if r < curr.priority:
                    continue
                v_curr = _require_nonneg(compute_lppm_value(curr.z, curr.q, r, spec, curr.t, T, lppm_params, dpa))
                if v_curr >= eta:
                    continue
                had_zfree_transition = True
                v_next = _require_nonneg(compute_lppm_value(nxt.z, nxt.q, r, spec, nxt.t, T, lppm_params, dpa))
                if curr.priority == r:
                    satisfied = (v_curr - v_next) >= eta
                else:
                    satisfied = v_next <= v_curr + p1_tol
                if not satisfied:
                    all_satisfied = False

        if had_zfree_transition:
            n_zfree += 1
            if all_satisfied:
                k_zfree += 1

    if n_zfree == 0:
        return ZFreeClosureResult(
            n_zfree=0, k_zfree=0, p_hat_closure=None, verifiable=False, gamma=gamma, eta=eta,
        )
    p_hat = _clopper_pearson_lower(k_zfree, n_zfree, gamma)
    return ZFreeClosureResult(
        n_zfree=n_zfree, k_zfree=k_zfree, p_hat_closure=p_hat, verifiable=True, gamma=gamma, eta=eta,
    )
