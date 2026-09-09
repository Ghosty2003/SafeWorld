"""
tests/test_lbsm_analytic.py

Stage-1 acceptance test for core/lbsm/ (SAFEWORLD L3, Appendix E.2's analytic
recurrence construction). This module implements the paper's Section 4.4 /
Theorem 5.6 machinery, distinct from cardreamer/eval_l3_live.py (a project-
internal sequential-zones probe) and utils/spec_analysis.py's "L3" complexity
level (Table 15) -- neither of those is what this module or test targets.

Primary acceptance criterion (per task spec): the concrete parameterization
gamma=0.5, sigma=0.1, d=2, k=3, r_F=0.30, ell=10, ||z0||^2=0.8 must reproduce
the paper's own reported warrant >= 0.92. A mismatch is a bug in this
implementation, not "an acceptable difference in reproduction method."
"""

from __future__ import annotations

import numpy as np
import pytest

from core.lbsm.collar import ball_crosses_boundary, eliminate_collar_stage1_analytic
from core.lbsm.drift import empirical_drift, exact_drift_e2, sample_truncated_gaussian_step, z_star_sq
from core.lbsm.ldba import accepting_mask, build_e2_recurrence_automaton, run_trajectory
from core.lbsm.model import AnalyticCertificate
from core.lbsm.verifier import verify_e2_recurrence_warrant

# Appendix E.2's concrete instance, used throughout.
GAMMA = 0.5
SIGMA = 0.1
D = 2
K = 3
R_F = 0.30
ELL = 10.0
Z0_NORM_SQ = 0.8


# ─── Primary acceptance test ────────────────────────────────────────────────

def test_e2_warrant_matches_paper():
    """The paper's own reported number: warrant >= 0.92."""
    result = verify_e2_recurrence_warrant(
        gamma=GAMMA, sigma=SIGMA, d=D, k=K, r_F=R_F, ell=ELL, z0_norm_sq=Z0_NORM_SQ,
    )
    assert result.warrant >= 0.92 - 1e-6, (
        f"warrant={result.warrant} does not reach the paper's 0.92 "
        f"(alpha={result.alpha}, alpha_collar={result.alpha_collar})"
    )
    # Exact match, not just "at least" -- alpha=0.08 exactly, alpha_collar=0
    # exactly, so warrant should be 0.92 to floating-point precision.
    assert result.warrant == pytest.approx(0.92, abs=1e-9)
    assert result.alpha == pytest.approx(0.08, abs=1e-9)
    assert result.confidence == 1.0


def test_alpha_collar_is_zero_and_actually_checked():
    """
    alpha_collar must come out to 0 via the elimination-route computation
    (core/lbsm/collar.py), not a hardcoded literal in the verifier.
    """
    result = verify_e2_recurrence_warrant(
        gamma=GAMMA, sigma=SIGMA, d=D, k=K, r_F=R_F, ell=ELL, z0_norm_sq=Z0_NORM_SQ,
    )
    assert result.alpha_collar == 0.0

    # Directly exercise the underlying primitive too: at mesh_delta=0 a ball
    # never crosses any boundary it isn't exactly centered on.
    assert eliminate_collar_stage1_analytic(R_F) == 0.0
    assert ball_crosses_boundary(anchor_norm=R_F, boundary_radius=R_F, mesh_delta=0.0) is False
    # Sanity check the primitive itself is not vacuously always-False: with a
    # positive mesh radius that spans the boundary, it MUST report crossing.
    assert ball_crosses_boundary(anchor_norm=R_F, boundary_radius=R_F, mesh_delta=0.05) is True


def test_drift_margins_match_paper_approximation():
    """
    Paper's own rounded figures: inner margin ~= 0.047 (I\\F, worst case at
    r_F), outer margin ~= 0.52 (C\\I, worst case at r_inv).
    """
    result = verify_e2_recurrence_warrant(
        gamma=GAMMA, sigma=SIGMA, d=D, k=K, r_F=R_F, ell=ELL, z0_norm_sq=Z0_NORM_SQ,
    )
    assert result.drift_margin_inner == pytest.approx(0.0475, abs=1e-3)
    assert result.drift_margin_outer == pytest.approx(0.52, abs=1e-2)
    assert result.b == pytest.approx(0.424264, abs=1e-5)
    assert result.r_inv == pytest.approx(0.848528, abs=1e-5)


# ─── Lemma E.24 precondition ────────────────────────────────────────────────

def test_lemma_e24_precondition_holds_for_paper_params():
    """r_F=0.30 < b~=0.424264, so the precondition for delta_exit > 0 holds."""
    result = verify_e2_recurrence_warrant(
        gamma=GAMMA, sigma=SIGMA, d=D, k=K, r_F=R_F, ell=ELL, z0_norm_sq=Z0_NORM_SQ,
    )
    assert result.lemma_e24_precondition_ok is True
    assert R_F < result.b


def test_lemma_e24_precondition_violation_raises():
    """
    r_F=0.45 sits between b~=0.4243 and r_inv~=0.8485: F is still strictly
    inside I (doesn't trip Remark E.25), but r_F >= b violates Lemma E.24 --
    must raise, not silently return an unsupported warrant.
    """
    with pytest.raises(ValueError, match="Lemma E.24"):
        verify_e2_recurrence_warrant(
            gamma=GAMMA, sigma=SIGMA, d=D, k=K, r_F=0.45, ell=ELL, z0_norm_sq=Z0_NORM_SQ,
        )


def test_remark_e25_degenerate_f_not_strictly_inside_i_raises():
    """r_F=0.9 > r_inv~=0.8485: F not strictly inside I -- must raise."""
    with pytest.raises(ValueError, match="Remark E.25"):
        verify_e2_recurrence_warrant(
            gamma=GAMMA, sigma=SIGMA, d=D, k=K, r_F=0.9, ell=ELL, z0_norm_sq=Z0_NORM_SQ,
        )


# ─── LDBA timing convention (destination-state, not source-state) ──────────

def test_ldba_accepting_set_uses_destination_state_not_source_state():
    """
    Construct a trajectory where p first holds at t=3. The CORRECT
    (destination-state / "entering x_t consumes L(z_t)") convention must
    report F_X first entered at t=3. A source-state implementation (judging
    acceptance on q_{t-1}, the state BEFORE consuming z_t -- i.e. the
    convention core/lppm/automaton.py::step_with_priority() uses, correctly,
    for L2's source-indexed (P1)-(P2)) would report it one step late, at
    t=4. If this test's primary assertion ever matched the "wrong_mask"
    constructed below instead of the paper-correct mask, that would mean the
    destination-state convention was accidentally replaced by a source-state
    one.
    """
    z_outside = np.array([0.5, 0.0])  # ||z||=0.5 > r_F=0.3 -> label p is false
    z_inside = np.array([0.1, 0.0])   # ||z||=0.1 <= r_F=0.3 -> label p is true
    zs = [z_outside, z_outside, z_outside, z_inside, z_inside]

    automaton = build_e2_recurrence_automaton(r_F=R_F)
    q_states = run_trajectory(automaton, zs)
    mask = accepting_mask(automaton, q_states)

    correct_mask = [False, False, False, True, True]
    assert mask == correct_mask, (
        f"got {mask}, expected {correct_mask} -- F_X must be entered exactly "
        "when the CURRENT step's label satisfies p (destination-state "
        "convention), not one step later."
    )

    # The wrong (source-state) convention: judge acceptance using q_{t-1}
    # (the state BEFORE consuming z_t) instead of q_t. Before any step, the
    # automaton is in its (non-accepting) initial condition.
    wrong_mask = [False] + mask[:-1]
    assert wrong_mask == [False, False, False, False, True]
    assert mask != wrong_mask, "test is vacuous if the two conventions coincide on this trajectory"


# ─── Empirical drift (Eq. 7) cross-check, diagnostic only ──────────────────

def test_empirical_drift_roughly_matches_exact_closed_form():
    """
    Eq.7's finite-kappa Monte-Carlo estimator, sampled from the actual
    truncated-Gaussian kernel, should approximately agree with
    exact_drift_e2() -- a sanity check that the closed form is not a
    transcription error. This is NOT the certified quantity (see
    core/lbsm/drift.py's module docstring); generous statistical tolerance,
    fixed seed for reproducibility.
    """
    rng = np.random.default_rng(0)
    b = K * SIGMA * (D ** 0.5)
    anchor = np.array([R_F, 0.0])  # worst-case radius used in the verifier
    kappa = 5000
    successors = [
        sample_truncated_gaussian_step(anchor, GAMMA, SIGMA, D, b, rng)
        for _ in range(kappa)
    ]
    cert = AnalyticCertificate(ell=ELL)
    empirical = empirical_drift(cert.U, anchor, successors)
    exact = exact_drift_e2(anchor, GAMMA, z_star_sq(D, SIGMA, GAMMA))

    assert empirical == pytest.approx(exact, abs=0.02), (
        f"empirical drift {empirical} strays too far from exact closed form {exact} "
        f"at kappa={kappa} -- check exact_drift_e2()'s derivation, not just re-seed."
    )


# ─── End-to-end pipeline sanity check (qualitative, not the certified number) ─

def test_pipeline_rollout_shows_genuine_recurrence_not_absorption():
    """
    Sample an actual rollout from the analytic kernel starting inside C, run
    it through the LDBA, and confirm the qualitative claim the whole
    construction rests on: F is entered AND later exited again (repeated
    exit-and-return), not permanently absorbed after one entry -- this is
    the empirical face of Lemma E.24 / the Box-Diamond-vs-Diamond-Box
    distinction (Remark E.25). Qualitative Monte-Carlo check, not a
    replacement for the closed-form warrant above.
    """
    rng = np.random.default_rng(1)
    b = K * SIGMA * (D ** 0.5)
    z = np.array([0.0, 0.0])
    T = 4000
    zs = [z]
    for _ in range(T - 1):
        z = sample_truncated_gaussian_step(z, GAMMA, SIGMA, D, b, rng)
        zs.append(z)

    automaton = build_e2_recurrence_automaton(r_F=R_F)
    q_states = run_trajectory(automaton, zs)
    mask = accepting_mask(automaton, q_states)

    assert any(mask), "F was never entered in this rollout -- check r_F/dynamics parameters"
    first_enter = mask.index(True)
    assert not all(mask[first_enter:]), (
        "once F was entered, the run never left again -- this rollout looks like "
        "Persistence (eventually-always-F), not genuine Recurrence (F infinitely "
        "often with exits in between)"
    )
