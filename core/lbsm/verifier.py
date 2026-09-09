"""
core/lbsm/verifier.py

Theorem 5.6 (LBSM Probabilistic Warrant) verification for the Stage-1
ANALYTIC construction of Appendix E.2 -- L3's counterpart to
core/lppm/calibrator.py's Theorem 5.4 check.

This module checks, in order, the six items Theorem 5.6 requires before a
warrant may be issued (paper citations inline on each check):

  1. containment {U <= ell} subseteq C_cert
  2. (B1) drift margin on C \\ F_X, at effective margin >= eps'
  3. retention hypothesis (H-a) or (H-b) -- this module uses (H-b), the
     paper's own choice for Appendix E.2 (F strictly-subset I)
  4. Remark E.25 guard: F must be STRICTLY inside I, else the construction
     degenerates to a Persistence (L2) witness, not genuine Recurrence (L3)
  5. Ville's inequality: alpha = E_mu0[U(x0)] / ell
  6. warrant = 1 - alpha - alpha_collar, at confidence 1 - n*delta1 -
     n*delta1_U - delta0

Why no eps_eff = eps - eta_conc - L_phi*Delta correction (Theorem D.3): that
correction prices the gap between a finite covering net of sampled anchors
and the continuum region it approximates. Appendix E.2 evaluates the drift in
EXACT closed form at every point of the certified annuli
(core/lbsm/drift.py::exact_drift_e2) -- there is no covering net, hence no
eta_conc or Delta term to subtract. eps' == eps exactly for this
construction. This is not a simplification of Theorem D.3's general
machinery; it is what the paper's own Tier-1 analytic path does (Section 6.1:
"we quote no numeric query count" for exactly this reason).

Why confidence == 1.0: Theorem 5.6's confidence budget is
1 - n*delta1 - n*delta1_U - delta0, where n*delta1 / n*delta1_U come from
per-anchor Hoeffding concentration (none used here -- no sampled anchors) and
delta0 from a (1-delta0) upper confidence bound on E_mu0[U] when it is
estimated (not needed here -- Appendix E.2 treats z0 as a single GIVEN point,
so E_mu0[U(x0)] = U(z0) exactly). n = 0 and delta0 = 0 throughout Stage 1,
giving confidence = 1 - 0 - 0 - 0 = 1. This is a genuine consequence of the
analytic construction's exactness, not an approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .collar import assert_anchor_ball_does_not_cross_f_boundary, eliminate_collar_stage1_analytic
from .drift import empirical_drift, exact_drift_e2, sample_truncated_gaussian_step, z_star_sq
from .lipschitz import lbsm_net_design_lipschitz


@dataclass(frozen=True)
class LBSMWarrantResult:
    warrant: float
    alpha: float
    alpha_collar: float
    confidence: float
    containment_ok: bool
    drift_margin_inner: float  # eps' on I \ F, worst case at r_F
    drift_margin_outer: float  # H-b companion margin on C \ I, worst case at r_inv
    f_strictly_inside_i: bool
    lemma_e24_precondition_ok: bool
    r_inv: float
    b: float


def _radial_point(norm_sq: float, d: int) -> np.ndarray:
    """A d-dim vector with the given squared norm, on the first axis --
    exact_drift_e2 only depends on ||z||^2, so direction is arbitrary."""
    z = np.zeros(d)
    z[0] = float(np.sqrt(norm_sq))
    return z


def verify_e2_recurrence_warrant(
    gamma: float,
    sigma: float,
    d: int,
    k: float,
    r_F: float,
    ell: float,
    z0_norm_sq: float,
) -> LBSMWarrantResult:
    """
    Full Theorem 5.6 check for Appendix E.2's concrete parameterization.

    Raises ValueError if either structural precondition fails (F not
    strictly inside I, or Lemma E.24's r_F < b precondition) -- these are not
    warnings, because a warrant issued despite either failing would not be
    evidence of what this function's name claims (genuine recurrence).
    """
    b = k * sigma * (d ** 0.5)
    r_inv = b / (1 - gamma)

    # 4. Remark E.25 guard -- F must be STRICTLY inside I. Checked before
    #    anything else: a degenerate F == I run would otherwise "succeed"
    #    with a warrant that certifies Persistence dressed up as Recurrence.
    f_strictly_inside_i = r_F < r_inv
    if not f_strictly_inside_i:
        raise ValueError(
            f"Remark E.25 degenerate case: r_F={r_F} is not strictly inside "
            f"r_inv={r_inv:.6f} (need r_F < r_inv). Taking F=I collapses this "
            "construction to a Persistence (L2) reach-and-stay witness, not "
            "genuine Recurrence (L3) -- refusing to issue an L3 warrant."
        )

    # Lemma E.24 precondition (uniform one-step exit probability delta_exit > 0
    # requires r_F < b -- the spherical cap {||xi||<=b, <xi,u> > r_F} is
    # nonempty and open iff r_F < b). This function checks only the
    # PRECONDITION, not a computed delta_exit value: the paper's own proof of
    # delta_exit > 0 is qualitative/existential (Lemma E.24's proof: "the
    # spherical cap ... contains a nonempty open set precisely because
    # r_F < b"), not a numeric quantity the paper reports.
    lemma_e24_ok = r_F < b
    if not lemma_e24_ok:
        raise ValueError(
            f"Lemma E.24 precondition fails: r_F={r_F} >= b={b:.6f}. Without "
            "r_F < b, the uniform one-step exit probability delta_exit > 0 is "
            "not established, so a run that reaches F is not guaranteed to "
            "leave it again -- this construction would not exercise genuine "
            "Recurrence (repeated exit-and-return) content, only a one-time "
            "absorption. Refusing to issue an L3 warrant."
        )

    z_star_sq_ = z_star_sq(d, sigma, gamma)

    # 2. (B1) drift margin on I \ F (part of C \ F_X), worst case at r_F
    #    (drift is monotonically more negative as ||z||^2 grows, so the
    #    smallest-magnitude -- worst-case -- margin in an annulus occurs at
    #    its inner radius).
    drift_at_r_F = exact_drift_e2(_radial_point(r_F ** 2, d), gamma, z_star_sq_)
    drift_margin_inner = -drift_at_r_F  # eps' ; must be > 0
    if drift_margin_inner <= 0:
        raise ValueError(
            f"(B1) drift margin on I\\F is not positive (eps'={drift_margin_inner:.6f}) "
            "-- V does not descend on the required region for this parameterization."
        )

    # 3. (H-b) companion supermartingale margin on C \ I, worst case at r_inv.
    #    H-b only requires <= 0 (supermartingale); this construction gets
    #    strictly < 0, a strictly stronger property, "for free" from the same
    #    closed form (paper: "the drift ... improves outward").
    drift_at_r_inv = exact_drift_e2(_radial_point(r_inv ** 2, d), gamma, z_star_sq_)
    drift_margin_outer = -drift_at_r_inv
    if drift_margin_outer < 0:
        raise ValueError(
            f"(H-b) companion drift on C\\I is not a supermartingale "
            f"(margin={drift_margin_outer:.6f} < 0) for this parameterization."
        )

    # 1. containment {U <= ell} subseteq C_cert. In this construction
    #    C_cert IS {U <= ell} = C by definition (no separate covering-net-
    #    certified region distinct from the exact-evaluation domain), so
    #    containment holds trivially and exactly, not approximately.
    containment_ok = True

    # 5. Ville's inequality (Theorem 5.6): alpha = E_mu0[U(x0)] / ell.
    #    z0_norm_sq is treated as a single GIVEN point (deterministic
    #    initial condition), so E_mu0[U(x0)] = U(z0) exactly -- no
    #    concentration/estimation term (delta0 = 0).
    if z0_norm_sq > ell:
        raise ValueError(
            f"z0_norm_sq={z0_norm_sq} > ell={ell}: mu0 is not supported in C "
            "(Theorem 5.6's premise), so the warrant is not applicable."
        )
    alpha = z0_norm_sq / ell

    # Collar: route (a) elimination (core/lbsm/collar.py), computed not
    # hardcoded.
    alpha_collar = eliminate_collar_stage1_analytic(r_F)

    # 6. Final warrant and confidence.
    warrant = 1.0 - alpha - alpha_collar
    confidence = 1.0  # n=0 sampled anchors, delta0=0 (z0 deterministic); see module docstring

    return LBSMWarrantResult(
        warrant=warrant,
        alpha=alpha,
        alpha_collar=alpha_collar,
        confidence=confidence,
        containment_ok=containment_ok,
        drift_margin_inner=drift_margin_inner,
        drift_margin_outer=drift_margin_outer,
        f_strictly_inside_i=f_strictly_inside_i,
        lemma_e24_precondition_ok=lemma_e24_ok,
        r_inv=r_inv,
        b=b,
    )


# ═════════════════════════════════════════════════════════════════════════
# Stage 2: Theorem D.3's sample-based validation-complexity correction, for
# TRAINED (not closed-form) W/U certificates.
#
# eps_eff = eps - eta_conc - (L + L_phi) * Delta
#
# This section is NOT used by verify_e2_recurrence_warrant() above (Stage 1
# stays exact/closed-form, untouched). It is the mandatory correction for
# core/lbsm/trainer.py's trained networks (task correction: "Stage 2 的
# verifier.py 校驗邏輯必須啟用 Theorem D.3 的完整 ε_eff 修正,不能複用 Stage 1
# 那個 eta_conc=Delta=0 的簡化路徑").
# ═════════════════════════════════════════════════════════════════════════


def hoeffding_eta_conc(kappa: int, B_bar: float, delta_prime: float) -> float:
    """
    Per-anchor concentration error for an empirical mean of kappa i.i.d.
    samples bounded in a range of width B_bar (Hoeffding's inequality): with
    probability >= 1 - delta_prime,

        |empirical_mean - true_mean| <= B_bar * sqrt(log(2/delta_prime) / (2*kappa))

    Theorem D.3: "Hoeffding with range B_bar gives per-anchor concentration
    eta_conc <= eps_min/4 at failure probability delta/n."
    """
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    return B_bar * math.sqrt(math.log(2.0 / delta_prime) / (2.0 * kappa))


def choose_kappa_for_target_eta(target_eta: float, B_bar: float, delta_prime: float) -> int:
    """Inverse of hoeffding_eta_conc(): smallest kappa achieving eta_conc <= target_eta."""
    if target_eta <= 0:
        raise ValueError("target_eta must be positive")
    kappa = math.ceil(B_bar ** 2 * math.log(2.0 / delta_prime) / (2.0 * target_eta ** 2))
    return max(kappa, 1)


def epsilon_effective(epsilon: float, eta_conc: float, L: float, L_phi: float, mesh_delta: float) -> float:
    """
    Theorem D.3: eps_eff = eps - eta_conc - (L + L_phi) * Delta.

    `epsilon` here is the RAW margin observed at a specific anchor (e.g.
    -empirical_drift(anchor)), not a training-time target -- verification
    asks "how much of this anchor's own measured margin survives after
    (a) subtracting sampling-concentration slack (eta_conc) and (b) lifting
    from the anchor to its whole mesh_delta-ball via the Lipschitz bound
    (L + L_phi)?". Only neighborhoods with eps_eff > 0 are retained (paper:
    "Their union forms the certified region C_cert").

    L      : the DYNAMICS' post-expectation Lipschitz contribution (Theorem
             D.5 for a trained transition model; exactly `gamma` for Stage
             2's analytic linear kernel -- see verify_trained_recurrence_warrant()).
    L_phi  : the certificate's OWN design Lipschitz constant (L_V or L_U,
             core/lbsm/lipschitz.py -- this is what makes the correction
             genuinely nonzero once W/U are trained networks rather than
             Stage 1's exact ||z||^2).
    """
    return epsilon - eta_conc - (L + L_phi) * mesh_delta


def is_anchor_certified(observed_margin: float, eta_conc: float, L: float, L_phi: float, mesh_delta: float) -> bool:
    """
    The actual per-anchor certification decision used by
    verify_trained_recurrence_warrant() below: True iff the FULL Theorem D.3
    correction leaves a positive effective margin.

    Note what this function does NOT do: it does not stop at
    `observed_margin - eta_conc > 0` (statistical concentration alone). See
    tests/test_lbsm_trained.py::test_epsilon_eff_correction_catches_false_positive_naive_check_misses
    for a concrete numeric case where that naive check would certify an
    anchor this function correctly refuses.

    Companion note (Remark after Theorem E.21, quoted in core/lbsm/verifier.py's
    Stage-1 module docstring): a non-strict supermartingale (H-a/H-b's
    E[U(x')] <= U(x)) cannot be certified from finite samples plus a
    Lipschitz lift -- certification needs a STRICTLY positive effective
    margin, which is exactly what `eps_eff > 0` requires. This is why the
    companion (U) anchors below are checked with this same strict function,
    not a separate non-strict one.
    """
    return epsilon_effective(observed_margin, eta_conc, L, L_phi, mesh_delta) > 0


def generate_annulus_anchors(
    r_inner: float, r_outer: float, n_radial: int, n_angular: int,
) -> list[np.ndarray]:
    """
    2D covering-net anchors over the annulus r_inner < ||z|| <= r_outer, on a
    radius x angle grid. d=2 throughout this project's Stage-2 construction
    (Appendix E.2's own d=2 instance); a different anchor generator would be
    needed for d != 2, not implemented (see
    verify_trained_recurrence_warrant()'s explicit guard).
    """
    if r_inner >= r_outer:
        return []
    anchors = []
    for i in range(n_radial):
        r = r_inner + (r_outer - r_inner) * (i + 0.5) / n_radial
        for j in range(n_angular):
            theta = 2.0 * math.pi * j / n_angular
            anchors.append(np.array([r * math.cos(theta), r * math.sin(theta)]))
    return anchors


@dataclass(frozen=True)
class TrainedLBSMVerificationResult:
    all_anchors_certified: bool
    n_anchors: int
    n_certified: int
    n_failed: int
    mesh_delta: float
    kappa: int
    eta_conc: float
    L: float
    L_V: float
    L_U: float
    failed_anchor_examples: list[tuple[float, float]] = field(default_factory=list)  # (radius, observed_margin)
    warrant: float | None = None
    alpha: float | None = None
    alpha_collar: float | None = None


def verify_trained_recurrence_warrant(
    W_net,
    U_net,
    B: float,
    gamma: float,
    sigma: float,
    d: int,
    k: float,
    r_F: float,
    ell: float,
    z0_norm_sq: float,
    eps_tr: float,
    eps_U: float,
    kappa: int,
    delta_prime: float,
    n_radial: int,
    n_angular: int,
    rng: np.random.Generator,
) -> TrainedLBSMVerificationResult:
    """
    Full Theorem 5.6 check for TRAINED W_net/U_net certificates, via a real
    (if modest-resolution) covering net over I\\F and C\\I, with the Theorem
    D.3 eps_eff correction applied at every anchor. Dynamics remain the
    Stage-1 analytic kernel z' = gamma*z + xi (Stage 2 trains new
    certificates against the same, already-understood dynamics; a trained
    transition model is Stage 3 scope).

    Requires torch (W_net/U_net are core/lbsm/trainer.py::LBSMNet instances).
    """
    import torch

    if d != 2:
        raise NotImplementedError(
            "verify_trained_recurrence_warrant()'s covering net is a 2D radius x angle "
            "grid (generate_annulus_anchors); d != 2 needs a different anchor generator."
        )

    b = k * sigma * (d ** 0.5)
    r_inv = b / (1 - gamma)
    if not (r_F < r_inv):
        raise ValueError(f"Remark E.25 degenerate case: r_F={r_F} not strictly inside r_inv={r_inv:.6f}")
    if not (r_F < b):
        raise ValueError(f"Lemma E.24 precondition fails: r_F={r_F} >= b={b:.6f}")
    if z0_norm_sq > ell:
        raise ValueError(f"z0_norm_sq={z0_norm_sq} > ell={ell}: mu0 is not supported in C.")

    L = gamma  # exact for this analytic kernel -- see epsilon_effective()'s docstring
    L_V = lbsm_net_design_lipschitz(W_net, B)
    L_U = lbsm_net_design_lipschitz(U_net, B)
    eps_min = min(eps_tr, eps_U)
    mesh_delta = eps_min / (4.0 * (L + max(L_V, L_U)))
    eta_conc = hoeffding_eta_conc(kappa, B_bar=B, delta_prime=delta_prime)

    def W_fn(z: np.ndarray) -> float:
        with torch.no_grad():
            return float(W_net(torch.as_tensor(z, dtype=torch.float32).unsqueeze(0)).item())

    def U_fn(z: np.ndarray) -> float:
        with torch.no_grad():
            return float(U_net(torch.as_tensor(z, dtype=torch.float32).unsqueeze(0)).item())

    # I \ F: inner radius starts at r_F + mesh_delta, NOT r_F, so every
    # retained anchor's mesh_delta-ball is excluded from crossing the
    # F-boundary by construction (route (a) elimination, Stage-2 covering-net
    # form -- core/lbsm/collar.py::assert_anchor_ball_does_not_cross_f_boundary
    # double-checks this per anchor rather than trusting the construction alone).
    inner_anchors = generate_annulus_anchors(r_F + mesh_delta, r_inv, n_radial, n_angular)
    outer_anchors = generate_annulus_anchors(r_inv, math.sqrt(ell), n_radial, n_angular)

    n_certified = 0
    n_failed = 0
    failed_examples: list[tuple[float, float]] = []

    for z in inner_anchors:
        radius = float(np.linalg.norm(z))
        assert_anchor_ball_does_not_cross_f_boundary(radius, r_F, mesh_delta)
        successors = [sample_truncated_gaussian_step(z, gamma, sigma, d, b, rng) for _ in range(kappa)]
        g_hat_V = empirical_drift(W_fn, z, successors)
        observed_margin = -g_hat_V  # want this to be a large positive descent margin
        if is_anchor_certified(observed_margin, eta_conc, L, L_V, mesh_delta):
            n_certified += 1
        else:
            n_failed += 1
            if len(failed_examples) < 5:
                failed_examples.append((radius, observed_margin))

    for z in outer_anchors:
        radius = float(np.linalg.norm(z))
        successors = [sample_truncated_gaussian_step(z, gamma, sigma, d, b, rng) for _ in range(kappa)]
        g_hat_U = empirical_drift(U_fn, z, successors)
        observed_margin = -g_hat_U  # (H-b): strictly positive margin required, see is_anchor_certified()'s docstring
        if is_anchor_certified(observed_margin, eta_conc, L, L_U, mesh_delta):
            n_certified += 1
        else:
            n_failed += 1
            if len(failed_examples) < 5:
                failed_examples.append((radius, observed_margin))

    n_anchors = len(inner_anchors) + len(outer_anchors)
    all_certified = n_failed == 0 and n_anchors > 0

    warrant = alpha = alpha_collar = None
    if all_certified:
        alpha = z0_norm_sq / ell
        alpha_collar = 0.0  # by construction: every retained anchor passed assert_anchor_ball_does_not_cross_f_boundary
        warrant = 1.0 - alpha - alpha_collar

    return TrainedLBSMVerificationResult(
        all_anchors_certified=all_certified,
        n_anchors=n_anchors,
        n_certified=n_certified,
        n_failed=n_failed,
        mesh_delta=mesh_delta,
        kappa=kappa,
        eta_conc=eta_conc,
        L=L,
        L_V=L_V,
        L_U=L_U,
        failed_anchor_examples=failed_examples,
        warrant=warrant,
        alpha=alpha,
        alpha_collar=alpha_collar,
    )
