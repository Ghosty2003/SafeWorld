"""
core/lbsm/drift.py

Two distinct things, not to be conflated (see the module-level note in
core/lbsm/verifier.py for which one the certified warrant actually uses):

1. empirical_drift() -- Eq. 7 (Section 4.4, "Learning candidate certificates"),
   the GENERAL finite-kappa Monte-Carlo drift estimator. Works for any
   certificate function and any resettable-generative kernel, sampled or
   trained. This is what Stage 2/3 will use, since a trained V/U has no
   closed form.

2. exact_drift_e2() -- Appendix E.2's EXACT closed-form one-step drift for
   the specific mean-reverting linear kernel z' = gamma*z + xi with U(z) =
   ||z||^2. No sampling, no concentration correction. The paper's own
   reported warrant number (0.92) is computed this way -- Appendix E.2
   states the drift is an "exact equality" and the construction "quotes no
   numeric query count" precisely because no covering-net sampling occurs.

core/lbsm/verifier.py uses (2) as the CERTIFIED quantity for the Stage-1
analytic construction (this is what reproduces the paper's 0.92 to floating-
point precision). (1) is exercised in tests/test_lbsm_analytic.py only as an
optional Monte-Carlo cross-check that the closed form is not a transcription
error -- it is never the certified value.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def sample_truncated_gaussian_step(
    z: np.ndarray,
    gamma: float,
    sigma: float,
    d: int,
    b: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Appendix E.2's dynamics: z' = gamma*z + xi, xi ~ N(0, sigma^2 I_d)
    conditioned on ||xi|| <= b (rejection sampling -- for k=3, d=2 the
    rejected mass is Pr[chi^2_2 > 2k^2] = e^-9 ~= 1.2e-4, so this converges
    essentially immediately in the parameter regime this module targets).
    """
    while True:
        xi = rng.normal(0.0, sigma, size=d)
        if float(np.dot(xi, xi)) <= b ** 2:
            return gamma * z + xi


def empirical_drift(
    certificate_fn: Callable[[np.ndarray], float],
    anchor: np.ndarray,
    successors: Sequence[np.ndarray],
) -> float:
    """
    Eq. 7: g_hat(x_i) = (1/kappa) * sum_j phi(x_i^(j)) - phi(x_i).

    `successors` are the kappa conditionally-independent draws from the
    resettable-generative oracle at `anchor` (Definition 4.1).
    """
    if not successors:
        raise ValueError("empirical_drift() requires at least one successor sample")
    mean_next = sum(certificate_fn(s) for s in successors) / len(successors)
    return mean_next - certificate_fn(anchor)


def z_star_sq(d: int, sigma: float, gamma: float) -> float:
    """
    The paper's CONSERVATIVE drift floor z_star^2 := d*sigma^2 / (1-gamma^2),
    an upper bound on the exact d*sigma_b^2/(1-gamma^2) (Appendix E.2:
    "conditioning a nonnegative variable on being small can only lower its
    mean", so sigma_b^2 <= sigma^2 always). Using the upper bound
    UNDERESTIMATES the true drift margin -- conservative, per the paper's own
    remark, never optimistic.
    """
    return d * sigma ** 2 / (1 - gamma ** 2)


def exact_drift_e2(z: np.ndarray, gamma: float, z_star_sq_: float) -> float:
    """
    Appendix E.2's exact one-step drift equality for U(z) = W(z) = ||z||^2
    under z' = gamma*z + xi (truncated-Gaussian xi):

        E[U(z') | z] - U(z) = -(1 - gamma^2) * (||z||^2 - z_star^2)

    Exact -- no sampling, no eta_conc, no Lipschitz-covering Delta term (see
    core/lbsm/verifier.py for why Theorem D.3's eps_eff = eps - eta_conc -
    L_phi*Delta correction is NOT applied here: it corrects for sampling a
    covering net of anchors, and this construction never builds one).
    """
    z_norm_sq = float(np.dot(z, z))
    return -(1 - gamma ** 2) * (z_norm_sq - z_star_sq_)
