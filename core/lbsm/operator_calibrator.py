"""Independent one-sided calibration for learned post-expectation maps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .operator_model import (
    AnchorSuccessorBatch,
    PostExpectationTrainingResult,
    anchor_fingerprint,
    assert_structural_certificate_bound,
    certificate_successor_means,
    evaluate_torch_scalar_fn,
)


def split_conformal_upper(scores: Sequence[float], alpha: float) -> float:
    """Finite-sample one-sided split-conformal upper quantile.

    Uses rank ceil((n+1)*(1-alpha)).  If that rank is n+1, finite calibration
    data cannot support the requested coverage and the safe result is +inf.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1")
    rank = math.ceil((values.size + 1) * (1.0 - alpha))
    if rank > values.size:
        return math.inf
    return float(np.partition(values, rank - 1)[rank - 1])


def one_sided_hoeffding_radius(bound: float, kappa: int, delta: float) -> float:
    """P(E[Y]-mean(Y_1..Y_k) > eta) <= delta for Y in [0,bound]."""
    if bound <= 0 or kappa <= 0:
        raise ValueError("bound and kappa must be positive")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie strictly between 0 and 1")
    return float(bound * math.sqrt(math.log(1.0 / delta) / (2.0 * kappa)))


@dataclass(frozen=True)
class PostExpectationCalibration:
    q_W: float
    q_U: float
    eta_W: float
    eta_U: float
    alpha_W: float
    alpha_U: float
    delta_mc_W: float
    delta_mc_U: float
    n_calibration: int
    kappa: int
    sampling_scope: str
    calibration_anchor_fingerprints: frozenset[str]

    @property
    def upper_correction_W(self) -> float:
        return self.q_W + self.eta_W

    @property
    def upper_correction_U(self) -> float:
        return self.q_U + self.eta_U

    @property
    def joint_operator_error_budget(self) -> float:
        # Union bound for the two independently calibrated scalar operators.
        return min(
            1.0,
            self.alpha_W + self.delta_mc_W + self.alpha_U + self.delta_mc_U,
        )


def calibrate_post_expectation_models(
    *,
    training: PostExpectationTrainingResult,
    W: Any,
    U: Any,
    calibration_batches: Sequence[AnchorSuccessorBatch],
    alpha_W: float = 0.05,
    alpha_U: float = 0.05,
    delta_mc_W: float = 0.01,
    delta_mc_U: float = 0.01,
) -> PostExpectationCalibration:
    """Calibrate upper residuals on anchors unseen by operator training."""
    if not calibration_batches:
        raise ValueError("calibration_batches must not be empty")
    batches = [batch.validated() for batch in calibration_batches]
    scopes = {batch.sampling_scope for batch in batches}
    if len(scopes) != 1 or None in scopes or "" in scopes:
        raise ValueError(
            "residual calibration batches must declare one non-empty sampling_scope"
        )
    sampling_scope = next(iter(scopes))
    assert_structural_certificate_bound(W, training.B_W, "W")
    assert_structural_certificate_bound(U, training.B_U, "U")
    fingerprints = frozenset(anchor_fingerprint(batch) for batch in batches)
    certificate_overlap = training.certificate_train_anchor_fingerprints & fingerprints
    operator_overlap = training.train_anchor_fingerprints & fingerprints
    if certificate_overlap or operator_overlap:
        raise ValueError(
            "residual_calibration overlaps an earlier split "
            f"(certificate_train={len(certificate_overlap)}, "
            f"operator_train={len(operator_overlap)}); "
            "distributional L3 requires independent splits"
        )
    kappas = {batch.successors.shape[0] for batch in batches}
    if len(kappas) != 1:
        raise ValueError("all residual-calibration batches must use the same kappa")
    kappa = kappas.pop()

    anchors = np.stack([batch.anchor for batch in batches])
    empirical_W = certificate_successor_means(
        W, batches, bound=training.B_W, certificate_name="W"
    )
    empirical_U = certificate_successor_means(
        U, batches, bound=training.B_U, certificate_name="U"
    )
    predicted_W = evaluate_torch_scalar_fn(training.H_W, anchors)
    predicted_U = evaluate_torch_scalar_fn(training.H_U, anchors)

    q_W = split_conformal_upper(empirical_W - predicted_W, alpha_W)
    q_U = split_conformal_upper(empirical_U - predicted_U, alpha_U)
    eta_W = one_sided_hoeffding_radius(training.B_W, kappa, delta_mc_W)
    eta_U = one_sided_hoeffding_radius(training.B_U, kappa, delta_mc_U)
    return PostExpectationCalibration(
        q_W=q_W,
        q_U=q_U,
        eta_W=eta_W,
        eta_U=eta_U,
        alpha_W=alpha_W,
        alpha_U=alpha_U,
        delta_mc_W=delta_mc_W,
        delta_mc_U=delta_mc_U,
        n_calibration=len(batches),
        kappa=kappa,
        sampling_scope=sampling_scope,
        calibration_anchor_fingerprints=fingerprints,
    )
