"""Distribution-scoped L3 drift validation.

This is intentionally separate from ``core.lbsm.verifier``.  It does not
establish an everywhere supermartingale premise and therefore does not invoke
Ville's inequality or emit the paper's global Theorem-5.6 warrant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from core.lppm.calibrator import _clopper_pearson_lower

from .operator_calibrator import PostExpectationCalibration
from .operator_model import (
    AnchorSuccessorBatch,
    PostExpectationTrainingResult,
    anchor_fingerprint,
    assert_structural_certificate_bound,
    evaluate_torch_scalar_fn,
)


L3_DISTRIBUTIONAL_WARRANT = "L3_DISTRIBUTIONAL_WARRANT"
INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class DistributionalL3Result:
    verdict: str
    n_warrant: int
    n_surrogate_conditions_met: int
    empirical_condition_rate: float
    cp_condition_mass_lower: float
    joint_operator_error_budget: float
    drift_valid_mass_lower: float
    warrant_threshold: float
    confidence: float
    eps_W: float
    eps_U: float
    failed_anchor_ids: tuple[str, ...]

    def is_warranted(self) -> bool:
        return self.verdict == L3_DISTRIBUTIONAL_WARRANT

    def summary(self) -> str:
        label = "DISTRIBUTIONAL WARRANT" if self.is_warranted() else "INCONCLUSIVE"
        return (
            f"[L3 distributional -- NOT Theorem-5.6 global L3] {label} | "
            f"drift_mass_lower={self.drift_valid_mass_lower:.3f} "
            f"surrogate_CP_lower={self.cp_condition_mass_lower:.3f} "
            f"operator_error_budget={self.joint_operator_error_budget:.3f} "
            f"n={self.n_warrant} confidence={self.confidence:.3f}"
        )


def verify_distributional_l3(
    *,
    training: PostExpectationTrainingResult,
    calibration: PostExpectationCalibration,
    W: Any,
    U: Any,
    warrant_batches: Sequence[AnchorSuccessorBatch],
    is_accepting: Callable[[np.ndarray], bool],
    in_retention_set: Callable[[np.ndarray], bool],
    eps_W: float,
    eps_U: float,
    gamma: float = 0.05,
    warrant_threshold: float = 0.80,
) -> DistributionalL3Result:
    """Lower-bound drift-condition mass under the warrant-anchor distribution.

    Four logical data roles must remain disjoint in a real pipeline: W/U
    certificate training, operator training, residual calibration, and this
    warrant split.  The first is owned by the certificate trainer; this
    function mechanically enforces the latter three by anchor content.

    ``drift_valid_mass_lower`` is a marginal, distribution-scoped lower bound.
    It is *not* a pathwise recurrence probability and must not be inserted into
    Ville's inequality.
    """
    if not warrant_batches:
        raise ValueError("warrant_batches must not be empty")
    if eps_W <= 0 or eps_U <= 0:
        raise ValueError("eps_W and eps_U must be strictly positive")
    if not (0.0 < gamma < 1.0):
        raise ValueError("gamma must lie strictly between 0 and 1")
    if not (0.0 <= warrant_threshold <= 1.0):
        raise ValueError("warrant_threshold must lie in [0,1]")

    batches = [batch.validated() for batch in warrant_batches]
    scopes = {batch.sampling_scope for batch in batches}
    if scopes != {calibration.sampling_scope}:
        raise ValueError(
            "warrant batches must use the same sampling_scope as residual calibration "
            f"({calibration.sampling_scope!r}); got {sorted(repr(scope) for scope in scopes)}"
        )
    assert_structural_certificate_bound(W, training.B_W, "W")
    assert_structural_certificate_bound(U, training.B_U, "U")
    fingerprints = frozenset(anchor_fingerprint(batch) for batch in batches)
    certificate_overlap = training.certificate_train_anchor_fingerprints & fingerprints
    train_overlap = training.train_anchor_fingerprints & fingerprints
    calib_overlap = calibration.calibration_anchor_fingerprints & fingerprints
    if certificate_overlap or train_overlap or calib_overlap:
        raise ValueError(
            "warrant split overlaps an earlier split "
            f"(certificate_train={len(certificate_overlap)}, "
            f"operator_train={len(train_overlap)}, residual_calibration={len(calib_overlap)})"
        )

    anchors = np.stack([batch.anchor for batch in batches])
    current_W = evaluate_torch_scalar_fn(W, anchors)
    current_U = evaluate_torch_scalar_fn(U, anchors)
    if np.any(current_W < -1e-6) or np.any(current_W > training.B_W + 1e-6):
        raise ValueError("W left its declared bounded range on the warrant split")
    if np.any(current_U < -1e-6) or np.any(current_U > training.B_U + 1e-6):
        raise ValueError("U left its declared bounded range on the warrant split")
    upper_next_W = (
        evaluate_torch_scalar_fn(training.H_W, anchors) + calibration.upper_correction_W
    )
    upper_next_U = (
        evaluate_torch_scalar_fn(training.H_U, anchors) + calibration.upper_correction_U
    )

    conditions_met = []
    failed_ids = []
    for i, batch in enumerate(batches):
        u_ok = bool(in_retention_set(batch.anchor)) or (
            upper_next_U[i] - current_U[i] <= -eps_U
        )
        w_ok = bool(is_accepting(batch.anchor)) or (
            upper_next_W[i] - current_W[i] <= -eps_W
        )
        ok = bool(u_ok and w_ok)
        conditions_met.append(ok)
        if not ok and len(failed_ids) < 10:
            failed_ids.append(batch.sample_id or anchor_fingerprint(batch))

    n = len(conditions_met)
    k = sum(conditions_met)
    cp_lower = _clopper_pearson_lower(k, n, gamma)
    operator_budget = calibration.joint_operator_error_budget
    drift_lower = max(0.0, cp_lower - operator_budget)
    verdict = (
        L3_DISTRIBUTIONAL_WARRANT
        if drift_lower >= warrant_threshold
        else INCONCLUSIVE
    )
    return DistributionalL3Result(
        verdict=verdict,
        n_warrant=n,
        n_surrogate_conditions_met=k,
        empirical_condition_rate=k / n,
        cp_condition_mass_lower=cp_lower,
        joint_operator_error_budget=operator_budget,
        drift_valid_mass_lower=drift_lower,
        warrant_threshold=warrant_threshold,
        confidence=1.0 - gamma,
        eps_W=eps_W,
        eps_U=eps_U,
        failed_anchor_ids=tuple(failed_ids),
    )
