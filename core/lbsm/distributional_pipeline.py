"""End-to-end orchestration for distribution-scoped L3 validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .distributional_trainer import (
    DistributionalCertificateTrainingResult,
    fit_distributional_certificates,
)
from .distributional_verifier import DistributionalL3Result, verify_distributional_l3
from .operator_calibrator import PostExpectationCalibration, calibrate_post_expectation_models
from .operator_model import (
    AnchorSuccessorBatch,
    PostExpectationTrainingResult,
    fit_post_expectation_models,
)


@dataclass(frozen=True)
class DistributionalL3PipelineResult:
    certificates: DistributionalCertificateTrainingResult
    operators: PostExpectationTrainingResult
    calibration: PostExpectationCalibration
    verification: DistributionalL3Result


def run_distributional_l3_pipeline(
    *,
    certificate_train_batches: Sequence[AnchorSuccessorBatch],
    operator_train_batches: Sequence[AnchorSuccessorBatch],
    residual_calibration_batches: Sequence[AnchorSuccessorBatch],
    warrant_batches: Sequence[AnchorSuccessorBatch],
    is_accepting: Callable[[np.ndarray], bool],
    in_retention_set: Callable[[np.ndarray], bool],
    core_inside: np.ndarray,
    core_boundary: np.ndarray,
    core_inside_upper: float,
    core_boundary_lower: float,
    seed: int,
    B_W: float,
    B_U: float,
    eps_W: float = 0.02,
    eps_U: float = 0.02,
    certificate_hidden_dim: int = 64,
    certificate_epochs: int = 500,
    certificate_successors_per_anchor: int = 16,
    operator_hidden_dim: int = 64,
    operator_epochs: int = 300,
    alpha_W: float = 0.05,
    alpha_U: float = 0.05,
    delta_mc_W: float = 0.01,
    delta_mc_U: float = 0.01,
    gamma: float = 0.05,
    warrant_threshold: float = 0.80,
) -> DistributionalL3PipelineResult:
    """Train, calibrate, and validate without a dynamics Lipschitz constant."""
    certificates = fit_distributional_certificates(
        train_batches=certificate_train_batches,
        is_accepting=is_accepting,
        in_retention_set=in_retention_set,
        core_inside=core_inside,
        core_boundary=core_boundary,
        core_inside_upper=core_inside_upper,
        core_boundary_lower=core_boundary_lower,
        seed=seed,
        B_W=B_W,
        B_U=B_U,
        hidden_dim=certificate_hidden_dim,
        n_epochs=certificate_epochs,
        max_successors_per_anchor=certificate_successors_per_anchor,
        eps_W=eps_W,
        eps_U=eps_U,
    )
    operators = fit_post_expectation_models(
        W=certificates.W,
        U=certificates.U,
        certificate_train_batches=certificate_train_batches,
        train_batches=operator_train_batches,
        seed=seed + 1,
        B_W=B_W,
        B_U=B_U,
        hidden_dim=operator_hidden_dim,
        n_epochs=operator_epochs,
    )
    calibration = calibrate_post_expectation_models(
        training=operators,
        W=certificates.W,
        U=certificates.U,
        calibration_batches=residual_calibration_batches,
        alpha_W=alpha_W,
        alpha_U=alpha_U,
        delta_mc_W=delta_mc_W,
        delta_mc_U=delta_mc_U,
    )
    verification = verify_distributional_l3(
        training=operators,
        calibration=calibration,
        W=certificates.W,
        U=certificates.U,
        warrant_batches=warrant_batches,
        is_accepting=is_accepting,
        in_retention_set=in_retention_set,
        eps_W=eps_W,
        eps_U=eps_U,
        gamma=gamma,
        warrant_threshold=warrant_threshold,
    )
    return DistributionalL3PipelineResult(
        certificates=certificates,
        operators=operators,
        calibration=calibration,
        verification=verification,
    )
