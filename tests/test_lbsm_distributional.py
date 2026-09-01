"""Tests for the learned post-expectation, distribution-scoped L3 path."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from core.lbsm.distributional_verifier import (
    INCONCLUSIVE,
    L3_DISTRIBUTIONAL_WARRANT,
    verify_distributional_l3,
)
from core.lbsm.distributional_pipeline import run_distributional_l3_pipeline
from core.lbsm.operator_calibrator import (
    calibrate_post_expectation_models,
    split_conformal_upper,
)
from core.lbsm.operator_model import (
    AnchorSuccessorBatch,
    PostExpectationTrainingResult,
    anchor_fingerprint,
    fit_post_expectation_models,
)


class SquareCertificate(torch.nn.Module):
    B = 1.0

    def forward(self, x):
        return x[:, 0] ** 2


class QuarterSquareOperator(torch.nn.Module):
    def forward(self, x):
        return 0.25 * x[:, 0] ** 2


def _deterministic_batches(start: int, count: int, *, kappa: int = 5000):
    batches = []
    # All anchors lie in [0.8, 1.0], but exact values are unique across splits.
    for offset in range(count):
        index = start + offset
        x = 0.8 + index * 0.0005
        anchor = np.asarray([x], dtype=np.float32)
        successors = np.full((kappa, 1), 0.5 * x, dtype=np.float32)
        batches.append(AnchorSuccessorBatch(
            anchor, successors, sample_id=f"x-{index}", sampling_scope="test-contracting",
        ))
    return batches


def _exact_training(train_batches):
    return PostExpectationTrainingResult(
        H_W=QuarterSquareOperator(),
        H_U=QuarterSquareOperator(),
        loss_history=(0.0,),
        n_train_anchors=len(train_batches),
        kappa_min=min(len(batch.successors) for batch in train_batches),
        certificate_train_anchor_fingerprints=frozenset(),
        train_anchor_fingerprints=frozenset(
            anchor_fingerprint(batch) for batch in train_batches
        ),
        B_W=1.0,
        B_U=1.0,
    )


def test_split_conformal_uses_finite_sample_corrected_rank():
    scores = list(range(20))
    # ceil(21 * .9) = 19: one-indexed rank 19, value 18.
    assert split_conformal_upper(scores, alpha=0.1) == 18.0
    # With only ten examples, 99% marginal coverage is unsupported.
    assert math.isinf(split_conformal_upper(scores[:10], alpha=0.01))


def test_operator_training_produces_bounded_scalar_models():
    W = SquareCertificate()
    batches = _deterministic_batches(0, 8, kappa=4)
    result = fit_post_expectation_models(
        W=W,
        U=W,
        certificate_train_batches=_deterministic_batches(50, 4, kappa=4),
        train_batches=batches,
        seed=7,
        B_W=1.0,
        B_U=1.0,
        hidden_dim=8,
        n_epochs=20,
    )
    probe = torch.tensor([[0.1], [0.9], [5.0]], dtype=torch.float32)
    with torch.no_grad():
        out_W = result.H_W(probe)
        out_U = result.H_U(probe)
    assert out_W.shape == (3,) and out_U.shape == (3,)
    assert torch.all((0.0 <= out_W) & (out_W <= 1.0))
    assert torch.all((0.0 <= out_U) & (out_U <= 1.0))
    assert len(result.loss_history) == 20


def test_end_to_end_pipeline_runs_with_four_disjoint_splits():
    certificate_train = _deterministic_batches(0, 6, kappa=4)
    operator_train = _deterministic_batches(20, 6, kappa=4)
    residual = _deterministic_batches(40, 20, kappa=4)
    warrant = _deterministic_batches(80, 12, kappa=4)
    result = run_distributional_l3_pipeline(
        certificate_train_batches=certificate_train,
        operator_train_batches=operator_train,
        residual_calibration_batches=residual,
        warrant_batches=warrant,
        is_accepting=lambda x: bool(x[0] <= 0.81),
        in_retention_set=lambda _x: False,
        core_inside=np.asarray([[0.80], [0.805]], dtype=np.float32),
        core_boundary=np.asarray([[0.95], [0.97]], dtype=np.float32),
        core_inside_upper=0.4,
        core_boundary_lower=0.6,
        seed=3,
        B_W=1.0,
        B_U=1.0,
        eps_W=0.01,
        eps_U=0.01,
        certificate_hidden_dim=8,
        certificate_epochs=3,
        operator_hidden_dim=8,
        operator_epochs=3,
    )
    assert result.certificates.n_train_anchors == 6
    assert result.operators.n_train_anchors == 6
    assert result.calibration.n_calibration == 20
    assert result.verification.n_warrant == 12
    assert result.verification.verdict in {L3_DISTRIBUTIONAL_WARRANT, INCONCLUSIVE}


def test_residual_calibration_rejects_operator_training_anchor_leakage():
    W = SquareCertificate()
    train = _deterministic_batches(0, 20)
    training = _exact_training(train)
    with pytest.raises(ValueError, match="independent splits"):
        calibrate_post_expectation_models(
            training=training,
            W=W,
            U=W,
            calibration_batches=[train[3]],
        )


def test_warrant_split_rejects_residual_calibration_leakage():
    W = SquareCertificate()
    train = _deterministic_batches(0, 20)
    residual = _deterministic_batches(30, 20)
    training = _exact_training(train)
    calibration = calibrate_post_expectation_models(
        training=training,
        W=W,
        U=W,
        calibration_batches=residual,
    )
    with pytest.raises(ValueError, match="warrant split overlaps"):
        verify_distributional_l3(
            training=training,
            calibration=calibration,
            W=W,
            U=W,
            warrant_batches=[residual[0]],
            is_accepting=lambda _x: False,
            in_retention_set=lambda _x: False,
            eps_W=0.2,
            eps_U=0.2,
        )


def test_contracting_system_gets_distributional_not_global_warrant():
    W = SquareCertificate()
    train = _deterministic_batches(0, 20)
    residual = _deterministic_batches(30, 20)
    warrant = _deterministic_batches(100, 100)
    training = _exact_training(train)
    calibration = calibrate_post_expectation_models(
        training=training,
        W=W,
        U=W,
        calibration_batches=residual,
        alpha_W=0.05,
        alpha_U=0.05,
        delta_mc_W=0.01,
        delta_mc_U=0.01,
    )
    result = verify_distributional_l3(
        training=training,
        calibration=calibration,
        W=W,
        U=W,
        warrant_batches=warrant,
        is_accepting=lambda _x: False,
        in_retention_set=lambda _x: False,
        eps_W=0.2,
        eps_U=0.2,
        gamma=0.05,
        warrant_threshold=0.80,
    )
    assert result.verdict == L3_DISTRIBUTIONAL_WARRANT
    assert result.is_warranted()
    assert result.n_surrogate_conditions_met == 100
    assert result.drift_valid_mass_lower >= 0.80
    summary = result.summary()
    assert "NOT Theorem-5.6 global L3" in summary
    assert "Ville" not in summary


def test_failed_drift_conditions_are_inconclusive_not_unsafe_or_global_safe():
    W = SquareCertificate()
    train = _deterministic_batches(0, 20)
    residual = _deterministic_batches(30, 20)
    warrant = _deterministic_batches(100, 30)
    training = _exact_training(train)
    calibration = calibrate_post_expectation_models(
        training=training, W=W, U=W, calibration_batches=residual,
    )
    result = verify_distributional_l3(
        training=training,
        calibration=calibration,
        W=W,
        U=W,
        warrant_batches=warrant,
        is_accepting=lambda _x: False,
        in_retention_set=lambda _x: False,
        eps_W=0.9,
        eps_U=0.9,
    )
    assert result.verdict == INCONCLUSIVE
    assert not result.is_warranted()
