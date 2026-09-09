"""Learned post-expectation operators for distribution-scoped L3 checks.

This module deliberately does *not* approximate the next latent state.  For a
fixed pair of LBSM certificates W/U it learns the two scalar maps

    H_W(x) = E[W(X') | X=x],   H_U(x) = E[U(X') | X=x].

That removes the real world model's global Lipschitz constant from the
validation path.  The learned maps are candidates, not proofs: independent
one-sided calibration is implemented in ``operator_calibrator.py``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass(frozen=True)
class AnchorSuccessorBatch:
    """One resettable-generative query location and its i.i.d. successors."""

    anchor: np.ndarray
    successors: np.ndarray
    sample_id: str | None = None
    sampling_scope: str | None = None

    def validated(self) -> "AnchorSuccessorBatch":
        anchor = np.asarray(self.anchor, dtype=np.float32)
        successors = np.asarray(self.successors, dtype=np.float32)
        if anchor.ndim != 1:
            raise ValueError("anchor must have shape (latent_dim,)")
        if successors.ndim != 2 or successors.shape[1] != anchor.shape[0]:
            raise ValueError("successors must have shape (kappa, latent_dim)")
        if successors.shape[0] == 0:
            raise ValueError("each anchor needs at least one successor")
        if not np.isfinite(anchor).all() or not np.isfinite(successors).all():
            raise ValueError("anchor/successors must contain only finite values")
        return AnchorSuccessorBatch(anchor, successors, self.sample_id, self.sampling_scope)


def anchor_fingerprint(batch: AnchorSuccessorBatch) -> str:
    """Content identity used to prevent train/calibration/warrant leakage."""
    anchor = np.ascontiguousarray(batch.validated().anchor)
    digest = hashlib.sha256()
    digest.update(str(anchor.shape).encode("ascii"))
    digest.update(anchor.tobytes())
    return digest.hexdigest()


def assert_anchor_splits_disjoint(
    left: Sequence[AnchorSuccessorBatch],
    right: Sequence[AnchorSuccessorBatch],
    *,
    left_name: str,
    right_name: str,
) -> None:
    overlap = {anchor_fingerprint(x) for x in left} & {anchor_fingerprint(x) for x in right}
    if overlap:
        raise ValueError(
            f"{left_name} and {right_name} share {len(overlap)} anchor(s); "
            "distributional L3 requires independent splits"
        )


PostExpectationNet = None

if nn is not None:

    class PostExpectationNet(nn.Module):
        """Small bounded scalar regressor for E[certificate(X') | X=x]."""

        def __init__(self, latent_dim: int, hidden_dim: int, bound: float):
            super().__init__()
            if bound <= 0:
                raise ValueError("bound must be positive")
            self.bound = float(bound)
            self.mlp = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.bound * torch.sigmoid(self.mlp(x).squeeze(-1))


@dataclass(frozen=True)
class PostExpectationTrainingResult:
    H_W: Any
    H_U: Any
    loss_history: tuple[float, ...]
    n_train_anchors: int
    kappa_min: int
    certificate_train_anchor_fingerprints: frozenset[str]
    train_anchor_fingerprints: frozenset[str]
    B_W: float
    B_U: float


def evaluate_torch_scalar_fn(fn: Any, values: np.ndarray) -> np.ndarray:
    """Evaluate a torch certificate/operator and return a flat float64 array."""
    if torch is None:
        raise RuntimeError("distributional L3 requires torch")
    tensor = torch.as_tensor(np.asarray(values), dtype=torch.float32)
    with torch.no_grad():
        output = fn(tensor)
    result = np.asarray(output.detach().cpu(), dtype=np.float64).reshape(-1)
    if result.shape[0] != tensor.shape[0]:
        raise ValueError("certificate/operator must return one scalar per input row")
    if not np.isfinite(result).all():
        raise ValueError("certificate/operator returned non-finite values")
    return result


def assert_structural_certificate_bound(certificate: Any, bound: float, name: str) -> None:
    """Require an auditable architectural bound, not sampled range evidence."""
    declared = getattr(certificate, "B", None)
    if declared is None:
        raise ValueError(
            f"{name} must expose a structural .B bound; observing bounded outputs "
            "on sampled points is insufficient for Hoeffding calibration"
        )
    if not math.isclose(float(declared), float(bound), rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError(f"{name}.B={declared} does not match declared bound {bound}")


def certificate_successor_means(
    certificate: Any,
    batches: Sequence[AnchorSuccessorBatch],
    *,
    bound: float | None = None,
    certificate_name: str = "certificate",
) -> np.ndarray:
    """Monte-Carlo targets used to learn a post-expectation operator."""
    targets = []
    for raw_batch in batches:
        batch = raw_batch.validated()
        values = evaluate_torch_scalar_fn(certificate, batch.successors)
        if bound is not None and (np.any(values < -1e-6) or np.any(values > bound + 1e-6)):
            raise ValueError(
                f"{certificate_name} left its declared [0, {bound}] range; "
                "the Hoeffding residual bound would be invalid"
            )
        targets.append(float(values.mean()))
    return np.asarray(targets, dtype=np.float32)


def fit_post_expectation_models(
    *,
    W: Any,
    U: Any,
    certificate_train_batches: Sequence[AnchorSuccessorBatch],
    train_batches: Sequence[AnchorSuccessorBatch],
    seed: int,
    B_W: float,
    B_U: float,
    hidden_dim: int = 64,
    n_epochs: int = 300,
    lr: float = 1e-3,
) -> PostExpectationTrainingResult:
    """Fit H_W/H_U on one split; calibration must use different anchors."""
    if torch is None or PostExpectationNet is None:
        raise RuntimeError("fit_post_expectation_models() requires torch")
    if not train_batches:
        raise ValueError("train_batches must not be empty")
    if not certificate_train_batches:
        raise ValueError(
            "certificate_train_batches must be supplied so certificate and operator "
            "training leakage can be rejected"
        )
    if n_epochs <= 0:
        raise ValueError("n_epochs must be positive")
    assert_structural_certificate_bound(W, B_W, "W")
    assert_structural_certificate_bound(U, B_U, "U")

    batches = [batch.validated() for batch in train_batches]
    certificate_batches = [batch.validated() for batch in certificate_train_batches]
    assert_anchor_splits_disjoint(
        certificate_batches,
        batches,
        left_name="certificate_train",
        right_name="operator_train",
    )
    latent_dims = {batch.anchor.shape[0] for batch in batches}
    if len(latent_dims) != 1:
        raise ValueError("all anchors must have the same latent dimension")
    latent_dim = latent_dims.pop()

    torch.manual_seed(seed)
    H_W = PostExpectationNet(latent_dim, hidden_dim, B_W)
    H_U = PostExpectationNet(latent_dim, hidden_dim, B_U)
    optimizer = torch.optim.Adam(list(H_W.parameters()) + list(H_U.parameters()), lr=lr)

    anchors = np.stack([batch.anchor for batch in batches])
    anchors_t = torch.as_tensor(anchors, dtype=torch.float32)
    target_W = torch.as_tensor(
        certificate_successor_means(W, batches, bound=B_W, certificate_name="W"),
        dtype=torch.float32,
    )
    target_U = torch.as_tensor(
        certificate_successor_means(U, batches, bound=B_U, certificate_name="U"),
        dtype=torch.float32,
    )

    history: list[float] = []
    for _ in range(n_epochs):
        optimizer.zero_grad()
        loss_W = torch.mean((H_W(anchors_t) - target_W) ** 2)
        loss_U = torch.mean((H_U(anchors_t) - target_U) ** 2)
        loss = loss_W + loss_U
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))

    return PostExpectationTrainingResult(
        H_W=H_W,
        H_U=H_U,
        loss_history=tuple(history),
        n_train_anchors=len(batches),
        kappa_min=min(batch.successors.shape[0] for batch in batches),
        certificate_train_anchor_fingerprints=frozenset(
            anchor_fingerprint(batch) for batch in certificate_batches
        ),
        train_anchor_fingerprints=frozenset(anchor_fingerprint(batch) for batch in batches),
        B_W=float(B_W),
        B_U=float(B_U),
    )
