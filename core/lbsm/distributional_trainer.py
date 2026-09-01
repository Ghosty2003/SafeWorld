"""Model-agnostic W/U training for the distribution-scoped L3 path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .operator_model import AnchorSuccessorBatch, anchor_fingerprint
from .trainer import LBSMNet

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class DistributionalCertificateTrainingResult:
    W: Any
    U: Any
    loss_history: tuple[float, ...]
    n_train_anchors: int
    train_anchor_fingerprints: frozenset[str]
    B_W: float
    B_U: float


def fit_distributional_certificates(
    *,
    train_batches: Sequence[AnchorSuccessorBatch],
    is_accepting: Callable[[np.ndarray], bool],
    in_retention_set: Callable[[np.ndarray], bool],
    core_inside: np.ndarray,
    core_boundary: np.ndarray,
    core_inside_upper: float,
    core_boundary_lower: float,
    seed: int,
    B_W: float,
    B_U: float,
    hidden_dim: int = 64,
    n_epochs: int = 500,
    lr: float = 1e-3,
    eps_W: float = 0.02,
    eps_U: float = 0.02,
    lambda_U: float = 1.0,
    lambda_core: float = 1.0,
    lambda_reg: float = 1e-4,
    max_successors_per_anchor: int | None = 16,
) -> DistributionalCertificateTrainingResult:
    """Train bounded W/U from arbitrary resettable-generative batches.

    A state vector may be a latent ``z`` or a caller-defined product-state
    encoding ``(z,q)``.  The core never interprets its coordinates.

    ``core_inside`` and ``core_boundary`` are mandatory because U=0 satisfies
    a non-increase loss trivially.  Their upper/lower targets shape a genuine
    operative sublevel set; omitting that obligation would turn recurrence
    training into a degenerate numerical exercise.
    """
    if torch is None or LBSMNet is None:
        raise RuntimeError("fit_distributional_certificates() requires torch")
    if not train_batches:
        raise ValueError("train_batches must not be empty")
    if n_epochs <= 0 or eps_W <= 0 or eps_U <= 0:
        raise ValueError("n_epochs and drift margins must be positive")
    if not (0.0 <= core_inside_upper < core_boundary_lower <= B_U):
        raise ValueError(
            "core targets must satisfy 0 <= inside_upper < boundary_lower <= B_U"
        )

    batches = [batch.validated() for batch in train_batches]
    dims = {batch.anchor.shape[0] for batch in batches}
    if len(dims) != 1:
        raise ValueError("all product-state vectors must have the same dimension")
    state_dim = dims.pop()
    inside = np.asarray(core_inside, dtype=np.float32)
    boundary = np.asarray(core_boundary, dtype=np.float32)
    if inside.ndim != 2 or boundary.ndim != 2:
        raise ValueError("core_inside/core_boundary must be 2-D arrays")
    if inside.shape[0] == 0 or boundary.shape[0] == 0:
        raise ValueError("core_inside/core_boundary must both be non-empty")
    if inside.shape[1] != state_dim or boundary.shape[1] != state_dim:
        raise ValueError("core points must use the same state encoding as anchors")

    torch.manual_seed(seed)
    W = LBSMNet(latent_dim=state_dim, hidden_dim=hidden_dim, B=B_W)
    U = LBSMNet(latent_dim=state_dim, hidden_dim=hidden_dim, B=B_U)
    optimizer = torch.optim.Adam(list(W.parameters()) + list(U.parameters()), lr=lr)

    anchors = torch.as_tensor(np.stack([batch.anchor for batch in batches]), dtype=torch.float32)
    kappas = {batch.successors.shape[0] for batch in batches}
    if len(kappas) != 1:
        raise ValueError("all certificate-training batches must use the same kappa")
    kappa = kappas.pop()
    if max_successors_per_anchor is not None:
        if max_successors_per_anchor <= 0:
            raise ValueError("max_successors_per_anchor must be positive or None")
        kappa = min(kappa, max_successors_per_anchor)
    successors = torch.as_tensor(
        np.stack([batch.successors[:kappa] for batch in batches]), dtype=torch.float32
    )
    accepting = torch.as_tensor(
        [bool(is_accepting(batch.anchor)) for batch in batches], dtype=torch.bool
    )
    retained = torch.as_tensor(
        [bool(in_retention_set(batch.anchor)) for batch in batches], dtype=torch.bool
    )
    inside_t = torch.as_tensor(inside, dtype=torch.float32)
    boundary_t = torch.as_tensor(boundary, dtype=torch.float32)

    history: list[float] = []
    for _ in range(n_epochs):
        optimizer.zero_grad()
        W_current = W(anchors)
        U_current = U(anchors)
        flat_successors = successors.reshape(-1, state_dim)
        W_next = W(flat_successors).reshape(len(batches), kappa).mean(dim=1)
        U_next = U(flat_successors).reshape(len(batches), kappa).mean(dim=1)
        drift_W = W_next - W_current
        drift_U = U_next - U_current

        non_accepting = (~accepting).float()
        loss_W = (non_accepting * torch.clamp(drift_W + eps_W, min=0.0)).mean()
        outside_retention = (~retained).float()
        loss_U = lambda_U * (
            outside_retention * torch.clamp(drift_U + eps_U, min=0.0)
        ).mean()
        loss_core = (
            torch.clamp(U(inside_t) - core_inside_upper, min=0.0).mean()
            + torch.clamp(core_boundary_lower - U(boundary_t), min=0.0).mean()
        )
        regularizer = sum((parameter ** 2).mean() for parameter in [*W.parameters(), *U.parameters()])
        loss = loss_W + loss_U + lambda_core * loss_core + lambda_reg * regularizer
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))

    return DistributionalCertificateTrainingResult(
        W=W,
        U=U,
        loss_history=tuple(history),
        n_train_anchors=len(batches),
        train_anchor_fingerprints=frozenset(anchor_fingerprint(batch) for batch in batches),
        B_W=float(B_W),
        B_U=float(B_U),
    )
