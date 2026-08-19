"""
core/lbsm/trainer.py

SAFEWORLD L3 -- training the LBSM return certificate W and companion
certificate U as small neural networks (Section 4.4 "Learning candidate
certificates", Eq. 7-8), on the SAME Stage-1 analytic kernel z' = gamma*z +
xi (Appendix E.2). Dynamics are not yet a trained world model -- that is
Stage 3 scope. This stage validates the TRAINING mechanics (does gradient
descent on Eq.8's loss actually produce usable certificates?), kept
deliberately separate from validating the theoretical pipeline (Stage 1,
exact/closed form) and from the sample-based Theorem D.3 verification of a
trained network (core/lbsm/verifier.py::verify_trained_recurrence_warrant,
Stage 2's other half).

Disjoint-split discipline (STRICTER than L2, per task correction)
--------------------------------------------------------------------
core/lppm/trainer.py::fit_lppm()'s disjointness check is opt-in (only runs
if calib_trajectories is passed at all -- see EXPERIMENT_CONFIG.md §8.7/
§8.12); once opted in, it now raises on overlap by default too (an
allow_overlap=True + overlap_reason escape hatch exists, forced and
auditable). This module's fit_lbsm() makes the check itself MANDATORY, with
no escape hatch at all: it always calls
core/lppm/verifier.py::find_trajectory_overlap() on (train_anchors,
val_anchors) and RAISES if any anchor is shared, no opt-out. Paper Appendix
C.2, verbatim: "The independent validation split is used to establish the
STATISTICAL INEQUALITIES needed by the final warrant" -- L3's validation
split is not a convenience; core/lbsm/verifier.py's Hoeffding/eps_eff
machinery is only sound if it runs on anchors the trained network never saw.
There is no reason for this to be more permissive than L2's already-known-
to-be-risky pattern (EXPERIMENT_CONFIG.md §8.7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.lppm.verifier import find_trajectory_overlap

from .drift import sample_truncated_gaussian_step

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


LBSMNet = None

if nn is not None:

    class LBSMNet(nn.Module):
        """
        Bounded-by-construction certificate network: B * sigmoid(MLP(z)).

        Architecturally bounded in [0, B] at INFERENCE time regardless of
        training -- not just a training-time constraint (the task spec
        explicitly warns against relying on that alone: "不要只是訓練時約束,
        推斷時可能超界"). sigmoid, not a clipped softplus: sigmoid is exactly
        bounded with a known, clean Lipschitz constant (0.25), which
        core/lbsm/lipschitz.py needs for the Theorem D.3 eps_eff correction;
        a clipped softplus's Lipschitz behavior at the clip boundary is
        messier to bound tightly.
        """

        def __init__(self, latent_dim: int, hidden_dim: int, B: float):
            super().__init__()
            self.B = B
            self.mlp = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, z: "torch.Tensor") -> "torch.Tensor":
            raw = self.mlp(z).squeeze(-1)
            return self.B * torch.sigmoid(raw)


@dataclass
class LBSMTrainingResult:
    W_net: Any
    U_net: Any
    loss_history: list[float]
    epochs_trained: int
    n_train_anchors: int
    n_val_anchors: int
    l_core_active: bool


def _random_point_with_norm_sq(rng: np.random.Generator, d: int, norm_sq: float) -> np.ndarray:
    direction = rng.normal(size=d)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    return direction * math.sqrt(max(norm_sq, 0.0))


def fit_lbsm(
    train_anchors: list[np.ndarray],
    val_anchors: list[np.ndarray],
    seed: int,
    r_F: float,
    ell: float,
    gamma: float,
    sigma: float,
    d: int,
    k: float,
    B: float,
    hidden_dim: int = 64,
    n_epochs: int = 500,
    lr: float = 1e-3,
    kappa_train: int = 8,
    eps_tr: float = 0.02,
    eps_U: float = 0.02,
    lambda_U: float = 1.0,
    lambda_core: float = 1.0,
    lambda_reg: float = 1e-4,
    core_margin: float = 0.5,
    n_core_samples: int = 64,
    use_l_core: bool = True,
) -> LBSMTrainingResult:
    """
    Eq. 8's joint training loss:

        L_train = (1/n_tr) sum_i [ 1{x_i not in F_X} * ReLU(g_hat_V(x_i) + eps_tr)
                                    + lambda_U * ReLU(g_hat_U(x_i) + eps_U) ]
                  + lambda_core * L_core + lambda_reg * R

    No separate loss term for (B2) (E[W(x')] <= B for x in F_X): W in [0,B]
    by construction (bounded architecture above) makes it hold structurally,
    matching the paper's own remark ("No separate loss for (B2) is required
    because W in [0,B] makes (B2) hold structurally").

    seed is REQUIRED (no default) -- deliberately, per the task correction
    referencing the L2 walker-walk asymmetric-seeding gap
    (EXPERIMENT_PARAMETERS.md §5, project memory): fit_lppm() / the tdmpc2
    eval script never seed torch, giving that pipeline weaker reproducibility
    than the cardreamer one. This function does not repeat that gap.

    use_l_core exists ONLY for tests/test_lbsm_trained.py's ablation (does
    omitting L_core actually risk the U=0 collapse the paper warns about?);
    it is not a knob production code should ever set False.

    Symbol note: Section 4.4's Eq. 7/8 (the "Learning candidate certificates"
    subsection) name the LBSM "V" (ĝ_V), while the rest of Section 4.4 and
    Appendix E.2 name it "W" -- this looks like a residual from the L2/LPPM
    section's own V_phi symbol, not a second certificate. This module follows
    W (matching core/lbsm/model.py and Stage 1) and just notes the paper-side
    inconsistency here so it isn't mistaken for a bug when cross-referencing
    Eq. 7/8 literally.
    """
    if torch is None or LBSMNet is None:
        raise RuntimeError("fit_lbsm() requires torch (no heuristic fallback for L3)")

    n_overlap = find_trajectory_overlap(train_anchors, val_anchors)
    if n_overlap:
        raise ValueError(
            f"fit_lbsm(): {n_overlap} anchor object(s) shared between train_anchors "
            "and val_anchors. L3's independent validation split is a load-bearing "
            "premise of Theorem 5.6 (Appendix C.2: the independent split "
            "'establish[es] the statistical inequalities needed by the final "
            "warrant'), stricter than L2's opt-in check (no allow_overlap escape "
            "hatch here). Refusing to train."
        )
    if not train_anchors:
        raise ValueError("fit_lbsm() requires at least one training anchor")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    b = k * sigma * (d ** 0.5)

    W_net = LBSMNet(latent_dim=d, hidden_dim=hidden_dim, B=B)
    U_net = LBSMNet(latent_dim=d, hidden_dim=hidden_dim, B=B)
    optimizer = torch.optim.Adam(list(W_net.parameters()) + list(U_net.parameters()), lr=lr)

    def label_p(z: np.ndarray) -> bool:
        return float(np.dot(z, z)) <= r_F ** 2

    # D_in / D_bdry for L_core (Eq. 8's sublevel-set shaping loss, "prevent
    # the degenerate choice U ≡ 0"). Generated from geometric knowledge of
    # the intended C = {||z||^2 <= ell} -- kept separate from
    # train_anchors/val_anchors (a shaping loss term, not part of the
    # drift-certification statistics the disjointness check protects).
    m = core_margin
    d_in = np.stack([
        _random_point_with_norm_sq(rng, d, rng.uniform(0.0, max(ell - 2 * m, 0.0)))
        for _ in range(n_core_samples)
    ])
    d_bdry = np.stack([
        _random_point_with_norm_sq(rng, d, rng.uniform(ell, ell + 2 * m))
        for _ in range(n_core_samples)
    ])
    d_in_t = torch.as_tensor(d_in, dtype=torch.float32)
    d_bdry_t = torch.as_tensor(d_bdry, dtype=torch.float32)

    train_anchors_arr = np.stack(train_anchors)
    train_anchors_t = torch.as_tensor(train_anchors_arr, dtype=torch.float32)
    train_labels = torch.as_tensor([label_p(z) for z in train_anchors], dtype=torch.bool)
    not_in_fx = (~train_labels).float()
    n_tr = len(train_anchors)

    loss_history: list[float] = []

    for _epoch in range(n_epochs):
        optimizer.zero_grad()

        # Eq. 7: empirical drift, resampled fresh successors each epoch (a
        # fresh resettable-generative draw per step, not a fixed cached set).
        successors = np.stack([
            [sample_truncated_gaussian_step(z, gamma, sigma, d, b, rng) for _ in range(kappa_train)]
            for z in train_anchors
        ])  # (n_tr, kappa_train, d)
        successors_t = torch.as_tensor(successors, dtype=torch.float32)

        V_curr = W_net(train_anchors_t)  # (n_tr,)
        V_next = W_net(successors_t.reshape(-1, d)).reshape(n_tr, kappa_train)
        g_hat_V = V_next.mean(dim=1) - V_curr  # Eq. 7

        U_curr = U_net(train_anchors_t)
        U_next = U_net(successors_t.reshape(-1, d)).reshape(n_tr, kappa_train)
        g_hat_U = U_next.mean(dim=1) - U_curr

        loss_V = (not_in_fx * torch.clamp(g_hat_V + eps_tr, min=0.0)).mean()
        loss_U = lambda_U * torch.clamp(g_hat_U + eps_U, min=0.0).mean()

        if use_l_core:
            u_in = U_net(d_in_t)
            u_bdry = U_net(d_bdry_t)
            l_core = (
                torch.clamp(u_in - (ell - m), min=0.0).mean()
                + torch.clamp((ell + m) - u_bdry, min=0.0).mean()
            )
        else:
            # SIMPLIFICATION: L_core disabled, ONLY for the deliberate
            # ablation test (tests/test_lbsm_trained.py) demonstrating why
            # the task spec calls it non-optional. Never set False otherwise.
            l_core = torch.tensor(0.0)

        reg = sum((p ** 2).mean() for p in list(W_net.parameters()) + list(U_net.parameters()))

        loss = loss_V + loss_U + lambda_core * l_core + lambda_reg * reg
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    return LBSMTrainingResult(
        W_net=W_net,
        U_net=U_net,
        loss_history=loss_history,
        epochs_trained=n_epochs,
        n_train_anchors=len(train_anchors),
        n_val_anchors=len(val_anchors),
        l_core_active=use_l_core,
    )
