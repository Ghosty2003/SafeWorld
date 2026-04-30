from __future__ import annotations

from typing import Any

from .loss import heuristic_epoch_loss, p1_loss, p2_loss, smoothness_penalty
from .model import NeuralLPPM, infer_feature_keys
from .verifier import run_product_trajectory

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


def fit_lppm(
    trajectories: list[list[dict[str, float]]],
    dpa,
    spec: dict,
    eta: float = 0.01,
    n_epochs: int = 300,
    lr: float = 1e-3,
    lambda_reg: float = 0.01,
) -> dict[str, Any]:
    odd_prios = dpa.odd_priorities
    all_transitions = []
    for traj in trajectories:
        path = run_product_trajectory(traj, dpa, spec)
        for i in range(len(path) - 1):
            all_transitions.append((path[i], path[i + 1]))

    if not all_transitions:
        return {
            "loss_history": [],
            "final_loss": 0.0,
            "n_transitions": 0,
            "epochs_trained": 0,
            "odd_priorities": odd_prios,
            "spec_id": spec.get("id", ""),
            "weights": None,
            "backend": "heuristic",
            "reason": "No transitions available for LPPM fitting.",
        }

    if torch is None or NeuralLPPM is None or F is None:
        return _fit_lppm_heuristic_fallback(all_transitions, dpa, spec, eta, n_epochs)

    feature_keys = infer_feature_keys(trajectories)
    state_to_idx = {state: idx for idx, state in enumerate(dpa.states)}
    odd_to_idx = {prio: idx for idx, prio in enumerate(odd_prios)}

    z_curr = torch.tensor(
        [[float(curr.z.get(key, 0.0)) for key in feature_keys] for curr, _ in all_transitions],
        dtype=torch.float32,
    )
    z_next = torch.tensor(
        [[float(nxt.z.get(key, 0.0)) for key in feature_keys] for _, nxt in all_transitions],
        dtype=torch.float32,
    )
    q_curr = torch.tensor([state_to_idx[curr.q] for curr, _ in all_transitions], dtype=torch.long)
    q_next = torch.tensor([state_to_idx[nxt.q] for _, nxt in all_transitions], dtype=torch.long)
    curr_priorities = torch.tensor([curr.priority for curr, _ in all_transitions], dtype=torch.long)

    hidden_dim = int(spec.get("lppm_hidden_dim", 128))
    q_embed_dim = int(spec.get("lppm_q_embed_dim", 16))
    device = torch.device("cpu")
    model = NeuralLPPM(
        latent_dim=len(feature_keys),
        n_states=len(dpa.states),
        n_heads=max(len(odd_prios), 1),
        hidden_dim=hidden_dim,
        q_embed_dim=q_embed_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    z_next = z_next.to(device)
    q_curr = q_curr.to(device)
    q_next = q_next.to(device)
    curr_priorities = curr_priorities.to(device)
    loss_history: list[float] = []

    for _epoch in range(n_epochs):
        optimizer.zero_grad()
        z_curr_epoch = z_curr.clone().to(device).requires_grad_(True)
        v_curr = model(z_curr_epoch, q_curr)
        v_next = model(z_next, q_next)
        loss = p1_loss(v_curr, v_next, curr_priorities, odd_to_idx)
        loss = loss + p2_loss(v_curr, v_next, curr_priorities, odd_to_idx, eta)
        loss = loss + lambda_reg * smoothness_penalty(v_curr, z_curr_epoch)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        loss_history.append(loss_value)
        if loss_value < 1e-6:
            break

    return {
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "n_transitions": len(all_transitions),
        "epochs_trained": len(loss_history),
        "odd_priorities": odd_prios,
        "spec_id": spec.get("id", ""),
        "weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "backend": "torch_mlp",
        "feature_keys": feature_keys,
        "state_to_idx": state_to_idx,
        "odd_to_idx": odd_to_idx,
        "hidden_dim": hidden_dim,
        "q_embed_dim": q_embed_dim,
    }


def _fit_lppm_heuristic_fallback(
    all_transitions,
    dpa,
    spec: dict[str, Any],
    eta: float,
    n_epochs: int,
) -> dict[str, Any]:
    odd_prios = dpa.odd_priorities
    loss_history: list[float] = []
    for _epoch in range(n_epochs):
        avg_loss = heuristic_epoch_loss(all_transitions, odd_prios, spec, dpa, eta)
        loss_history.append(avg_loss)
        if avg_loss < 1e-6:
            break
    return {
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "n_transitions": len(all_transitions),
        "epochs_trained": len(loss_history),
        "odd_priorities": odd_prios,
        "spec_id": spec.get("id", ""),
        "weights": None,
        "backend": "heuristic",
        "reason": "PyTorch unavailable; used heuristic LPPM fallback.",
    }
