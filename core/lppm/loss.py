from __future__ import annotations

from typing import Any

from .model import compute_lppm_value

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


def p1_loss(v_curr, v_next, curr_priorities, odd_to_idx: dict[int, int]):
    total = torch.zeros((), dtype=torch.float32, device=v_curr.device)
    for rp, rp_idx in odd_to_idx.items():
        mask = curr_priorities < rp
        if torch.any(mask):
            total = total + F.relu(v_next[mask, rp_idx] - v_curr[mask, rp_idx]).mean()
    return total


def p2_loss(v_curr, v_next, curr_priorities, odd_to_idx: dict[int, int], eta: float):
    total = torch.zeros((), dtype=torch.float32, device=v_curr.device)
    for rp, rp_idx in odd_to_idx.items():
        mask = curr_priorities == rp
        if torch.any(mask):
            total = total + F.relu(v_next[mask, rp_idx] - v_curr[mask, rp_idx] + eta).mean()
    return total


def smoothness_penalty(v_curr, z_curr):
    grad = torch.autograd.grad(v_curr.sum(), z_curr, create_graph=True)[0]
    return grad.pow(2).sum(dim=-1).mean()


def heuristic_epoch_loss(
    all_transitions,
    odd_prios: list[int],
    spec: dict[str, Any],
    dpa,
    eta: float,
) -> float:
    total_loss = 0.0
    T = max((p.t for p, _ in all_transitions), default=50) + 1
    for curr, nxt in all_transitions:
        r = curr.priority
        for rp in odd_prios:
            vc = compute_lppm_value(curr.z, curr.q, rp, spec, curr.t, T, dpa=dpa)
            vn = compute_lppm_value(nxt.z, nxt.q, rp, spec, nxt.t, T, dpa=dpa)
            if rp > r:
                total_loss += max(0.0, vn - vc)
            if rp == r:
                total_loss += max(0.0, vn - vc + eta)
    return total_loss / max(len(all_transitions), 1)
