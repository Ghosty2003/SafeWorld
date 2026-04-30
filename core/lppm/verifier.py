from __future__ import annotations

import math
from dataclasses import dataclass

from .automaton import ParityAutomaton, ProductState, extract_active_aps
from .model import compute_lppm_value


def run_product_trajectory(
    trajectory: list[dict[str, float]],
    dpa: ParityAutomaton,
    spec: dict,
) -> list[ProductState]:
    q = dpa.initial
    product_path: list[ProductState] = []
    for t, z in enumerate(trajectory):
        active_aps = extract_active_aps(z, spec)
        q_next, priority = dpa.step_with_priority(q, active_aps)
        product_path.append(ProductState(t=t, z=z, q=q, q_next=q_next, priority=priority))
        q = q_next
    return product_path


@dataclass
class PathwiseResult:
    satisfied: bool
    p1_violations: int
    p2_violations: int
    total_transitions: int
    min_descent_margin: float
    conformity_score: float


def check_pathwise_conditions(
    product_path: list[ProductState],
    dpa: ParityAutomaton,
    spec: dict,
    eta: float = 0.01,
    lppm_params: dict | None = None,
) -> PathwiseResult:
    odd_prios = dpa.odd_priorities
    T = len(product_path)
    p1_viols = p2_viols = 0
    min_descent = math.inf

    for i in range(T - 1):
        curr = product_path[i]
        nxt = product_path[i + 1]
        for r in odd_prios:
            v_curr = compute_lppm_value(curr.z, curr.q, r, spec, i, T, lppm_params, dpa)
            v_next = compute_lppm_value(nxt.z, nxt.q, r, spec, i + 1, T, lppm_params, dpa)
            if curr.priority == r:
                margin = v_curr - v_next
                min_descent = min(min_descent, margin)
                if margin < eta:
                    p2_viols += 1
            elif r > curr.priority and v_next > v_curr + 1e-6:
                p1_viols += 1

    total = max(T - 1, 1)
    satisfied = (p1_viols == 0) and (p2_viols == 0)
    return PathwiseResult(
        satisfied=satisfied,
        p1_violations=p1_viols,
        p2_violations=p2_viols,
        total_transitions=total,
        min_descent_margin=min_descent if min_descent != math.inf else 0.0,
        conformity_score=1.0 if satisfied else 0.0,
    )
