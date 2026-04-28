"""
core/stl_monitor.py

SAFEWORLD – Latent Monitor (Section 4.2)

Implements quantitative STL robustness semantics (Definition 3.4) directly on
latent trajectories.  No environment access is required; only the formula tree
and a list of state-dicts are needed.

Public API
----------
compute_robustness(formula, trajectory, t=0) -> float
    Evaluate ρ(φ, τ, t) for a single trajectory.

monitor_rollouts(formula, trajectories) -> MonitorResult
    Evaluate all N rollouts and return the SAFEWORLD Latent Monitor summary,
    including ρ* (worst-case margin) and the witnessing rollout index.

Algorithm cost: O(N · T · |φ|)  (linear in rollouts, horizon, formula size)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


# ─── core robustness recursion ────────────────────────────────────────────────

def compute_robustness(
    formula: dict,
    trajectory: list[dict[str, float]],
    t: int = 0,
) -> float:
    """
    Recursively evaluate ρ(formula, trajectory, t).

    Parameters
    ----------
    formula    : STL/LTL formula parse-tree dict (see specs/stl_specs.py for schema).
    trajectory : list of T state-dicts  {dim -> float}.
                 Missing keys default to 0.0.
    t          : current time index.

    Returns
    -------
    Robustness value ρ ∈ ℝ.
    Positive  → formula satisfied with margin |ρ|.
    Negative  → formula violated  with margin |ρ|.
    """
    if not formula or not trajectory:
        return 0.0

    ftype = formula["type"]

    # ── atomic predicate ─────────────────────────────────────────────────────
    if ftype == "atom":
        if t >= len(trajectory):
            return -math.inf
        val = trajectory[t].get(formula["dim"], 0.0)
        thr = formula["threshold"]
        return val - thr if formula["op"] == ">" else thr - val

    # ── negation ─────────────────────────────────────────────────────────────
    if ftype == "not":
        return -compute_robustness(formula["child"], trajectory, t)

    # ── conjunction ──────────────────────────────────────────────────────────
    if ftype == "and":
        return min(
            compute_robustness(formula["left"],  trajectory, t),
            compute_robustness(formula["right"], trajectory, t),
        )

    # ── disjunction ──────────────────────────────────────────────────────────
    if ftype == "or":
        return max(
            compute_robustness(formula["left"],  trajectory, t),
            compute_robustness(formula["right"], trajectory, t),
        )

    # ── implication ──────────────────────────────────────────────────────────
    if ftype == "implies":
        return max(
            -compute_robustness(formula["left"], trajectory, t),
            compute_robustness(formula["right"], trajectory, t),
        )

    # ── next ─────────────────────────────────────────────────────────────────
    if ftype == "next":
        return compute_robustness(formula["child"], trajectory, t + 1)

    T = len(trajectory)

    # ── always  □[a,b] φ ─────────────────────────────────────────────────────
    if ftype == "always":
        a, b = formula["a"], formula["b"]
        lo, hi = t + a, min(t + b, T - 1)
        if lo > T - 1:
            return math.inf          # vacuously true (no steps to check)
        worst = math.inf
        for tp in range(lo, hi + 1):
            worst = min(worst, compute_robustness(formula["child"], trajectory, tp))
        return worst

    # ── eventually  ♢[a,b] φ ─────────────────────────────────────────────────
    if ftype == "eventually":
        a, b = formula["a"], formula["b"]
        lo, hi = t + a, min(t + b, T - 1)
        if lo > T - 1:
            return -math.inf         # no steps available → vacuously false
        best = -math.inf
        for tp in range(lo, hi + 1):
            best = max(best, compute_robustness(formula["child"], trajectory, tp))
        return best

    # ── until  φ U[a,b] ψ ───────────────────────────────────────────────────
    if ftype == "until":
        a, b = formula["a"], formula["b"]
        lo, hi = t + a, min(t + b, T - 1)
        best = -math.inf
        for tp in range(lo, hi + 1):
            psi_rob = compute_robustness(formula["right"], trajectory, tp)
            phi_min = math.inf
            for tpp in range(t, tp):
                phi_min = min(phi_min, compute_robustness(formula["left"], trajectory, tpp))
            if phi_min == math.inf:   # no prefix steps → only ψ matters
                phi_min = psi_rob
            best = max(best, min(psi_rob, phi_min))
        return best

    raise ValueError(f"Unknown formula type: '{ftype}'")


# ─── batch monitor over N rollouts ───────────────────────────────────────────

@dataclass
class MonitorResult:
    """Output of monitor_rollouts() – the SAFEWORLD Latent Monitor summary."""

    margins: list[float]
    """ρ(φ, τ_i, 0) for every rollout i ∈ [N]."""

    rho_star: float
    """ρ* = min_i margins[i]  (worst-case STL margin, Algorithm 1 line 2)."""

    witness_idx: int
    """Index of the rollout achieving ρ*."""

    n_rollouts: int
    """Total number of rollouts evaluated."""

    n_satisfied: int
    """Number of rollouts with ρ > 0."""

    mean_margin: float
    """Mean robustness across all rollouts."""

    std_margin: float
    """Std-dev of robustness across all rollouts."""

    formula_type: str = ""
    """Top-level formula type for bookkeeping."""

    raw_margins_per_step: list[list[float]] = field(default_factory=list)
    """
    Optional: per-step robustness ρ(φ, τ_i, t) for each rollout i and time t.
    Populated only when compute_per_step=True.
    """

    def is_violated(self) -> bool:
        return self.rho_star < 0

    def satisfaction_rate(self) -> float:
        return self.n_satisfied / self.n_rollouts if self.n_rollouts > 0 else 0.0

    def summary(self) -> str:
        verdict = "VIOLATION" if self.is_violated() else "SATISFYING"
        return (
            f"[LatentMonitor] {verdict} | "
            f"ρ*={self.rho_star:.4f}  mean={self.mean_margin:.4f}  "
            f"sat={self.n_satisfied}/{self.n_rollouts}"
        )


def monitor_rollouts(
    formula: dict,
    trajectories: Sequence[list[dict[str, float]]],
    compute_per_step: bool = False,
) -> MonitorResult:
    """
    Evaluate STL robustness for all N rollouts (Algorithm 1, lines 1-2).

    Parameters
    ----------
    formula         : STL formula parse tree.
    trajectories    : N trajectories, each a list of T state-dicts.
    compute_per_step: If True, also compute ρ(φ, τ_i, t) for all t and store in
                      MonitorResult.raw_margins_per_step.

    Returns
    -------
    MonitorResult with ρ*, witness index, and summary statistics.
    """
    if not trajectories:
        raise ValueError("trajectories must be non-empty")

    margins: list[float] = []
    per_step: list[list[float]] = []

    for tau in trajectories:
        rho = compute_robustness(formula, tau, t=0)
        margins.append(rho)

        if compute_per_step:
            step_rhos = [compute_robustness(formula, tau, t=t) for t in range(len(tau))]
            per_step.append(step_rhos)

    rho_star   = min(margins)
    witness    = margins.index(rho_star)
    n_sat      = sum(1 for m in margins if m > 0)
    mean_m     = sum(margins) / len(margins)
    variance   = sum((m - mean_m) ** 2 for m in margins) / len(margins)
    std_m      = math.sqrt(variance)

    return MonitorResult(
        margins=margins,
        rho_star=rho_star,
        witness_idx=witness,
        n_rollouts=len(margins),
        n_satisfied=n_sat,
        mean_margin=mean_m,
        std_margin=std_m,
        formula_type=formula.get("type", ""),
        raw_margins_per_step=per_step,
    )


# ─── robustness evolution (diagnostic, Section 6.13) ─────────────────────────

def robustness_evolution(
    formula: dict,
    trajectory: list[dict[str, float]],
) -> list[float]:
    """
    Return [ρ(φ, τ, 0), ρ(φ, τ, 1), ..., ρ(φ, τ, T-1)].
    Useful for plotting how robustness evolves along a single rollout.
    """
    return [compute_robustness(formula, trajectory, t) for t in range(len(trajectory))]


def net_robustness(rho_star: float, c_hat_err: float) -> float:
    """
    ρ_net = ρ* - ĉ_err  (Equation 2 from the paper).
    Positive ρ_net means the safety margin exceeds the model/environment mismatch budget.
    """
    return rho_star - c_hat_err
