"""
core/transfer_calibrator.py

SAFEWORLD – Transfer Calibrator (Section 4.2, Section 5, Corollary 5.2)

将 world model 在 latent space 得到的 STL robustness 结果，通过 conformal prediction
校准后转移到真实环境，给出有统计保证的结论。

核心思想
--------
World model 不是真实环境的完美复现，decoder error 和 rollout drift 会让模型侧的
STL robustness margin 在真实环境中失效。Transfer Calibrator 做两件事：

1. 用 split conformal prediction 估计模型误差上界 ĉ_err（从 paired rollouts 得到）：
       ĉ_err = Quantile_{1-δ_err}(cerr(τ^M_i, τ^E_i))
   其中 cerr = max_{j∈AP} max_t |r^M_j(t) - r^E_j(t)|  (Definition 3.5)

2. 计算 net robustness：
       ρ_net = ρ* - ĉ_err  (Equation 2)
   若 ρ_net > 0，则 Pr[τ^E |= φ] ≥ 1 - δ_cp - δ_err  (Corollary 5.2)

Public API
----------
compute_atomic_distortion(model_traj, env_traj, formula_aps) -> float
    cerr for a single paired rollout (Definition 3.5).

fit_conformal_error_budget(paired_rollouts, formula_aps, delta_err) -> float
    Estimate ĉ_err from N paired (model, environment) rollout pairs.

calibrate_robustness_quantile(margins, delta_cp) -> float
    Split-conformal quantile q̂_δ from N model-side robustness values.

transfer_verdict(rho_star, c_hat_err, delta_cp, delta_err) -> TransferResult
    Combine everything into a single verdict with statistical guarantees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from utils.task_parser import evaluate_predicate


# ─── paired rollout distortion (Definition 3.5) ──────────────────────────────

def compute_atomic_distortion(
    model_traj:  list[dict[str, float]],
    env_traj:    list[dict[str, float]],
    formula_aps: list[str],
    formula:     dict | None = None,
    predicate_map: dict[str, dict] | None = None,
) -> float:
    """
    Compute cerr(τ^M, τ^E) for a single paired rollout.

    cerr = max_{j ∈ AP(φ)} max_{t ∈ [0,T-1]} |r^M_j(t) - r^E_j(t)|

    where r_j(t) is the scalar atomic robustness signal for predicate j at step t.
    If `formula` is provided, r_j is the full formula robustness; otherwise it is
    the raw dim value from the state dict.

    Parameters
    ----------
    model_traj   : T state-dicts from the world model.
    env_traj     : T state-dicts from the true environment (same initial conditions).
    formula_aps  : list of AP key strings referenced by the specification.
    formula      : (optional) full formula tree – if given, uses compute_robustness
                   for each AP's sub-formula instead of raw values.
    """
    T = min(len(model_traj), len(env_traj))
    if T == 0:
        return 0.0

    max_distortion = 0.0
    for ap_key in formula_aps:
        for t in range(T):
            if predicate_map and ap_key in predicate_map:
                r_model = evaluate_predicate(model_traj[t], predicate_map[ap_key])
                r_env = evaluate_predicate(env_traj[t], predicate_map[ap_key])
            else:
                r_model = model_traj[t].get(ap_key, 0.0)
                r_env = env_traj[t].get(ap_key, 0.0)
            distortion = abs(r_model - r_env)
            if distortion > max_distortion:
                max_distortion = distortion

    return max_distortion


# ─── conformal error budget estimation ───────────────────────────────────────

def fit_conformal_error_budget(
    paired_rollouts: list[tuple[list[dict], list[dict]]],
    formula_aps:     list[str],
    delta_err:       float = 0.05,
    formula:         dict | None = None,
    predicate_map:   dict[str, dict] | None = None,
) -> float:
    """
    Estimate ĉ_err via split conformal prediction (Algorithm 1, line 3).

    Given N paired (model, environment) rollout pairs, computes the
    (1 - δ_err) quantile of per-pair cerr values.

    Parameters
    ----------
    paired_rollouts : list of (model_trajectory, env_trajectory) pairs.
    formula_aps     : AP keys referenced by the spec.
    delta_err       : failure probability for error bound (default 0.05).
    formula         : optional – full formula tree for robustness-based distortion.

    Returns
    -------
    ĉ_err : conformal upper bound on cerr.
            With probability ≥ 1-δ_err, cerr(τ^M_new, τ^E_new) ≤ ĉ_err
            for a fresh exchangeable paired rollout.
    """
    if not paired_rollouts:
        return 0.0

    scores = [
        compute_atomic_distortion(m, e, formula_aps, formula, predicate_map)
        for m, e in paired_rollouts
    ]

    n = len(scores)
    # Split-conformal quantile at level ceil((1-δ)(n+1))/n
    sorted_scores = sorted(scores)
    idx = math.ceil((1.0 - delta_err) * (n + 1)) - 1
    idx = max(0, min(idx, n - 1))
    return sorted_scores[idx]


# ─── robustness quantile (for PAC bound on model-side margin) ────────────────

def calibrate_robustness_quantile(
    margins:  Sequence[float],
    delta_cp: float = 0.05,
) -> float:
    """
    Compute q̂_δ = split-conformal (1-δ_cp) quantile of the N model robustness values.

    Theorem D.4 guarantees:
        Pr[ρ(φ, τ^M_{N+1}, 0) ≥ q̂_δ] ≥ 1 - δ_cp
    for a fresh exchangeable rollout.

    Parameters
    ----------
    margins  : list of N robustness values ρ(φ, τ_i, 0).
    delta_cp : conformal coverage failure probability (default 0.05).

    Returns
    -------
    q̂_δ : (1-δ_cp) quantile of the calibration margins.
    """
    if not margins:
        return -math.inf
    n = len(margins)
    sorted_m = sorted(margins)
    idx = math.ceil((1.0 - delta_cp) * (n + 1)) - 1
    idx = max(0, min(idx, n - 1))
    return sorted_m[idx]


# ─── main transfer verdict ────────────────────────────────────────────────────

@dataclass
class TransferResult:
    """
    Complete output of the Transfer Calibrator.

    Two guarantee paths (Theorem 5.5 / Corollary 5.2):

    Strict path (transfers()):
        ρ_net = ρ* - ĉ_err > 0
        Every sampled rollout satisfied the spec with margin ≥ ĉ_err.
        Confidence: 1 - δ_err  (no δ_cp charge — worst-case over all rollouts).

    Conformal path (transfers_cp()):
        ρ_net_cp = q̂_δ - ĉ_err > 0
        A (1-δ_cp) fraction of rollouts satisfied the spec with margin ≥ ĉ_err.
        Confidence: 1 - δ_cp - δ_err  (Theorem 5.5 PAC-CP guarantee).

    The strict path is the special case δ_cp → 0 (q̂_0 = ρ*).
    """

    # raw inputs
    rho_star:    float   # ρ* = min_i ρ(φ, τ^M_i, 0)
    c_hat_err:   float   # ĉ_err = conformal model-error upper bound
    q_hat:       float   # q̂_δ  = (1-δ_cp) quantile of per-rollout robustness
    delta_cp:    float   # coverage failure prob for model-side CP bound
    delta_err:   float   # coverage failure prob for error budget

    # derived — strict path
    rho_net:     float   # ρ_net    = ρ* - ĉ_err  (Equation 2)
    confidence:  float   # 1 - δ_cp - δ_err

    # derived — conformal path (Theorem 5.5)
    rho_net_cp:  float   # ρ_net_cp = q̂_δ - ĉ_err

    # per-AP error budget breakdown (optional)
    per_ap_distortion: dict[str, float] | None = None

    def transfers(self) -> bool:
        """
        Strict transfer: ρ_net > 0.
        All N rollouts satisfied the spec with margin > ĉ_err.
        Confidence ~ 1 - δ_err.
        """
        return self.rho_net > 0

    def transfers_cp(self) -> bool:
        """
        Conformal transfer (Theorem 5.5): ρ_net_cp > 0.
        With probability ≥ 1 - δ_cp - δ_err, a fresh environment rollout satisfies the spec.
        """
        return self.rho_net_cp > 0

    def effective_confidence(self) -> float:
        """
        Confidence level of the strongest applicable guarantee.
        Strict path charges only δ_err; conformal path charges δ_cp + δ_err.
        """
        if self.transfers():
            return max(0.0, 1.0 - self.delta_err)
        if self.transfers_cp():
            return max(0.0, 1.0 - self.delta_cp - self.delta_err)
        return 0.0

    def summary(self) -> str:
        if self.transfers():
            status = "TRANSFERS (strict) ✓"
        elif self.transfers_cp():
            status = "TRANSFERS (conformal) ✓"
        else:
            status = "INSUFFICIENT MARGIN ✗"
        return (
            f"[TransferCalibrator] {status}\n"
            f"  ρ*={self.rho_star:.4f}  q̂_δ={self.q_hat:.4f}  ĉ_err={self.c_hat_err:.4f}\n"
            f"  ρ_net={self.rho_net:+.4f} (strict)   "
            f"ρ_net_cp={self.rho_net_cp:+.4f} (conformal)\n"
            f"  confidence={self.effective_confidence():.3f}  "
            f"(δ_cp={self.delta_cp}, δ_err={self.delta_err})"
        )


def transfer_verdict(
    rho_star:       float,
    c_hat_err:      float,
    delta_cp:       float = 0.05,
    delta_err:      float = 0.05,
    margins:        list[float] | None = None,
) -> TransferResult:
    """
    Combine ρ* and ĉ_err into the transfer verdict (Algorithm 1, lines 3-5).

    Parameters
    ----------
    rho_star   : worst-case STL margin from Latent Monitor.
    c_hat_err  : conformal model error bound from fit_conformal_error_budget().
    delta_cp   : confidence level for robustness quantile.
    delta_err  : confidence level for error budget.
    margins    : if provided, also compute q̂_δ from these.

    Returns
    -------
    TransferResult with rho_net and statistical confidence.
    """
    rho_net    = rho_star - c_hat_err
    confidence = max(0.0, 1.0 - delta_cp - delta_err)
    q_hat      = calibrate_robustness_quantile(margins, delta_cp) if margins else rho_star
    rho_net_cp = q_hat - c_hat_err

    return TransferResult(
        rho_star=rho_star,
        c_hat_err=c_hat_err,
        q_hat=q_hat,
        delta_cp=delta_cp,
        delta_err=delta_err,
        rho_net=rho_net,
        confidence=confidence,
        rho_net_cp=rho_net_cp,
    )


# ─── Lipschitz-based error budget (Corollary D.3) ────────────────────────────

def lipschitz_error_budget(
    latent_mismatch: float,
    lipschitz_constants: dict[str, float],
    formula_aps: list[str],
) -> float:
    """
    Upper-bound cerr via latent-space mismatch when AP readout maps are Lipschitz.

    ĉ_err ≤ max_{j ∈ AP(φ)} L_j · ε_z

    where ε_z = max_t ‖z^M_t - z^E_t‖  and  L_j is the Lipschitz constant of r_j.

    Use this when paired environment rollouts are unavailable.

    Parameters
    ----------
    latent_mismatch      : ε_z, the worst-case latent L2 distance between model
                           and environment rollouts.
    lipschitz_constants  : dict mapping AP key -> L_j constant.
    formula_aps          : AP keys referenced by the spec.
    """
    return max(
        lipschitz_constants.get(ap, 1.0) * latent_mismatch
        for ap in formula_aps
    )


# ─── helper: compute per-AP distortion breakdown ─────────────────────────────

def per_ap_distortion_breakdown(
    model_traj:  list[dict[str, float]],
    env_traj:    list[dict[str, float]],
    formula_aps: list[str],
) -> dict[str, float]:
    """
    Return the maximum distortion per AP key (for diagnostic reporting, Table 22).

    Returns
    -------
    dict mapping each AP key -> max_t |r^M_j(t) - r^E_j(t)|
    """
    T = min(len(model_traj), len(env_traj))
    result = {}
    for ap_key in formula_aps:
        worst = 0.0
        for t in range(T):
            diff = abs(model_traj[t].get(ap_key, 0.0) - env_traj[t].get(ap_key, 0.0))
            if diff > worst:
                worst = diff
        result[ap_key] = worst
    return result
