from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from scipy.stats import beta as _beta_dist

from .config import DEFAULT_LPPM_CONFIG
from .verifier import (
    PathwiseResult,
    ZFreeClosureResult,
    check_pathwise_conditions,
    find_trajectory_overlap,
    run_product_trajectory,
    verify_zfree_closure,
)


@dataclass
class LPPMResult:
    pathwise: list[PathwiseResult]
    p_hat_gamma: float
    satisfaction_rate: float
    avg_descent_margin: float
    training_info: dict = field(default_factory=dict)
    warrant_threshold: float = 0.80
    p1_tolerance: float = 0.0
    # External-review fix: Theorem 5.4's exact minimal premise (P1-P2 on
    # Z_free-source transitions only), computed alongside p_hat_gamma but
    # deliberately kept SEPARATE -- p_hat_gamma remains a lower bound on the
    # strictly stronger whole-trajectory event C(tau); zfree_closure answers
    # the narrower, Theorem-5.4-exact question. See
    # core/lppm/verifier.py::verify_zfree_closure()'s docstring.
    zfree_closure: ZFreeClosureResult | None = None

    def is_warranted(self) -> bool:
        return self.p_hat_gamma >= self.warrant_threshold

    def summary(self) -> str:
        status = "WARRANT ✓" if self.is_warranted() else "NOT WARRANTED"
        lines = [
            f"[LPPM] {status} | "
            f"p̂_γ={self.p_hat_gamma:.3f}  "
            f"sat_rate={self.satisfaction_rate:.3f}  "
            f"avg_descent={self.avg_descent_margin:.4f}  "
            f"threshold={self.warrant_threshold:.2f}"
        ]
        if self.zfree_closure is not None:
            if self.zfree_closure.verifiable:
                lines.append(
                    f"[Z_free closure] p̂_closure={self.zfree_closure.p_hat_closure:.3f}  "
                    f"n={self.zfree_closure.n_zfree}  k={self.zfree_closure.k_zfree}  "
                    "(Theorem 5.4 minimal premise -- separate from p̂_γ above, not merged)"
                )
            else:
                lines.append(
                    "[Z_free closure] UNVERIFIABLE -- no trajectory had a source state "
                    "in Z_free; p̂_γ above remains valid only against the stronger "
                    "whole-trajectory event, not Theorem 5.4's minimal premise"
                )
        return "\n".join(lines)


def calibrate_lppm(
    trajectories: list[list[dict[str, float]]],
    dpa,
    spec: dict,
    gamma: float = 0.05,
    eta: float = DEFAULT_LPPM_CONFIG.eta,
    warrant_threshold: float = 0.80,
    lppm_params: dict | None = None,
    p1_tol: float = 0.0,
    train_trajectories: list[list[dict[str, float]]] | None = None,
    allow_overlap: bool = False,
    overlap_reason: str | None = None,
) -> LPPMResult:
    """
    p1_tol : see check_pathwise_conditions() docstring. Default 0.0 preserves
      old strict behavior. Derive from data, not a hardcoded guess, when a
      trained V has collapsed to a near-constant output (report the value
      used in LPPMResult.p1_tolerance alongside any p_hat_gamma it affected).

    train_trajectories : opt-in disjointness check, mirror of fit_lppm()'s.
      Pass the training split here to verify it shares no trajectory object
      with `trajectories`. Off by default (None): no check, no behavior
      change for callers that don't pass it.

    allow_overlap, overlap_reason : External-review fix -- overlap detected
      when train_trajectories is passed now RAISES ValueError by default
      (previously only warned). Pass allow_overlap=True with a non-empty
      overlap_reason to proceed anyway -- a forced, auditable opt-out.
    """
    if train_trajectories is not None:
        n_overlap = find_trajectory_overlap(trajectories, train_trajectories)
        if n_overlap:
            if not allow_overlap:
                raise ValueError(
                    f"calibrate_lppm(): {n_overlap} trajectory object(s) also "
                    "present in train_trajectories -- the calibration split is "
                    "not disjoint from the training split, which breaks the "
                    "exchangeability premise Theorem 5.4's p_hat_gamma relies on. "
                    "Pass allow_overlap=True with a non-empty overlap_reason to "
                    "proceed anyway (e.g. a deliberate ablation)."
                )
            if not overlap_reason:
                raise ValueError(
                    "calibrate_lppm(): allow_overlap=True requires a non-empty "
                    "overlap_reason explaining why this violation is intentional "
                    "-- silent overrides are not permitted."
                )
            warnings.warn(
                f"calibrate_lppm(): {n_overlap} trajectory object(s) also "
                f"present in train_trajectories, allowed via allow_overlap=True: "
                f"{overlap_reason}",
                stacklevel=2,
            )

    pathwise_results: list[PathwiseResult] = []
    for traj in trajectories:
        path = run_product_trajectory(traj, dpa, spec)
        pathwise_results.append(check_pathwise_conditions(path, dpa, spec, eta, lppm_params, p1_tol))

    n = len(pathwise_results)
    k = sum(1 for pw in pathwise_results if pw.satisfied)
    sat_rate = k / n if n > 0 else 0.0
    p_hat = _clopper_pearson_lower(k, n, gamma)
    avg_descent = (
        sum(pw.min_descent_margin for pw in pathwise_results) / n
        if n > 0 else 0.0
    )
    # External-review fix: compute Theorem 5.4's exact minimal-premise
    # diagnostic alongside p_hat_gamma so every real caller sees it, instead
    # of it only existing for whoever separately remembers to call
    # verify_zfree_closure() themselves. Kept as its own LPPMResult field --
    # never blended into p_hat_gamma itself.
    zfree_closure = verify_zfree_closure(
        trajectories, dpa, spec, gamma=gamma, eta=eta, lppm_params=lppm_params, p1_tol=p1_tol,
    )
    return LPPMResult(
        pathwise=pathwise_results,
        p_hat_gamma=p_hat,
        satisfaction_rate=sat_rate,
        avg_descent_margin=avg_descent,
        warrant_threshold=warrant_threshold,
        p1_tolerance=p1_tol,
        zfree_closure=zfree_closure,
    )


def _clopper_pearson_lower(k: int, n: int, gamma: float) -> float:
    """
    Exact one-sided (1-gamma) Clopper-Pearson lower confidence bound:
        p_hat_gamma = Beta^{-1}(gamma; k, n-k+1),   0 when k=0.
    (Previously a hardcoded-z Wald normal approximation that silently ignored
    gamma except at k=n; replaced to match the paper's exact-CP definition.)
    """
    if n == 0 or k == 0:
        return 0.0
    return float(_beta_dist.ppf(gamma, k, n - k + 1))
