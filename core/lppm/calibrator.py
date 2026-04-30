from __future__ import annotations

import math
from dataclasses import dataclass, field

from .verifier import PathwiseResult, check_pathwise_conditions, run_product_trajectory


@dataclass
class LPPMResult:
    pathwise: list[PathwiseResult]
    p_hat_gamma: float
    satisfaction_rate: float
    avg_descent_margin: float
    training_info: dict = field(default_factory=dict)
    warrant_threshold: float = 0.80

    def is_warranted(self) -> bool:
        return self.p_hat_gamma >= self.warrant_threshold

    def summary(self) -> str:
        status = "WARRANT ✓" if self.is_warranted() else "NOT WARRANTED"
        return (
            f"[LPPM] {status} | "
            f"p̂_γ={self.p_hat_gamma:.3f}  "
            f"sat_rate={self.satisfaction_rate:.3f}  "
            f"avg_descent={self.avg_descent_margin:.4f}  "
            f"threshold={self.warrant_threshold:.2f}"
        )


def calibrate_lppm(
    trajectories: list[list[dict[str, float]]],
    dpa,
    spec: dict,
    gamma: float = 0.05,
    eta: float = 0.01,
    warrant_threshold: float = 0.80,
    lppm_params: dict | None = None,
) -> LPPMResult:
    pathwise_results: list[PathwiseResult] = []
    for traj in trajectories:
        path = run_product_trajectory(traj, dpa, spec)
        pathwise_results.append(check_pathwise_conditions(path, dpa, spec, eta, lppm_params))

    n = len(pathwise_results)
    k = sum(1 for pw in pathwise_results if pw.satisfied)
    sat_rate = k / n if n > 0 else 0.0
    p_hat = _clopper_pearson_lower(k, n, gamma)
    avg_descent = (
        sum(pw.min_descent_margin for pw in pathwise_results) / n
        if n > 0 else 0.0
    )
    return LPPMResult(
        pathwise=pathwise_results,
        p_hat_gamma=p_hat,
        satisfaction_rate=sat_rate,
        avg_descent_margin=avg_descent,
        warrant_threshold=warrant_threshold,
    )


def _clopper_pearson_lower(k: int, n: int, gamma: float) -> float:
    if n == 0 or k == 0:
        return 0.0
    if k == n:
        return (1.0 - gamma) ** (1.0 / n)
    p_hat = k / n
    z = 1.645
    se = math.sqrt(p_hat * (1 - p_hat) / n)
    return max(0.0, p_hat - z * se)
