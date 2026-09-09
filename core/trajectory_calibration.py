"""
core/trajectory_calibration.py

Trajectory-level statistical safety rate -- an INDEPENDENT, WEAKER
certificate path from the L2 co-Büchi progress-measure certificate
(Theorem 5.2/5.4, core/lppm/). Deliberately NOT placed under core/lppm/:
this does not train any network, does not build a parity automaton, and
does not carry the finite-prefix-absorption-witness structure Theorem
5.4's guarantee relies on -- it is a plain STL-monitor-style boolean
check on raw AP values, calibrated with the same Clopper-Pearson bound
used elsewhere in this project (reused from core.lppm.calibrator, not
reimplemented).

C(tau) = 1[the predicate holds at EVERY timestep of tau] -- e.g.
G(hazard_dist>0) or G(height>h_min) evaluated directly on the sampled
trajectory, with no automaton/certificate-network involved at all.

Why this exists: ltl_hazard_avoidance and ltl_height_safety's L2
progress-measure certificate path was diagnosed and confirmed
STRUCTURALLY infeasible on both real checkpoints this project has (not a
training-configuration bug -- three independent remediation strategies
were tried and ruled out; see EXPERIMENT_CONFIG.md §8.17/§8.18/§8.19/§8.20
for the full diagnostic chain). The root tension: p2_loss's gradient
signal requires odd-priority (trap-adjacent) transitions, but a
competent policy's whole practical value is making those transitions rare
-- so the specs L2 verification would be MOST useful for certifying are
structurally the hardest to train a non-degenerate progress-measure
certificate for. This module is the fallback path for exactly that case.

IMPORTANT -- guarantee-strength wording discipline: a result from this
module is NEVER a "warrant" in the Theorem-5.4 sense. Report it as a
"trajectory-level safety rate" / p_hat_safety, and always alongside the
explicit caveat in TrajectorySafetyRateResult.summary(). It says only:
"under the SAME deployment distribution these trajectories were drawn
from, the empirical fraction of ENTIRE sampled trajectories with zero
violations, lower-bounded at confidence (1-gamma), is p_hat_safety." It
carries no per-step ranking-function structure, no absorption-witness
argument, and no infinite-horizon extension beyond the sampled horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .lppm.calibrator import _clopper_pearson_lower


@dataclass
class TrajectorySafetyRateResult:
    n: int
    k: int
    p_hat_safety: float
    gamma: float

    def summary(self) -> str:
        return (
            f"[Trajectory-level safety rate -- NOT a Theorem-5.4 warrant] "
            f"p_hat_safety={self.p_hat_safety:.4f}  k={self.k}/{self.n}  gamma={self.gamma}  "
            "(lower bound on Pr[entire trajectory has zero violations] under this "
            "deployment distribution only -- no progress-measure/absorption-witness "
            "structure, do not compare directly against a p_hat_gamma warrant threshold "
            "as if the two were the same guarantee type)"
        )


def compute_trajectory_safety_rate(
    trajectories: list[list[dict[str, float]]],
    is_safe_step: Callable[[dict[str, float]], bool],
    gamma: float = 0.05,
) -> TrajectorySafetyRateResult:
    """
    C(tau) = 1[is_safe_step(z_t) holds for every t in tau]. Reports the
    exact one-sided Clopper-Pearson lower bound on Pr[C(tau)=1]
    (core.lppm.calibrator._clopper_pearson_lower -- reused, not
    reimplemented) over the given trajectory sample.

    No training, no automaton, no certificate network -- a direct boolean
    check on the raw per-step AP dict, evaluated once per trajectory.
    """
    n = len(trajectories)
    k = sum(1 for traj in trajectories if all(is_safe_step(step) for step in traj))
    p_hat = _clopper_pearson_lower(k, n, gamma)
    return TrajectorySafetyRateResult(n=n, k=k, p_hat_safety=p_hat, gamma=gamma)
