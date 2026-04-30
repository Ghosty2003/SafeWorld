"""
wrappers/random_wrapper.py

RandomWorldModel wrapper – built-in test/baseline wrapper.
Produces controlled synthetic latent trajectories without any ML dependencies.
Used in the paper's main experimental tables to validate the verification
algorithms independently of GPU training variability (Section 6.1).

Fidelity parameter controls trajectory safety:
    fidelity = 1.0  → very safe trajectories (large positive STL margins)
    fidelity = 0.5  → marginal trajectories  (margins near zero)
    fidelity = 0.0  → unsafe trajectories    (frequent violations)
"""

from __future__ import annotations

import math
from configs.settings import RolloutConfig
from .base import WorldModelWrapper


def _pseudo_rng(seed: int, i: int) -> float:
    """Deterministic float in [0,1] from seed and index."""
    x = math.sin(seed * 9301.0 + i * 49297.0) * 0.5 + 0.5
    return x


class RandomWorldModelWrapper(WorldModelWrapper):
    """
    Deterministic pseudo-random world model for algorithm validation.
    No external dependencies required.
    """

    def load(self, **kwargs) -> None:
        pass  # nothing to load

    def sample_rollouts(
        self,
        config: RolloutConfig | None = None,
    ) -> list[list[dict[str, float]]]:
        cfg = config or self.config
        spec_type = cfg.extra.get("spec_type", "always_safe")
        fidelity  = cfg.extra.get("fidelity", 0.75)

        return [
            self._make_trajectory(
                spec_type=spec_type,
                T=cfg.horizon,
                seed=cfg.seed + i * 7,
                fidelity=fidelity,
            )
            for i in range(cfg.n_rollouts)
        ]

    def _make_trajectory(
        self,
        spec_type: str,
        T: int,
        seed: int,
        fidelity: float,
    ) -> list[dict[str, float]]:
        traj = []
        for t in range(T):
            base  = _pseudo_rng(seed, t)
            noise = (_pseudo_rng(seed, t + 100) - 0.5) * (1.0 - fidelity) * 0.4
            frac  = t / T

            state: dict[str, float] = {}

            if spec_type == "always_safe":
                state["hazard_dist"]  = base * fidelity + noise
                state["velocity"]     = (1.0 - base) * (1.5 - fidelity)

            elif spec_type == "eventually_goal":
                state["goal_dist"]    = 1.0 - frac * fidelity - noise * 0.3
                state["hazard_dist"]  = base * fidelity + noise

            elif spec_type == "always_eventually":
                state["zone_a"]       = math.sin(t * 0.7 + seed) * 0.5 * fidelity + 0.5 + noise
                state["hazard_dist"]  = base * fidelity + noise

            elif spec_type == "sequential":
                state["zone_a"]       = 1.0 if (0.2 < frac < 0.5 and fidelity > 0.5) else noise * 0.3
                state["zone_b"]       = 1.0 if (frac > 0.65 and fidelity > 0.5) else noise * 0.3
                state["hazard_dist"]  = base * fidelity + noise

            elif spec_type == "full_mission":
                state["hazard_dist"]  = base * fidelity + noise
                state["velocity"]     = (1.0 - base) * (1.5 - fidelity)
                state["zone_a"]       = 1.0 if (0.15 < frac < 0.45 and fidelity > 0.5) else noise * 0.2
                state["zone_b"]       = 1.0 if (frac > 0.55 and fidelity > 0.5) else noise * 0.2
                state["zone_c"]       = 1.0 if abs(math.sin(t * 0.5 + seed)) > (1.0 - fidelity * 0.8) else 0.0
                state["near_obstacle"]= base * (1.0 - fidelity) * 0.5 - 0.1
                state["goal_dist"]    = 1.0 - frac * fidelity - noise * 0.3

            else:
                # Generic fallback: fill all known keys
                state = {
                    "hazard_dist":   base * fidelity + noise,
                    "velocity":      (1.0 - base) * (1.5 - fidelity),
                    "goal_dist":     1.0 - frac * fidelity,
                    "near_obstacle": base * 0.3 - 0.1,
                    "near_human":    base * 0.2 - 0.05,
                    "zone_a":        1.0 if (0.2 < frac < 0.5) else 0.0,
                    "zone_b":        1.0 if frac > 0.6 else 0.0,
                    "zone_c":        1.0 if abs(math.sin(t * 0.5)) > 0.6 else 0.0,
                    "carrying":      0.0,
                }

            # Fill any missing standard keys with safe defaults
            for key in self.ap_keys():
                state.setdefault(key, 0.0)

            traj.append(state)

        return traj

    def ap_keys(self) -> list[str]:
        return [
            "hazard_dist", "velocity", "goal_dist",
            "near_obstacle", "near_human",
            "zone_a", "zone_b", "zone_c", "carrying",
        ]
