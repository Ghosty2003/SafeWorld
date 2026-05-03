"""
wrappers/base.py

Abstract base class for SAFEWORLD world-model wrappers.

Any wrapper must produce latent trajectories as lists of state-dicts:
    trajectory = [
        {"hazard_dist": 0.4, "velocity": 0.3, "zone_a": 0.1, ...},  # t=0
        {"hazard_dist": 0.3, "velocity": 0.4, "zone_a": 0.6, ...},  # t=1
        ...
    ]
Keys must match the "dim" fields in your chosen specification's formula tree.

AP key convention:
    hazard_dist     float  – signed dist to hazard (>0 safe, <0 inside)
    velocity        float  – speed scalar
    goal_dist       float  – signed dist to goal (<0 inside goal radius)
    near_obstacle   float  – proximity to obstacle (>0 far, <0 close)
    near_human      float  – proximity to human
    zone_a/b/c      float  – zone membership (>0.5 = inside)
    carrying        float  – 1.0 if holding object, 0.0 otherwise
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

from configs.settings import RolloutConfig


@dataclass
class ReplayStep:
    """
    One step of a decode-and-replay comparison.

    model_obs   : The world model's predicted observation at this step.
                  For obs-space models (SimplePointGoal2) this is the direct
                  decoder output (next_obs).
                  For latent-space models (DreamerV3) this is the RSSM decoder
                  output; None when the decoder is unavailable (simulation mode).
    env_obs     : The real simulator observation after applying the same action.
    action      : The action executed at this step.
    obs_rmse    : sqrt(mean((model_obs - env_obs)^2)) across obs dimensions.
                  None when model_obs is unavailable.
    ap_errors   : Absolute difference |model_semantic[k] - env_semantic[k]| per AP.
    """

    t:               int
    action:          np.ndarray
    model_obs:       np.ndarray | None
    env_obs:         np.ndarray
    model_semantic:  dict[str, float]
    env_semantic:    dict[str, float]
    obs_rmse:        float | None
    ap_errors:       dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.ap_errors:
            keys = set(self.model_semantic) & set(self.env_semantic)
            self.ap_errors = {
                k: abs(self.model_semantic[k] - self.env_semantic[k]) for k in keys
            }

    def max_ap_error(self) -> float:
        return max(self.ap_errors.values()) if self.ap_errors else 0.0


class WorldModelWrapper(abc.ABC):
    """
    Protocol every world-model wrapper must implement.

    Subclass and implement:
        load()            – load weights / connect to server.
        sample_rollouts() – return N latent trajectories of length T.
        ap_keys()         – declare which AP keys this wrapper produces.
        close()           – release resources (optional).
    """

    def __init__(self, config: RolloutConfig | None = None):
        self.config = config or RolloutConfig()

    # ── required ──────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def load(self, **kwargs) -> None:
        """Load model weights or connect to an external server."""

    @abc.abstractmethod
    def sample_rollouts(
        self,
        config: RolloutConfig | None = None,
    ) -> list[list[dict[str, float]]]:
        """
        Sample N latent rollouts of length T.

        Returns
        -------
        List of N trajectories; each trajectory is a list of T state-dicts.
        """

    @abc.abstractmethod
    def ap_keys(self) -> list[str]:
        """Return the AP dimension keys this wrapper provides."""

    # ── optional ──────────────────────────────────────────────────────────────

    def sample_paired_rollouts(
        self,
        config: RolloutConfig | None = None,
    ) -> list[tuple[list[dict[str, float]], list[dict[str, float]]]]:
        """
        Optional paired (model, environment) rollouts for transfer calibration.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement paired environment rollouts."
        )

    def decode_and_replay(
        self,
        config: RolloutConfig | None = None,
        closed_loop: bool = False,
    ) -> list[list[ReplayStep]]:
        """
        Run model rollouts, replay the exact same actions in the real simulator,
        and return per-step comparison records.

        Parameters
        ----------
        config      : rollout configuration (uses self.config if None).
        closed_loop : if True, feed the model's own predicted obs back as input
                      for the next step — measures cumulative drift.
                      If False (default), feed the real env obs back each step —
                      measures per-step prediction accuracy.

        Returns
        -------
        List of N rollouts; each rollout is a list of ReplayStep records.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement decode_and_replay."
        )

    def close(self) -> None:
        """Release GPU memory, close sockets, etc."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── helpers ───────────────────────────────────────────────────────────────

    def validate_trajectories(
        self,
        trajectories: list[list[dict[str, float]]],
        required_keys: list[str],
    ) -> list[str]:
        """
        Check trajectories contain all required AP keys.
        Returns list of warning strings (empty if all good).
        """
        warnings = []
        provided = set(self.ap_keys())
        missing_from_wrapper = [k for k in required_keys if k not in provided]
        if missing_from_wrapper:
            warnings.append(
                f"Wrapper '{type(self).__name__}' does not declare AP keys: "
                f"{missing_from_wrapper}. Those dims will default to 0.0."
            )
        for i, traj in enumerate(trajectories):
            for t, state in enumerate(traj):
                for key in required_keys:
                    if key not in state:
                        warnings.append(
                            f"Trajectory {i} step {t}: missing key '{key}' – will default to 0.0."
                        )
        return warnings
