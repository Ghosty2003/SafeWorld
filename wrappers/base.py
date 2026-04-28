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
from typing import Any


@dataclass
class RolloutConfig:
    """Parameters controlling how rollouts are sampled."""

    horizon:      int  = 50
    """Number of latent steps T per rollout."""

    n_rollouts:   int  = 20
    """Number of independent rollouts N."""

    action_source: str = "random"
    """
    'random'  – uniform random actions (broad latent coverage).
    'policy'  – use the model's own actor.
    'zeros'   – zero actions (debugging).
    """

    seed:         int  = 0
    """Random seed for reproducibility."""

    device:       str  = "cpu"
    """Torch device string ('cpu', 'cuda:0', etc.)."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Wrapper-specific kwargs (e.g. env_name, checkpoint_path, use_decoder)."""


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
