from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RolloutConfig:
    """Runtime rollout settings shared by all world-model wrappers."""

    horizon: int = 50
    """Number of latent/environment steps per rollout."""

    n_rollouts: int = 20
    """Number of independent rollouts."""

    action_source: str = "random"
    """
    Action sampling mode. Common values are:
    random, env, zeros, adversarial, policy.
    """

    seed: int = 0
    """Base random seed."""

    device: str = "cpu"
    """Torch device string, when a wrapper uses torch."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Wrapper-specific runtime settings."""
