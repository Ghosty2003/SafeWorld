from __future__ import annotations

import numpy as np

from environment import EnvWrapper, rollout_env
from environment.adapters import safety_point_goal_adapter
from configs.settings import RolloutConfig
from .base import WorldModelWrapper


class SafetyPointGoalWrapper(WorldModelWrapper):
    """
    Environment-backed wrapper for Safety-Gym-style point-goal tasks such as
    `SafetyPointGoal1-v0`.

    It connects the environment to the current SAFEWORLD V2 safety evaluation
    flow by converting raw environment observations into semantic state dicts
    like `hazard_dist`, `goal_dist`, and `velocity`.
    """

    def __init__(self, config: RolloutConfig | None = None):
        super().__init__(config)
        self.env: EnvWrapper | None = None
        self.env_name = "SafetyPointGoal1-v0"

    def load(self, **kwargs) -> None:
        env_name = kwargs.get("env_name", self.config.extra.get("env_name", self.env_name))
        env_kwargs = dict(self.config.extra.get("env_kwargs", {}))
        env_kwargs.update(kwargs.get("env_kwargs", {}))
        self.env_name = env_name
        self.env = EnvWrapper(env_name, **env_kwargs)

    def sample_rollouts(
        self,
        config: RolloutConfig | None = None,
    ) -> list[list[dict[str, float]]]:
        cfg = config or self.config
        if self.env is None:
            self.load()
        assert self.env is not None

        trajectories: list[list[dict[str, float]]] = []
        for i in range(cfg.n_rollouts):
            actions = self._sample_action_sequence(cfg.horizon, cfg.seed + i)
            traj = rollout_env(
                self.env,
                actions,
                state_adapter=safety_point_goal_adapter,
                reset_kwargs=cfg.extra.get("reset_kwargs"),
            )
            trajectories.append(traj)
        return trajectories

    def ap_keys(self) -> list[str]:
        return [
            "hazard_dist",
            "goal_dist",
            "velocity",
            "near_obstacle",
            "near_human",
            "zone_a",
            "zone_b",
            "zone_c",
            "carrying",
        ]

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def _sample_action_sequence(self, horizon: int, seed: int) -> list[np.ndarray | int | float]:
        assert self.env is not None
        rng = np.random.default_rng(seed)
        action_space = self.env.action_space
        mode = self.config.extra.get("action_source", self.config.action_source)
        actions = []
        for _ in range(horizon):
            if mode == "zeros":
                actions.append(np.zeros(action_space.shape, dtype=getattr(action_space, "dtype", np.float32)))
            elif hasattr(action_space, "n"):
                actions.append(int(rng.integers(0, action_space.n)))
            else:
                low = np.asarray(action_space.low, dtype=float)
                high = np.asarray(action_space.high, dtype=float)
                sample = rng.uniform(low, high).astype(getattr(action_space, "dtype", np.float32))
                actions.append(sample)
        return actions
