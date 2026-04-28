from __future__ import annotations


class EnvWrapper:
    def __init__(self, env_name: str, **kwargs):
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError(
                "gymnasium is required for environment-backed SAFEWORLD rollouts."
            ) from exc
        self.env_name = env_name
        self.env = gym.make(env_name, **kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()
