from __future__ import annotations


class EnvWrapper:
    def __init__(self, env_name: str, **kwargs):
        self.env_name = env_name
        self.backend = "gymnasium"
        use_safety_gymnasium = kwargs.pop("use_safety_gymnasium", None)
        safety_env_name = kwargs.pop("safety_env_name", None) or _safety_env_id(env_name)

        if use_safety_gymnasium is True or (
            use_safety_gymnasium is None and _looks_like_safety_env(env_name)
        ):
            try:
                self.env = _make_safety_gymnasium_env(safety_env_name, **kwargs)
                self.backend = "safety_gymnasium"
            except ImportError:
                if use_safety_gymnasium is True:
                    raise
                self.env = _make_gymnasium_env(env_name, **kwargs)
            except Exception:
                if use_safety_gymnasium is True:
                    raise
                self.env = _make_gymnasium_env(env_name, **kwargs)
        else:
            self.env = _make_gymnasium_env(env_name, **kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        result = self.env.step(action)
        if isinstance(result, tuple) and len(result) == 6:
            obs, reward, cost, terminated, truncated, info = result
            info = dict(info or {})
            info.setdefault("cost", cost)
            return obs, reward, terminated, truncated, info
        return result

    def close(self):
        self.env.close()


def _make_gymnasium_env(env_name: str, **kwargs):
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError(
            "gymnasium is required for environment-backed SAFEWORLD rollouts."
        ) from exc
    return gym.make(env_name, **kwargs)


def _make_safety_gymnasium_env(env_name: str, **kwargs):
    try:
        import safety_gymnasium
        from safety_gymnasium.wrappers import SafetyGymnasium2Gymnasium
    except ImportError as exc:
        raise ImportError(
            "safety_gymnasium is required for SafetyPointGoal paired rollouts."
        ) from exc
    return SafetyGymnasium2Gymnasium(safety_gymnasium.make(env_name, **kwargs))


def _looks_like_safety_env(env_name: str) -> bool:
    return env_name.startswith("Safety") and env_name.endswith("-v0")


def _safety_env_id(env_name: str) -> str:
    if env_name.endswith("Gymnasium-v0"):
        return f"{env_name[:-len('Gymnasium-v0')]}-v0"
    return env_name
