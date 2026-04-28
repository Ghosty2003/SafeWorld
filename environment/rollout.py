from __future__ import annotations

from typing import Any, Callable


def rollout_env(
    env,
    actions: list[Any],
    state_adapter: Callable[[Any], dict[str, float]] | None = None,
    reset_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    obs, info = env.reset(**(reset_kwargs or {}))
    trajectory: list[dict[str, float]] = []
    prev_obs = obs

    for action in actions:
        obs, _, terminated, truncated, info = env.step(action)
        state = (
            _adapt_state(state_adapter, obs, info, prev_obs, action)
            if state_adapter else obs
        )
        trajectory.append(state)
        prev_obs = obs
        if terminated or truncated:
            break

    return trajectory


def _adapt_state(
    state_adapter: Callable,
    obs: Any,
    info: dict[str, Any] | None,
    prev_obs: Any,
    action: Any,
) -> dict[str, float]:
    try:
        return state_adapter(obs, info=info, prev_obs=prev_obs, action=action)
    except TypeError:
        return state_adapter(obs)
