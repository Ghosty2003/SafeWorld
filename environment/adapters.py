from __future__ import annotations

import math
from typing import Any

import numpy as np


def safety_point_goal_adapter(
    obs: Any,
    *,
    info: dict[str, Any] | None = None,
    prev_obs: Any = None,
    action: Any = None,
) -> dict[str, float]:
    """
    Convert a SafetyPointGoal-style observation into the semantic state dict used
    by SAFEWORLD V2 verification.

    This adapter is intentionally defensive because different Safety-Gym builds
    expose slightly different observation/info layouts. It prefers `info` when
    available, then falls back to dict observations, and finally to coarse array
    heuristics when only a flat vector is available.
    """
    info = info or {}

    agent_pos = (
        _extract_vector(info, ("agent_pos", "robot_pos", "position"))
        or _extract_vector(obs, ("agent_pos", "robot_pos", "position"))
    )
    goal_pos = (
        _extract_vector(info, ("goal_pos", "goal_position"))
        or _extract_vector(obs, ("goal_pos", "goal_position"))
    )
    hazards = (
        _extract_points(info, ("hazards", "hazard_positions"))
        or _extract_points(obs, ("hazards", "hazard_positions"))
    )
    velocity_vec = (
        _extract_vector(info, ("velocity", "vel", "robot_vel"))
        or _extract_vector(obs, ("velocity", "vel", "robot_vel"))
    )

    goal_dist = _goal_distance(agent_pos, goal_pos, obs, info)
    hazard_dist = _nearest_distance(agent_pos, hazards)
    near_obstacle = _near_obstacle_signal(agent_pos, hazards, obs, info)
    velocity = _velocity_magnitude(velocity_vec, prev_obs, obs, info)

    state = {
        "hazard_dist": hazard_dist,
        "goal_dist": goal_dist,
        "velocity": velocity,
        "near_obstacle": near_obstacle,
        "near_human": float(_scalar(info, ("near_human",), default=0.0)),
        "zone_a": float(_scalar(info, ("zone_a",), default=0.0)),
        "zone_b": float(_scalar(info, ("zone_b",), default=0.0)),
        "zone_c": float(_scalar(info, ("zone_c",), default=0.0)),
        "carrying": float(_scalar(info, ("carrying",), default=0.0)),
    }
    return state


def _goal_distance(agent_pos, goal_pos, obs: Any, info: dict[str, Any]) -> float:
    explicit = _scalar(info, ("goal_dist", "goal_distance"), default=None)
    if explicit is None:
        explicit = _scalar(obs, ("goal_dist", "goal_distance"), default=None)
    if explicit is not None:
        return float(explicit)
    if agent_pos is not None and goal_pos is not None:
        return float(np.linalg.norm(np.asarray(agent_pos) - np.asarray(goal_pos)))
    arr = _as_array(obs)
    if arr is not None and arr.size >= 2:
        return float(np.linalg.norm(arr[:2]))
    return 1.0


def _near_obstacle_signal(agent_pos, hazards, obs: Any, info: dict[str, Any]) -> float:
    explicit = _scalar(info, ("near_obstacle", "obstacle_margin"), default=None)
    if explicit is None:
        explicit = _scalar(obs, ("near_obstacle", "obstacle_margin"), default=None)
    if explicit is not None:
        return float(explicit)
    hazard_dist = _nearest_distance(agent_pos, hazards)
    if math.isfinite(hazard_dist):
        return float(hazard_dist)
    arr = _as_array(obs)
    if arr is not None and arr.size > 4:
        return float(arr[4])
    return 1.0


def _velocity_magnitude(velocity_vec, prev_obs: Any, obs: Any, info: dict[str, Any]) -> float:
    explicit = _scalar(info, ("speed", "velocity_mag"), default=None)
    if explicit is not None:
        return float(explicit)
    if velocity_vec is not None:
        return float(np.linalg.norm(np.asarray(velocity_vec, dtype=float)))
    prev_arr = _as_array(prev_obs)
    curr_arr = _as_array(obs)
    if prev_arr is not None and curr_arr is not None and prev_arr.size >= 2 and curr_arr.size >= 2:
        delta = curr_arr[:2] - prev_arr[:2]
        return float(np.linalg.norm(delta))
    return 0.0


def _nearest_distance(agent_pos, points) -> float:
    if agent_pos is None or not points:
        return 1.0
    ap = np.asarray(agent_pos, dtype=float)
    return float(min(np.linalg.norm(ap - np.asarray(point, dtype=float)) for point in points))


def _extract_vector(source: Any, keys: tuple[str, ...]):
    if not isinstance(source, dict):
        return None
    for key in keys:
        if key in source:
            value = np.asarray(source[key], dtype=float).reshape(-1)
            if value.size >= 2:
                return value[:2]
    return None


def _extract_points(source: Any, keys: tuple[str, ...]):
    if not isinstance(source, dict):
        return None
    for key in keys:
        if key not in source:
            continue
        value = np.asarray(source[key], dtype=float)
        if value.ndim == 2 and value.shape[1] >= 2:
            return [row[:2] for row in value]
    return None


def _scalar(source: Any, keys: tuple[str, ...], default=None):
    if isinstance(source, dict):
        for key in keys:
            if key in source:
                value = np.asarray(source[key]).reshape(-1)
                if value.size:
                    return float(value[0])
    return default


def _as_array(obs: Any):
    if isinstance(obs, dict):
        return None
    arr = np.asarray(obs, dtype=float).reshape(-1)
    return arr if arr.size else None
