from __future__ import annotations

import math
from typing import Any

import numpy as np


class CarryingTracker:
    """
    Stateful pick-and-place carrying signal.

    Enters carrying=1.0 when the agent steps inside button_1's radius,
    drops to carrying=0.0 when it steps inside button_2's radius, and
    holds its previous value otherwise.  Create once per episode (or call
    reset() at episode boundaries) and pass into safety_point_goal_adapter
    on every step.

    Example
    -------
    config = {"button1_pos": [1, 2], "button2_pos": [-1, -2], "button_radius": 0.3}
    tracker = CarryingTracker.from_config(config)
    state = safety_point_goal_adapter(obs, info=info, tracker=tracker)
    """

    def __init__(
        self,
        button1_pos: np.ndarray,
        button2_pos: np.ndarray,
        button_radius: float,
    ) -> None:
        self._b1 = np.asarray(button1_pos, dtype=float).reshape(-1)[:2]
        self._b2 = np.asarray(button2_pos, dtype=float).reshape(-1)[:2]
        self._r = float(button_radius)
        self._carrying = 0.0

    @classmethod
    def from_config(cls, config: dict) -> CarryingTracker:
        return cls(
            button1_pos=config["button1_pos"],
            button2_pos=config["button2_pos"],
            button_radius=config["button_radius"],
        )

    def reset(self) -> None:
        """Reset to no-carrying state at the start of a new episode."""
        self._carrying = 0.0

    def update(self, agent_pos) -> float:
        """
        Update carrying state from a single agent position and return the new value.

        Parameters
        ----------
        agent_pos : array-like, shape (2,) or longer — only the first two elements are used.

        Returns
        -------
        0.0 or 1.0
        """
        pos = np.asarray(agent_pos, dtype=float).reshape(-1)[:2]
        if np.linalg.norm(pos - self._b1) < self._r:
            self._carrying = 1.0
        elif np.linalg.norm(pos - self._b2) < self._r:
            self._carrying = 0.0
        return self._carrying

    def update_batch(self, agent_positions) -> np.ndarray:
        """
        Vectorised variant for N agents simultaneously.

        Parameters
        ----------
        agent_positions : array-like, shape (N, 2+)

        Returns
        -------
        np.ndarray of shape (N,) with 0.0 / 1.0 values.
        The tracker's own scalar state is NOT modified by this call.
        """
        pos = np.asarray(agent_positions, dtype=float)[:, :2]          # (N, 2)
        in_b1 = np.linalg.norm(pos - self._b1, axis=1) < self._r      # (N,) bool
        in_b2 = np.linalg.norm(pos - self._b2, axis=1) < self._r      # (N,) bool
        carrying = np.full(len(pos), self._carrying)
        carrying[in_b2] = 0.0
        carrying[in_b1] = 1.0                                           # b1 wins if both true
        return carrying.astype(float)

    @property
    def carrying(self) -> float:
        return self._carrying


def safety_point_goal_adapter(
    obs: Any,
    *,
    info: dict[str, Any] | None = None,
    prev_obs: Any = None,
    action: Any = None,
    tracker: CarryingTracker | None = None,
) -> dict[str, float]:
    """
    Convert a SafetyPointGoal-style observation into the semantic state dict used
    by SAFEWORLD V2 verification.

    This adapter is intentionally defensive because different Safety-Gym builds
    expose slightly different observation/info layouts. It prefers `info` when
    available, then falls back to dict observations, and finally to coarse array
    heuristics when only a flat vector is available.

    Parameters
    ----------
    tracker : CarryingTracker, optional
        If provided, carrying is derived from agent position using the tracker's
        stateful button-zone logic.  If None, carrying falls back to info["carrying"]
        (or 0.0 when absent).
    """
    info = info or {}

    agent_pos = (
        _extract_vector(info, ("agent_pos", "robot_pos", "position"))
        or _extract_vector(obs, ("agent_pos", "robot_pos", "position"))
        or _obs_xy(obs)
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
    hazard_dist = _hazard_margin(agent_pos, hazards, info)
    near_obstacle = _near_obstacle_signal(agent_pos, hazards, obs, info)
    velocity = _velocity_magnitude(velocity_vec, prev_obs, obs, info)

    if tracker is not None and agent_pos is not None:
        carrying = tracker.update(agent_pos)
    else:
        carrying = float(_scalar(info, ("carrying",), default=0.0))

    state = {
        "hazard_dist": hazard_dist,
        "goal_dist": goal_dist,
        "velocity": velocity,
        "near_obstacle": near_obstacle,
        "near_human": float(_scalar(info, ("near_human",), default=0.0)),
        "zone_a": float(_scalar(info, ("zone_a",), default=0.0)),
        "zone_b": float(_scalar(info, ("zone_b",), default=0.0)),
        "zone_c": float(_scalar(info, ("zone_c",), default=0.0)),
        "carrying": carrying,
    }
    return state


def _obs_xy(obs: Any):
    """Extract the first two elements of a flat obs array as agent xy."""
    arr = _as_array(obs)
    if arr is not None and arr.size >= 2:
        return arr[:2]
    return None


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


def _hazard_margin(agent_pos, hazards, info: dict[str, Any]) -> float:
    explicit = _scalar(info, ("hazard_dist", "hazard_distance"), default=None)
    if explicit is not None:
        return float(explicit)
    cost = _scalar(info, ("cost_hazards", "cost"), default=None)
    if cost is not None:
        return 0.5 - float(cost)
    return _nearest_distance(agent_pos, hazards)


def _velocity_magnitude(velocity_vec, prev_obs: Any, obs: Any, info: dict[str, Any]) -> float:
    explicit = _scalar(info, ("speed", "velocity_mag"), default=None)
    if explicit is not None:
        return float(explicit)
    if velocity_vec is not None:
        return float(np.linalg.norm(np.asarray(velocity_vec, dtype=float)))
    curr_arr = _as_array(obs)
    if curr_arr is not None and curr_arr.size >= 6:
        return float(np.linalg.norm(curr_arr[3:6]))
    prev_arr = _as_array(prev_obs)
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
