from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .rollout import RolloutConfig


DEFAULT_SETTINGS_DIR = Path(__file__).resolve().parent


def settings_path_for_model(model_type: str) -> Path:
    return DEFAULT_SETTINGS_DIR / f"{model_type}.json"


def load_settings_config(path_or_obj: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if path_or_obj is None:
        return {}
    if isinstance(path_or_obj, dict):
        return dict(path_or_obj)
    path = Path(path_or_obj)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Settings config must be a JSON object: {path}")
    data["_source_path"] = str(path)
    return data


def build_rollout_config(settings: dict[str, Any]) -> RolloutConfig:
    rollout = dict(settings.get("rollout", {}))
    model = dict(settings.get("model", {}))
    environment = dict(settings.get("environment", {}))
    extra = dict(settings.get("extra", {}))

    model_type = model.get("type")
    env_name = environment.get("name")
    if model_type is not None:
        extra.setdefault("model_type", model_type)
    if env_name is not None:
        extra.setdefault("env_name", env_name)
    if "kwargs" in environment:
        extra.setdefault("env_kwargs", dict(environment["kwargs"] or {}))
    if "reset_kwargs" in environment:
        extra.setdefault("reset_kwargs", dict(environment["reset_kwargs"] or {}))
    if model.get("checkpoint_path") is not None:
        extra.setdefault("checkpoint_path", model["checkpoint_path"])

    return RolloutConfig(
        horizon=int(rollout.get("horizon", 50)),
        n_rollouts=int(rollout.get("n_rollouts", rollout.get("num_samples", 20))),
        action_source=str(rollout.get("action_source", "random")),
        seed=int(rollout.get("seed", 0)),
        device=str(rollout.get("device", "cpu")),
        extra=extra,
    )
