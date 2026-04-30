from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


SAFEGOALPOINT2_ENV_ID = "SafetyPointGoal2Gymnasium-v0"
SAFETY_SAFEGOALPOINT2_ENV_ID = "SafetyPointGoal2-v0"
OFFLINE_SAFEGOALPOINT2_ENV_ID = "OfflinePointGoal2Gymnasium-v0"


@dataclass(frozen=True)
class SafeGoalPoint2EnvStatus:
    env_id: str
    gymnasium_installed: bool
    safety_gymnasium_installed: bool
    dsrl_installed: bool
    h5py_installed: bool
    registered: bool
    error: str | None = None

    @property
    def can_make_env(self) -> bool:
        return self.registered and self.error is None

    @property
    def missing_packages(self) -> list[str]:
        missing = []
        if not self.gymnasium_installed:
            missing.append("gymnasium")
        if not self.safety_gymnasium_installed:
            missing.append("safety-gymnasium")
        return missing


def check_safegoalpoint2_env(env_id: str = SAFEGOALPOINT2_ENV_ID) -> SafeGoalPoint2EnvStatus:
    """
    Check whether the SafetyPointGoal2 environment can be created in the active
    Python environment. Prefer the safety_gymnasium constructor because recent
    Safety-Gymnasium installs expose SafetyPointGoal2 as `SafetyPointGoal2-v0`
    and convert it with `SafetyGymnasium2Gymnasium`.
    """
    gymnasium_installed = find_spec("gymnasium") is not None
    safety_gymnasium_installed = find_spec("safety_gymnasium") is not None
    dsrl_installed = find_spec("dsrl") is not None
    h5py_installed = find_spec("h5py") is not None

    if not gymnasium_installed and not safety_gymnasium_installed:
        return SafeGoalPoint2EnvStatus(
            env_id=env_id,
            gymnasium_installed=False,
            safety_gymnasium_installed=safety_gymnasium_installed,
            dsrl_installed=dsrl_installed,
            h5py_installed=h5py_installed,
            registered=False,
            error="neither gymnasium nor safety_gymnasium is installed",
        )

    if safety_gymnasium_installed:
        safety_env_id = _safety_env_id(env_id)
        try:
            import safety_gymnasium
            from safety_gymnasium.wrappers import SafetyGymnasium2Gymnasium

            env = SafetyGymnasium2Gymnasium(safety_gymnasium.make(safety_env_id))
            env.close()
            return SafeGoalPoint2EnvStatus(
                env_id=safety_env_id,
                gymnasium_installed=gymnasium_installed,
                safety_gymnasium_installed=True,
                dsrl_installed=dsrl_installed,
                h5py_installed=h5py_installed,
                registered=True,
            )
        except Exception as exc:
            safety_error = f"{type(exc).__name__}: {exc}"
    else:
        safety_error = "safety_gymnasium is not installed"

    if not gymnasium_installed:
        return SafeGoalPoint2EnvStatus(
            env_id=env_id,
            gymnasium_installed=False,
            safety_gymnasium_installed=safety_gymnasium_installed,
            dsrl_installed=dsrl_installed,
            h5py_installed=h5py_installed,
            registered=False,
            error=safety_error,
        )

    try:
        import gymnasium as gym

        if safety_gymnasium_installed:
            import safety_gymnasium  # noqa: F401

        registered = env_id in {spec.id for spec in gym.envs.registry.values()}
        if not registered:
            return SafeGoalPoint2EnvStatus(
                env_id=env_id,
                gymnasium_installed=True,
                safety_gymnasium_installed=safety_gymnasium_installed,
                dsrl_installed=dsrl_installed,
                h5py_installed=h5py_installed,
                registered=False,
                error=f"{env_id} is not registered",
            )
        env = gym.make(env_id)
        env.close()
        return SafeGoalPoint2EnvStatus(
            env_id=env_id,
            gymnasium_installed=True,
            safety_gymnasium_installed=safety_gymnasium_installed,
            dsrl_installed=dsrl_installed,
            h5py_installed=h5py_installed,
            registered=True,
        )
    except Exception as exc:
        return SafeGoalPoint2EnvStatus(
            env_id=env_id,
            gymnasium_installed=True,
            safety_gymnasium_installed=safety_gymnasium_installed,
            dsrl_installed=dsrl_installed,
            h5py_installed=h5py_installed,
            registered=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _safety_env_id(env_id: str) -> str:
    if env_id.endswith("Gymnasium-v0"):
        return f"{env_id[:-len('Gymnasium-v0')]}-v0"
    return env_id
