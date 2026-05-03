from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from environment import EnvWrapper
from environment.adapters import safety_point_goal_adapter
from configs.settings import RolloutConfig
from .base import WorldModelWrapper


DEFAULT_SNAPSHOT = Path(
    "/home/chenmg93@netid.washington.edu/.cache/huggingface/hub/"
    "models--helenant--simple_pointgoal2_worldmodel/snapshots/"
    "d9158d06d2eea9940a354c02eee63bf175e08d21/simple_pointgoal2_worldmodel"
)
DEFAULT_CHECKPOINT = DEFAULT_SNAPSHOT / "checkpoints/simple_world_model.pt"


class SimplePointGoal2Model(nn.Module):
    """
    Deterministic MLP world model:
        (obs, action) -> delta_obs, reward, cost, done_logit

    The 256-d output of `backbone` is the smallest latent-like representation.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head_delta_obs = nn.Linear(hidden_dim, obs_dim)
        self.head_reward = nn.Linear(hidden_dim, 1)
        self.head_cost = nn.Linear(hidden_dim, 1)
        self.head_done = nn.Linear(hidden_dim, 1)

    def encode(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.backbone(torch.cat([obs, action], dim=-1))

    def decode_from_latent(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "delta_obs": self.head_delta_obs(h),
            "reward": self.head_reward(h).squeeze(-1),
            "cost": self.head_cost(h).squeeze(-1),
            "done_logit": self.head_done(h).squeeze(-1),
        }

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.decode_from_latent(self.encode(obs, action))


@dataclass
class LatentUnderstanding:
    latent: np.ndarray
    reward: float
    cost: float
    done_probability: float
    hazard_avoidance_score: float
    goal_achievement_score: float
    top_cost_latent_dims: list[tuple[int, float]]
    top_reward_latent_dims: list[tuple[int, float]]
    top_done_latent_dims: list[tuple[int, float]]


class SimplePointGoal2WorldModelWrapper(WorldModelWrapper):
    """
    Wrapper for `helenant/simple_pointgoal2_worldmodel`.

    It exposes the model's 256-d hidden activation through `encode()` and can
    roll the deterministic MLP forward from a flat 60-d observation plus 2-d
    action. For SAFEWORLD, reward/cost/done are interpreted as the model's
    learned signals for goal achievement, hazard/contact risk, and termination.
    """

    def __init__(self, config: RolloutConfig | None = None):
        super().__init__(config)
        self.model: SimplePointGoal2Model | None = None
        self.device = torch.device(self.config.device)
        self.checkpoint_path = DEFAULT_CHECKPOINT
        self.env_name = "SafetyPointGoal2Gymnasium-v0"
        self.obs_dim: int | None = None
        self.act_dim: int | None = None
        self.hidden_dim: int | None = None
        self._obs_mean: np.ndarray | None = None
        self._obs_std: np.ndarray | None = None
        self._act_mean: np.ndarray | None = None
        self._act_std: np.ndarray | None = None
        self.env: EnvWrapper | None = None

    def load(self, **kwargs) -> None:
        checkpoint_path = kwargs.get("checkpoint_path")
        if checkpoint_path is None:
            checkpoint_path = self.config.extra.get("checkpoint_path", self.checkpoint_path)
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.device = torch.device(kwargs.get("device", self.config.device))
        self.env_name = kwargs.get(
            "env_name",
            self.config.extra.get("env_name", self.env_name),
        )

        _install_numpy_core_pickle_alias()
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.obs_dim = int(ckpt["obs_dim"])
        self.act_dim = int(ckpt["act_dim"])
        self.hidden_dim = int(ckpt["hidden_dim"])
        self.model = SimplePointGoal2Model(self.obs_dim, self.act_dim, self.hidden_dim)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self._obs_mean = np.asarray(ckpt["obs_mean"], dtype=np.float32)
        self._obs_std = np.asarray(ckpt["obs_std"], dtype=np.float32)
        self._act_mean = np.asarray(ckpt["act_mean"], dtype=np.float32)
        self._act_std = np.asarray(ckpt["act_std"], dtype=np.float32)

    def encode(self, obs: Any, action: Any) -> np.ndarray:
        self._ensure_loaded()
        obs_t, act_t, squeeze = self._prepare_inputs(obs, action)
        assert self.model is not None
        with torch.no_grad():
            h = self.model.encode(obs_t, act_t).cpu().numpy()
        return h[0] if squeeze else h

    def predict(self, obs: Any, action: Any) -> dict[str, Any]:
        self._ensure_loaded()
        obs_np = self._as_batch(obs, self.obs_dim, "obs")
        obs_t, act_t, squeeze = self._prepare_inputs(obs, action)
        assert self.model is not None
        assert self._obs_std is not None
        with torch.no_grad():
            out = self.model(obs_t, act_t)
        delta_obs_raw = out["delta_obs"].cpu().numpy() * self._obs_std
        next_obs = obs_np + delta_obs_raw
        result = {
            "next_obs": next_obs,
            "reward": out["reward"].cpu().numpy(),
            "cost": out["cost"].cpu().numpy(),
            "done_probability": torch.sigmoid(out["done_logit"]).cpu().numpy(),
        }
        if squeeze:
            return {
                "next_obs": result["next_obs"][0],
                "reward": float(result["reward"][0]),
                "cost": float(result["cost"][0]),
                "done_probability": float(result["done_probability"][0]),
            }
        return result

    def understand(self, obs: Any, action: Any, top_k: int = 8) -> LatentUnderstanding:
        """
        Explain one transition through latent head sensitivities.

        `top_*_latent_dims` are absolute gradients of each scalar head with
        respect to h. Larger entries are the hidden dimensions that most affect
        the model's hazard-cost, goal-reward, or done predictions locally.
        """
        self._ensure_loaded()
        obs_t, act_t, _ = self._prepare_inputs(obs, action)
        assert self.model is not None

        with torch.enable_grad():
            h = self.model.encode(obs_t[:1], act_t[:1]).detach().requires_grad_(True)
            out = self.model.decode_from_latent(h)
            reward = out["reward"][0]
            cost = out["cost"][0]
            done_logit = out["done_logit"][0]
            done_prob = torch.sigmoid(done_logit)
            cost_grad = torch.autograd.grad(cost, h, retain_graph=True)[0][0]
            reward_grad = torch.autograd.grad(reward, h, retain_graph=True)[0][0]
            done_grad = torch.autograd.grad(done_logit, h)[0][0]

        cost_value = float(cost.detach().cpu())
        reward_value = float(reward.detach().cpu())
        done_value = float(done_prob.detach().cpu())
        return LatentUnderstanding(
            latent=h.detach().cpu().numpy()[0],
            reward=reward_value,
            cost=cost_value,
            done_probability=done_value,
            hazard_avoidance_score=float(-cost_value),
            goal_achievement_score=reward_value,
            top_cost_latent_dims=self._top_abs_dims(cost_grad, top_k),
            top_reward_latent_dims=self._top_abs_dims(reward_grad, top_k),
            top_done_latent_dims=self._top_abs_dims(done_grad, top_k),
        )

    def sample_rollouts(
        self,
        config: RolloutConfig | None = None,
    ) -> list[list[dict[str, float]]]:
        cfg = config or self.config
        self._ensure_loaded()

        trajectories: list[list[dict[str, float]]] = []
        for i in range(cfg.n_rollouts):
            rng = np.random.default_rng(cfg.seed + i)
            obs = self._initial_obs(rng)
            traj: list[dict[str, float]] = []
            for _ in range(cfg.horizon):
                action = self._sample_action(rng)
                pred = self.predict(obs, action)
                next_obs = np.asarray(pred["next_obs"], dtype=np.float32)
                state = self._model_state(next_obs, pred)
                traj.append(state)
                obs = next_obs
                if cfg.extra.get("stop_on_done", False) and pred["done_probability"] >= cfg.extra.get("done_threshold", 0.5):
                    break
            trajectories.append(traj)
        return trajectories

    def sample_paired_rollouts(
        self,
        config: RolloutConfig | None = None,
    ) -> list[tuple[list[dict[str, float]], list[dict[str, float]]]]:
        """
        Roll the model and Gymnasium env from the same actions.

        This requires `safety_gymnasium` to be installed so the environment ID
        `SafetyPointGoal2Gymnasium-v0` is registered.
        """
        cfg = config or self.config
        self._ensure_loaded()
        self._ensure_env()
        assert self.env is not None

        pairs: list[tuple[list[dict[str, float]], list[dict[str, float]]]] = []
        for i in range(cfg.n_rollouts):
            rng = np.random.default_rng(cfg.seed + i)
            obs, info = self.env.reset(**cfg.extra.get("reset_kwargs", {}))
            obs_arr = self._flatten_obs(obs)
            model_traj: list[dict[str, float]] = []
            env_traj: list[dict[str, float]] = []
            prev_obs = obs
            for _ in range(cfg.horizon):
                action = self._sample_action(rng)
                pred = self.predict(obs_arr, action)
                model_next = np.asarray(pred["next_obs"], dtype=np.float32)
                model_traj.append(self._model_state(model_next, pred))

                obs, _, terminated, truncated, info = self.env.step(action)
                env_traj.append(
                    safety_point_goal_adapter(obs, info=info, prev_obs=prev_obs, action=action)
                )
                prev_obs = obs
                obs_arr = self._flatten_obs(obs)
                if terminated or truncated:
                    break
            pairs.append((model_traj, env_traj))
        return pairs

    def decode_and_replay(
        self,
        config: RolloutConfig | None = None,
        closed_loop: bool = False,
    ) -> list[list]:
        """
        Run N rollouts, replay the exact same actions in the real simulator,
        and return per-step ReplayStep records.

        This model is obs-space based: predict(obs, action) → next_obs directly,
        so decode is free — model_obs IS the decoder output.

        Parameters
        ----------
        closed_loop : False (default) — feed real env obs to model each step;
                      measures one-step prediction accuracy.
                      True — feed model's own next_obs back; measures drift.
        """
        from .base import ReplayStep

        cfg = config or self.config
        self._ensure_loaded()
        self._ensure_env()
        assert self.env is not None

        all_rollouts: list[list[ReplayStep]] = []

        for i in range(cfg.n_rollouts):
            rng       = np.random.default_rng(cfg.seed + i)
            obs, info = self.env.reset(**cfg.extra.get("reset_kwargs", {}))
            obs_arr   = self._flatten_obs(obs)
            model_obs_arr = obs_arr.copy()   # model starts from same initial obs
            prev_obs  = obs
            steps: list[ReplayStep] = []

            for t in range(cfg.horizon):
                action = self._sample_action(rng)

                # ── model side: predict in obs space ─────────────────────────
                pred           = self.predict(model_obs_arr, action)
                model_next_obs = np.asarray(pred["next_obs"], dtype=np.float32)
                model_semantic = self._model_state(model_next_obs, pred)

                # ── env side: real simulator step ─────────────────────────────
                obs, _, terminated, truncated, info = self.env.step(action)
                env_obs_arr  = self._flatten_obs(obs)
                env_semantic = safety_point_goal_adapter(
                    obs, info=info, prev_obs=prev_obs, action=action
                )

                # ── per-step obs error ────────────────────────────────────────
                obs_rmse = float(
                    np.sqrt(np.mean((model_next_obs - env_obs_arr) ** 2))
                )

                steps.append(ReplayStep(
                    t              = t,
                    action         = np.asarray(action, dtype=np.float32),
                    model_obs      = model_next_obs,
                    env_obs        = env_obs_arr,
                    model_semantic = model_semantic,
                    env_semantic   = env_semantic,
                    obs_rmse       = obs_rmse,
                ))

                prev_obs = obs
                if closed_loop:
                    model_obs_arr = model_next_obs   # drift mode
                else:
                    model_obs_arr = env_obs_arr      # open-loop: ground truth each step

                if terminated or truncated:
                    break

            all_rollouts.append(steps)

        return all_rollouts

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
            "model_reward",
            "model_cost",
            "done_probability",
            "hazard_avoidance_score",
            "goal_achievement_score",
        ]

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.load()

    def _ensure_env(self) -> None:
        if self.env is None:
            env_kwargs = dict(self.config.extra.get("env_kwargs", {}))
            if self.env_name.startswith("SafetyPointGoal2"):
                env_kwargs.setdefault("use_safety_gymnasium", True)
                env_kwargs.setdefault("safety_env_name", "SafetyPointGoal2-v0")
            self.env = EnvWrapper(self.env_name, **env_kwargs)

    def _prepare_inputs(self, obs: Any, action: Any) -> tuple[torch.Tensor, torch.Tensor, bool]:
        obs_np = self._as_batch(obs, self.obs_dim, "obs")
        act_np = self._as_batch(action, self.act_dim, "action")
        squeeze = np.asarray(obs, dtype=np.float32).ndim == 1
        assert self._obs_mean is not None and self._obs_std is not None
        assert self._act_mean is not None and self._act_std is not None
        obs_norm = (obs_np - self._obs_mean) / self._obs_std
        act_norm = (act_np - self._act_mean) / self._act_std
        obs_t = torch.tensor(obs_norm, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(act_norm, dtype=torch.float32, device=self.device)
        return obs_t, act_t, squeeze

    @staticmethod
    def _as_batch(value: Any, expected_dim: int | None, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None]
        if expected_dim is not None and arr.shape[-1] != expected_dim:
            raise ValueError(f"{name} has dim {arr.shape[-1]}, expected {expected_dim}.")
        return arr

    def _initial_obs(self, rng: np.random.Generator) -> np.ndarray:
        assert self.obs_dim is not None
        assert self._obs_mean is not None and self._obs_std is not None
        if "initial_obs" in self.config.extra:
            return np.asarray(self.config.extra["initial_obs"], dtype=np.float32)
        z = rng.normal(size=self.obs_dim).astype(np.float32)
        return self._obs_mean + z * self._obs_std

    def _sample_action(self, rng: np.random.Generator) -> np.ndarray:
        assert self.act_dim is not None
        mode = self.config.extra.get("action_source", self.config.action_source)
        if mode == "zeros":
            return np.zeros(self.act_dim, dtype=np.float32)
        if mode == "adversarial":
            action = np.zeros(self.act_dim, dtype=np.float32)
            action[0] = 1.0
            return action
        if mode == "env" and self.env is not None:
            return np.asarray(self.env.action_space.sample(), dtype=np.float32)
        low = np.asarray(self.config.extra.get("action_low", -1.0), dtype=np.float32)
        high = np.asarray(self.config.extra.get("action_high", 1.0), dtype=np.float32)
        return rng.uniform(low, high, size=self.act_dim).astype(np.float32)

    def _model_state(self, obs: np.ndarray, pred: dict[str, Any]) -> dict[str, float]:
        semantic = safety_point_goal_adapter(obs)
        reward = float(pred["reward"])
        cost = float(pred["cost"])
        done_probability = float(pred["done_probability"])
        hazard_margin = 0.5 - cost
        semantic.update(
            {
                "hazard_dist": hazard_margin,
                "near_obstacle": hazard_margin,
                "model_reward": reward,
                "model_cost": cost,
                "done_probability": done_probability,
                "hazard_avoidance_score": -cost,
                "goal_achievement_score": reward,
            }
        )
        return semantic

    def _flatten_obs(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            pieces = [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in sorted(obs)]
            arr = np.concatenate(pieces)
        else:
            arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.obs_dim is not None and arr.size != self.obs_dim:
            raise ValueError(f"Environment observation has dim {arr.size}, expected {self.obs_dim}.")
        return arr

    @staticmethod
    def _top_abs_dims(values: torch.Tensor, top_k: int) -> list[tuple[int, float]]:
        arr = values.detach().abs().cpu().numpy()
        k = max(0, min(int(top_k), arr.size))
        if k == 0:
            return []
        idx = np.argpartition(-arr, np.arange(k))[:k]
        idx = idx[np.argsort(-arr[idx])]
        return [(int(i), float(arr[i])) for i in idx]


def _install_numpy_core_pickle_alias() -> None:
    """
    Allow checkpoints pickled with NumPy 2.x (`numpy._core.*`) to load under
    NumPy 1.x (`numpy.core.*`).
    """
    if importlib.util.find_spec("numpy._core") is not None:
        return
    try:
        import numpy.core as np_core
        import numpy.core.multiarray as np_multiarray
        import numpy.core.numeric as np_numeric
        import numpy.core.umath as np_umath
    except ImportError:
        return
    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)
    sys.modules.setdefault("numpy._core.umath", np_umath)
