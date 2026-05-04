"""
wrappers/goal2_dreamer_wrapper.py

SAFEWORLD wrapper for the Goal2 DreamerV3-style offline world model
(FanZhangg/dreamv3-learned2, trained with SafeWorld-Benchmark-main).

Model architecture (from training code)
────────────────────────────────────────
  Encoder   obs(60) → embed(512)          [3-layer MLP, SiLU+LN]
  RSSM      h=512 (GRUCell)  z=32×32=1024 (flat one-hot categorical)
              prior:     h       → z_flat  (imagination — no obs needed)
              posterior: h,embed → z_flat  (encoding real obs)
            feat = cat(h, z_flat) = 1536-dim
  Decoder
              obs_decoder  feat → obs(60)
              reward_head  feat → scalar
              aux_decoder  feat → {cost, speed, goal_distance,
                                   nearest_hazard_distance,
                                   nearest_vase_distance, human_distance}

SAFEWORLD latent protocol
─────────────────────────
  Paper (§3): "d = 32 corresponding to the deterministic GRU hidden state h_t.
  The stochastic component z_t … is not used directly because threshold-based
  predicates require continuous-valued inputs."

  This wrapper exposes two modes via `latent_mode`:

    "h_only" (default, paper-faithful)
        _feat(h, z) = cat(h, zeros(z_flat))   — z zeroed before aux decoder
        RSSM transitions still use the real (h, z) pair for accurate dynamics.
        The aux decoder receives h with z=0, which is a linear approximation of
        the h-only decoder that would require retraining.  Paper reports a 3–7 %
        F1 drop vs. the full-feat decoder; acceptable for threshold-predicate use.

    "feat" (full, non-paper)
        _feat(h, z) = cat(h, z)                — current full 1536-dim feature
        Maximises aux-decoder quality but passes categorical z through
        threshold predicates (technically violates the paper's continuous
        assumption — use only for ablation / comparison).

AP mapping  (aux decoder → SAFEWORLD AP dict)
─────────────────────────────────────────────
  nearest_hazard_distance  →  hazard_dist   = dist - hazard_safe_dist
  speed                    →  velocity
  goal_distance            →  goal_dist     = dist - goal_reach_radius
  nearest_vase_distance    →  near_obstacle = dist - obstacle_safe_dist
  human_distance           →  near_human    = human_near_threshold - dist

Thresholds come from the environment config JSON
(configs/environments/goal2.json) via `env_config["ap_thresholds"]`.

model_dir must point to the dreamer_world_model package directory:
  SafeWorld-Benchmark-main/SafeWorld-Benchmark-main/training/dreamer_world_model
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from environment.adapters import safety_point_goal_adapter, CarryingTracker
from configs.settings import RolloutConfig
from .base import ReplayStep, WorldModelWrapper


# ─── model loading ────────────────────────────────────────────────────────────

def _import_dreamer_package(model_dir: str | Path):
    """
    Import WorldModel and WorldModelConfig from the dreamer_world_model package.

    model_dir must be the dreamer_world_model/ directory itself (it has __init__.py).
    We add its parent to sys.path and import the package by its directory name.
    """
    model_dir = Path(model_dir).resolve()
    parent    = str(model_dir.parent)
    pkg_name  = model_dir.name   # "dreamer_world_model"

    if parent not in sys.path:
        sys.path.insert(0, parent)

    pkg     = importlib.import_module(pkg_name)
    wm_mod  = importlib.import_module(f"{pkg_name}.world_model")
    cfg_mod = importlib.import_module(f"{pkg_name}.config")
    return wm_mod.WorldModel, cfg_mod.WorldModelConfig


# ─── main wrapper ─────────────────────────────────────────────────────────────

class Goal2WorldModelWrapper(WorldModelWrapper):
    """
    SAFEWORLD wrapper for the Goal2 DreamerV3-style offline world model.

    Quick start
    -----------
    from wrappers import Goal2WorldModelWrapper
    from configs.settings import RolloutConfig

    cfg = RolloutConfig(
        horizon=50, n_rollouts=20, seed=42,
        extra={
            "checkpoint_path": "/path/to/ckpt_0500000.pt",
            "model_dir":       "/path/to/dreamer_world_model",
        }
    )
    w = Goal2WorldModelWrapper(cfg)
    w.load(env_config=json.load(open("configs/environments/goal2.json")))
    trajectories = w.sample_rollouts()
    """

    def __init__(self, config: RolloutConfig | None = None, latent_mode: str = "h_only"):
        """
        Parameters
        ----------
        config       : RolloutConfig
        latent_mode  : "h_only" (paper-faithful, default) or "feat" (full cat(h,z))
                       See module docstring for details.
        """
        super().__init__(config)
        self.model:     Any          = None
        self.device:    torch.device = torch.device("cpu")
        self.env_name:  str          = "SafetyPointGoal2Gymnasium-v0"
        self.env_config: dict        = {}

        if latent_mode not in ("h_only", "feat"):
            raise ValueError(f"latent_mode must be 'h_only' or 'feat', got {latent_mode!r}")
        self._latent_mode: str = latent_mode

        # Resolved from WorldModelConfig after load()
        self._deter_dim: int = 512
        self._z_flat:    int = 1024   # stoch_dim * stoch_classes

        # Normalisation stats (none in this checkpoint)
        self._obs_mean: np.ndarray | None = None
        self._obs_std:  np.ndarray | None = None

        # AP conversion thresholds (from env_config)
        self._ap_thresholds: dict[str, float] = {}

        # Gymnasium env for paired rollouts / replay
        self._gym_env: Any = None

        # CarryingTracker (activated when env_config has button_zones.enabled=true)
        self._tracker: CarryingTracker | None = None

    # ── loading ───────────────────────────────────────────────────────────────

    def load(self, **kwargs) -> None:
        """
        Load checkpoint and build the WorldModel.

        Parameters
        ----------
        checkpoint_path : str | Path      path to .pt file
        model_dir       : str | Path      dreamer_world_model package directory
        env_config      : dict            loaded goal2.json  (env params + model_arch)
        device          : str             "cpu" | "cuda" | "cuda:0"
        env_name        : str             Gymnasium env ID override
        """
        ckpt_path  = kwargs.get("checkpoint_path") or self.config.extra.get("checkpoint_path")
        model_dir  = kwargs.get("model_dir")       or self.config.extra.get("model_dir")
        env_config = kwargs.get("env_config")      or {}
        device_str = kwargs.get("device")          or self.config.extra.get("device", "cpu")

        self.device     = torch.device(device_str)
        self.env_name   = kwargs.get("env_name") or self.config.extra.get("env_name", self.env_name)
        self.env_config = env_config

        # latent_mode can be overridden at load() time or via env_config
        mode = (
            kwargs.get("latent_mode")
            or env_config.get("latent_mode")
            or self._latent_mode
        )
        if mode not in ("h_only", "feat"):
            raise ValueError(f"latent_mode must be 'h_only' or 'feat', got {mode!r}")
        self._latent_mode = mode

        self._ap_thresholds = env_config.get("ap_thresholds", {})

        # CarryingTracker
        zones = env_config.get("button_zones", {})
        self._tracker = CarryingTracker.from_config(zones) if zones.get("enabled") else None

        # ── load checkpoint ───────────────────────────────────────────────────
        if ckpt_path is None:
            raise ValueError(
                "Goal2WorldModelWrapper.load() requires checkpoint_path. "
                "Pass it as a kwarg or set config.extra['checkpoint_path']."
            )
        if model_dir is None:
            raise ValueError(
                "Goal2WorldModelWrapper.load() requires model_dir pointing to "
                "the dreamer_world_model package directory."
            )

        ckpt = torch.load(Path(ckpt_path).expanduser(),
                          map_location=self.device, weights_only=False)

        state_dict = ckpt["model"] if "model" in ckpt else ckpt

        # ── import WorldModel + WorldModelConfig from training package ─────────
        WorldModel, WorldModelConfig = _import_dreamer_package(model_dir)

        # Build WorldModelConfig from env_config["model_arch"] + training defaults
        arch = env_config.get("model_arch", {})
        cfg = WorldModelConfig(
            obs_dim        = int(arch.get("obs_dim",        60)),
            act_dim        = int(arch.get("act_dim",        2)),
            deter_dim      = int(arch.get("deter_dim",      512)),
            stoch_dim      = int(arch.get("stoch_dim",      32)),
            stoch_classes  = int(arch.get("stoch_classes",  32)),
            enc_hidden     = list(arch.get("enc_hidden",    [512, 512, 512])),
            dec_hidden     = list(arch.get("dec_hidden",    [512, 512, 512])),
            aux_keys       = list(arch.get("aux_keys", [
                "cost", "speed", "goal_distance",
                "nearest_hazard_distance", "nearest_vase_distance",
                "human_distance",
            ])),
        )

        self._deter_dim = cfg.deter_dim
        self._z_flat    = cfg.stoch_dim * cfg.stoch_classes

        self.model = WorldModel(cfg)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Optional normalisation stats
        if "obs_mean" in ckpt:
            self._obs_mean = np.asarray(ckpt["obs_mean"], dtype=np.float32)
            self._obs_std  = np.asarray(ckpt["obs_std"],  dtype=np.float32)

    # ── RSSM helpers ──────────────────────────────────────────────────────────
    # z is kept FLAT: (1, stoch_dim * stoch_classes) = (1, 1024)
    # This matches the training code's RSSM.initial_state() convention.

    def _init_rssm_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, self._deter_dim, device=self.device)
        z = torch.zeros(1, self._z_flat,    device=self.device)
        return h, z

    def _rssm_imagine_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prior step — no observation. Returns (h_next, z_next)."""
        h_next, z_next, _prior_logits = self.model.rssm.step_prior(h, z, action)
        return h_next, z_next

    def _rssm_encode_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior step — encode real obs. Returns (h_next, z_next)."""
        obs_norm = self._normalise_obs(obs)
        embed    = self.model.encoder(obs_norm)         # (1, embed_dim)
        h_next, z_next, _post, _prior = self.model.rssm.step_posterior(h, z, action, embed)
        return h_next, z_next

    def _feat(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Build the 1536-dim feature vector fed to all decoders.

        h_only mode (paper-faithful): z is zeroed out so only the continuous
        deterministic state h influences threshold predicates and AP values.
        The RSSM transitions (step_prior / step_posterior) always receive the
        real z so rollout dynamics remain accurate.

        feat mode: full cat(h, z) — best decoder accuracy, but z is
        categorical (technically violates continuous-input assumption).
        """
        if self._latent_mode == "h_only":
            z_zeros = torch.zeros_like(z)
            return torch.cat([h, z_zeros], dim=-1)
        return torch.cat([h, z], dim=-1)

    def _normalise_obs(self, obs_t: torch.Tensor) -> torch.Tensor:
        if self._obs_mean is not None:
            mean = torch.tensor(self._obs_mean, device=self.device)
            std  = torch.tensor(self._obs_std,  device=self.device).clamp(min=1e-6)
            return (obs_t - mean) / std
        return obs_t

    # ── AP decoding ───────────────────────────────────────────────────────────

    def _decode_aps(self, feat: torch.Tensor) -> dict[str, float]:
        """Run aux decoder on feat (1, 1536) → SAFEWORLD AP dict."""
        with torch.no_grad():
            # aux_decoder.forward() returns {key: tensor shape (1,) or scalar}
            aux: dict[str, torch.Tensor] = self.model.aux_decoder(feat)

        def _s(key: str) -> float:
            return float(aux[key].squeeze())

        thr = self._ap_thresholds
        return {
            "hazard_dist":   _s("nearest_hazard_distance") - thr.get("hazard_safe_dist",   0.25),
            "goal_dist":     _s("goal_distance")           - thr.get("goal_reach_radius",  0.30),
            "near_obstacle": _s("nearest_vase_distance")   - thr.get("obstacle_safe_dist", 0.20),
            "near_human":    thr.get("human_near_threshold", 1.0) - _s("human_distance"),
            "velocity":      _s("speed"),
            "model_cost":    _s("cost"),
            "zone_a": 0.0, "zone_b": 0.0, "zone_c": 0.0,
            "carrying": 0.0,
        }

    def _decode_obs(self, feat: torch.Tensor) -> np.ndarray:
        """Decode feat → reconstructed 60-d observation."""
        with torch.no_grad():
            obs_t = self.model.obs_decoder(feat)    # (1, 60)
        return obs_t.squeeze(0).cpu().numpy()

    # ── action sampling ───────────────────────────────────────────────────────

    def _load_oracle_episodes(self, level_filter: str | None = None) -> list[dict[str, np.ndarray]]:
        """Load obs+action sequences from satisfied success-bucket episode JSONs.

        Only loads episodes where satisfied=True (agent actually completed the
        task), skipping near_success episodes where the task was not fulfilled.

        Parameters
        ----------
        level_filter : e.g. "L2" — only load episodes whose directory name
                       contains this string.  None loads everything.
        """
        import json as _json
        episodes_dir = Path(
            self.config.extra.get(
                "oracle_episodes_dir",
                "/home/fanz23@netid.washington.edu/Desktop/SafeWorld-Benchmark"
                "/datasets/goal2_master/safeworld-goal2-master/episodes",
            )
        )
        filter_str = level_filter or self.config.extra.get("oracle_level_filter")
        seen_ids: set[str] = set()
        episodes: list[dict[str, np.ndarray]] = []
        for ep_file in sorted(episodes_dir.rglob("*success*.json")):
            if filter_str and filter_str not in str(ep_file):
                continue
            try:
                ep = _json.loads(ep_file.read_text(encoding="utf-8"))
                # skip episodes where the agent didn't actually satisfy the spec
                if not ep.get("satisfied", True):
                    continue
                ep_id = ep.get("episode_id", str(ep_file))
                if ep_id in seen_ids:
                    continue
                seen_ids.add(ep_id)
                acts = np.array(ep["action"], dtype=np.float32)
                obs  = np.array(ep["obs"],    dtype=np.float32)
                if len(acts) > 0 and len(obs) > 0:
                    episodes.append({"obs": obs, "action": acts})
            except Exception:
                continue
        return episodes

    def _sample_action(self, rng: np.random.Generator) -> np.ndarray:
        act_dim = self.model.cfg.act_dim
        return rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)

    def _action_tensor(self, action: np.ndarray) -> torch.Tensor:
        return torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)

    # ── rollout methods ───────────────────────────────────────────────────────

    def sample_rollouts(
        self,
        config: RolloutConfig | None = None,
    ) -> list[list[dict[str, float]]]:
        """Sample N rollouts and extract AP signals.

        oracle mode (action_source == "oracle"):
            Replays obs+action from real success episodes through the RSSM
            posterior (encode real obs at every step) so h_t captures the
            true world state.  This is the only mode that gives meaningful
            AP values for STL verification.

        random mode:
            Pure RSSM prior imagination from (h=0, z=0) with uniform-random
            actions.  AP values are effectively uninformative baselines.
        """
        self._ensure_loaded()
        cfg = config or self.config

        action_source = cfg.extra.get("action_source", getattr(cfg, "action_source", "random"))

        if action_source == "oracle":
            return self._sample_rollouts_oracle(cfg)

        # ── random / prior imagination ────────────────────────────────────────
        trajectories: list[list[dict[str, float]]] = []
        for i in range(cfg.n_rollouts):
            rng  = np.random.default_rng(cfg.seed + i)
            h, z = self._init_rssm_state()
            traj: list[dict[str, float]] = []
            with torch.no_grad():
                for _ in range(cfg.horizon):
                    feat = self._feat(h, z)
                    aps  = self._decode_aps(feat)
                    action = self._sample_action(rng)
                    h, z = self._rssm_imagine_step(h, z, self._action_tensor(action))
                    traj.append(aps)
            trajectories.append(traj)
        return trajectories

    def _sample_rollouts_oracle(
        self,
        cfg: RolloutConfig,
    ) -> list[list[dict[str, float]]]:
        """
        Oracle posterior rollouts: encode the real obs at every step via
        RSSM posterior so h_t has full knowledge of the world state.
        AP signals are then decoded from the posterior feature.

        Each rollout uses one deduplicated episode (cycling if n_rollouts >
        number of unique episodes).  The trajectory is truncated to
        min(horizon, episode_length) steps.
        """
        episodes = self._load_oracle_episodes(
            level_filter=cfg.extra.get("oracle_level_filter")
        )
        if not episodes:
            # Fallback: load all episodes if level filter yields nothing (e.g. L7 has no data)
            episodes = self._load_oracle_episodes(level_filter=None)
        if not episodes:
            raise RuntimeError(
                "action_source='oracle' but no success-bucket episode JSONs found. "
                "Check oracle_episodes_dir in config.extra."
            )

        trajectories: list[list[dict[str, float]]] = []
        for i in range(cfg.n_rollouts):
            ep      = episodes[i % len(episodes)]
            ep_obs  = ep["obs"]      # (T, 60)
            ep_acts = ep["action"]   # (T, 2)
            T       = min(cfg.horizon, len(ep_acts), len(ep_obs))

            h, z = self._init_rssm_state()
            traj: list[dict[str, float]] = []

            with torch.no_grad():
                for t in range(T):
                    obs_t = torch.tensor(
                        ep_obs[t][None], dtype=torch.float32, device=self.device
                    )
                    act_t = self._action_tensor(ep_acts[t])

                    # Posterior: encode real obs → update (h, z)
                    h, z = self._rssm_encode_step(h, z, act_t, obs_t)

                    feat = self._feat(h, z)
                    aps  = self._decode_aps(feat)
                    traj.append(aps)

            trajectories.append(traj)
        return trajectories

    def sample_paired_rollouts(
        self,
        config: RolloutConfig | None = None,
    ) -> list[tuple[list[dict[str, float]], list[dict[str, float]]]]:
        """
        Run imagination rollouts and replay the same actions in the real env.
        Used for conformal transfer calibration.

        Supports oracle action_source the same way as sample_rollouts().
        """
        self._ensure_loaded()
        self._ensure_env()
        cfg = config or self.config

        action_source = cfg.extra.get("action_source", getattr(cfg, "action_source", "random"))
        oracle_actions: list[np.ndarray] | None = None
        if action_source == "oracle":
            oracle_actions = self._load_oracle_actions()
            if not oracle_actions:
                raise RuntimeError(
                    "action_source='oracle' but no success-bucket episode JSONs found."
                )

        pairs: list[tuple[list[dict[str, float]], list[dict[str, float]]]] = []
        for i in range(cfg.n_rollouts):
            rng = np.random.default_rng(cfg.seed + i)
            obs, info = self._gym_env.reset(**cfg.extra.get("reset_kwargs", {}))
            if self._tracker is not None:
                self._tracker.reset()
            prev_obs = obs

            ep_actions: np.ndarray | None = None
            if oracle_actions is not None:
                ep_actions = oracle_actions[i % len(oracle_actions)]

            h, z = self._init_rssm_state()
            model_traj: list[dict[str, float]] = []
            env_traj:   list[dict[str, float]] = []

            for t in range(cfg.horizon):
                if ep_actions is not None and t < len(ep_actions):
                    action = ep_actions[t]
                else:
                    action = self._sample_action(rng)

                with torch.no_grad():
                    feat      = self._feat(h, z)
                    model_aps = self._decode_aps(feat)
                    h, z      = self._rssm_imagine_step(h, z, self._action_tensor(action))
                model_traj.append(model_aps)

                obs, _, terminated, truncated, info = self._gym_env.step(action)
                env_traj.append(
                    safety_point_goal_adapter(
                        obs, info=info, prev_obs=prev_obs, tracker=self._tracker,
                    )
                )
                prev_obs = obs
                if terminated or truncated:
                    break

            pairs.append((model_traj, env_traj))
        return pairs

    # ── decode and replay ─────────────────────────────────────────────────────

    def decode_and_replay(
        self,
        config: RolloutConfig | None = None,
        closed_loop: bool = False,
    ) -> list[list[ReplayStep]]:
        """
        Imagine rollouts, decode each latent to 60-d obs, replay actions in the
        real simulator, and return per-step comparison records.

        closed_loop=False: use RSSM posterior on real env obs (single-step accuracy).
        closed_loop=True:  keep RSSM prior throughout (cumulative drift).
        """
        self._ensure_loaded()
        self._ensure_env()
        cfg = config or self.config

        all_rollouts: list[list[ReplayStep]] = []

        for i in range(cfg.n_rollouts):
            rng  = np.random.default_rng(cfg.seed + i)
            obs, info = self._gym_env.reset(**cfg.extra.get("reset_kwargs", {}))
            if self._tracker is not None:
                self._tracker.reset()
            prev_obs = obs

            h, z  = self._init_rssm_state()
            steps: list[ReplayStep] = []

            for t in range(cfg.horizon):
                action   = self._sample_action(rng)
                action_t = self._action_tensor(action)
                obs_arr  = np.asarray(obs, dtype=np.float32).reshape(-1)

                with torch.no_grad():
                    feat         = self._feat(h, z)
                    model_aps    = self._decode_aps(feat)
                    model_obs_np = self._decode_obs(feat)       # (60,) reconstructed

                    if closed_loop:
                        h, z = self._rssm_imagine_step(h, z, action_t)
                    else:
                        obs_t = torch.tensor(obs_arr, dtype=torch.float32,
                                             device=self.device).unsqueeze(0)
                        h, z = self._rssm_encode_step(h, z, action_t, obs_t)

                obs, _, terminated, truncated, info = self._gym_env.step(action)
                env_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
                env_aps = safety_point_goal_adapter(
                    obs, info=info, prev_obs=prev_obs, tracker=self._tracker,
                )

                obs_rmse = float(np.sqrt(np.mean((model_obs_np - env_arr) ** 2)))
                steps.append(ReplayStep(
                    t              = t,
                    action         = action,
                    model_obs      = model_obs_np,
                    env_obs        = env_arr,
                    model_semantic = model_aps,
                    env_semantic   = env_aps,
                    obs_rmse       = obs_rmse,
                ))

                prev_obs = obs
                if terminated or truncated:
                    break

            all_rollouts.append(steps)
        return all_rollouts

    # ── required abstract methods ─────────────────────────────────────────────

    def ap_keys(self) -> list[str]:
        return [
            "hazard_dist", "goal_dist", "velocity",
            "near_obstacle", "near_human",
            "zone_a", "zone_b", "zone_c", "carrying",
            "model_cost",
        ]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "Goal2WorldModelWrapper: call load() before sampling rollouts."
            )

    def _ensure_env(self) -> None:
        if self._gym_env is not None:
            return
        import gymnasium as gym
        env_kwargs = dict(self.env_config.get("env_kwargs", {}))
        env_kwargs.update(self.config.extra.get("env_kwargs", {}))
        self._gym_env = gym.make(self.env_name, **env_kwargs)

    def close(self) -> None:
        if self._gym_env is not None:
            self._gym_env.close()
            self._gym_env = None
        self.model = None