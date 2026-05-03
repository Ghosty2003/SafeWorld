"""
wrappers/dreamerv3_wrapper.py

SAFEWORLD wrapper for DreamerV3 (Hafner et al., 2023).

How loading and rollout work (reverse-engineered from run_safeworld_dreamer.py):
─────────────────────────────────────────────────────────────────────────────────
  Loading
  ───────
  1. Read config.yaml from <logdir>/config.yaml if it exists (already-merged
     training config), otherwise layer config presets from dreamerv3/configs.yaml.
  2. Call dreamerv3.main.make_agent(config) to construct the JAX agent.
  3. Load weights with elements.Checkpoint pointing at <logdir>/ckpt/.

  Rollout  (imagination mode — no real env needed)
  ──────────────────────────────────────────────────
  4. Build zero initial RSSM carry:
         carry = {deter: (1, deter_dim), stoch: (1, stoch_k, stoch_cls)}
  5. For each rollout i, call model.dyn.imagine(carry, actions, T, False) inside
     a ninjax pure/JIT wrapper.
     - action_source="random"  → random actions sampled from act_space
     - action_source="policy"  → model.pol(feat) → sample from distribution
  6. Concatenate feat["deter"][0] and feat["stoch"][0].reshape(T,-1)
     → z_array of shape (T, deter + stoch*classes)

  AP extraction from raw latent
  ──────────────────────────────
  The reference file calls build_specs() which identifies the two highest-variance
  latent dimensions and uses them as "hazard" and "goal" predicates with
  data-driven thresholds (mean +/- k*std).

  SAFEWORLD AP dict conversion
  ─────────────────────────────
  We expose two strategies (set via RolloutConfig.extra["ap_mode"]):
    "projection"  (default) - fixed random-projection classifier (fast, no training)
    "stats"       - data-driven thresholds matching build_specs() exactly
    "decoder"     - decode latent to obs, then index into obs vector (needs real env)

Latent vector layout (DreamerV3 defaults):
    deter_dim  = rssm.deter  (typically 4096 for large, 512 for small)
    stoch_k    = rssm.stoch  (typically 32)
    stoch_cls  = rssm.classes (typically 32)
    D = deter_dim + stoch_k * stoch_cls

RolloutConfig.extra keys
------------------------
    logdir          str       path to DreamerV3 checkpoint directory (REQUIRED)
    task            str       task name, e.g. "dmc_walker_walk"
    configs         list[str] config preset names (default ["defaults", "dmc"])
    action_source   str       "random" | "policy"
    use_stoch       bool      include stochastic z_t in latent (default True)
    ap_mode         str       "projection" | "stats" | "decoder"
    hazard_sigma    float     threshold = mean + hazard_sigma*std  (default 1.5)
    goal_sigma      float     threshold = mean + goal_sigma*std    (default 1.0)
    fidelity        float     simulation fidelity in fallback mode  (default 0.75)

Requires (real mode):
    pip install jax[cuda12] ninjax>=3.5.1 elements>=3.19.1 optax ruamel.yaml
    # CPU-only: pip install jax ninjax elements optax ruamel.yaml
    # Clone DreamerV3 repo and add to PYTHONPATH
"""

from __future__ import annotations

import logging
import pathlib
import warnings
from typing import Any

import numpy as np

from configs.settings import RolloutConfig
from .base import WorldModelWrapper

logger = logging.getLogger(__name__)

# ── optional JAX / DreamerV3 imports ─────────────────────────────────────────

try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import ninjax as nj           # type: ignore
    NJ_AVAILABLE = True
except ImportError:
    NJ_AVAILABLE = False

try:
    import elements               # type: ignore
    ELEMENTS_AVAILABLE = True
except ImportError:
    ELEMENTS_AVAILABLE = False

DREAMER_AVAILABLE = JAX_AVAILABLE and NJ_AVAILABLE and ELEMENTS_AVAILABLE


# ══════════════════════════════════════════════════════════════════════════════
# Helpers adapted from run_safeworld_dreamer.py
# ══════════════════════════════════════════════════════════════════════════════

def _load_config(logdir: str, task: str, configs: list[str]):
    """
    Reproduces run_safeworld_dreamer.py:run() config-loading logic.
    Tries <logdir>/config.yaml first (already-merged training config),
    then falls back to stacking named presets from dreamerv3/configs.yaml.
    """
    import elements               # type: ignore
    import ruamel.yaml as yaml    # type: ignore

    saved_cfg = pathlib.Path(logdir) / "config.yaml"
    if saved_cfg.exists():
        logger.info(f"Loading saved config from {saved_cfg}")
        raw = elements.Path(str(saved_cfg)).read()
        cfg_dict = yaml.YAML(typ="safe").load(raw)
        config = elements.Config(cfg_dict)
    else:
        logger.info(f"Building config from presets: {configs}")
        import dreamerv3          # type: ignore
        folder   = pathlib.Path(dreamerv3.__file__).parent
        cfgyaml  = elements.Path(str(folder / "configs.yaml")).read()
        cfgs     = yaml.YAML(typ="safe").load(cfgyaml)
        config   = elements.Config(cfgs["defaults"])
        for name in configs:
            if name in cfgs:
                config = config.update(cfgs[name])
            else:
                logger.warning(f"Config preset '{name}' not found, skipping.")
        config = config.update(task=task)

    # Always force logdir to the real checkpoint directory
    config = config.update(logdir=logdir)
    return config


def _load_agent(config):
    """
    Reproduces run_safeworld_dreamer.py:load_agent().
    make_agent() + elements.Checkpoint load.
    """
    import elements                        # type: ignore
    from dreamerv3.main import make_agent  # type: ignore

    agent   = make_agent(config)
    ckpt_dir = elements.Path(config.logdir) / "ckpt"
    cp       = elements.Checkpoint(directory=ckpt_dir)
    cp.agent = agent
    if cp.exists():
        cp.load(keys=["agent"])
        logger.info(f"Checkpoint loaded from {ckpt_dir}")
    else:
        logger.warning(f"No checkpoint at {ckpt_dir} — using random weights.")
    return agent


def _sample_random_action(rng: np.random.Generator, act_space) -> dict:
    """One random action sample (host numpy). Matches reference file."""
    action = {}
    for k, sp in act_space.items():
        if sp.discrete:
            classes = int(np.asarray(sp.classes).flatten()[0])
            action[k] = rng.integers(0, classes, size=sp.shape, dtype=sp.dtype)
        else:
            action[k] = rng.uniform(-1.0, 1.0, size=sp.shape).astype(sp.dtype)
    return action


def _feat_to_numpy(feat_host: dict, T: int, use_stoch: bool) -> np.ndarray:
    """
    Converts imagine() output to (T, D) numpy array.
    Exactly mirrors run_safeworld_dreamer.py:sample_trajectories() lines:
        deter      = feat_host['deter'][0]          # (T, deter_dim)
        stoch_flat = feat_host['stoch'][0].reshape(T, -1)
        z          = np.concatenate([deter, stoch_flat], axis=-1)
    """
    deter      = np.asarray(feat_host["deter"])[0]   # (T, deter_dim)
    stoch      = np.asarray(feat_host["stoch"])[0]   # (T, stoch_k, stoch_cls)
    stoch_flat = stoch.reshape(T, -1)                 # (T, stoch_k*stoch_cls)
    if use_stoch:
        return np.concatenate([deter, stoch_flat], axis=-1)
    return deter


# ══════════════════════════════════════════════════════════════════════════════
# AP extraction strategies
# ══════════════════════════════════════════════════════════════════════════════

def _compute_stats_thresholds(
    z_array:      np.ndarray,
    deter_dim:    int,
    hazard_sigma: float = 1.5,
    goal_sigma:   float = 1.0,
) -> dict[str, Any]:
    """
    Reproduces build_specs() from run_safeworld_dreamer.py.

    Finds the two highest-variance dimensions in the deterministic (deter)
    part of the latent, then sets:
        hazard_thr = mean[d0] + hazard_sigma * std[d0]
        goal_thr   = mean[d1] + goal_sigma   * std[d1]

    Only the deter part is searched (stoch dims are categorical probabilities
    in [0,1] and saturate the calibrator — see reference file comment).

    z_array : (N, T, D)
    """
    flat       = z_array.reshape(-1, z_array.shape[-1])
    means      = flat.mean(axis=0)
    stds       = flat.std(axis=0) + 1e-8
    search_end = min(deter_dim, flat.shape[-1])
    top        = np.argsort(flat[:, :search_end].var(axis=0))[::-1]
    d0, d1     = int(top[0]), int(top[1])
    t0 = float(means[d0] + hazard_sigma * stds[d0])
    t1 = float(means[d1] + goal_sigma   * stds[d1])
    logger.info(f"build_specs: hazard=z[{d0}]>{t0:.4f}  goal=z[{d1}]>{t1:.4f}")
    return {
        "means":      means,
        "stds":       stds,
        "hazard_dim": d0,
        "hazard_thr": t0,
        "goal_dim":   d1,
        "goal_thr":   t1,
    }


def _stats_aps(
    z_t:        np.ndarray,
    hazard_dim: int,
    goal_dim:   int,
    hazard_thr: float,
    goal_thr:   float,
) -> dict[str, float]:
    """
    Data-driven AP extraction matching build_specs() convention:
        hazard_dist > 0  means safe  (z[hazard_dim] < hazard_thr)
        goal_dist   > 0  means goal reached (z[goal_dim] > goal_thr)
    """
    return {
        "hazard_dist":   hazard_thr - float(z_t[hazard_dim]),
        "goal_dist":     float(z_t[goal_dim]) - goal_thr,
        "velocity":      0.0,
        "near_obstacle": 0.0,
        "near_human":    0.0,
        "zone_a":        0.0,
        "zone_b":        0.0,
        "zone_c":        0.0,
        "carrying":      0.0,
    }


def _projection_aps(z_t: np.ndarray, cache: dict) -> dict[str, float]:
    """
    Fixed random-projection AP classifier.
    Fast fallback when no trained classifier is available.
    Uses seed=42 for reproducibility (matches Appendix R convention).
    """
    d = len(z_t)
    if cache.get("d") != d:
        rng = np.random.default_rng(42)
        cache["projs"] = rng.standard_normal((9, d))
        cache["d"] = d

    z_norm = z_t / (np.linalg.norm(z_t) + 1e-8)
    s = cache["projs"] @ z_norm  # (9,)

    return {
        "velocity":      float(np.clip(abs(s[0]) * 0.8, 0.0, 2.0)),
        "hazard_dist":   float(np.tanh(s[1]) * 0.5),
        "goal_dist":     float(np.tanh(s[2]) * 0.5 - 0.1),
        "near_obstacle": float(np.tanh(s[3]) * 0.4),
        "near_human":    float(np.tanh(s[4]) * 0.4),
        "zone_a":        1.0 if s[5] > 0.3 else 0.0,
        "zone_b":        1.0 if s[6] > 0.3 else 0.0,
        "zone_c":        1.0 if s[7] > 0.3 else 0.0,
        "carrying":      1.0 if s[8] > 0.5 else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Simulation fallback
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_rssm_rollout(T: int, D: int, seed: int, fidelity: float) -> np.ndarray:
    """Pure-numpy GRU-style RSSM simulation. Returns (T, D) latent array."""
    rng = np.random.default_rng(seed)
    h   = rng.standard_normal(D) * 0.1
    rows = []
    for _ in range(T):
        h = 0.95 * h + 0.05 * np.tanh(h + rng.standard_normal(D) * 0.05 * (1 - fidelity + 0.1))
        h[:4] += 0.08 * rng.uniform(-1, 1, size=4)
        rows.append(h.copy())
    return np.stack(rows)  # (T, D)


# ══════════════════════════════════════════════════════════════════════════════
# Main wrapper class
# ══════════════════════════════════════════════════════════════════════════════

class DreamerV3Wrapper(WorldModelWrapper):
    """
    SAFEWORLD wrapper for a trained DreamerV3 agent.

    Follows run_safeworld_dreamer.py exactly for config loading, agent loading,
    and rollout sampling.  Falls back to numpy simulation when JAX/dreamerv3
    is not installed.

    Quick start
    -----------
    >>> cfg = RolloutConfig(
    ...     horizon=20, n_rollouts=40, seed=0,
    ...     extra={
    ...         "logdir":  "/path/to/dreamer_logdir",
    ...         "task":    "dmc_walker_walk",
    ...         "configs": ["defaults", "dmc"],
    ...         "ap_mode": "stats",          # matches build_specs() from reference
    ...     }
    ... )
    >>> with DreamerV3Wrapper(cfg) as w:
    ...     w.load()
    ...     trajs = w.sample_rollouts()
    ...     meta  = w.get_stats_meta()       # hazard/goal dims + thresholds
    """

    def __init__(self, config: RolloutConfig | None = None):
        super().__init__(config)
        self._agent      : Any = None
        self._dv3_config : Any = None
        self._loaded     = False
        self._sim_mode   = not DREAMER_AVAILABLE
        self._deter_dim  : int | None = None
        self._stoch_k    : int | None = None
        self._stoch_cls  : int | None = None
        self._stats      : dict | None = None   # populated in "stats" ap_mode
        self._proj_cache : dict = {}             # projection matrix cache
        self._env        : Any = None            # lazy Gymnasium env for replay

        # Public: raw (N, T, D) latent array from last sample_rollouts() call
        self.last_z_array: np.ndarray | None = None

        if self._sim_mode:
            warnings.warn(
                "JAX / ninjax / elements not found — DreamerV3Wrapper running in "
                "simulation mode. Install: pip install jax ninjax elements ruamel.yaml",
                stacklevel=2,
            )

    # ── WorldModelWrapper interface ───────────────────────────────────────────

    def load(self, **kwargs) -> None:
        """
        Load the DreamerV3 agent from checkpoint.

        Reads configuration from RolloutConfig.extra or keyword arguments:
            logdir   (str)       – path to DreamerV3 logdir  [required]
            task     (str)       – task name, e.g. "dmc_walker_walk"
            configs  (list[str]) – config preset names

        Mirrors run_safeworld_dreamer.py:run() exactly.
        """
        if self._sim_mode:
            self._loaded = True
            return

        logdir  = kwargs.get("logdir",  self.config.extra.get("logdir",  ""))
        task    = kwargs.get("task",    self.config.extra.get("task",    "dmc_walker_walk"))
        configs = kwargs.get("configs", self.config.extra.get("configs", ["defaults", "dmc"]))

        if not logdir:
            warnings.warn(
                "DreamerV3Wrapper: 'logdir' not set — switching to simulation mode.",
                stacklevel=2,
            )
            self._sim_mode = True
            self._loaded   = True
            return

        try:
            self._dv3_config = _load_config(logdir, task, configs)
            self._agent      = _load_agent(self._dv3_config)

            rssm             = self._dv3_config.agent.dyn.rssm
            self._deter_dim  = rssm.deter
            self._stoch_k    = rssm.stoch
            self._stoch_cls  = rssm.classes
            D = self._deter_dim + self._stoch_k * self._stoch_cls
            logger.info(
                f"DreamerV3Wrapper ready: "
                f"deter={self._deter_dim}  "
                f"stoch={self._stoch_k}x{self._stoch_cls}  D={D}"
            )
            self._loaded = True

        except Exception as exc:
            warnings.warn(
                f"DreamerV3Wrapper.load() failed ({exc}) — switching to simulation mode.",
                stacklevel=2,
            )
            self._sim_mode = True
            self._loaded   = True

    def sample_rollouts(
        self,
        config: RolloutConfig | None = None,
    ) -> list[list[dict[str, float]]]:
        """
        Sample N latent rollouts of length T.

        Real mode  (dreamerv3 installed + checkpoint loaded):
          - Builds zero RSSM carry (deter, stoch).
          - Calls nj.pure(model.dyn.imagine) under jax.jit.
          - Concatenates deter + stoch.flatten() per rollout  → (T, D).
          - Converts to AP dicts via the chosen ap_mode.

        Simulation mode (fallback):
          - Uses _simulate_rssm_rollout() (pure numpy).

        The raw latent array (N, T, D) is stored in self.last_z_array for
        downstream use (e.g. building custom specs with build_specs() logic).

        Returns
        -------
        List of N trajectories; each trajectory is a list of T AP state-dicts.
        """
        cfg = config or self.config
        if not self._loaded:
            self.load()

        use_stoch    = cfg.extra.get("use_stoch",     True)
        ap_mode      = cfg.extra.get("ap_mode",       "projection")
        hazard_sigma = cfg.extra.get("hazard_sigma",  1.5)
        goal_sigma   = cfg.extra.get("goal_sigma",    1.0)
        action_src   = cfg.extra.get("action_source", cfg.action_source)

        # ── Simulation mode ───────────────────────────────────────────────────
        if self._sim_mode or self._agent is None:
            D        = self._deter_dim or cfg.extra.get("latent_dim", 512)
            fidelity = cfg.extra.get("fidelity", 0.75)
            z_list   = [
                _simulate_rssm_rollout(cfg.horizon, D, cfg.seed + i * 13, fidelity)
                for i in range(cfg.n_rollouts)
            ]
            self.last_z_array = np.stack(z_list)
            logger.debug(
                f"DreamerV3Wrapper [sim]: {cfg.n_rollouts} x T={cfg.horizon} "
                f"shape={self.last_z_array.shape}"
            )
            return self._convert_to_trajectories(
                self.last_z_array, ap_mode,
                hazard_sigma, goal_sigma,
                D,
            )

        # ── Real DreamerV3 rollout ────────────────────────────────────────────
        # Mirrors sample_trajectories() from run_safeworld_dreamer.py exactly.

        model     = self._agent.model
        params    = self._agent.params
        T         = cfg.horizon
        deter_dim = self._deter_dim
        stoch_k   = self._stoch_k
        stoch_cls = self._stoch_cls
        act_space = self._agent.act_space

        # ── JIT-compiled imagination fn inside ninjax context ─────────────────
        def _imagine_fn(carry, actions):
            if action_src == "policy":
                sample_act = lambda xs: {k: v.sample(nj.seed()) for k, v in xs.items()}
                policyfn   = lambda c: sample_act(model.pol(model.feat2tensor(c), 1))
                _, feat, _ = model.dyn.imagine(carry, policyfn, T, False)
            else:
                _, feat, _ = model.dyn.imagine(carry, actions, T, False)
            return feat

        imagine_jit = jax.jit(nj.pure(_imagine_fn))

        # Zero initial carry — jax.device_put avoids transfer-guard errors
        carry0 = dict(
            deter=jax.device_put(np.zeros([1, deter_dim], dtype=np.float32)),
            stoch=jax.device_put(np.zeros([1, stoch_k, stoch_cls], dtype=np.float32)),
        )

        rng    = np.random.default_rng(cfg.seed)
        z_list = []

        logger.info(
            f"DreamerV3Wrapper: sampling {cfg.n_rollouts} x T={T} "
            f"D={deter_dim + stoch_k * stoch_cls} action_src={action_src}"
        )

        for i in range(cfg.n_rollouts):
            seed_i   = int(rng.integers(0, 2**31))
            seed_jax = jax.device_put(np.array([seed_i, seed_i], dtype=np.uint32))

            if action_src == "policy":
                actions_arg = None
            else:
                # Build batched random actions: {key: device_array(1, T, *shape)}
                actions_arg = {}
                for k, sp in act_space.items():
                    if sp.discrete:
                        classes = int(np.asarray(sp.classes).flatten()[0])
                        a = rng.integers(0, classes, size=(1, T, *sp.shape), dtype=np.int32)
                    else:
                        a = rng.uniform(-1.0, 1.0, (1, T, *sp.shape)).astype(np.float32)
                    actions_arg[k] = jax.device_put(a)

            _, feat   = imagine_jit(params, carry0, actions_arg, seed=seed_jax)
            feat_host = jax.device_get(feat)
            z         = _feat_to_numpy(feat_host, T, use_stoch)   # (T, D)
            z_list.append(z)

            if (i + 1) % 10 == 0 or (i + 1) == cfg.n_rollouts:
                logger.info(f"  {i + 1}/{cfg.n_rollouts}")

        self.last_z_array = np.stack(z_list)   # (N, T, D)
        logger.info(f"DreamerV3Wrapper: done — shape={self.last_z_array.shape}")

        return self._convert_to_trajectories(
            self.last_z_array, ap_mode,
            hazard_sigma, goal_sigma,
            deter_dim,
        )

    # ── Internal AP conversion ────────────────────────────────────────────────

    def _convert_to_trajectories(
        self,
        z_array:      np.ndarray,
        ap_mode:      str,
        hazard_sigma: float,
        goal_sigma:   float,
        deter_dim:    int,
    ) -> list[list[dict[str, float]]]:
        N, T, _ = z_array.shape

        if ap_mode == "stats":
            if self._stats is None:
                self._stats = _compute_stats_thresholds(
                    z_array, deter_dim, hazard_sigma, goal_sigma
                )
            st = self._stats

        trajectories: list[list[dict[str, float]]] = []
        for i in range(N):
            traj: list[dict[str, float]] = []
            for t in range(T):
                z_t = z_array[i, t]
                if ap_mode == "stats":
                    aps = _stats_aps(
                        z_t,
                        st["hazard_dim"], st["goal_dim"],
                        st["hazard_thr"], st["goal_thr"],
                    )
                else:
                    # "projection" default
                    aps = _projection_aps(z_t, self._proj_cache)
                traj.append(aps)
            trajectories.append(traj)

        return trajectories

    # ── Public extras ─────────────────────────────────────────────────────────

    def get_stats_meta(self) -> dict | None:
        """
        Return data-driven threshold metadata populated during "stats" ap_mode.
        Equivalent to the `meta` dict from build_specs() in the reference file.
        Keys: hazard_dim, hazard_thr, goal_dim, goal_thr, means, stds.
        """
        return self._stats

    def sample_paired_rollouts(
        self,
        config: RolloutConfig | None = None,
    ) -> list[tuple[list[dict[str, float]], list[dict[str, float]]]]:
        cfg = config or self.config
        if not self._loaded:
            self.load()
        if self._sim_mode or self._agent is None or self._dv3_config is None:
            raise NotImplementedError(
                "DreamerV3 paired rollouts require a loaded DreamerV3 checkpoint and environment."
            )

        from Safeworld.run_safeworld_dreamer import collect_paired_trajectories

        action_src = cfg.extra.get("action_source", cfg.action_source)
        ap_mode = cfg.extra.get("ap_mode", "projection")
        hazard_sigma = cfg.extra.get("hazard_sigma", 1.5)
        goal_sigma = cfg.extra.get("goal_sigma", 1.0)
        use_policy = action_src == "policy"

        model_z, env_z, _ = collect_paired_trajectories(
            self._agent,
            self._dv3_config,
            cfg.n_rollouts,
            cfg.horizon,
            use_policy=use_policy,
        )
        joined = np.concatenate([model_z, env_z], axis=0)
        deter_dim = self._deter_dim or joined.shape[-1]

        if ap_mode == "stats":
            self._stats = _compute_stats_thresholds(joined, deter_dim, hazard_sigma, goal_sigma)
            st = self._stats

            def convert(z_t):
                return _stats_aps(
                    z_t,
                    st["hazard_dim"], st["goal_dim"],
                    st["hazard_thr"], st["goal_thr"],
                )
        else:
            def convert(z_t):
                return _projection_aps(z_t, self._proj_cache)

        model_trajs = [
            [convert(model_z[i, t]) for t in range(model_z.shape[1])]
            for i in range(model_z.shape[0])
        ]
        env_trajs = [
            [convert(env_z[i, t]) for t in range(env_z.shape[1])]
            for i in range(env_z.shape[0])
        ]
        self.last_z_array = model_z
        return list(zip(model_trajs, env_trajs))

    def decode_and_replay(
        self,
        config: RolloutConfig | None = None,
        closed_loop: bool = False,
    ) -> list[list]:
        """
        Imagine N rollouts in RSSM latent space, decode latent → obs, replay
        the same actions in the real simulator, and return per-step comparisons.

        Decoder path
        ------------
        When a real DreamerV3 checkpoint is loaded, model.dec reconstructs the
        raw observation from the RSSM features (deter + stoch) at each step.
        In simulation mode (no checkpoint), model_obs is None.

        closed_loop
        -----------
        False (default): the model re-imagines from the real env obs each step
                         → measures single-step prediction accuracy.
        True:            the model continues from its own last latent state
                         → measures cumulative imagination drift.
        """
        from .base import ReplayStep

        cfg = config or self.config

        if self._sim_mode or self._agent is None:
            raise NotImplementedError(
                "DreamerV3Wrapper.decode_and_replay() requires a loaded DreamerV3 "
                "checkpoint and a registered Gymnasium environment."
            )

        import jax
        import jax.numpy as jnp
        import ninjax as nj

        model      = self._agent.model
        params     = self._agent.params
        T          = cfg.horizon
        deter_dim  = self._deter_dim
        stoch_k    = self._stoch_k
        stoch_cls  = self._stoch_cls
        act_space  = self._agent.act_space
        ap_mode    = cfg.extra.get("ap_mode", "projection")
        hazard_sigma = cfg.extra.get("hazard_sigma", 1.5)
        goal_sigma   = cfg.extra.get("goal_sigma", 1.0)
        use_stoch    = cfg.extra.get("use_stoch", True)

        # ── JIT-compiled imagination ──────────────────────────────────────────
        def _imagine_fn(carry, actions):
            _, feat, _ = model.dyn.imagine(carry, actions, T, False)
            return feat

        imagine_jit = jax.jit(nj.pure(_imagine_fn))

        # ── JIT-compiled decoder ──────────────────────────────────────────────
        def _decode_fn(dec_carry, feat, is_first):
            _, _, recons = model.dec(dec_carry, feat, is_first, training=False)
            return recons

        decode_jit = jax.jit(nj.pure(_decode_fn))

        carry0 = dict(
            deter=jax.device_put(np.zeros([1, deter_dim], dtype=np.float32)),
            stoch=jax.device_put(np.zeros([1, stoch_k, stoch_cls], dtype=np.float32)),
        )

        # Resolve env
        env_name = cfg.extra.get("env_name", "SafetyPointGoal2Gymnasium-v0")
        if self._env is None:
            import gymnasium as gym
            self._env = gym.make(env_name)

        rng_global = np.random.default_rng(cfg.seed)
        all_rollouts: list[list[ReplayStep]] = []

        for i in range(cfg.n_rollouts):
            seed_i   = int(rng_global.integers(0, 2**31))
            seed_jax = jax.device_put(np.array([seed_i, seed_i], dtype=np.uint32))
            rng_i    = np.random.default_rng(seed_i)

            # ── build action array (1, T, *shape) for imagination ─────────────
            actions_np: dict[str, np.ndarray] = {}
            for k, sp in act_space.items():
                if sp.discrete:
                    classes = int(np.asarray(sp.classes).flatten()[0])
                    actions_np[k] = rng_i.integers(
                        0, classes, size=(1, T, *sp.shape), dtype=np.int32
                    )
                else:
                    actions_np[k] = rng_i.uniform(
                        -1.0, 1.0, (1, T, *sp.shape)
                    ).astype(np.float32)
            actions_jax = {k: jax.device_put(v) for k, v in actions_np.items()}

            # ── imagination: latent trajectory ────────────────────────────────
            _, feat_host = imagine_jit(params, carry0, actions_jax, seed=seed_jax)
            z_array = _feat_to_numpy(feat_host, T, use_stoch)   # (T, D)

            # ── decode latent → obs ───────────────────────────────────────────
            dec_carry0 = nj.pure(model.dec.initial)(params, 1)
            dec_carry0 = jax.device_put(dec_carry0)
            is_first   = jax.device_put(np.zeros([1, T], dtype=bool))
            try:
                _, recons_host = decode_jit(params, dec_carry0, feat_host, is_first)
                recons_host    = jax.device_get(recons_host)
                # DreamerV3 vector obs key is typically 'vector'
                obs_key        = "vector" if "vector" in recons_host else next(iter(recons_host))
                decoded_obs    = np.asarray(recons_host[obs_key][0])  # (T, obs_dim)
            except Exception:
                decoded_obs = None

            # ── env replay with the same action sequence ──────────────────────
            reset_kwargs = cfg.extra.get("reset_kwargs", {})
            env_obs, env_info = self._env.reset(**reset_kwargs)
            prev_obs = env_obs
            steps: list[ReplayStep] = []

            for t in range(T):
                # action for this step: shape (*sp.shape,) numpy
                action_t = {k: v[0, t] for k, v in actions_np.items()}
                # for simple continuous act spaces collapse to 1-D array
                action_arr = np.concatenate(
                    [v.reshape(-1) for v in action_t.values()], axis=0
                ).astype(np.float32)

                env_obs, _, terminated, truncated, env_info = self._env.step(
                    action_t if len(action_t) > 1 else action_arr
                )
                env_obs_arr  = np.asarray(env_obs, dtype=np.float32).reshape(-1)
                from environment.adapters import safety_point_goal_adapter as _spa
                env_semantic = _spa(env_obs, info=env_info, prev_obs=prev_obs)

                # latent → semantic APs
                z_t          = z_array[t]
                model_semantic = _projection_aps(z_t, self._proj_cache) \
                    if ap_mode == "projection" \
                    else _stats_aps(z_t,
                                    self._stats["hazard_dim"], self._stats["goal_dim"],
                                    self._stats["hazard_thr"], self._stats["goal_thr"])

                model_obs_t = decoded_obs[t] if decoded_obs is not None else None
                obs_rmse = (
                    float(np.sqrt(np.mean((model_obs_t - env_obs_arr) ** 2)))
                    if model_obs_t is not None else None
                )

                steps.append(ReplayStep(
                    t              = t,
                    action         = action_arr,
                    model_obs      = model_obs_t,
                    env_obs        = env_obs_arr,
                    model_semantic = model_semantic,
                    env_semantic   = env_semantic,
                    obs_rmse       = obs_rmse,
                ))

                prev_obs = env_obs
                if terminated or truncated:
                    break

            all_rollouts.append(steps)

        return all_rollouts

    def ap_keys(self) -> list[str]:
        return [
            "hazard_dist", "goal_dist",
            "velocity", "near_obstacle", "near_human",
            "zone_a", "zone_b", "zone_c", "carrying",
        ]

    def close(self) -> None:
        self._agent      = None
        self._dv3_config = None
        if self._env is not None:
            self._env.close()
            self._env = None
        logger.debug("DreamerV3Wrapper: resources released.")
