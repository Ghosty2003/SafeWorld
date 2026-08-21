"""
wrappers/safedreamer_wrapper.py

SAFEWORLD wrapper for SafeDreamer (Huang et al., 2024), OSRP-Vector variant.
Repo: https://github.com/PKU-Alignment/SafeDreamer

Verified against the actual checkpoint before writing this file
──────────────────────────────────────────────────────────────────────────────
Unlike dreamerv3_wrapper.py (written for the vanilla DreamerV3 repo's
`dreamerv3.main.make_agent` / `model.dyn.imagine` API), SafeDreamer's own
`SafeDreamer/agent.py` exposes a different surface:

    SafeDreamer.Agent(obs_space, act_space, step, config)   # jaxagent.Wrapper
        .agent.wm.imagine(policy, start, horizon)            # f: Z x A -> Delta(Z)
        .agent.wm.heads['decoder'](traj)                     # g: Z -> O  (vector-only MLP)
        .agent.wm.heads['rew'](traj)                         # symlog_disc, 255 bins
        .agent.wm.rssm.initial(batch)                        # rho: initial latent dist

We inspected the actual `.ckpt` file (pickle of a flat {param_path: ndarray}
dict, embodied.core.Checkpoint format) with `pickle.load` before writing any
of this:
    - obs dim  = 29   (agent/wm/enc/mlp/h0/kernel: (29,256); dec output: (256,29))
    - act dim  = 2    (task_behavior/ac/actor/dist_out/*: (256,2))
    - rssm     = deter=256, stoch=16 x classes=16   (matches configs.yaml osrp_vector)
    - heads present : dec (vector only, no cnn), rew (symlog_disc/255), cont (binary)
    - heads ABSENT  : cost  -- osrp_vector config has use_cost=False, confirmed by
                       the absence of any `agent/wm/cost/*` key in the checkpoint.
This means hazard/goal APs CANNOT come from a cost head for this checkpoint.
They must be derived from the decoded 29-dim observation vector, same as the
real-environment adapter already does. We reuse
`environment.adapters.safety_point_goal_adapter` for both, so the model side
and the real-env side of Transfer Calibrator (Definition 3.5) go through the
identical extractor -- required by cardreamer/README.md rule #5.

Loading requires the real SafeDreamer source tree on disk (not pip-installed):
`agent.py` uses relative imports (`from . import behaviors`), so the parent
of the `SafeDreamer/` package directory must be added to sys.path and the
package imported as `import SafeDreamer.agent`, exactly as `SafeDreamer/train.py`
does via its own sys.path manipulation.

Deviations / honesty notes (read before trusting output)
──────────────────────────────────────────────────────────────────────────────
1. Config reconstruction is NOT guaranteed identical to the original training
   run. The `20240307-010600_osrp_vector_...ckpt` file has no sibling
   `config.yaml` on disk (the `logdir/` folders found locally are from a
   *different*, later re-run). We rebuild the config from the *current*
   `SafeDreamer/configs.yaml` (`defaults` + `osrp_vector` preset). This is a
   best-effort reconstruction, cross-checked against the checkpoint's actual
   parameter shapes (obs=29, act=2, deter=256, stoch=16x16 all matched) --
   but non-architectural fields (loss weights, lr, etc, irrelevant at
   inference time) could differ if configs.yaml changed since training.
2. AP keys `near_human`, `zone_a`, `zone_b`, `zone_c`, `carrying` are NOT
   meaningful for a SafetyPointGoal task (no buttons/zones/humans exist in
   this environment) and are emitted as UNCERTAIN_SENTINEL (-999.0) rather
   than the adapter's silent 0.0 default, per wrappers/README.md rule #1.
   main.py's spec-AP-aware sentinel gate will route any spec that actually
   needs one of those keys to INCONCLUSIVE_perception instead of a false
   verdict.
3. `reward` is decoded via the symlog two-hot inverse (`DiscDist.mode()` in
   SafeDreamer's own jaxutils.py) -- never read the raw 255-way head output
   as a scalar.
4. This wrapper verifies the WORLD MODEL's imagined rollouts. It says
   nothing about the trained policy's real-environment behavior; that is a
   different, already-published number (SafeDreamer's own eval logs).
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from configs.settings import RolloutConfig
from .base import WorldModelWrapper

logger = logging.getLogger(__name__)

UNCERTAIN_SENTINEL = -999.0

# AP keys this wrapper can genuinely ground for a plain SafetyPointGoal task.
_GROUNDED_AP_KEYS = ("hazard_dist", "goal_dist", "velocity", "near_obstacle")
# AP keys that have no corresponding concept in a plain PointGoal task (no
# buttons/zones/humans exist) -- always emitted as UNCERTAIN_SENTINEL.
_UNGROUNDED_AP_KEYS = ("near_human", "zone_a", "zone_b", "zone_c", "carrying")

# Slice layout of the 29-dim 'observation' vector for safetygymcoor_SafetyPointGoal*-v0,
# confirmed from SafetyGymCoor's own debug printout at env-construction time
# (`key_to_slice <name> <slice> <sample_values>`, seen verbatim when _make_env()
# runs). NOT verified for other tasks/robots -- e.g. SafetyCarGoal1 uses a
# different robot with different sensors, so this layout must be re-confirmed
# (rerun with a fresh env construction and read the printed key_to_slice lines)
# before reusing it there.
_POINTGOAL_COOR_SLICES = {
    "velocimeter": slice(0, 2),
    "accelerometer": slice(2, 3),
    "gyro": slice(3, 4),
    "magnetometer": slice(4, 7),
    "goal": slice(7, 9),
    "hazards": slice(9, 25),   # 8 hazards x (dx, dy), egocentric
    "robot_m": slice(25, 27),
    "robot": slice(27, 29),
}
# From models/detailinfo.md (measured against the real simulator geometry):
_HAZARD_RADIUS = 0.20   # hazard_radius -- AP triggers once inside this distance
_GOAL_RADIUS = 0.30     # goal_radius


def _pointgoal_coor_aps(obs_29: np.ndarray) -> dict[str, float]:
    """
    AP extraction specific to safetygymcoor_SafetyPointGoal*-v0's 29-dim vector
    observation, indexing the confirmed slice layout directly.

    Do NOT reuse environment.adapters.safety_point_goal_adapter for this: that
    adapter expects a dict-shaped obs or an `info` dict carrying hazard
    positions, and silently falls back to a hardcoded 1.0 default distance
    when given a bare flat array -- confirmed by trial, it produced a constant
    fake hazard_dist=1.0 across every step of a real imagined rollout instead
    of raising or signalling missing data. That is exactly the "proxy instead
    of sentinel" failure wrappers/README.md rule #1 warns about.
    """
    goal_vec = obs_29[_POINTGOAL_COOR_SLICES["goal"]]
    hazard_vecs = obs_29[_POINTGOAL_COOR_SLICES["hazards"]].reshape(-1, 2)
    vel_vec = obs_29[_POINTGOAL_COOR_SLICES["velocimeter"]]

    goal_dist = float(np.linalg.norm(goal_vec)) - _GOAL_RADIUS
    hazard_dist = float(np.min(np.linalg.norm(hazard_vecs, axis=-1))) - _HAZARD_RADIUS
    velocity = float(np.linalg.norm(vel_vec))

    return {
        "hazard_dist": hazard_dist,
        "goal_dist": goal_dist,
        "velocity": velocity,
        "near_obstacle": hazard_dist,  # hazards are the only obstacle class in this task
    }


def _aps_with_sentinel(obs_29: np.ndarray) -> dict[str, float]:
    """_pointgoal_coor_aps() plus the UNCERTAIN_SENTINEL overlay for AP keys
    this task has no concept of. Shared by sample_rollouts() and
    sample_paired_rollouts() so both go through the identical extractor."""
    aps = _pointgoal_coor_aps(obs_29)
    for k in _UNGROUNDED_AP_KEYS:
        aps[k] = UNCERTAIN_SENTINEL
    return aps

try:
    import jax  # noqa: F401
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

SAFEDREAMER_AVAILABLE = JAX_AVAILABLE  # refined once we actually try the import


def _ensure_on_path(repo_root: str) -> None:
    """
    repo_root must be the SafeDreamer checkout root, i.e. the directory that
    directly contains the `SafeDreamer/` package folder (the one with
    agent.py, embodied/, ninjax.py, configs.yaml).
    Example: /home/user/Documents/SafeDreamer

    Two paths must be on sys.path, matching SafeDreamer/train.py's own
    sys.path manipulation:
      - repo_root itself            -> lets `import SafeDreamer.agent` work
                                        (relative imports inside agent.py
                                        resolve against the SafeDreamer package).
      - repo_root/SafeDreamer        -> lets `import embodied` resolve as a
                                        top-level bare import, since agent.py
                                        does `import embodied` (absolute, not
                                        relative) and `embodied/` is bundled
                                        one level down, not pip-installed.
    """
    root = Path(repo_root).resolve()
    for p in (str(root), str(root / "SafeDreamer")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _build_config(sd_module, method: str, task: str, extra_overrides: dict | None = None):
    """
    Reproduces SafeDreamer/train.py:main()'s config assembly:
        config = defaults
        config = config.update(configs.yaml[method-preset])
        config = config.update(method=..., task=..., <cli overrides>)
    without touching argv/Flags (we are not launching a training run).
    """
    import embodied  # type: ignore

    configs = sd_module.Agent.configs  # pre-parsed configs.yaml, keyed by preset name
    if method not in configs:
        raise ValueError(
            f"SafeDreamer preset '{method}' not found in configs.yaml. "
            f"Available: {sorted(configs.keys())}"
        )
    config = embodied.Config(configs["defaults"])
    config = config.update(configs[method])
    overrides = {"method": method, "task": task}
    if extra_overrides:
        overrides.update(extra_overrides)
    config = config.update(**overrides)
    return config


def _make_env(config, task: str):
    """
    Minimal re-implementation of SafeDreamer/train.py:make_env() for a single
    non-parallel environment -- we only need it for obs_space / act_space,
    never for stepping (imagination rollouts never touch the real simulator).
    """
    import importlib

    import SafeDreamer  # type: ignore

    suite, task_name = task.split("_", 1)
    ctor_paths = {
        "safetygym": "embodied.envs.safetygym:SafetyGym",
        "safetygymcoor": "embodied.envs.safetygymcoor:SafetyGymCoor",
        "safetygymmujoco": "embodied.envs.safetygym_mujoco:SafetyGymMujoco",
    }
    if suite not in ctor_paths:
        raise ValueError(
            f"safedreamer_wrapper only wired up safetygym* suites, got '{suite}'. "
            f"Extend ctor_paths if you need another suite."
        )
    module_name, cls_name = ctor_paths[suite].split(":")
    module = importlib.import_module(module_name)
    ctor = getattr(module, cls_name)

    kwargs = dict(config.env.get(suite, {}))
    kwargs["platform"] = config.jax.platform
    env = ctor(task_name, **kwargs)
    return SafeDreamer.wrap_env(env, config)


class SafeDreamerWrapper(WorldModelWrapper):
    """
    SAFEWORLD wrapper for a trained SafeDreamer OSRP-Vector agent.

    Quick start
    -----------
    >>> cfg = RolloutConfig(
    ...     horizon=50, n_rollouts=20, seed=0,
    ...     extra={
    ...         "repo_root": "/home/user/Documents/SafeDreamer",
    ...         "checkpoint_path": "/home/user/Documents/SafeDreamer/checkpoint/"
    ...                             "20240307-010600_osrp_vector_safetygymcoor_"
    ...                             "SafetyPointGoal1-v0_0.ckpt",
    ...         "method": "osrp_vector",
    ...         "task": "safetygymcoor_SafetyPointGoal1-v0",
    ...         "action_source": "random",   # or "policy"
    ...     },
    ... )
    >>> with SafeDreamerWrapper(cfg) as w:
    ...     w.load()
    ...     trajs = w.sample_rollouts()
    """

    def __init__(self, config: RolloutConfig | None = None):
        super().__init__(config)
        self._agent: Any = None          # jaxagent.Wrapper-wrapped Agent
        self._config: Any = None         # embodied.Config used to build the agent
        self._env: Any = None            # kept only for obs_space/act_space + spec
        self._loaded = False
        self._sim_mode = not JAX_AVAILABLE

        self.last_z_array: np.ndarray | None = None
        """Raw (N, T, D) latent array (deter concat stoch) from the last sample_rollouts()."""

        if self._sim_mode:
            warnings.warn(
                "jax not importable -- SafeDreamerWrapper cannot run. "
                "Activate the 'safedreamer' conda env.",
                stacklevel=2,
            )

    # ── WorldModelWrapper interface ───────────────────────────────────────────

    def load(self, **kwargs) -> None:
        if self._sim_mode:
            raise RuntimeError(
                "SafeDreamerWrapper.load() called without jax available. "
                "This wrapper has no simulation fallback -- it exists to verify "
                "a real checkpoint, not to approximate one."
            )

        extra = self.config.extra
        repo_root = kwargs.get("repo_root", extra.get("repo_root"))
        checkpoint_path = kwargs.get("checkpoint_path", extra.get("checkpoint_path"))
        method = kwargs.get("method", extra.get("method", "osrp_vector"))
        task = kwargs.get("task", extra.get("task", "safetygymcoor_SafetyPointGoal1-v0"))
        config_overrides = kwargs.get("config_overrides", extra.get("config_overrides", {}))

        if not repo_root:
            raise ValueError("SafeDreamerWrapper: 'repo_root' is required (path to the "
                              "SafeDreamer checkout, parent of the SafeDreamer/ package dir).")
        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise ValueError(f"SafeDreamerWrapper: checkpoint not found at {checkpoint_path!r}.")

        _ensure_on_path(repo_root)
        import embodied  # type: ignore
        import SafeDreamer as sd_module  # type: ignore

        self._config = _build_config(sd_module, method, task, config_overrides)

        logger.info("SafeDreamerWrapper: constructing env for obs_space/act_space only "
                     f"(task={task}) -- not used for stepping.")
        self._env = _make_env(self._config, task)

        step = embodied.Counter()
        agent = sd_module.Agent(self._env.obs_space, self._env.act_space, step, self._config)

        checkpoint = embodied.Checkpoint()
        checkpoint.agent = agent
        checkpoint.load(checkpoint_path, keys=["agent"])
        logger.info(f"SafeDreamerWrapper: loaded checkpoint {checkpoint_path}")

        self._agent = agent
        self._loaded = True

    # ── rollout sampling ──────────────────────────────────────────────────────

    def sample_rollouts(
        self,
        config: RolloutConfig | None = None,
    ) -> list[list[dict[str, float]]]:
        cfg = config or self.config
        if not self._loaded:
            self.load()

        z_array, obs_array = self._imagine(cfg)   # (N,T,D), (N,T,29)
        self.last_z_array = z_array
        return self._decode_to_trajectories(obs_array)

    def _imagine(self, cfg: RolloutConfig):
        """
        Runs `self._agent.agent.wm.imagine(policy, start, horizon)` under
        nj.pure + jax.jit, and decodes every step through the vector decoder
        head in the same pass (cheap -- MLP only, no CNN).

        Returns
        -------
        z_array   : (N, T, D) float32   deter concat stoch, D = 256 + 16*16
        obs_array : (N, T, 29) float32  decoded 'observation' vectors
        """
        import jax
        import numpy as np

        # CRITICAL: must be `SafeDreamer.ninjax`, NOT a bare `import ninjax`.
        # `_ensure_on_path()` puts SafeDreamer/SafeDreamer on sys.path so that a
        # bare `import ninjax` also succeeds -- but that creates a SECOND,
        # separate module object distinct from the one SafeDreamer.agent itself
        # uses internally (`from . import ninjax`, i.e. `SafeDreamer.ninjax`).
        # ninjax keeps its state (CONTEXT, SCOPE) as plain module-level globals,
        # so the two module objects have two independent, never-synced CONTEXT
        # dicts. Calling nj.pure()/nj.jit() on the wrong one sets up a context
        # that the real modules (created under SafeDreamer.ninjax) never see,
        # producing a misleading "Wrap impure functions in pure() before
        # running them" RuntimeError from deep inside wm.rssm.initial() even
        # though pure()/jit() were used correctly. Confirmed by direct probing:
        # `import ninjax as a; import SafeDreamer.ninjax as b; a is b` -> False.
        import SafeDreamer.ninjax as nj  # type: ignore

        raw_agent = self._agent.agent   # unwrap jaxagent.Wrapper -> raw nj.Module Agent
        wm = raw_agent.wm
        params = self._agent.varibs if hasattr(self._agent, "varibs") else None
        if params is None:
            raise RuntimeError(
                "Could not find loaded parameters on the wrapped Agent "
                "(expected `.varibs`, jaxagent.JAXAgent's parameter store)."
            )

        T = cfg.horizon
        action_source = cfg.extra.get("action_source", cfg.action_source)
        act_space = self._env.act_space
        # 'action' is the only key task_behavior policies use; reset/log_* keys excluded.
        act_key = "action"

        def _random_policy_fn(feat_or_state, *_):
            sp = act_space[act_key]
            shape = feat_or_state["deter"].shape[:1] + sp.shape
            return jax.random.uniform(nj.rng(), shape, minval=-1.0, maxval=1.0)

        def _encode_start(obs0_np):
            # Encoder bridge (matches sample_paired_rollouts()'s _encode_and_imagine):
            # ground the imagination's starting latent in a REAL env observation
            # instead of the model's fixed wm.rssm.initial() point. Necessary for
            # the calibration (ĉ_err, computed from encoder-bridge-started paired
            # rollouts) to be statistically applicable to rho* -- confirmed by
            # trial that the two populations differ hugely (84/100 vs 7/20 spec
            # satisfaction) when rho* rollouts used the fixed initial() point
            # while calibration rollouts used real-encoded starts; conformal
            # calibration requires calibration and test data to be exchangeable
            # (same distribution), which a fixed vs. diverse starting point
            # violates.
            obs_for_model = {
                "observation": jax.numpy.asarray(obs0_np, dtype=jax.numpy.float32)[None],
                "is_terminal": jax.numpy.zeros((1,), dtype=jax.numpy.float32),
                "is_first": jax.numpy.ones((1,), dtype=bool),
            }
            processed = raw_agent.preprocess(obs_for_model)
            embed = wm.encoder(processed)
            zero_act = jax.numpy.zeros((1, act_space[act_key].shape[0]), dtype=jax.numpy.float32)
            start, _ = wm.rssm.obs_step(wm.rssm.initial(1), zero_act, embed, processed["is_first"])
            return start

        def _imagine_fn_random(obs0_np):
            start = _encode_start(obs0_np)
            start["is_terminal"] = jax.numpy.zeros((1,), dtype=bool)
            traj = wm.imagine(_random_policy_fn, start, T)
            decoded = wm.heads["decoder"](traj)
            obs = decoded["observation"].mode()
            return traj, obs

        def _imagine_fn_policy(obs0_np):
            # osrp_vector's deployed policy is NOT task_behavior.ac.actor --
            # Agent.policy(obs, state, mode='eval') in agent.py checks
            # config.expl_behavior and, for planner types (CEMPlanner,
            # CCEPlanner, PIDPlanner -- osrp_vector uses CCEPlanner), dispatches
            # to expl_behavior.policy() instead of the plain actor network.
            # Confirmed by trial: sampling directly from task_behavior.ac.actor
            # produced a 0/5 hazard_avoidance satisfaction rate that does not
            # reflect what this checkpoint actually does when deployed.
            #
            # CCEPlanner.policy(latent, planner_state) is a receding-horizon
            # MPC-style planner: it runs its OWN nested wm.imagine() calls
            # internally to score candidate action sequences and returns only
            # the first action of the refined plan, plus an updated
            # planner_state (action_mean/action_std) to warm-start the next
            # call. This carried state has no place in wm.imagine()'s own
            # scan-based unroll, so we drive the outer T-step rollout with a
            # plain Python loop instead, calling img_step by hand -- this
            # mirrors what Agent.policy() does per real step (minus the real
            # observation encoding, which -- unlike the random-action path
            # above -- IS now grounded via _encode_start() too).
            latent = _encode_start(obs0_np)
            planner = raw_agent.expl_behavior
            planner_state = planner.initial(1)
            latents = [latent]
            for _ in range(T):
                outs, planner_state = planner.policy(latent, planner_state)
                latent = wm.rssm.img_step(latent, outs["action"])
                latents.append(latent)
            keys = latents[0].keys()
            traj = {k: jax.numpy.stack([lat[k] for lat in latents], 0) for k in keys}
            decoded = wm.heads["decoder"](traj)
            obs = decoded["observation"].mode()
            return traj, obs

        _imagine_fn = _imagine_fn_policy if action_source == "policy" else _imagine_fn_random

        # nj.pure(fun) => purified(state, rng, *args, **kwargs) -> (out, state).
        # `fun` itself must take no positional args here; randomness flows through
        # the threaded ninjax Context and is read inside via nj.rng().
        # IMPORTANT: must use ninjax's own nj.jit(), not raw jax.jit() -- nj.jit
        # does a separate create=True/ignore=True "init" pass to discover which
        # state keys a pure function creates before the real "apply" pass; a bare
        # jax.jit() skips that bookkeeping and the ninjax Context never gets
        # threaded through jit's trace correctly (confirmed by trial: raw jax.jit
        # here raises "Wrap impure functions in pure() before running them" from
        # inside wm.rssm.initial()). This mirrors jaxagent.py:_transform()'s own
        # `nj.jit(nj.pure(self.agent.policy))` pattern exactly.
        imagine_jit = nj.jit(nj.pure(_imagine_fn))

        rng = np.random.default_rng(cfg.seed)
        zero_action = np.zeros(act_space[act_key].shape, dtype=np.float32)
        z_list, obs_list = [], []
        for i in range(cfg.n_rollouts):
            # Real env reset -> encoder bridge start point (see _encode_start()
            # above). Each of the N rollouts gets its OWN fresh real starting
            # scenario, matching the diversity of sample_paired_rollouts()'s
            # calibration data -- this is what makes rho* (computed here) and
            # c_hat_err (computed from paired rollouts) exchangeable.
            obs0 = self._env.step({"reset": np.array(True), "action": zero_action})
            obs0_dp = jax.device_put(obs0["observation"].astype(np.float32))

            # jax_transfer_guard='disallow' (set in jaxagent.py) forbids implicit
            # host->device transfers, including inside jnp.array(); must use
            # jax.device_put() explicitly on a plain numpy array first.
            seed_i = jax.device_put(np.array(
                [int(rng.integers(0, 2**31)), int(rng.integers(0, 2**31))], dtype=np.uint32
            ))
            (traj, obs), _ = imagine_jit(params, seed_i, obs0_dp)
            traj_host = jax.device_get(traj)
            obs_raw = np.asarray(jax.device_get(obs))
            # WorldModel.imagine() builds traj via jaxutils.scan(), which stacks
            # along a NEW LEADING axis -- so every field in `traj` (and anything
            # computed from it, like the decoded obs) is TIME-FIRST:
            # (T+1, batch, feat), not (batch, T+1, feat). Indexing [0] here
            # (i.e. batch axis first) was wrong and silently collapsed the
            # whole rollout down to its first timestep only -- confirmed by
            # trial: it produced T=1 trajectories instead of T=horizon.
            obs_host = obs_raw[:, 0] if obs_raw.ndim == 3 else obs_raw

            deter = np.asarray(traj_host["deter"])[:, 0] if traj_host["deter"].ndim == 3 else np.asarray(traj_host["deter"])
            stoch = np.asarray(traj_host["stoch"])
            stoch = stoch[:, 0] if stoch.ndim == 4 else stoch
            T_actual = deter.shape[0]
            z = np.concatenate([deter, stoch.reshape(T_actual, -1)], axis=-1)

            # traj has T+1 steps: index 0 is the (deterministic/learned) RSSM
            # initial state z0 ~ rho, indices 1..T are the T imagined steps.
            # We keep [:T] -> [z0, imagined_1, ..., imagined_{T-1}], i.e. T
            # steps starting from z0, matching the paper's tau=(z0,...,z_{T-1})
            # -- NOTE this z0 is a fixed/learned point, not a sampled
            # distribution; see module docstring deviations.
            z_list.append(z[:T])
            obs_list.append(obs_host[:T])

            if (i + 1) % 10 == 0 or (i + 1) == cfg.n_rollouts:
                logger.info(f"SafeDreamerWrapper: imagined {i + 1}/{cfg.n_rollouts}")

        return np.stack(z_list), np.stack(obs_list)

    def _decode_to_trajectories(self, obs_array: np.ndarray) -> list[list[dict[str, float]]]:
        N, T, _ = obs_array.shape
        trajectories: list[list[dict[str, float]]] = []
        for i in range(N):
            traj = [_aps_with_sentinel(obs_array[i, t]) for t in range(T)]
            trajectories.append(traj)
        return trajectories

    def ap_keys(self) -> list[str]:
        return list(_GROUNDED_AP_KEYS) + list(_UNGROUNDED_AP_KEYS)

    # ── paired rollouts for Transfer Calibrator (Definition 3.5) ─────────────

    def sample_paired_rollouts(
        self,
        config: RolloutConfig | None = None,
    ) -> list[tuple[list[dict[str, float]], list[dict[str, float]]]]:
        """
        Generate N paired (model, environment) rollouts for conformal c_err
        calibration (Definition 3.5). Each pair shares:
          - the SAME real starting observation. Confirmed by trial that
            passing "the same seed number" to the real env and to the
            imagination does NOT align them -- they are two unrelated random
            systems (numpy RandomState for safety_gymnasium's object
            placement vs. wm.rssm.initial()'s fixed learned point, which does
            not consume the seed at all). The only way to align them is to
            take the real env's actual first observation and encode it into
            a latent via wm.encoder + wm.rssm.obs_step, then imagine forward
            from THAT latent instead of from wm.rssm.initial().
          - the SAME action sequence, applied to the real env via env.step()
            and to the imagined rollout via rssm.img_step().

        Only action_source="random" is implemented. "policy" (CCEPlanner) is
        a genuine closed-loop re-planner that chooses each action from the
        CURRENT real observation -- pairing it correctly requires stepping
        the real env and a shadow imagination in lockstep (real obs -> real
        CCEPlanner decision -> apply to both real env and shadow img_step),
        which is not implemented here.
        """
        cfg = config or self.config
        if not self._loaded:
            self.load()

        action_source = cfg.extra.get("action_source", cfg.action_source)
        if action_source != "random":
            raise NotImplementedError(
                f"SafeDreamerWrapper.sample_paired_rollouts() only supports "
                f"action_source='random', got {action_source!r}. Pairing a "
                f"CCEPlanner-driven rollout requires real/imagined lockstep "
                f"stepping, which is not implemented."
            )

        import jax
        import SafeDreamer.ninjax as nj  # see _imagine() for why this exact import matters

        raw_agent = self._agent.agent
        wm = raw_agent.wm
        params = self._agent.varibs
        env = self._env
        T = cfg.horizon
        act_dim = env.act_space["action"].shape[0]

        def _encode_and_imagine(obs0_np, actions_np):
            obs_for_model = {
                "observation": jax.numpy.asarray(obs0_np, dtype=jax.numpy.float32)[None],
                "is_terminal": jax.numpy.zeros((1,), dtype=jax.numpy.float32),
                "is_first": jax.numpy.ones((1,), dtype=bool),
            }
            processed = raw_agent.preprocess(obs_for_model)
            embed = wm.encoder(processed)
            zero_act = jax.numpy.zeros((1, act_dim), dtype=jax.numpy.float32)
            latent, _ = wm.rssm.obs_step(
                wm.rssm.initial(1), zero_act, embed, processed["is_first"]
            )
            latents = [latent]
            for t in range(actions_np.shape[0]):
                a = jax.numpy.asarray(actions_np[t], dtype=jax.numpy.float32)[None]
                latent = wm.rssm.img_step(latent, a)
                latents.append(latent)
            keys = latents[0].keys()
            traj = {k: jax.numpy.stack([lat[k] for lat in latents], 0) for k in keys}
            decoded = wm.heads["decoder"](traj)
            return decoded["observation"].mode()

        encode_imagine_jit = nj.jit(nj.pure(_encode_and_imagine))

        rng = np.random.default_rng(cfg.seed)
        pairs: list[tuple[list[dict[str, float]], list[dict[str, float]]]] = []

        for i in range(cfg.n_rollouts):
            zero_action = np.zeros(act_dim, dtype=np.float32)
            obs0 = env.step({"reset": np.array(True), "action": zero_action})
            action_seq = rng.uniform(-1.0, 1.0, size=(T, act_dim)).astype(np.float32)

            # Real replay: step the SAME env forward with action_seq.
            real_obs_list = [obs0["observation"].astype(np.float32)]
            for t in range(T):
                o = env.step({"reset": np.array(False), "action": action_seq[t]})
                real_obs_list.append(o["observation"].astype(np.float32))
            real_obs_arr = np.stack(real_obs_list)  # (T+1, 29)

            # Imagined: encode obs0 -> z0, then img_step through action_seq.
            seed_i = jax.device_put(np.array(
                [int(rng.integers(0, 2**31)), int(rng.integers(0, 2**31))], dtype=np.uint32
            ))
            obs0_dp = jax.device_put(obs0["observation"].astype(np.float32))
            actions_dp = jax.device_put(action_seq)
            imagined_obs, _ = encode_imagine_jit(params, seed_i, obs0_dp, actions_dp)
            imagined_obs_host = np.asarray(jax.device_get(imagined_obs))[:, 0]  # (T+1, 29)

            model_traj = [_aps_with_sentinel(imagined_obs_host[t]) for t in range(T + 1)]
            env_traj = [_aps_with_sentinel(real_obs_arr[t]) for t in range(T + 1)]
            pairs.append((model_traj, env_traj))

            if (i + 1) % 5 == 0 or (i + 1) == cfg.n_rollouts:
                logger.info(f"SafeDreamerWrapper: paired rollout {i + 1}/{cfg.n_rollouts}")

        return pairs

    def close(self) -> None:
        self._agent = None
        self._config = None
        self._env = None
        logger.debug("SafeDreamerWrapper: resources released.")
