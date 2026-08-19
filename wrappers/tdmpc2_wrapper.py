"""
wrappers/tdmpc2_wrapper.py

SAFEWORLD wrapper for TD-MPC2 (Hansen et al., 2023) / dm_control walker
task family (walker-walk / walker-run / walker-stand and the project's
custom -backwards checkpoint variants).

Validated before this file was written (scratchpad, not reproduced here):
  - Official TD-MPC2 agent (encoder/dynamics/reward/pi/Q, SimNorm, CEM/MPPI
    planner) loads correctly from the released .pt checkpoints via the
    upstream repo's own code path (parse_cfg -> make_env -> TDMPC2 ->
    agent.load), tensordict pinned to 0.8.3 to match the checkpoint format.
  - `plan_from_z()` below (the CEM/MPPI planner entered directly at a given
    latent z instead of an obs) was regression-tested against the official,
    unmodified `TDMPC2._plan(obs)` in tdmpc2_plan_consistency_test.py: a
    fixed real-obs sequence fed to both, multi-step (t0=False chained
    across steps, exercising the `_prev_mean` warm-start state -- the
    flagged risk), same seed before each step, eval_mode=True AND
    eval_mode=False. Result: action and _prev_mean matched EXACTLY
    (0.00e+00 max abs diff) at every step. plan_from_z is therefore used
    as-is; it is a literal copy of _plan()'s body (tdmpc2.py:138-206) minus
    the initial `z = self.model.encode(obs, task)` line -- not a
    reconstruction.
  - Height-probe validity (C0/C1) for walker-walk/seed=3: see
    wrappers/tdmpc2_probes.py docstring. Height probes are NOT part of
    this wrapper -- see "AP extraction is external" below.

AP extraction is external
--------------------------
This wrapper's only job is producing latent trajectories (encode / next /
sample_rollouts). It does not hardcode a height probe, an upright probe, or
any other AP-specific logic. Callers attach a pluggable extractor:

    from wrappers.tdmpc2_probes import fit_height_probe, make_height_ap_extractor
    probe = fit_height_probe("/path/to/posterior.npz")
    w.set_ap_extractor(make_height_ap_extractor(probe), keys=["height"])
    trajectories = w.sample_rollouts(cfg, action_source="mpc_plan")

action_source has no default
-----------------------------
Unlike the sibling wrappers (which read action_source from RolloutConfig
with a silent fallback default), sample_rollouts() here requires it as an
explicit keyword argument with NO default. This is deliberate: "mpc_plan"
(real CEM/MPPI planning, matching actual TD-MPC2 deployment) and "pi_prior"
(bare deterministic policy, tanh(mean), NOT how this checkpoint is actually
deployed) answer different questions, and which one was used must never be
an implicit choice. Every trajectory produced with action_source="pi_prior"
is tagged with a `_approx_pi_prior: 1.0` marker in every step dict (mirrors
the `_perception_uncertain` marker convention in cardreamer_wrapper.py) so
downstream code cannot mistake it for a real-deployment rollout. Use
has_pi_prior_approx() to filter/route.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any, Callable, Literal

import numpy as np

from configs.settings import RolloutConfig
from .base import WorldModelWrapper

logger = logging.getLogger(__name__)

TDMPC2_SRC_DEFAULT = "/tmp/claude-1000/-home-bot-SafeWorld/e841d332-2bd3-48b3-932f-f67fc6de35a2/scratchpad/tdmpc2_src/tdmpc2"


# ══════════════════════════════════════════════════════════════════════════════
# plan_from_z -- regression-tested against official _plan(), see module docstring.
# Do not modify without re-running tdmpc2_plan_consistency_test.py.
# ══════════════════════════════════════════════════════════════════════════════

def plan_from_z(agent: Any, z0, t0: bool, eval_mode: bool, task):
    """Literal copy of TDMPC2._plan (tdmpc2.py:138-206), entered at a given
    latent z0 instead of encoding an obs. z0: [1, latent_dim] torch tensor."""
    import torch
    from common import math as tdmath  # noqa: PLC0415 (tdmpc2 src on sys.path by load())

    cfg = agent.cfg
    with torch.no_grad():
        z = z0
        if cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(cfg.horizon, cfg.num_pi_trajs, cfg.action_dim, device=agent.device)
            _z = z.repeat(cfg.num_pi_trajs, 1)
            for t in range(cfg.horizon - 1):
                pi_actions[t], _ = agent.model.pi(_z, task)
                _z = agent.model.next(_z, pi_actions[t], task)
            pi_actions[-1], _ = agent.model.pi(_z, task)

        z = z.repeat(cfg.num_samples, 1)
        mean = torch.zeros(cfg.horizon, cfg.action_dim, device=agent.device)
        std = torch.full((cfg.horizon, cfg.action_dim), cfg.max_std, dtype=torch.float, device=agent.device)
        if not t0:
            mean[:-1] = agent._prev_mean[1:]
        actions = torch.empty(cfg.horizon, cfg.num_samples, cfg.action_dim, device=agent.device)
        if cfg.num_pi_trajs > 0:
            actions[:, :cfg.num_pi_trajs] = pi_actions

        for _ in range(cfg.iterations):
            r = torch.randn(cfg.horizon, cfg.num_samples - cfg.num_pi_trajs, cfg.action_dim, device=std.device)
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, cfg.num_pi_trajs:] = actions_sample
            if cfg.multitask:
                actions = actions * agent.model._action_masks[task]

            value = agent._estimate_value(z, actions, task).nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1), cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            max_value = elite_value.max(0).values
            score = torch.exp(cfg.temperature * (elite_value - max_value))
            score = score / score.sum(0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
            std = std.clamp(cfg.min_std, cfg.max_std)
            if cfg.multitask:
                mean = mean * agent.model._action_masks[task]
                std = std * agent.model._action_masks[task]

        rand_idx = tdmath.gumbel_softmax_sample(score.squeeze(1))
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        a, std_ = actions[0], std[0]
        if not eval_mode:
            a = a + std_ * torch.randn(cfg.action_dim, device=std_.device)
        agent._prev_mean.copy_(mean)
        return a.clamp(-1, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Non-deployment-approximation marker (mirrors _perception_uncertain convention)
# ══════════════════════════════════════════════════════════════════════════════

def has_pi_prior_approx(trajectory: list[dict[str, float]]) -> bool:
    """True if this trajectory was produced with action_source='pi_prior' --
    a bare-policy approximation of deployment behavior, not real MPC-planned
    deployment. Route accordingly before treating results as primary safety
    evidence."""
    return any(step.get("_approx_pi_prior", 0.0) == 1.0 for step in trajectory)


# ══════════════════════════════════════════════════════════════════════════════
# Main wrapper class
# ══════════════════════════════════════════════════════════════════════════════

class TDMPC2Wrapper(WorldModelWrapper):
    """
    SAFEWORLD wrapper for a released TD-MPC2 checkpoint (walker task family).

    Quick start
    -----------
    >>> w = TDMPC2Wrapper()
    >>> w.load(checkpoint="/home/bot/SafeWorld/models/walker-walk-3.pt", task="walker-walk")
    >>> from wrappers.tdmpc2_probes import fit_height_probe, make_height_ap_extractor
    >>> probe = fit_height_probe("/path/to/posterior.npz")
    >>> w.set_ap_extractor(make_height_ap_extractor(probe), keys=["height"])
    >>> cfg = RolloutConfig(horizon=100, n_rollouts=20, seed=0)
    >>> trajs = w.sample_rollouts(cfg, action_source="mpc_plan")
    """

    def __init__(
        self,
        config: RolloutConfig | None = None,
        *,
        tdmpc2_src: str = TDMPC2_SRC_DEFAULT,
    ):
        super().__init__(config)
        self._tdmpc2_src = tdmpc2_src
        self._agent: Any = None
        self._env: Any = None
        self._cfg: Any = None
        self._task: str = ""
        self._dmw: Any = None            # DMControlWrapper instance (obs conversion helper)
        self._physics: Any = None        # dm_control Physics (get_state/set_state)
        self._dm_task: Any = None        # dm_control Task (get_observation)
        self._ap_extractor: Callable[[np.ndarray], dict[str, float]] | None = None
        self._ap_keys: list[str] = []

        # Public: raw (N, T, D) latent array from the last sample_latent_rollouts() call.
        self.last_z_array: np.ndarray | None = None

    # ── WorldModelWrapper interface ───────────────────────────────────────────

    def load(
        self,
        checkpoint: str = "/home/bot/SafeWorld/models/walker-walk-3.pt",
        task: str = "walker-walk",
        seed: int = 0,
        **kwargs,
    ) -> None:
        """
        Load the official TD-MPC2 agent + dm_control env, exactly matching the
        validated scratchpad loading sequence (parse_cfg -> make_env -> TDMPC2
        -> agent.load, tensordict==0.8.3 pinned to match the checkpoint format).
        """
        import sys
        if self._tdmpc2_src not in sys.path:
            sys.path.insert(0, self._tdmpc2_src)
        os.environ.setdefault("MUJOCO_GL", "egl")

        import hydra  # noqa: PLC0415
        from omegaconf import OmegaConf  # noqa: PLC0415

        hydra.utils.get_original_cwd = lambda: os.getcwd()

        from common.parser import parse_cfg  # noqa: PLC0415
        from envs import make_env  # noqa: PLC0415
        from tdmpc2 import TDMPC2  # noqa: PLC0415

        cfg = OmegaConf.load(f"{self._tdmpc2_src}/config.yaml")
        cfg.task = task
        cfg.checkpoint = checkpoint
        cfg.seed = seed
        cfg.compile = False
        cfg.obs = "state"
        cfg = parse_cfg(cfg)

        self._cfg = cfg
        self._task = task
        self._env = make_env(cfg)

        self._agent = TDMPC2(cfg)
        self._agent.load(checkpoint)
        self._agent.model.eval()

        # Direct physics/task handles for exact anchor restoration (see
        # sample_latent_rollouts_from_states). Confirmed access path: env
        # (TensorWrapper) . env (Timeout) . env (DMControlWrapper) . env
        # (action_scale.Wrapper) . physics -- dm_control.suite.walker.Physics,
        # with get_state()/set_state() giving lossless (~1e-8) round-trips.
        try:
            self._dmw = self._env.env.env
            self._physics = self._dmw.env.physics
            self._dm_task = self._dmw.env.task
        except AttributeError:
            self._dmw = self._physics = self._dm_task = None

        logger.info(f"TDMPC2Wrapper loaded: task={task} device={self._agent.device}")

    def set_ap_extractor(
        self,
        extractor: Callable[[np.ndarray], dict[str, float]],
        keys: list[str],
    ) -> None:
        """
        Attach an externally-fit AP extractor (e.g. from wrappers.tdmpc2_probes).
        This wrapper contains no probe-specific logic of its own -- see module
        docstring "AP extraction is external".
        """
        self._ap_extractor = extractor
        self._ap_keys = list(keys)

    def ap_keys(self) -> list[str]:
        return list(self._ap_keys)

    # ── Latent primitives (the wrapper's actual responsibility) ──────────────

    def encode(self, obs):
        """obs (raw torch tensor, as returned by env.reset()/env.step()) -> z [1, latent_dim]."""
        obs_dev = obs.to(self._agent.device, non_blocking=True)
        return self._agent.model.encode(obs_dev.unsqueeze(0), None)

    def next(self, z, a):
        """z [1, latent_dim], a [1, action_dim] -> z' [1, latent_dim]."""
        return self._agent.model.next(z, a, None)

    def pi_prior_action(self, z):
        """Bare deterministic policy action: tanh(mean). NOT how this checkpoint
        is actually deployed (real deployment uses CEM/MPPI, i.e. mpc_plan)."""
        import torch
        with torch.no_grad():
            mean, _ = self._agent.model._pi(z).chunk(2, dim=-1)
            return torch.tanh(mean)

    def plan_action(self, z, t0: bool, eval_mode: bool = True):
        """Real CEM/MPPI action from latent z. See plan_from_z() docstring for
        the regression-test evidence this matches official TDMPC2._plan()."""
        return plan_from_z(self._agent, z, t0=t0, eval_mode=eval_mode, task=None).unsqueeze(0)

    def _real_height(self, obs) -> float:
        return float(obs[14])  # dm_control walker obs layout: orientations(14)+height+velocity

    def _obs_from_physics_state(self, state: np.ndarray):
        """Restore an EXACT real environment state via physics.set_state()
        (lossless, ~1e-8 round-trip; see tdmpc2_recollect_with_physics_state.py)
        and reconstruct the obs the same way env.reset()/env.step() would.
        Requires load() to have resolved self._physics/self._dm_task (the
        dm_control access path confirmed for walker-walk's wrapper stack)."""
        if self._physics is None or self._dm_task is None:
            raise RuntimeError(
                "physics/task handles unavailable -- _obs_from_physics_state() "
                "requires the confirmed DMControlWrapper access path from load()."
            )
        self._env.reset()  # establishes a valid physics/task context to overwrite
        self._physics.set_state(state)
        self._physics.forward()
        obs_dict = self._dm_task.get_observation(self._physics)
        return self._dmw._obs_to_array(obs_dict)

    def _select_action(self, z, step_idx: int, action_source: str):
        if action_source == "mpc_plan":
            return self.plan_action(z, t0=(step_idx == 0), eval_mode=True)
        elif action_source == "pi_prior":
            return self.pi_prior_action(z)
        raise ValueError(f"Unknown action_source: {action_source!r}")

    def _imagine_from_obs(self, obs, action_source: str, horizon: int) -> np.ndarray:
        """Encode obs -> anchor z, then imagine `horizon` steps forward (t=1..T).
        Shared by both anchor-sourcing strategies (burn_in and physics-state)
        so the imagination loop itself -- including t0 handling -- has one
        implementation, not two independently-maintained copies."""
        import torch
        with torch.no_grad():
            z = self.encode(obs)
            z_seq = []
            for step_idx in range(horizon):
                a = self._select_action(z, step_idx, action_source)
                z = self.next(z, a)
                z_seq.append(z[0].cpu().numpy())
        return np.stack(z_seq)

    # ── Rollout sampling ──────────────────────────────────────────────────────

    # Empirically confirmed (tdmpc2_anchor_distribution_check.py, 2026-07):
    # dm_control walker-walk's env.reset() is a DEGENERATE single point --
    # torso height is exactly 1.3000 (std=0.0000) across 50 resets, despite
    # initialize_episode() randomizing joint angles. The validated (C) curve
    # (see tdmpc2_probes.py docstring) used 200 anchors taken every 20 steps
    # across 8 full 500-step MPC episodes: mean=1.286 std=0.024
    # range=[1.204, 1.366]. burn_in=0 does NOT reach that distribution -- it
    # reaches a single unvisited-in-C1 point. Default burn_in is therefore a
    # randomized range matching the C1 anchor step span (0..480 over a
    # 500-step episode), NOT 0.
    DEFAULT_BURN_IN_RANGE: tuple[int, int] = (0, 480)

    def sample_latent_rollouts(
        self,
        config: RolloutConfig | None = None,
        *,
        action_source: Literal["mpc_plan", "pi_prior"],
        burn_in: int | tuple[int, int] = DEFAULT_BURN_IN_RANGE,
    ) -> tuple[np.ndarray, dict]:
        """
        Sample N latent rollouts of length T. This is the wrapper's actual
        responsibility: encode/next (+ CEM plan_from_z for "mpc_plan"). No AP
        extraction happens here -- see sample_rollouts() / tdmpc2_probes.py.

        action_source:
          "mpc_plan"  -- real CEM/MPPI replan at every step (regression-tested
                         against official TDMPC2._plan(); matches actual
                         deployment behavior).
          "pi_prior"  -- bare deterministic policy (tanh(mean)), no planning.
                         NOT how this checkpoint is deployed.
        burn_in : int, or (lo, hi) range (default (0, 480)). Number of real-
          environment steps (using the SAME action_source mechanism) taken
          before the anchor z is recorded. If a range, each rollout draws its
          own burn_in uniformly from [lo, hi] (seeded by cfg.seed) so anchors
          are spread across the episode, matching the distribution the C1
          height-probe drift measurement was actually validated on -- a
          single fixed burn_in (especially 0) reaches only one point on the
          walking manifold, NOT the distribution results were validated on.
          Passing a single int opts out of this coverage deliberately; do so
          only if you have separately re-validated drift on that distribution.

        burn_in coverage is approximate, not exact
        -------------------------------------------
        A bootstrap check (tdmpc2_anchor_bootstrap_check.py) resampled the
        200 C1-validated real anchors (n=200, B=5000) to get the expected
        sampling distribution of std/min at n=200 under the SAME underlying
        distribution: std in [0.0219, 0.0297] at the 5th-max range. The
        burn_in=(0,480) anchor height std measured at n=200 was 0.0306 --
        above the entire bootstrap range, i.e. NOT explainable as ordinary
        sampling noise. Step-count-based burn-in is therefore a real but
        approximate proxy for "a state on the deployment manifold", not a
        validated match to the C1 anchor distribution. For anything feeding
        a formal verdict (as opposed to exploratory sampling), prefer
        sample_latent_rollouts_from_states() with the actual saved C1
        physics states instead -- exact, not approximate.

        Returns
        -------
        z_array : (N, T, latent_dim) numpy array of imagined latents
                  (t=1..T; the anchor itself is not included).
        meta    : {"action_source", "burn_in_range", "anchors": [{"rollout",
                  "burn_in"}, ...]}  -- anchors[i]["burn_in"] is the ACTUAL
                  step count used for rollout i, so downstream per-episode /
                  per-phase diagnostics (as done for the (C) curve) stay
                  reproducible.
        """
        import torch

        if self._agent is None:
            raise RuntimeError("Call load() before sample_latent_rollouts().")

        cfg = config or self.config
        N, T = cfg.n_rollouts, cfg.horizon
        rng = np.random.default_rng(cfg.seed)

        z_rollouts = []
        anchor_meta = []
        with torch.no_grad():
            for i in range(N):
                bi = int(rng.integers(burn_in[0], burn_in[1] + 1)) \
                    if isinstance(burn_in, tuple) else burn_in

                obs = self._env.reset()
                step_idx = 0
                for _ in range(bi):
                    z = self.encode(obs)
                    a = self._select_action(z, step_idx, action_source)
                    obs, _, done, _ = self._env.step(a[0].cpu())
                    step_idx += 1
                    if done:
                        break

                z = self.encode(obs)
                z_seq = []
                for _ in range(T):
                    a = self._select_action(z, step_idx, action_source)
                    z = self.next(z, a)
                    z_seq.append(z[0].cpu().numpy())
                    step_idx += 1
                z_rollouts.append(np.stack(z_seq))
                anchor_meta.append({"rollout": i, "burn_in": bi})

        z_array = np.stack(z_rollouts)  # (N, T, D)
        self.last_z_array = z_array
        meta = {"action_source": action_source, "burn_in_range": burn_in, "anchors": anchor_meta}
        return z_array, meta

    def sample_latent_rollouts_from_states(
        self,
        physics_states: np.ndarray | str,
        *,
        action_source: Literal["mpc_plan", "pi_prior"],
        horizon: int,
    ) -> tuple[np.ndarray, dict]:
        """
        Sample latent rollouts anchored at EXACT saved real-environment
        physics states, instead of approximating an anchor via a randomized
        burn_in step count (see sample_latent_rollouts()'s "burn_in coverage
        is approximate, not exact" note -- a bootstrap check showed burn_in's
        anchor-height std is NOT explainable as sampling noise around the
        validated C1 distribution). This method removes that approximation
        entirely: each anchor IS one of the real states the height probe was
        actually validated on (physics.set_state() + physics.forward(),
        confirmed lossless to ~1e-8; see
        tdmpc2_recollect_with_physics_state.py).

        physics_states : (N, state_dim) array, or a path to an .npz with a
          "physics_state" array (as saved by
          tdmpc2_recollect_with_physics_state.py). Each row must be a state
          this exact checkpoint's dm_control Physics.get_state() produced
          (state layout is task/model-specific).
        action_source, horizon : see sample_latent_rollouts().

        Each anchor starts its own imagination fresh (t0=True at the first
        imagined step), matching how the validated (C) drift curve was
        computed -- not a continuation of whatever CEM state existed during
        the original real episode.

        Returns
        -------
        z_array : (N, horizon, latent_dim) numpy array of imagined latents.
        meta    : {"action_source", "n_anchors"}
        """
        if self._agent is None:
            raise RuntimeError("Call load() before sample_latent_rollouts_from_states().")
        if self._physics is None:
            raise RuntimeError(
                "physics handle unavailable -- this checkpoint/env's wrapper "
                "stack didn't resolve the expected DMControlWrapper access path."
            )

        if isinstance(physics_states, str):
            physics_states = np.load(physics_states)["physics_state"]

        z_rollouts = []
        for i in range(len(physics_states)):
            obs = self._obs_from_physics_state(physics_states[i])
            z_rollouts.append(self._imagine_from_obs(obs, action_source, horizon))

        z_array = np.stack(z_rollouts)  # (N, horizon, D)
        self.last_z_array = z_array
        meta = {"action_source": action_source, "n_anchors": len(physics_states)}
        return z_array, meta

    def sample_rollouts(
        self,
        config: RolloutConfig | None = None,
        *,
        action_source: Literal["mpc_plan", "pi_prior"],
        ap_extractor: Callable[[np.ndarray], dict[str, float]] | None = None,
    ) -> list[list[dict[str, float]]]:
        """
        Sample N rollouts and convert to AP-dict trajectories via an
        externally-supplied extractor (no default AP scheme is baked in).

        action_source has NO default -- see module docstring. Trajectories
        produced with action_source="pi_prior" are tagged with
        `_approx_pi_prior: 1.0` in every step; check with has_pi_prior_approx()
        before treating results as primary deployment-safety evidence.
        """
        extractor = ap_extractor or self._ap_extractor
        if extractor is None:
            raise RuntimeError(
                "No ap_extractor attached. Call set_ap_extractor(...) or pass "
                "ap_extractor= explicitly -- TDMPC2Wrapper does not hardcode any "
                "AP-extraction logic (see wrappers/tdmpc2_probes.py)."
            )

        z_array, meta = self.sample_latent_rollouts(config, action_source=action_source)
        N, T, _ = z_array.shape

        trajectories: list[list[dict[str, float]]] = []
        for i in range(N):
            traj: list[dict[str, float]] = []
            for t in range(T):
                aps = extractor(z_array[i, t])
                if action_source == "pi_prior":
                    aps = {**aps, "_approx_pi_prior": 1.0}
                traj.append(aps)
            trajectories.append(traj)
        return trajectories

    def close(self) -> None:
        self._agent = None
        self._env = None
        self._cfg = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.debug("TDMPC2Wrapper: resources released.")
