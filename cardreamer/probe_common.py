"""
probe_common.py — Shared helpers for CarDreamer latent-probe verification.

Everything probe-related lives in this package:
  collect_probe_data.py   four_lane progress probe (cum_wpt, trajectory-level only)
  collect_goal_probe.py   roundabout position probe (x,y — step-level validated)
  eval_l2_roundabout.py   L2 safe_goal_reach verdict (CV hazard + probe goal)
  eval_l3_live.py         L3 sequential-zones verdict (80-step, live anchors)
  offline_checks.py       L3/L5 offline analyses on saved probe data

Task geometry (carla_roundabout, Town03, from tasks.yaml ego_path):
  GOAL_XY : final ego_path point
  RING_C / RING_R : circulation-ring circle fit through the three mid-route points
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/home/bot/CarDreamer")
sys.path.insert(0, "/home/bot/SafeWorld")

# ── carla_roundabout task geometry ────────────────────────────────────────────

GOAL_XY  = np.array([4.2, -43.2])
RING_C   = np.array([-6.4, -6.3])
RING_R   = 20.9
ROUTE    = np.array([[-52.6, 1.0], [-23.0, 7.5], [-17.0, 11.7],
                     [13.3, -13.2], [7.6, -21.8], [4.2, -43.2]])

GOAL_PROBE_DATA = "/tmp/goal_probe_data"
ROUNDABOUT_CKPT = "/home/bot/CarDreamer/logdir/carla_roundabout/checkpoint.ckpt"
# frozen verified snapshot (training overwrites the logdir checkpoint):
ROUNDABOUT_CKPT_FROZEN = \
    "/home/bot/SafeWorld/verified_checkpoints/carla_roundabout_20260713.ckpt"


# ── probe data / fitting ──────────────────────────────────────────────────────

def load_posterior(save_dir: str = GOAL_PROBE_DATA):
    """Return (h, xy, ep, step, ep_summary) with NaN-position rows removed."""
    post = np.load(f"{save_dir}/posterior.npz")
    ok = ~np.isnan(post["xy"]).any(axis=1)
    return (post["h"][ok], post["xy"][ok], post["ep"][ok], post["step"][ok],
            post["ep_summary"])


def load_prior(save_dir: str = GOAL_PROBE_DATA):
    """Return (h_prior (M,H,hdim), anchor_ep, anchor_step)."""
    prior = np.load(f"{save_dir}/prior.npz")
    return prior["h"], prior["anchor_ep"], prior["anchor_step"]


def fit_position_probe(save_dir: str = GOAL_PROBE_DATA, alpha: float = 10.0):
    """Ridge probe h -> (x, y), fit on all posterior data."""
    from sklearn.linear_model import Ridge
    h, xy, *_ = load_posterior(save_dir)
    return Ridge(alpha=alpha).fit(h, xy)


def position_lookup(save_dir: str = GOAL_PROBE_DATA) -> dict:
    """{(episode, step): real xy} for aligning prior predictions to real futures."""
    _, xy, eps, st, _ = load_posterior(save_dir)
    return {(int(e), int(s)): xy[i] for i, (e, s) in enumerate(zip(eps, st))}


# ── path geometry ─────────────────────────────────────────────────────────────

def path_dist(p: np.ndarray, path: np.ndarray = ROUTE) -> np.ndarray:
    """Distance from point(s) (...,2) to a polyline. NOTE: the raw ROUTE
    polyline is the chord path — real driving follows road arcs and deviates
    up to ~21m from it (see verification doc N.4). Use only with that caveat,
    or pass an empirical centerline."""
    p = np.atleast_2d(p)
    seg = np.diff(path, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    best = np.full(len(p), np.inf)
    for j in range(len(seg)):
        t = np.clip(((p - path[j]) @ seg[j]) / seglen[j] ** 2, 0.0, 1.0)
        best = np.minimum(best,
                          np.linalg.norm(p - (path[j] + t[:, None] * seg[j]), axis=1))
    return best


def in_ring(p: np.ndarray, slack: float = 4.0) -> np.ndarray:
    """zone_a: inside/near the circulation ring."""
    return np.linalg.norm(np.atleast_2d(p) - RING_C, axis=-1) < RING_R + slack


def near_goal(p: np.ndarray, radius: float = 8.0) -> np.ndarray:
    """zone_b: near the exit/goal."""
    return np.linalg.norm(np.atleast_2d(p) - GOAL_XY, axis=-1) < radius


# ── imagination builders (nj.jit, burn-in from real obs) ─────────────────────

def build_imagine_fn(jax_agent, horizon: int, n_acts: int, burn_in: int,
                     decode: bool = False):
    """
    f(varibs, rng, obs_seq, act_seq) -> deter_seq            (decode=False)
                                     -> (images_u8, deter)   (decode=True)
      obs_seq : (n, burn_in+1, 128, 128, 3) uint8
      act_seq : (n, burn_in, n_acts) float32 one-hot
      deter   : (horizon, n, hdim);  images_u8: (horizon, n, 128, 128, 3)
    Burn-in: cold obs_step at t=0, then real (obs, action) steps, then imagine.
    """
    from dreamerv3 import ninjax as nj
    import jax.numpy as jnp
    from wrappers.cardreamer_wrapper import _get_actor

    wm, actor = jax_agent.agent.wm, _get_actor(jax_agent)
    dev = jax_agent.policy_devices[0]

    def _fn(obs_seq, act_seq):
        nb = obs_seq.shape[0]
        img0 = obs_seq[:, 0].astype(jnp.float32) / 255.0
        latent = wm.rssm.initial(nb)
        latent, _ = wm.rssm.obs_step(latent, jnp.zeros((nb, n_acts), jnp.float32),
                                     wm.encoder({"birdeye_wpt": img0}),
                                     jnp.ones((nb,), bool))
        for t in range(1, burn_in + 1):
            imgt = obs_seq[:, t].astype(jnp.float32) / 255.0
            latent, _ = wm.rssm.obs_step(latent, act_seq[:, t - 1],
                                         wm.encoder({"birdeye_wpt": imgt}),
                                         jnp.zeros((nb,), bool))
        latent["is_terminal"] = jnp.zeros((nb,), jnp.float32)

        def policy(state):
            return actor(state).sample(seed=nj.rng())

        traj = wm.imagine(policy, latent, horizon)
        if not decode:
            return traj["deter"][1:]
        decoded = wm.heads["decoder"](traj)
        imgs = jnp.clip(jnp.round(decoded["birdeye_wpt"].mode() * 255.0),
                        0, 255).astype(jnp.uint8)
        return imgs[1:], traj["deter"][1:]

    return nj.jit(nj.pure(_fn), device=dev)


def build_imagine_from_latent_fn(jax_agent, horizon: int):
    """
    f(varibs, rng, deter, stoch, logit) -> deter_seq (horizon, n, hdim).
    Start imagination from an existing posterior latent (e.g. captured from
    jax_agent.policy() while driving a real episode).
    """
    from dreamerv3 import ninjax as nj
    import jax.numpy as jnp
    from wrappers.cardreamer_wrapper import _get_actor

    wm, actor = jax_agent.agent.wm, _get_actor(jax_agent)
    dev = jax_agent.policy_devices[0]

    def _fn(deter, stoch, logit):
        start = {"deter": deter, "stoch": stoch, "logit": logit,
                 "is_terminal": jnp.zeros((deter.shape[0],), jnp.float32)}

        def policy(state):
            return actor(state).sample(seed=nj.rng())

        return wm.imagine(policy, start, horizon)["deter"][1:]

    return nj.jit(nj.pure(_fn), device=dev)


# ── CARLA episode driving helpers ─────────────────────────────────────────────

def wrap_obs(raw_obs, is_first: bool):
    img = raw_obs["birdeye_wpt"] if isinstance(raw_obs, dict) else raw_obs
    return {
        "birdeye_wpt": img[np.newaxis],
        "is_first":    np.array([is_first], dtype=bool),
        "is_last":     np.array([False],    dtype=bool),
        "is_terminal": np.array([False],    dtype=bool),
        "reward":      np.array([0.0],      dtype=np.float32),
    }
