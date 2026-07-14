"""
eval_l3_live.py — Formal L3 stl_sequential_zones verdict on carla_roundabout.

Anchors are harvested from LIVE CARLA episodes (deployment distribution;
the training process manages/rotates the replay buffer, so replay anchors
are not reliable while training runs). 80-step imagination (hrz_required=79).

IMPORTANT: use a verification-dedicated CARLA on a DISTANT port (default 2100)
— CARLA binds ports N..N+2, and a nearby verification instance can crash the
training server (see verification doc, ops incident 2026-07-13).

  python eval_l3_live.py [--port 2100] [--episodes 5] [--checkpoint ...]
"""

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "/home/bot/SafeWorld")
from cardreamer.probe_common import (GOAL_XY, ROUNDABOUT_CKPT_FROZEN,
                                           build_imagine_fn, fit_position_probe,
                                           in_ring, near_goal, wrap_obs)

H, CHUNK, ANCHOR_EVERY = 80, 20, 10


def main(args):
    import jax
    from wrappers.cardreamer_wrapper import CarDreamerWrapper
    from configs.settings import RolloutConfig
    from specs.stl_specs import get_stl_spec_by_id
    from core.stl_monitor import monitor_rollouts

    probe = fit_position_probe()

    print("[1/3] load model ...", flush=True)
    cfg = RolloutConfig(n_rollouts=CHUNK, horizon=H, seed=0, action_source="actor")
    w = CarDreamerWrapper(cfg)
    w.load(checkpoint_path=args.checkpoint)
    ja = w._jax_agent
    dev = ja.policy_devices[0]
    bi = w._burn_in

    print(f"[2/3] harvest anchors from {args.episodes} live episodes "
          f"(CARLA :{args.port}) ...", flush=True)
    import car_dreamer
    env, _ = car_dreamer.create_task(
        "carla_roundabout", ["--env.world.carla_port", str(args.port)])
    anchors = []
    for ep in range(args.episodes):
        raw = env.reset(); done = False; step = 0; pol = None
        frames, acts = [], []
        while not done:
            outs, pol = ja.policy(wrap_obs(raw, step == 0), pol, mode="eval")
            aoh = np.asarray(outs["action"])[0].astype(np.float32)
            frames.append(raw["birdeye_wpt"].copy()); acts.append(aoh)
            raw, _r, done, _i = env.step(int(np.argmax(aoh)))
            if step >= bi and step % ANCHOR_EVERY == 0:
                anchors.append((np.stack(frames[step - bi:step + 1]),
                                np.stack(acts[step - bi:step])))
            step += 1
        print(f"  ep {ep+1}: len={step} anchors={len(anchors)}", flush=True)
        time.sleep(2.0)
    env.close()

    print(f"[3/3] imagine {H} steps from {len(anchors)} anchors ...", flush=True)
    fn = build_imagine_fn(ja, H, w._n_acts, bi, decode=False)
    trajs, start_pos = [], []
    for s in range(0, len(anchors), CHUNK):
        batch = anchors[s:s + CHUNK]
        if len(batch) < CHUNK:                      # keep one JIT signature
            batch = batch + batch[:CHUNK - len(batch)]
        rng = ja._next_rngs(ja.policy_devices)
        o = jax.device_put(np.stack([b[0] for b in batch]), dev)
        a = jax.device_put(np.stack([b[1] for b in batch]), dev)
        deter, _ = fn(ja.varibs, rng, o, a)
        deter = np.asarray(jax.device_get(deter))
        n_real = min(CHUNK, len(anchors) - s)
        pred = probe.predict(deter.reshape(-1, deter.shape[-1])).reshape(H, CHUNK, 2)
        for i in range(n_real):
            pos = pred[:, i]
            za = in_ring(pos).astype(float)
            zb = near_goal(pos).astype(float)
            trajs.append([{"zone_a": za[t], "zone_b": zb[t]} for t in range(H)])
            start_pos.append(pos[0])

    spec = get_stl_spec_by_id("stl_sequential_zones")
    res = monitor_rollouts(spec["formula"], trajs)
    margins = np.array(res.margins)
    n = len(trajs)
    print(f"\nL3 stl_sequential_zones (live anchors n={n}, horizon={H}):")
    print(f"  SAT {int((margins>0).sum())}/{n}   rho*={res.rho_star:+.3f}   "
          f"verdict: {'WARRANT' if res.rho_star > 0 else 'VIOLATION'}")
    d_goal = np.linalg.norm(np.array(start_pos) - GOAL_XY, axis=1)
    live = (d_goal > 8.0) & (d_goal < 45.0)
    print(f"  anchors d_goal p10/p50/p90: {np.percentile(d_goal,10):.1f}/"
          f"{np.percentile(d_goal,50):.1f}/{np.percentile(d_goal,90):.1f}")
    if live.any():
        ml = margins[live]
        print(f"  physically-live anchors (8<d<45m, n={int(live.sum())}): "
              f"SAT {int((ml>0).sum())}/{int(live.sum())}")
    np.savez("/tmp/l3_roundabout.npz", margins=margins,
             start_pos=np.array(start_pos))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=2100)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--checkpoint", default=ROUNDABOUT_CKPT_FROZEN)
    main(p.parse_args())
