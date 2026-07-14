"""
collect_goal_probe.py — Position probe + goal-reach detection for carla_roundabout

Goal (from task definition ego_path, Town03 world coords): (4.2, -43.2)
Labels: exact ego world position (ego_x, ego_y) from env info — every step.

Part A (collect, needs CARLA on Town03):
  Drive N real episodes with the roundabout checkpoint.
  Per step  : posterior deter h_t  +  (ego_x, ego_y)
  Per anchor: imagination prior deter sequence (HORIZON steps) + (ep, step) index
              so prior predictions can be aligned with the episode's REAL future.

Part B (check, no CARLA):
  1. Ridge probe h -> (x,y), GroupKFold by episode: R^2, RMSE [m]
  2. Prior-side positional error vs aligned real future, by imagination depth
  3. Goal-reach detection: model (probe on prior) vs real (env coords),
     confusion matrix over anchors with a full HORIZON of real future.

Usage:
  conda activate cardreamer && cd /home/bot/SafeWorld
  python collect_goal_probe.py --episodes 20                # collect + check
  python collect_goal_probe.py --check_only                 # re-run analysis
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, "/home/bot/CarDreamer")
sys.path.insert(0, "/home/bot/SafeWorld")

SAVE_DIR     = "/tmp/goal_probe_data"
TASK         = "carla_roundabout"
CKPT         = "/home/bot/CarDreamer/logdir/carla_roundabout/checkpoint.ckpt"
GOAL_XY      = np.array([4.2, -43.2])   # last ego_path point (tasks.yaml)
GOAL_RADIUS  = 5.0                       # metres: reached iff dist < this
BURN_IN      = 3
HORIZON      = 50
ANCHOR_EVERY = 10


from cardreamer.probe_common import (  # shared, was duplicated here
    build_imagine_from_latent_fn as build_imagine_fn,
    wrap_obs as _wrap_obs,
)


def collect(args):
    import jax
    os.makedirs(args.save_dir, exist_ok=True)

    print("[1/3] Loading roundabout checkpoint ...", flush=True)
    from wrappers.cardreamer_wrapper import CarDreamerWrapper
    from configs.settings import RolloutConfig
    cfg = RolloutConfig(n_rollouts=1, horizon=HORIZON, seed=0, action_source="actor")
    w = CarDreamerWrapper(cfg)
    w.load(checkpoint_path=CKPT)
    jax_agent = w._jax_agent

    print("[2/3] Connecting to CARLA (Town03) ...", flush=True)
    import car_dreamer
    env, _ = car_dreamer.create_task(TASK)

    print("[3/3] Building imagination JIT ...", flush=True)
    imagine_fn = build_imagine_fn(jax_agent, HORIZON)

    post_h, post_xy, post_ep, post_step = [], [], [], []
    prior_h, anchor_ep, anchor_step = [], [], []
    ep_summary = []

    print(f"\nCollecting {args.episodes} episodes ...\n", flush=True)
    for ep in range(args.episodes):
        raw_obs, info = env.reset(), {}
        done, step = False, 0
        is_dest = is_col = False
        pol_state = None

        while not done:
            obs_dict = _wrap_obs(raw_obs, step == 0)
            outs, pol_state = jax_agent.policy(obs_dict, pol_state, mode="eval")
            latent = pol_state[0][0]
            h_np = np.asarray(latent["deter"])[0]

            # step env FIRST to get this step's info (ego position after action)?
            # No — record position BEFORE stepping: env info from previous step.
            # At step 0 info is empty; use planner start point instead.
            act_oh  = np.asarray(outs["action"])[0]
            raw_obs, _r, done, info = env.step(int(np.argmax(act_oh)))

            # info now corresponds to the state h_t transitioned into is (t+1);
            # pair h_t (posterior after obs_t) with ego position of obs_{t+1}'s
            # info is off by one step (0.1s, <1m) — acceptable at probe scale,
            # and consistent across posterior and real-future alignment.
            post_h.append(h_np)
            post_xy.append([float(info.get("ego_x", np.nan)),
                            float(info.get("ego_y", np.nan))])
            post_ep.append(ep)
            post_step.append(step)

            if step >= BURN_IN and step % ANCHOR_EVERY == 0:
                dev = jax_agent.policy_devices[0]
                rng = jax_agent._next_rngs(jax_agent.policy_devices)
                deter = jax.device_put(np.asarray(latent["deter"]), dev)
                stoch = jax.device_put(np.asarray(latent["stoch"]), dev)
                logit = jax.device_put(np.asarray(latent["logit"]), dev)
                h_prior_jax, _ = imagine_fn(jax_agent.varibs, rng, deter, stoch, logit)
                prior_h.append(np.asarray(jax.device_get(h_prior_jax))[:, 0, :])
                anchor_ep.append(ep)
                anchor_step.append(step)

            if info.get("is_collision", False):
                is_col = True
            if info.get("destination_reached", False):
                is_dest = True
            step += 1

        ep_summary.append((step, int(is_dest), int(is_col)))
        print(f"  ep {ep+1:3d}/{args.episodes}  len={step:4d}  "
              f"dest={is_dest}  col={is_col}  anchors={len(prior_h)}", flush=True)

    env.close()

    np.savez(os.path.join(args.save_dir, "posterior.npz"),
             h=np.array(post_h), xy=np.array(post_xy),
             ep=np.array(post_ep), step=np.array(post_step),
             ep_summary=np.array(ep_summary, dtype=float))
    np.savez(os.path.join(args.save_dir, "prior.npz"),
             h=np.array(prior_h),
             anchor_ep=np.array(anchor_ep), anchor_step=np.array(anchor_step))
    print(f"\nSaved to {args.save_dir}: {len(post_h)} posterior steps, "
          f"{len(prior_h)} anchors", flush=True)


def check(save_dir):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    post  = np.load(os.path.join(save_dir, "posterior.npz"))
    prior = np.load(os.path.join(save_dir, "prior.npz"))
    h, xy   = post["h"], post["xy"]
    eps, st = post["ep"], post["step"]
    hp      = prior["h"]                     # (M, HORIZON, h_dim)
    a_ep, a_st = prior["anchor_ep"], prior["anchor_step"]

    ok = ~np.isnan(xy).any(axis=1)
    h, xy, eps, st = h[ok], xy[ok], eps[ok], st[ok]

    print("\n" + "="*66)
    print("  Position Probe (h -> world x,y)  +  Goal-Reach Transferability")
    print(f"  goal={tuple(GOAL_XY)}  radius={GOAL_RADIUS}m  horizon={HORIZON}")
    print("="*66)

    # ── C0: posterior probe quality (episode-grouped CV) ─────────────────────
    gkf = GroupKFold(n_splits=min(5, len(np.unique(eps))))
    errs = []
    for tr, te in gkf.split(h, xy, groups=eps):
        probe = Ridge(alpha=10.0).fit(h[tr], xy[tr])
        errs.append(np.linalg.norm(probe.predict(h[te]) - xy[te], axis=1))
    errs = np.concatenate(errs)
    ss_res = (errs**2).sum()
    ss_tot = ((xy - xy.mean(0))**2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"\n  [C0] posterior probe: R²={r2:.3f}  "
          f"RMSE={np.sqrt((errs**2).mean()):.2f}m  p50={np.percentile(errs,50):.2f}m  "
          f"p90={np.percentile(errs,90):.2f}m"
          f"   -> {'PASS' if r2 > 0.7 else 'FAIL'} (need R²>0.7)")

    probe = Ridge(alpha=10.0).fit(h, xy)     # final probe on all posterior data

    # ── index real positions by (ep, step) for alignment ─────────────────────
    pos_lookup = {(int(e), int(s)): xy[i] for i, (e, s) in enumerate(zip(eps, st))}

    # ── C1: prior positional error vs aligned real future ────────────────────
    pred = probe.predict(hp.reshape(-1, hp.shape[-1])).reshape(hp.shape[0], HORIZON, 2)
    by_depth = {k: [] for k in (1, 5, 10, 15, 25, 50)}
    model_reach, real_reach = [], []
    n_full = 0
    for i in range(len(hp)):
        real_future = [pos_lookup.get((int(a_ep[i]), int(a_st[i]) + k))
                       for k in range(1, HORIZON + 1)]
        for k in by_depth:
            rf = real_future[k - 1]
            if rf is not None:
                by_depth[k].append(np.linalg.norm(pred[i, k - 1] - rf))
        # goal detection over the window (only anchors with full real future)
        if all(r is not None for r in real_future):
            n_full += 1
            m_d = np.linalg.norm(pred[i] - GOAL_XY, axis=1).min()
            r_d = np.linalg.norm(np.array(real_future) - GOAL_XY, axis=1).min()
            model_reach.append(m_d < GOAL_RADIUS)
            real_reach.append(r_d < GOAL_RADIUS)

    print(f"\n  [C1] prior position error vs real future (aligned, metres):")
    for k, v in by_depth.items():
        if v:
            print(f"       depth {k:3d}: p50={np.percentile(v,50):6.2f}  "
                  f"p90={np.percentile(v,90):6.2f}  n={len(v)}")

    # ── C1b: decision-relevant error metrics ─────────────────────────────────
    # (a) dist-to-goal error |‖pred−G‖ − ‖real−G‖| — the exact quantity the
    #     goal predicate depends on (≤ Euclidean error by triangle inequality)
    # (b) arc-length progress error — longitudinal error along ego_path
    path = np.array([[-52.6, 1.0], [-23.0, 7.5], [-17.0, 11.7],
                     [13.3, -13.2], [7.6, -21.8], [4.2, -43.2]])
    seg   = np.diff(path, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seglen)])

    def arclen(p):
        """Project point(s) (...,2) onto the polyline, return arc-length [m]."""
        p = np.atleast_2d(p)
        best_d = np.full(len(p), np.inf); best_s = np.zeros(len(p))
        for j in range(len(seg)):
            t = np.clip(((p - path[j]) @ seg[j]) / (seglen[j]**2), 0.0, 1.0)
            proj = path[j] + t[:, None] * seg[j]
            d = np.linalg.norm(p - proj, axis=1)
            upd = d < best_d
            best_d[upd] = d[upd]
            best_s[upd] = cumlen[j] + t[upd] * seglen[j]
        return best_s

    gd_err = {k: [] for k in (1, 5, 10, 15, 25, 50)}
    al_err = {k: [] for k in (1, 5, 10, 15, 25, 50)}
    for i in range(len(hp)):
        for k in gd_err:
            rf = pos_lookup.get((int(a_ep[i]), int(a_st[i]) + k))
            if rf is not None:
                pd_ = pred[i, k - 1]
                gd_err[k].append(abs(np.linalg.norm(pd_ - GOAL_XY)
                                     - np.linalg.norm(rf - GOAL_XY)))
                al_err[k].append(abs(arclen(pd_)[0] - arclen(rf)[0]))
    print(f"\n  [C1b] decision-relevant errors (p50 / p90, metres):")
    print(f"        {'depth':>5s}  {'dist-to-goal':>18s}  {'arc-progress':>18s}")
    for k in gd_err:
        if gd_err[k]:
            print(f"        {k:5d}  "
                  f"{np.percentile(gd_err[k],50):7.2f} /{np.percentile(gd_err[k],90):7.2f}   "
                  f"{np.percentile(al_err[k],50):7.2f} /{np.percentile(al_err[k],90):7.2f}")

    # ── C2: goal-reach agreement ──────────────────────────────────────────────
    m = np.array(model_reach); r = np.array(real_reach)
    if n_full:
        tp = int((m & r).sum());  fp = int((m & ~r).sum())
        fn = int((~m & r).sum()); tn = int((~m & ~r).sum())
        agree = (tp + tn) / n_full
        print(f"\n  [C2] goal-reach (F<{GOAL_RADIUS}m within {HORIZON} steps), "
              f"{n_full} full-future anchors:")
        print(f"       model∧real={tp}  model-only(FP)={fp}  "
              f"real-only(FN)={fn}  neither={tn}")
        print(f"       agreement={agree:.1%}  "
              f"-> {'PASS' if agree > 0.8 else 'MARGINAL' if agree > 0.6 else 'FAIL'}")
    else:
        print("\n  [C2] no anchors with full real future — episodes too short?")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",   type=int, default=20)
    p.add_argument("--save_dir",   default=SAVE_DIR)
    p.add_argument("--check_only", action="store_true")
    args = p.parse_args()
    if not args.check_only:
        collect(args)
    check(args.save_dir)
