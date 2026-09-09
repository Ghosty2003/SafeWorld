"""
collect_goal_probe_fourlane.py — Position probe + goal-reach + speed feasibility
for carla_four_lane.

Goal: all 4 lanes share the same destination y (-58); x varies by lane
(5.8/9.0/12.2/15.6) and doesn't matter for "reached the end of the road".
So goal_dist(t) = |ego_y(t) - (-58)|, radius in metres.

Also collects real speed_norm (ground truth, CARLA velocity magnitude) at
every step, to test whether velocity is recoverable as a finite-difference
of consecutive PROBED positions (v_probe = |pos(t+1)-pos(t)| / dt, dt=0.1s) —
a different extraction path from the reward-based one already ruled out.

Part A (collect, needs CARLA on Town04):
  Per step  : posterior deter h_t + (ego_x, ego_y, speed_norm)
  Per anchor: imagination prior deter sequence (HORIZON steps), for goal C1/C2.

Part B (check, no CARLA):
  1. Ridge probe h -> (x,y), GroupKFold: R^2, RMSE [m]                     (C0)
  2. Prior-side positional error vs aligned real future, by depth          (C1)
  3. Goal-reach: model (probe on prior) vs real, confusion matrix          (C2)
  4. Velocity feasibility: real posterior-only, v_probe vs real speed_norm (V1)
     -- this is the "test if speed can be used" deliverable.

Usage:
  conda activate cardreamer && cd /home/bot/SafeWorld
  python cardreamer/collect_goal_probe_fourlane.py --episodes 20
  python cardreamer/collect_goal_probe_fourlane.py --check_only
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, "/home/bot/CarDreamer")
sys.path.insert(0, "/home/bot/SafeWorld")

SAVE_DIR     = "/tmp/goal_probe_data_fourlane"
TASK         = "carla_four_lane"
CKPT         = "/home/bot/CarDreamer/logdir/carla_four_lane/checkpoint.ckpt"
GOAL_Y       = -58.0     # shared destination y for all 4 lanes (tasks.yaml lane_end_points)
GOAL_RADIUS  = 5.0        # metres: reached iff |y - GOAL_Y| < this
BURN_IN      = 3
HORIZON      = 50
ANCHOR_EVERY = 10
DT           = 0.1        # fixed_delta_seconds (common.yaml)


from cardreamer.probe_common import (
    build_imagine_from_latent_fn as build_imagine_fn,
    wrap_obs as _wrap_obs,
)


def collect(args):
    import jax
    os.makedirs(args.save_dir, exist_ok=True)

    print("[1/3] Loading four_lane checkpoint ...", flush=True)
    from wrappers.cardreamer_wrapper import CarDreamerWrapper
    from configs.settings import RolloutConfig
    cfg = RolloutConfig(n_rollouts=1, horizon=HORIZON, seed=0, action_source="actor")
    w = CarDreamerWrapper(cfg)
    w.load(checkpoint_path=CKPT)
    jax_agent = w._jax_agent

    print("[2/3] Connecting to CARLA (Town04) ...", flush=True)
    import car_dreamer
    env, _ = car_dreamer.create_task(TASK)

    print("[3/3] Building imagination JIT ...", flush=True)
    imagine_fn = build_imagine_fn(jax_agent, HORIZON)

    post_h, post_xy, post_speed, post_ep, post_step = [], [], [], [], []
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

            act_oh = np.asarray(outs["action"])[0]
            raw_obs, _r, done, info = env.step(int(np.argmax(act_oh)))

            post_h.append(h_np)
            post_xy.append([float(info.get("ego_x", np.nan)),
                            float(info.get("ego_y", np.nan))])
            post_speed.append(float(info.get("speed_norm", np.nan)))
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
             h=np.array(post_h), xy=np.array(post_xy), speed=np.array(post_speed),
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
    h, xy, speed = post["h"], post["xy"], post["speed"]
    eps, st = post["ep"], post["step"]
    hp      = prior["h"]
    a_ep, a_st = prior["anchor_ep"], prior["anchor_step"]

    ok = ~np.isnan(xy).any(axis=1) & ~np.isnan(speed)
    h, xy, speed, eps, st = h[ok], xy[ok], speed[ok], eps[ok], st[ok]

    print("\n" + "=" * 66)
    print("  four_lane Position Probe + Goal-Reach + Speed Feasibility")
    print(f"  goal: |y - ({GOAL_Y})| < {GOAL_RADIUS}m   horizon={HORIZON}")
    print("=" * 66)

    # ── C0: posterior probe quality (episode-grouped CV) ─────────────────────
    gkf = GroupKFold(n_splits=min(5, len(np.unique(eps))))
    errs = []
    for tr, te in gkf.split(h, xy, groups=eps):
        probe = Ridge(alpha=10.0).fit(h[tr], xy[tr])
        errs.append(np.linalg.norm(probe.predict(h[te]) - xy[te], axis=1))
    errs = np.concatenate(errs)
    ss_res = (errs ** 2).sum()
    ss_tot = ((xy - xy.mean(0)) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"\n  [C0] posterior probe: R²={r2:.3f}  "
        f"RMSE={np.sqrt((errs**2).mean()):.2f}m  p50={np.percentile(errs,50):.2f}m  "
        f"p90={np.percentile(errs,90):.2f}m"
        f"   -> {'PASS' if r2 > 0.7 else 'FAIL'} (need R²>0.7)")

    probe = Ridge(alpha=10.0).fit(h, xy)

    # ── V1: velocity feasibility (real posterior data only, no imagination) ──
    # v_probe(t) = |probe(h_{t+1}) - probe(h_t)| / dt, compared to real speed_norm(t)
    print(f"\n  [V1] Velocity feasibility (derived from consecutive PROBED "
        f"positions, dt={DT}s):")
    pred_xy = probe.predict(h)
    v_probe_list, v_real_list = [], []
    for e in np.unique(eps):
        sel = np.where(eps == e)[0]
        order = np.argsort(st[sel])
        sel = sel[order]
        s_sorted = st[sel]
        for i in range(len(sel) - 1):
            if s_sorted[i + 1] - s_sorted[i] != 1:
                continue   # only consecutive real steps
            dx = pred_xy[sel[i + 1]] - pred_xy[sel[i]]
            v_probe_list.append(np.linalg.norm(dx) / DT)
            v_real_list.append(speed[sel[i]])
    v_probe_arr = np.array(v_probe_list)
    v_real_arr = np.array(v_real_list)
    err = np.abs(v_probe_arr - v_real_arr)
    corr = np.corrcoef(v_probe_arr, v_real_arr)[0, 1]
    print(f"       n_pairs={len(v_probe_arr)}")
    print(f"       real speed_norm:  mean={v_real_arr.mean():.2f}  "
        f"p50={np.percentile(v_real_arr,50):.2f}  p90={np.percentile(v_real_arr,90):.2f} m/s")
    print(f"       derived v_probe:  mean={v_probe_arr.mean():.2f}  "
        f"p50={np.percentile(v_probe_arr,50):.2f}  p90={np.percentile(v_probe_arr,90):.2f} m/s")
    print(f"       abs error:        mean={err.mean():.2f}  p50={np.percentile(err,50):.2f}  "
        f"p90={np.percentile(err,90):.2f} m/s")
    print(f"       correlation(v_probe, v_real) = {corr:.3f}")
    snr = (v_real_arr.std() / err.std()) if err.std() > 0 else float("inf")
    print(f"       signal/noise ratio (std(real speed) / std(error)) = {snr:.2f}")
    usable = corr > 0.7 and snr > 2.0
    print(f"       -> {'USABLE (corr>0.7, SNR>2)' if usable else 'NOT RELIABLE ENOUGH'} "
        f"for a step-level stl_speed_limit-style spec")

    # ── index real positions by (ep, step) for goal alignment ────────────────
    pos_lookup = {(int(e), int(s)): xy[i] for i, (e, s) in enumerate(zip(eps, st))}

    def goal_dist(p):
        return abs(np.atleast_2d(p)[:, 1] - GOAL_Y)

    # ── C1: prior positional error vs aligned real future ────────────────────
    pred = probe.predict(hp.reshape(-1, hp.shape[-1])).reshape(hp.shape[0], HORIZON, 2)
    by_depth = {k: [] for k in (1, 5, 10, 15, 25, 50)}
    model_reach, real_reach, start_gd = [], [], []
    n_full = 0
    for i in range(len(hp)):
        real_future = [pos_lookup.get((int(a_ep[i]), int(a_st[i]) + k))
                       for k in range(1, HORIZON + 1)]
        for k in by_depth:
            rf = real_future[k - 1]
            if rf is not None:
                by_depth[k].append(np.linalg.norm(pred[i, k - 1] - rf))
        anchor_xy = pos_lookup.get((int(a_ep[i]), int(a_st[i])))
        if anchor_xy is not None:
            start_gd.append(goal_dist(anchor_xy)[0])
        if all(r is not None for r in real_future):
            n_full += 1
            m_d = goal_dist(pred[i]).min()
            r_d = goal_dist(np.array(real_future)).min()
            model_reach.append(m_d < GOAL_RADIUS)
            real_reach.append(r_d < GOAL_RADIUS)

    print(f"\n  [C1] prior position error vs real future (aligned, metres):")
    for k, v in by_depth.items():
        if v:
            print(f"       depth {k:3d}: p50={np.percentile(v,50):6.2f}  "
                f"p90={np.percentile(v,90):6.2f}  n={len(v)}")

    if start_gd:
        start_gd = np.array(start_gd)
        print(f"\n  anchor |y-goal_y| distribution: p10={np.percentile(start_gd,10):.1f} "
            f"p50={np.percentile(start_gd,50):.1f} p90={np.percentile(start_gd,90):.1f}")

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
        print(f"       agreement={agree:.1%}  real_base_rate={r.mean():.1%}  "
            f"model_base_rate={m.mean():.1%}")
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
