"""
calibrate_cerr.py — Properly paired cerr calibration for CarDreamer (Definition 3.5)

For each anchor step t of a real CARLA episode:
  shared prefix : real frames [t-burn_in .. t] + real actions  → RSSM posterior
  model side    : imagine `horizon` steps from that posterior, decode, CV-extract
  env side      : the episode's actual next `horizon` real frames, CV-extract

Both sides use the SAME extract_aps_from_image, so CV bias cancels and the
per-pair distortion  cerr_i = max_t |hazard^M(t) - hazard^E(t)|  isolates
model error (encoder/RSSM/decoder + rollout drift).

ĉ_err = split-conformal (1-δ_err) UPPER quantile of {cerr_i}  (error upper bound).

Usage:
  conda activate cardreamer
  cd /home/bot/SafeWorld
  python calibrate_cerr.py [--episodes 6] [--horizon 50] [--stride 50]
                           [--delta_err 0.05] [--rho_star -0.628]
"""

import sys, argparse, math, time
import numpy as np

sys.path.insert(0, "/home/bot/CarDreamer")
sys.path.insert(0, "/home/bot/SafeWorld")

UNCERTAIN_SENTINEL = -999.0
AP_KEY  = "hazard_dist"
CHUNK   = 20   # imagination batch size (bounds decoder GPU memory)


def load_wrapper():
    from wrappers.cardreamer_wrapper import CarDreamerWrapper
    from configs.settings import RolloutConfig
    cfg = RolloutConfig(n_rollouts=CHUNK, horizon=50, seed=0, action_source="actor")
    w = CarDreamerWrapper(cfg)
    w.load()
    return w


def make_env():
    import car_dreamer
    env, _ = car_dreamer.create_task("carla_four_lane")
    return env


def wrap_obs(raw_obs, is_first):
    return {
        "birdeye_wpt": raw_obs["birdeye_wpt"][None],
        "is_first":    np.array([is_first], dtype=bool),
        "is_last":     np.array([False],    dtype=bool),
        "is_terminal": np.array([False],    dtype=bool),
        "reward":      np.array([0.0],      dtype=np.float32),
    }


def reset_with_retry(env, retries=3, delay=5.0):
    for attempt in range(retries):
        try:
            return env.reset()
        except RuntimeError as e:
            if attempt < retries - 1:
                print(f"  [CARLA] reset failed ({e}), retrying in {delay}s ...", flush=True)
                time.sleep(delay)
            else:
                raise


def run_episode(jax_agent, env):
    """Drive one real episode; return (frames uint8 list, onehot action list)."""
    raw_obs   = reset_with_retry(env)
    pol_state = None
    frames, acts = [], []
    done, step = False, 0
    while not done:
        outs, pol_state = jax_agent.policy(wrap_obs(raw_obs, step == 0),
                                           pol_state, mode="eval")
        act_oh = np.asarray(outs["action"])[0].astype(np.float32)  # (n_acts,)
        frames.append(raw_obs["birdeye_wpt"].copy())
        acts.append(act_oh)
        raw_obs, _r, done, _info = env.step(int(np.argmax(act_oh)))
        step += 1
    return frames, acts


def main(args):
    import jax
    from wrappers.cardreamer_wrapper import extract_aps_from_image, VEHICLE_COLOR_TOL

    def cv_extract(img):
        return extract_aps_from_image(img, color_tol=VEHICLE_COLOR_TOL,
                                      bbox_inflate_px=0)

    print("\n" + "=" * 65)
    print("  Paired cerr calibration (Definition 3.5) — hazard_dist")
    print(f"  burn-in prefix → {{model: imagine {args.horizon}, env: real {args.horizon}}}")
    print("=" * 65 + "\n")

    print("[1/4] Loading model ...", flush=True)
    w = load_wrapper()
    jax_agent = w._jax_agent
    burn_in   = w._burn_in
    dev       = jax_agent.policy_devices[0]

    print("[2/4] Connecting to CARLA ...", flush=True)
    env = make_env()

    # ── Drive real episodes, harvest anchors ─────────────────────────────────
    print(f"[3/4] Driving {args.episodes} real episodes ...", flush=True)
    anchors = []   # (obs_seq (burn_in+1,H,W,C), act_seq (burn_in,n_acts), real_frames [horizon])
    for ep in range(args.episodes):
        # CARLA tick can time out under GPU contention (XLA autotune); retry
        # the whole episode on a fresh reset rather than dying.
        for attempt in range(3):
            try:
                frames, acts = run_episode(jax_agent, env)
                break
            except RuntimeError as e:
                print(f"  [CARLA] episode failed ({e}); retry {attempt+1}/3 in 15s",
                      flush=True)
                time.sleep(15.0)
        else:
            print("  [CARLA] giving up on this episode"); continue
        n_anchor_before = len(anchors)
        t = burn_in
        while t + args.horizon < len(frames):
            anchors.append((
                np.stack(frames[t - burn_in: t + 1]),
                np.stack(acts[t - burn_in: t]),
                frames[t + 1: t + 1 + args.horizon],
            ))
            t += args.stride
        print(f"  ep {ep+1:2d}: len={len(frames):4d}  "
              f"anchors+={len(anchors)-n_anchor_before}  total={len(anchors)}", flush=True)
        if ep < args.episodes - 1:
            time.sleep(2.0)
    env.close()

    if not anchors:
        print("No anchors collected (episodes too short?)"); return

    # ── Imagination from each anchor's posterior, in chunks ──────────────────
    print(f"\n[4/4] Imagining {args.horizon} steps from {len(anchors)} anchors "
          f"(chunks of {CHUNK}) ...", flush=True)
    scores, dropped = [], 0
    series_m, series_e = [], []   # raw per-step hazard_dist, NaN where SENTINEL
    for start in range(0, len(anchors), CHUNK):
        batch = anchors[start:start + CHUNK]
        m = len(batch)
        fn  = w._get_fn("actor", m, args.horizon)
        rng = jax_agent._next_rngs(jax_agent.policy_devices)
        obs_jax = jax.device_put(np.stack([a[0] for a in batch]), dev)
        act_jax = jax.device_put(np.stack([a[1] for a in batch]), dev)
        images_u8, _ = fn(jax_agent.varibs, rng, obs_jax, act_jax)
        images_np = np.asarray(jax.device_get(images_u8))  # (horizon+1, m, H, W, C)
        if w._bgr_observations:
            images_np = images_np[..., ::-1]

        for i in range(m):
            sm = np.full(args.horizon, np.nan)
            se = np.full(args.horizon, np.nan)
            for t in range(args.horizon):
                r_m = cv_extract(images_np[t + 1, i]).get(AP_KEY, UNCERTAIN_SENTINEL)
                r_e = cv_extract(batch[i][2][t]).get(AP_KEY, UNCERTAIN_SENTINEL)
                if r_m != UNCERTAIN_SENTINEL:
                    sm[t] = r_m
                if r_e != UNCERTAIN_SENTINEL:
                    se[t] = r_e
            series_m.append(sm)
            series_e.append(se)
            diffs = np.abs(sm - se)
            if np.isnan(diffs).all():
                dropped += 1
            else:
                scores.append(np.nanmax(diffs))
        print(f"  chunk {start//CHUNK + 1}: {len(scores)} pairs scored", flush=True)

    # ── Conformal upper quantile ──────────────────────────────────────────────
    n = len(scores)
    sorted_s = sorted(scores)
    idx = min(max(math.ceil((1.0 - args.delta_err) * (n + 1)) - 1, 0), n - 1)
    c_hat = sorted_s[idx]

    print("\n" + "=" * 65)
    print(f"  cerr over {n} paired rollouts ({dropped} dropped, all-SENTINEL)")
    print("=" * 65)
    print(f"  mean={np.mean(scores):.3f}  p50={np.percentile(scores,50):.3f}  "
          f"p90={np.percentile(scores,90):.3f}  max={max(scores):.3f}")
    print(f"  ĉ_err (1-δ={1-args.delta_err:.2f} conformal upper): {c_hat:.3f}")

    sm = np.array(series_m)   # (n_pairs, horizon) raw hazard_dist, NaN = SENTINEL
    se = np.array(series_e)
    np.savez("cerr_calibration.npz", scores=np.array(scores),
             series_model=sm, series_env=se,
             c_hat_err=c_hat, delta_err=args.delta_err,
             horizon=args.horizon, burn_in=burn_in)
    print("  saved → cerr_calibration.npz")

    def conformal_upper(vals, delta):
        vals = sorted(v for v in vals if not np.isnan(v))
        k = len(vals)
        if k == 0:
            return float("nan")
        j = min(max(math.ceil((1.0 - delta) * (k + 1)) - 1, 0), k - 1)
        return vals[j]

    # ── step-level cerr vs horizon (Def 3.5, max_t |h^M - h^E|) ──────────────
    sd = np.abs(sm - se)
    print(f"\n  Step-level cerr(h) = max_t≤h |h^M - h^E|  (Def 3.5):")
    print(f"  Formula-level cerr(h) = |min_t≤h h^M - min_t≤h h^E|  "
          f"(distortion of G-robustness):")
    for h in (5, 10, 15, 20, 30, 40, args.horizon):
        if h > args.horizon:
            continue
        step_c  = np.nanmax(sd[:, :h], axis=1)
        form_c  = np.abs(np.nanmin(sm[:, :h], axis=1) - np.nanmin(se[:, :h], axis=1))
        print(f"    h={h:3d}:  step p50={np.nanpercentile(step_c,50):7.3f} "
              f"ĉ={conformal_upper(step_c, args.delta_err):7.3f}   |   "
              f"formula p50={np.nanpercentile(form_c,50):7.3f} "
              f"ĉ={conformal_upper(form_c, args.delta_err):7.3f}")

    if args.rho_star is not None:
        rho_net = args.rho_star - c_hat
        print(f"\n  Transfer check vs model-side ρ*={args.rho_star:+.3f}:")
        print(f"  ρ_net = ρ* - ĉ_err = {rho_net:+.3f}  → "
              f"{'TRANSFERS (strict)' if rho_net > 0 else 'does NOT transfer'}")
        print(f"  (rerun main.py with --c_hat {c_hat:.3f} for the full verdict)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",  type=int,   default=6)
    p.add_argument("--horizon",   type=int,   default=50)
    p.add_argument("--stride",    type=int,   default=50)
    p.add_argument("--delta_err", type=float, default=0.05)
    p.add_argument("--rho_star",  type=float, default=None,
                   help="model-side ρ* to combine with the new ĉ_err")
    main(p.parse_args())
