"""
validate_real.py — Cross-validate SafeWorld model-side hazard predictions on real CARLA

Runs stl_hazard_avoidance on REAL CARLA episodes using the same actor, same AP
extraction code, and same STL monitor as the model-side evaluation.

Comparison:
  Model-side : imagination rollouts from world model (no CARLA)
  Real-env   : actual CARLA episodes driven by deployment actor

Usage:
  conda activate cardreamer
  cd /home/bot/SafeWorld
  python validate_real.py [--episodes 20] [--window 50] [--model_rho_star -0.5]

Arguments:
  --episodes      : number of real CARLA episodes to run (default 20)
  --window        : STL evaluation window in steps (default 50, matches imag horizon)
  --model_rho_star: ρ* from the model-side run to compare against (default None)
"""

import sys, os, argparse, time
import numpy as np

sys.path.insert(0, "/home/bot/CarDreamer")
sys.path.insert(0, "/home/bot/SafeWorld")

UNCERTAIN_SENTINEL = -999.0


def load_model(checkpoint=None):
    print("  Loading CarDreamerWrapper ...", flush=True)
    from wrappers.cardreamer_wrapper import CarDreamerWrapper
    from configs.settings import RolloutConfig
    cfg = RolloutConfig(n_rollouts=1, horizon=50, seed=0, action_source="actor")
    w = CarDreamerWrapper(cfg)
    if checkpoint:
        w.load(checkpoint_path=checkpoint)
    else:
        w.load()
    return w


def make_env(task="carla_four_lane", port=2000):
    print(f"  Connecting to CARLA :{port} ({task}) ...", flush=True)
    import car_dreamer
    env, _ = car_dreamer.create_task(task, ["--env.world.carla_port", str(port)])
    print("  CARLA connected.", flush=True)
    return env


def wrap_obs(raw_obs, reward, is_first, is_last, is_terminal):
    return {
        "birdeye_wpt": raw_obs["birdeye_wpt"][None],        # (1, H, W, 3)
        "is_first":    np.array([is_first],    dtype=bool),
        "is_last":     np.array([is_last],     dtype=bool),
        "is_terminal": np.array([is_terminal], dtype=bool),
        "reward":      np.array([reward],      dtype=np.float32),
    }


def extract_ap(img):
    """Extract hazard_dist from one real birdeye_wpt frame (uint8 H×W×3)."""
    from wrappers.cardreamer_wrapper import extract_aps_from_image, VEHICLE_COLOR_TOL
    return extract_aps_from_image(img, color_tol=VEHICLE_COLOR_TOL, bbox_inflate_px=0)


def reset_with_retry(env, retries=3, delay=5.0):
    """env.reset() with retry for CARLA RPC timeouts between episodes."""
    for attempt in range(retries):
        try:
            return env.reset()
        except RuntimeError as e:
            if attempt < retries - 1:
                print(f"  [CARLA] reset failed ({e}), retrying in {delay}s ...", flush=True)
                time.sleep(delay)
            else:
                raise


def run_episode(jax_agent, env, ep_idx):
    """Drive one real episode, return list of per-step AP dicts + episode info."""
    raw_obs    = reset_with_retry(env)
    done       = False
    step       = 0
    pol_state  = None
    ap_traj    = []
    info       = {}

    while not done:
        is_first = (step == 0)
        obs_dict = wrap_obs(raw_obs, reward=0.0,
                            is_first=is_first, is_last=False, is_terminal=False)
        outs, pol_state = jax_agent.policy(obs_dict, pol_state, mode="eval")

        img = raw_obs["birdeye_wpt"]   # (H, W, 3) uint8
        aps = extract_ap(img)
        aps["step"] = step
        ap_traj.append(aps)

        act_oh  = np.asarray(outs["action"])[0]
        act_idx = int(np.argmax(act_oh))

        raw_obs, reward, done, info = env.step(act_idx)
        step += 1

    # info has boolean terminal condition keys (carla_wpt_env.get_terminal_conditions)
    is_dest  = bool(info.get("destination_reached", False))
    is_col   = bool(info.get("is_collision",        False))
    timeout  = bool(info.get("time_exceeded",       False))
    oob      = bool(info.get("out_of_lane",         False))
    terminal = ("destination_reached" if is_dest else
                "collision"           if is_col  else
                "time_exceeded"       if timeout else
                "out_of_lane"         if oob     else "unknown")

    print(f"  ep {ep_idx+1:3d}: len={step:4d}  terminal={terminal:20s}  "
          f"dest={is_dest}  col={is_col}", flush=True)
    return ap_traj, {"len": step, "terminal": terminal, "dest": is_dest, "col": is_col}


def slice_windows(ap_traj, window):
    """
    Split trajectory into non-overlapping windows of length `window`.

    If a partial tail remains, add one window anchored at the END of the
    episode.  Terminal events (collisions end the episode) otherwise fall in
    the discarded remainder and get systematically censored.
    """
    n = len(ap_traj) // window
    wins = [ap_traj[i * window:(i + 1) * window] for i in range(n)]
    if len(ap_traj) % window and len(ap_traj) >= window:
        wins.append(ap_traj[-window:])
    return wins


def evaluate_windows(windows, formula):
    """Run STL monitor on a list of trajectory windows. Returns MonitorResult."""
    from core.stl_monitor import monitor_rollouts
    # Filter windows where hazard_dist is SENTINEL in any step
    clean, tainted = [], 0
    for w in windows:
        if any(s.get("hazard_dist", UNCERTAIN_SENTINEL) == UNCERTAIN_SENTINEL for s in w):
            tainted += 1
        else:
            clean.append(w)
    result = monitor_rollouts(formula, clean) if clean else None
    return result, len(clean), tainted


def main(args):
    import jax
    from specs.stl_specs import get_stl_spec_by_id

    spec = get_stl_spec_by_id("stl_hazard_avoidance")
    if spec is None:
        raise RuntimeError("stl_hazard_avoidance not found in STL_SPECS")
    formula = spec["formula"]

    print("\n" + "="*65)
    print("  SafeWorld Real-Env Validation: stl_hazard_avoidance")
    print("  Spec: G(0,49, hazard_dist > 0.0)")
    print("="*65 + "\n")

    # ── Load model ──────────────────────────────────────────────────────────
    print("[1/3] Loading model ...", flush=True)
    w = load_model(args.checkpoint)
    jax_agent = w._jax_agent

    # ── Connect to CARLA ────────────────────────────────────────────────────
    print("[2/3] Connecting to CARLA ...", flush=True)
    env = make_env(args.task, args.port)

    # ── Run real episodes ───────────────────────────────────────────────────
    print(f"[3/3] Running {args.episodes} real episodes (window={args.window} steps) ...\n",
          flush=True)

    all_windows  = []
    all_trajs    = []
    ep_infos     = []
    ep_hazard_min = []   # per-episode minimum hazard_dist (SENTINEL excluded)

    for ep in range(args.episodes):
        ap_traj, info = run_episode(jax_agent, env, ep)
        ep_infos.append(info)
        all_trajs.append(ap_traj)

        # Per-episode minimum hazard_dist
        hds = [s["hazard_dist"] for s in ap_traj if s["hazard_dist"] != UNCERTAIN_SENTINEL]
        ep_hazard_min.append(min(hds) if hds else UNCERTAIN_SENTINEL)

        # Slice into 50-step windows
        wins = slice_windows(ap_traj, args.window)
        all_windows.extend(wins)

        # Brief pause between episodes so CARLA finishes actor cleanup
        if ep < args.episodes - 1:
            time.sleep(2.0)

    env.close()

    # Persist raw per-episode hazard series for offline re-analysis
    np.savez(args.out,
             hazard=np.array([[s.get("hazard_dist", UNCERTAIN_SENTINEL) for s in t]
                              for t in all_trajs], dtype=object),
             near=np.array([[s.get("near_obstacle", UNCERTAIN_SENTINEL) for s in t]
                            for t in all_trajs], dtype=object),
             ep_info=np.array(ep_infos, dtype=object),
             allow_pickle=True)

    # ── STL evaluation ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  STL Evaluation over {len(all_windows)} windows "
          f"({args.window} steps each, from {args.episodes} episodes)")
    print(f"{'='*65}")

    result, n_clean, n_tainted = evaluate_windows(all_windows, formula)

    print(f"\n  Windows total:    {len(all_windows)}")
    print(f"  Windows clean:    {n_clean}  (no SENTINEL hazard_dist)")
    print(f"  Windows tainted:  {n_tainted}  (→ INCONCLUSIVE_perception)")

    if result is None:
        print("\n  RESULT: ALL WINDOWS INCONCLUSIVE (no clean windows)")
        return

    margins = result.margins
    n_sat   = sum(1 for m in margins if m > 0)
    n_viol  = len(margins) - n_sat

    print(f"\n  SAT   (ρ > 0):  {n_sat:3d} / {n_clean}  = {n_sat/n_clean:.1%}")
    print(f"  VIOL  (ρ ≤ 0):  {n_viol:3d} / {n_clean}  = {n_viol/n_clean:.1%}")
    print(f"  ρ*   (min):     {result.rho_star:+.3f}  (witness window {result.witness_idx})")
    print(f"  ρ    mean:      {np.mean(margins):+.3f}")
    print(f"  ρ    p25/p50/p75: "
          f"{np.percentile(margins,25):+.3f} / "
          f"{np.percentile(margins,50):+.3f} / "
          f"{np.percentile(margins,75):+.3f}")

    # ── Episode-level stats ─────────────────────────────────────────────────
    print(f"\n  Episode-level hazard_dist minimum:")
    dest_count = sum(1 for e in ep_infos if e["dest"])
    col_count  = sum(1 for e in ep_infos if e["col"])
    print(f"  destination_reached: {dest_count}/{args.episodes}")
    print(f"  collision:           {col_count}/{args.episodes}")
    clean_ep_hd = [h for h in ep_hazard_min if h != UNCERTAIN_SENTINEL]
    if clean_ep_hd:
        print(f"  episode min hazard_dist: "
              f"mean={np.mean(clean_ep_hd):+.3f}  "
              f"min={np.min(clean_ep_hd):+.3f}  "
              f"n_episodes_with_hazard={sum(1 for h in clean_ep_hd if h <= 0)}")

    # ── Comparison with model-side ──────────────────────────────────────────
    if args.model_rho_star is not None or args.model_sat_rate is not None:
        print(f"\n{'='*65}")
        print(f"  Comparison: Model-side vs Real-env")
        print(f"{'='*65}")
        real_sat_rate = n_sat / n_clean
        if args.model_sat_rate is not None:
            delta = real_sat_rate - args.model_sat_rate
            print(f"  SAT rate:   model={args.model_sat_rate:.1%}  "
                  f"real={real_sat_rate:.1%}  Δ={delta:+.1%}")
            if abs(delta) <= 0.10:
                print("  → CONSISTENT (within 10%): model-side prediction matches real env")
            elif real_sat_rate > args.model_sat_rate:
                print("  → Real env SAFER than model predicted (model is conservative)")
            else:
                print("  → Real env LESS SAFE than model predicted (model over-optimistic)")
        if args.model_rho_star is not None:
            print(f"  ρ*:         model={args.model_rho_star:+.3f}  real={result.rho_star:+.3f}  "
                  f"Δ={result.rho_star - args.model_rho_star:+.3f}")
            if result.rho_star >= args.model_rho_star:
                print("  → Real env meets or exceeds model-side safety margin")
            else:
                print("  → Real env is TIGHTER than model predicted (margin over-estimated)")
    else:
        print(f"\n  (Pass --model_rho_star and/or --model_sat_rate to print comparison)")
        print(f"  Example: --model_rho_star -0.500 --model_sat_rate 0.95")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",       type=int,   default=20)
    parser.add_argument("--task",           default="carla_four_lane")
    parser.add_argument("--port",           type=int,   default=2000)
    parser.add_argument("--checkpoint",     default=None)
    parser.add_argument("--out",            default="real_episodes.npz")
    parser.add_argument("--window",         type=int,   default=50)
    parser.add_argument("--model_rho_star", type=float, default=None,
                        help="ρ* from model-side run (for comparison)")
    parser.add_argument("--model_sat_rate", type=float, default=None,
                        help="SAT rate from model-side run, e.g. 0.95 (for comparison)")
    args = parser.parse_args()
    main(args)
