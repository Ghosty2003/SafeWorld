"""
diag_cardreamer.py — AP extraction diagnostic for CarDreamer stl_safe_goal_reach

Checks:
  1. Channel order (BGR vs RGB) via multi-frame vote
  2. Per-rollout robustness distribution (SAT/VIOL breakdown)
  3. Witness rollout step-by-step AP trace
  4. goal_dist / hazard_dist distributions across all rollouts
  5. Sentinel contamination count

Run:
    conda activate cardreamer
    python diag_cardreamer.py [--spec stl_safe_goal_reach] [--n 20]
"""

import sys
import argparse
import numpy as np
sys.path.insert(0, "/home/bot/CarDreamer")

from wrappers import CarDreamerWrapper
from wrappers.cardreamer_wrapper import UNCERTAIN_SENTINEL, verify_channel_order_multi
from core.stl_monitor import monitor_rollouts
from specs import get_spec_by_id


def fmt(v: float) -> str:
    if v == UNCERTAIN_SENTINEL:
        return "  SENTINEL"
    return f"{v:+8.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec",    default="stl_safe_goal_reach")
    parser.add_argument("--n",       type=int, default=20)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--seed",    type=int, default=0)
    args = parser.parse_args()

    # ── 1. Load wrapper ---------------------------------------------------------
    from configs.settings import RolloutConfig
    roll_cfg = RolloutConfig(
        n_rollouts=args.n, horizon=args.horizon, seed=args.seed, action_source="actor"
    )
    w = CarDreamerWrapper(roll_cfg)
    w.load()
    print(f"\n{'='*65}")
    print(f" DiagCarDreamer  spec={args.spec}  N={args.n}  T={args.horizon}")
    print(f"{'='*65}\n")

    # ── 2. Channel order check --------------------------------------------------
    print("[1/5] Channel order check (multi-frame vote) ...")
    imgs = w.decode_sample(n=5, horizon=10)
    flat = [imgs[i][j] for i in range(5) for j in range(10)]
    ch = verify_channel_order_multi(flat)
    print(f"      BGR votes={ch['votes_bgr']}  RGB votes={ch['votes_rgb']}"
          f"  inconclusive={ch['inconclusive_frames']}")
    print(f"      → {ch['recommendation']}")
    if ch['note']:
        print(f"      Note: {ch['note']}")

    # ── 3. Sample rollouts ------------------------------------------------------
    print(f"\n[2/5] Sampling {args.n} rollouts (T={args.horizon}) ...")
    import time
    t0 = time.perf_counter()
    rollouts = w.sample_rollouts()
    elapsed = time.perf_counter() - t0
    print(f"      Done in {elapsed:.2f}s")

    # ── 4. Sanity-check first decoded frame -------------------------------------
    print("\n[3/5] Spot-checking first frame of rollout 0 ...")
    frame0 = flat[0]
    non_black = (frame0.astype(np.int32).sum(axis=-1) > 10).sum()
    print(f"      non-black pixels in decode_sample[0][0]: {non_black} / {128*128}")
    if non_black < 100:
        print("      ⚠ Frame looks mostly black — decoder may not be working correctly")
    else:
        print("      Frame has visual content (OK)")

    # ── 5. STL monitor ----------------------------------------------------------
    print(f"\n[4/5] STL monitor ({args.spec}) ...")
    spec = get_spec_by_id(args.spec)
    if spec is None:
        print(f"  ✗ Unknown spec '{args.spec}'. Available: {[s['id'] for s in __import__('specs').ALL_SPECS]}")
        return
    monitor = monitor_rollouts(spec["formula"], rollouts)

    print(f"      ρ*={monitor.rho_star:+.4f}  "
          f"sat={monitor.n_satisfied}/{monitor.n_rollouts}  "
          f"witness=#{monitor.witness_idx}")

    # Per-rollout summary table
    print(f"\n      {'i':>3}  {'ρ':>8}  {'label':>5}  "
          f"{'goal_min':>9}  {'goal_max':>9}  {'g_snt':>5}  "
          f"{'haz_min':>8}  {'haz_max':>8}  {'h_snt':>5}")
    print("      " + "-" * 72)

    for i, (margin, traj) in enumerate(zip(monitor.margins, rollouts)):
        goal_vals = [s["goal_dist"]    for s in traj]
        haz_vals  = [s["hazard_dist"]  for s in traj]
        g_sentinel = sum(1 for v in goal_vals if v == UNCERTAIN_SENTINEL)
        h_sentinel = sum(1 for v in haz_vals  if v == UNCERTAIN_SENTINEL)
        g_real = [v for v in goal_vals if v != UNCERTAIN_SENTINEL]
        h_real = [v for v in haz_vals  if v != UNCERTAIN_SENTINEL]
        g_min  = min(g_real) if g_real else float("nan")
        g_max  = max(g_real) if g_real else float("nan")
        h_min  = min(h_real) if h_real else float("nan")
        h_max  = max(h_real) if h_real else float("nan")
        label  = "SAT" if margin > 0 else "VIOL"
        marker = " ← witness" if i == monitor.witness_idx else ""
        print(f"      {i:3d}  {margin:+8.3f}  {label:>5}  "
              f"{g_min:+9.3f}  {g_max:+9.3f}  {g_sentinel:5d}  "
              f"{h_min:+8.3f}  {h_max:+8.3f}  {h_sentinel:5d}"
              f"{marker}")

    # ── 6. Witness rollout step-by-step trace -----------------------------------
    widx = monitor.witness_idx
    print(f"\n[5/5] Witness rollout #{widx} step-by-step AP trace (ρ={monitor.rho_star:+.4f})")
    print(f"      {'t':>4}  {'goal_dist':>10}  {'hazard_dist':>11}  {'near_obs':>9}  {'velocity':>9}")
    print("      " + "-" * 55)
    for t, aps in enumerate(rollouts[widx]):
        g  = aps.get("goal_dist",    UNCERTAIN_SENTINEL)
        h  = aps.get("hazard_dist",  UNCERTAIN_SENTINEL)
        no = aps.get("near_obstacle",UNCERTAIN_SENTINEL)
        v  = aps.get("velocity",     UNCERTAIN_SENTINEL)
        flag = ""
        if g == UNCERTAIN_SENTINEL or h == UNCERTAIN_SENTINEL:
            flag = "  ← SENTINEL"
        elif g < -50 or h < -50:
            flag = "  ← extreme value"
        print(f"      {t:4d}  {fmt(g)}  {fmt(h)}  {fmt(no)}  {fmt(v)}{flag}")

    # ── 7. Aggregate AP statistics ---------------------------------------------
    print("\n──── Aggregate AP stats (across all rollouts, non-sentinel steps) ────")
    for key in ("goal_dist", "hazard_dist", "near_obstacle"):
        vals = [
            s[key]
            for traj in rollouts
            for s in traj
            if s.get(key, UNCERTAIN_SENTINEL) != UNCERTAIN_SENTINEL
        ]
        snt_count = sum(
            1
            for traj in rollouts
            for s in traj
            if s.get(key, UNCERTAIN_SENTINEL) == UNCERTAIN_SENTINEL
        )
        total = args.n * args.horizon
        if vals:
            print(f"  {key:20s}: n={len(vals):5d}/{total}  "
                  f"sentinel={snt_count:4d}  "
                  f"min={min(vals):+7.3f}  "
                  f"max={max(vals):+7.3f}  "
                  f"mean={sum(vals)/len(vals):+7.3f}")
        else:
            print(f"  {key:20s}: ALL {total} steps are SENTINEL — extraction completely failed")

    print("\n──── Diagnosis hints ─────────────────────────────────────────────────")
    goal_snt_total = sum(
        1 for traj in rollouts for s in traj
        if s.get("goal_dist", UNCERTAIN_SENTINEL) == UNCERTAIN_SENTINEL
    )
    if goal_snt_total > args.n * args.horizon * 0.5:
        print(f"  ⚠ goal_dist SENTINEL rate {goal_snt_total}/{args.n*args.horizon} "
              f"({100*goal_snt_total/(args.n*args.horizon):.0f}%) — "
              f"waypoint detection is failing on most frames.")
        print(f"    Check channel order (above) and MIN_CLUSTER_PX threshold.")

    if ch['votes_bgr'] + ch['votes_rgb'] < 5:
        print(f"  ⚠ Only {ch['votes_bgr']+ch['votes_rgb']} frames had detectable waypoints "
              f"for channel-order vote. BGR status unconfirmed.")

    bad_g = [i for i, m in enumerate(monitor.margins) if rollouts[i][0].get("goal_dist", 0) == UNCERTAIN_SENTINEL]
    if bad_g:
        print(f"  ⚠ Rollouts where goal_dist[t=0] is SENTINEL: {bad_g[:10]}")

    rho_arr = np.array(monitor.margins)
    if rho_arr.max() > 0 and rho_arr.min() < 0:
        spread = rho_arr.max() - rho_arr.min()
        print(f"  ρ range: [{rho_arr.min():+.3f}, {rho_arr.max():+.3f}]  "
              f"spread={spread:.3f}  — mixed SAT/VIOL suggests partial AP extraction failure")

    print()


if __name__ == "__main__":
    main()
