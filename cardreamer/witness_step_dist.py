"""
witness_step_dist.py — Violation-step distribution across seeds (artifact check)

For each of N seeds, run the standard model-side hazard evaluation (20 rollouts
x 50 steps, actor policy) and locate the witness rollout (min robustness).
Record WHERE the violation happens:

  first_seen  : first step where a vehicle enters view (hazard < cap)
  first_viol  : first step where hazard_dist <= 0 (spec G(hazard>0) violated)
  min_step    : step of the closest approach (argmin hazard)
  jump15      : max |Δhazard| in steps 12..18 (discontinuity at the actor
                training-horizon boundary would show up here)

Interpretation:
  dispersed first_viol across seeds  → genuine dynamics (approach timing varies
                                       with initial conditions)
  clustered at horizon end (44-49)   → suspect end-of-imagination drift artifact
  jump15 comparable to typical step  → no boundary discontinuity

Usage:
  conda activate cardreamer && cd /home/bot/SafeWorld
  python witness_step_dist.py [--seeds 20] [--n 20] [--horizon 50]
"""

import sys, argparse
import numpy as np

sys.path.insert(0, "/home/bot/CarDreamer")
sys.path.insert(0, "/home/bot/SafeWorld")

UNCERTAIN_SENTINEL = -999.0
AP_KEY = "hazard_dist"


def main(args):
    from wrappers.cardreamer_wrapper import CarDreamerWrapper
    from configs.settings import RolloutConfig

    print("[1/2] Loading model ...", flush=True)
    base = RolloutConfig(n_rollouts=args.n, horizon=args.horizon,
                         seed=0, action_source="actor")
    w = CarDreamerWrapper(base)
    if args.checkpoint:
        w.load(checkpoint_path=args.checkpoint)
    else:
        w.load()

    print(f"[2/2] Running {args.seeds} seeds x {args.n} rollouts ...\n", flush=True)
    rows = []
    for s in range(args.seeds):
        cfg = RolloutConfig(n_rollouts=args.n, horizon=args.horizon,
                            seed=s, action_source="actor")
        trajs = w.sample_rollouts(cfg)

        # hazard series per rollout; robustness of G(hazard>0) = min_t hazard
        margins = []
        series  = []
        for traj in trajs:
            h = np.array([st.get(AP_KEY, UNCERTAIN_SENTINEL) for st in traj])
            h = np.where(h == UNCERTAIN_SENTINEL, np.nan, h)
            series.append(h)
            margins.append(np.nanmin(h))
        wi = int(np.nanargmin(margins))
        h  = series[wi]
        cap = np.nanmax(h)  # "no vehicle" reading for this run

        viol_steps = np.where(h <= 0)[0]
        seen_steps = np.where(h < cap - 1e-6)[0]
        d = np.abs(np.diff(h))
        jump15  = np.nanmax(d[12:18]) if len(d) > 18 else np.nan
        typ_jmp = np.nanmedian(d)

        rows.append({
            "seed": s,
            "rho_star": float(margins[wi]),
            "n_viol_rollouts": int(sum(1 for m in margins if m <= 0)),
            "first_seen": int(seen_steps[0]) if len(seen_steps) else -1,
            "first_viol": int(viol_steps[0]) if len(viol_steps) else -1,
            "min_step": int(np.nanargmin(h)),
            "jump15": float(jump15),
            "typ_jump": float(typ_jmp),
        })
        r = rows[-1]
        print(f"  seed {s:2d}: ρ*={r['rho_star']:+7.3f}  viol_rollouts={r['n_viol_rollouts']}  "
              f"first_seen={r['first_seen']:3d}  first_viol={r['first_viol']:3d}  "
              f"min_step={r['min_step']:3d}  jump15={r['jump15']:.3f} "
              f"(typ {r['typ_jump']:.3f})", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    fv = [r["first_viol"] for r in rows if r["first_viol"] >= 0]
    ms = [r["min_step"]  for r in rows]
    print("\n" + "=" * 65)
    print(f"  Violation-step distribution over {len(rows)} seeds "
          f"({len(fv)} with ρ*<=0)")
    print("=" * 65)
    if fv:
        print(f"  first_viol: min={min(fv)}  p25={np.percentile(fv,25):.0f}  "
              f"p50={np.percentile(fv,50):.0f}  p75={np.percentile(fv,75):.0f}  max={max(fv)}")
        print(f"    ≤15: {sum(1 for v in fv if v <= 15)}   "
              f"16-43: {sum(1 for v in fv if 15 < v < 44)}   "
              f"44-49: {sum(1 for v in fv if v >= 44)}")
    print(f"  min_step  : min={min(ms)}  p50={np.percentile(ms,50):.0f}  max={max(ms)}")
    j15 = [r["jump15"] for r in rows if not np.isnan(r["jump15"])]
    tj  = [r["typ_jump"] for r in rows]
    print(f"  jump15 (max |Δh| @steps 12-18): p50={np.percentile(j15,50):.3f}  "
          f"max={max(j15):.3f}   vs typical step |Δh| p50={np.percentile(tj,50):.3f}")

    np.savez(args.out, rows=rows)
    print(f"  saved → {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",   type=int, default=20)
    p.add_argument("--n",       type=int, default=20)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--checkpoint", default=None,
                   help="checkpoint path (default: carla_four_lane logdir)")
    p.add_argument("--out", default="witness_step_dist.npz")
    main(p.parse_args())
