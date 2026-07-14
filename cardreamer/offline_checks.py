"""
offline_checks.py — Offline probe analyses on saved goal-probe data (no CARLA/GPU).

Merges the former scratchpad scripts l3_check.py / l5_check.py / l5_semantics.py.

  python offline_checks.py l3      # sequential-zone (A then B) model-vs-real agreement
  python offline_checks.py l5      # corridor spec at its own decision scale (polyline)
  python offline_checks.py l5sem   # L5 semantic validity: arc decomposition +
                                   # empirical-centerline re-test (the check that
                                   # reclassified L5 as trivially-SAT, doc N.4)

Requires collect_goal_probe.py output in /tmp/goal_probe_data.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, "/home/bot/SafeWorld")
from cardreamer.probe_common import (GOAL_XY, ROUTE, fit_position_probe,
                                           in_ring, load_posterior, load_prior,
                                           near_goal, path_dist,
                                           position_lookup)

H = 50


def _prior_predictions():
    probe = fit_position_probe()
    hp, a_ep, a_st = load_prior()
    pred = probe.predict(hp.reshape(-1, hp.shape[-1])).reshape(hp.shape[0], H, 2)
    return pred, a_ep, a_st, position_lookup()


def _real_future(lut, ep, step, horizon=H):
    fut = [lut.get((int(ep), int(step) + k)) for k in range(1, horizon + 1)]
    return [f for f in fut if f is not None and not np.isnan(f).any()]


def check_l3():
    """Sequential F(A then B): model (probe on prior) vs real, per anchor."""
    pred, a_ep, a_st, lut = _prior_predictions()

    def seq_holds(traj):
        a = in_ring(traj); b = near_goal(traj)
        ta = np.where(a)[0]
        return bool(len(ta) and b[ta[0] + 1:].any())

    m_seq, r_seq = [], []
    for i in range(len(pred)):
        fut = _real_future(lut, a_ep[i], a_st[i])
        if len(fut) < 10:
            continue
        m_seq.append(seq_holds(pred[i, :len(fut)]))
        r_seq.append(seq_holds(np.array(fut)))
    m, r = np.array(m_seq), np.array(r_seq)
    tp = int((m & r).sum()); fp = int((m & ~r).sum())
    fn = int((~m & r).sum()); tn = int((~m & ~r).sum())
    n = len(m)
    print(f"L3 sequential (ring then goal): anchors={n}  "
          f"base rate real={r.mean():.1%} model={m.mean():.1%}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}  agreement={(tp+tn)/n:.1%}  "
          f"TPR={tp/max(tp+fn,1):.1%}  FPR={fp/max(fp+tn,1):.1%}")


def _l5_holds(pd):
    """G(0,40, F(0,9, pd<6))"""
    for t in range(min(41, len(pd))):
        w = pd[t:t + 10]
        if len(w) == 0 or not (w < 6.0).any():
            return False
    return True


def check_l5():
    """Corridor spec at its own scale: boundary-band occupancy/flip + verdicts.
    Uses the CHORD polyline — see check_l5sem for why this semantics is invalid."""
    pred, a_ep, a_st, lut = _prior_predictions()
    _, xy, *_ = load_posterior()

    pdp = path_dist(xy)
    print(f"real polyline path_dist: p50={np.percentile(pdp,50):.2f}  "
          f"p90={np.percentile(pdp,90):.2f}  >6m: {(pdp>6).mean():.1%}  "
          f"band[4,8]m: {((pdp>=4)&(pdp<=8)).mean():.2%}")

    m_v, r_v, band_agree, lat_err = [], [], [], []
    for i in range(len(pred)):
        fut = _real_future(lut, a_ep[i], a_st[i])
        if len(fut) < H:
            continue
        real = np.array(fut); model = pred[i, :len(fut)]
        pdr, pdm = path_dist(real), path_dist(model)
        m_v.append(_l5_holds(pdm)); r_v.append(_l5_holds(pdr))
        inb = (pdr >= 4) & (pdr <= 8)
        if inb.any():
            band_agree.append(((pdm < 6) == (pdr < 6))[inb].mean())
        lat_err.extend(np.abs(pdm - pdr))
    m, r = np.array(m_v), np.array(r_v)
    tp = int((m & r).sum()); fp = int((m & ~r).sum())
    fn = int((~m & r).sum()); tn = int((~m & ~r).sum())
    print(f"L5 verdict (polyline pd<6): n={len(m)}  real_true={r.mean():.1%}  "
          f"TP={tp} FP={fp} FN={fn} TN={tn}  agreement={(tp+tn)/max(len(m),1):.1%}")
    print(f"  in-band step agreement: "
          f"{np.mean(band_agree) if band_agree else float('nan'):.1%}   "
          f"lateral probe error p50={np.percentile(lat_err,50):.2f}m "
          f"p90={np.percentile(lat_err,90):.2f}m")


def check_l5sem():
    """Semantic validity of the corridor spec (the check that killed L5):
    ① are polyline pd>6 steps route geometry (arc-vs-chord)?
    ② empirical-centerline (leave-one-episode-out) lateral spread
    ③ L5 verdict under corrected reference."""
    pred, a_ep, a_st, lut = _prior_predictions()
    _, xy, eps, st, ep_sum = load_posterior()

    pdp = path_dist(xy)
    off = pdp > 6.0
    dC = np.linalg.norm(xy - np.array([-6.4, -6.3]), axis=1)
    on_arc = np.abs(dC - 20.9) < 6.0
    print(f"① polyline pd>6: {off.mean():.1%} of steps; of these on ring annulus: "
          f"{on_arc[off].mean():.1%} (crude — ② is the decisive test)")

    dest_eps = [i for i in range(len(ep_sum)) if ep_sum[i, 1] > 0]

    def centerline(exclude_ep):
        return xy[np.isin(eps, [e for e in dest_eps if e != exclude_ep])][::3]

    def pd_emp(p, ref):
        return np.array([np.linalg.norm(ref - q, axis=1).min()
                         for q in np.atleast_2d(p)])

    vals = []
    for e in np.unique(eps):
        ref = centerline(e)
        idx = np.where(eps == e)[0][::5]
        vals.extend(pd_emp(xy[idx], ref))
    vals = np.array(vals)
    print(f"② empirical-centerline spread: p50={np.percentile(vals,50):.2f}m  "
          f"p99={np.percentile(vals,99):.2f}m  >6m: {(vals>6).mean():.2%}")

    m_v, r_v = [], []
    for i in range(len(pred)):
        fut = _real_future(lut, a_ep[i], a_st[i])
        if len(fut) < H:
            continue
        ref = centerline(int(a_ep[i]))
        m_v.append(_l5_holds(pd_emp(pred[i, :len(fut)], ref)))
        r_v.append(_l5_holds(pd_emp(np.array(fut), ref)))
    m, r = np.array(m_v), np.array(r_v)
    print(f"③ L5 verdict (empirical centerline): n={len(m)}  "
          f"real_true={r.mean():.1%}  model_true={m.mean():.1%}  "
          f"→ trivially-SAT at 6m; meaningful thresholds are below probe "
          f"lateral resolution (see doc N.4)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("check", choices=["l3", "l5", "l5sem"])
    args = p.parse_args()
    {"l3": check_l3, "l5": check_l5, "l5sem": check_l5sem}[args.check]()
