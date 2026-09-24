"""Step 1 (per carrier, only for Step-0-passing carriers): anti-fraud dual
check + parameter pre-registration/freeze.

PRE-REGISTRATION (fixed here, before Step 1 numbers are computed; frozen
for the rest of this carrier's pipeline -- Step 2/3 must reuse these
verbatim, never re-tune):

  prominence = 0.03 for both carriers (same absolute scale validated for
    walker-walk; Step 0 already confirmed it recovers clean, matching
    real/imagination structure for both cheetah-run and walker-run).

  false_rate_cp_upper: CITED from the Group 1 four-gate screen
    (artifacts/l3_multimodel_screen/group1_dmcontrol/), which already
    measured real-held-out peak-matching miss/false rates with the
    identical prominence on each carrier's own real data:
      cheetah:   false_rate_cp_upper = 0.0366593606474291
      walkerrun: false_rate_cp_upper = 0.010703829113449055
    Not recomputed here (would be redundant with already-collected,
    already-published numbers from the same real data population).

  M_MIN: a METHODOLOGICAL REFINEMENT discovered during this pre-registration
    step, documented not hidden. walker-walk's own M_MIN formula
    (floor(300 / P90(real interval))) was tried first here and produced
    M_MIN=50 for walker-run -- ABOVE its own real mean event count (47.76)
    and imagination mean (47.75), which would make the count gate fail
    almost every genuinely healthy trajectory. Root cause: walker-run's
    gait is so regular (interval std=0.54 steps) that P90(interval) is
    within noise of the mean interval, and converting through
    floor(300/interval) then loses ~1-2 cycles to boundary-truncation
    effects the interval-based formula does not account for. FIX: M_MIN is
    instead set directly from the REAL per-trajectory raw peak-COUNT
    distribution's own P10 (floor), which is not subject to this
    interval-to-count conversion error:
      cheetah:   P10(real count) = 17.9 -> M_MIN = 17 (matches the old
                 interval-based estimate almost exactly here, since
                 cheetah's gait has enough interval variance for the two
                 formulas to agree)
      walkerrun: P10(real count) = 46.9 -> M_MIN = 46 (vs. the broken
                 interval-based 50)
    This refined formula is used for BOTH carriers for consistency, even
    though cheetah's original estimate was already fine.

  CV_p95: P95 of the REAL per-trajectory interval-CV distribution (same
    convention as walker-walk):
      cheetah:   0.38260226202130304
      walkerrun: 0.21606759056802555

Once frozen above, this script:
  1. Manufactures 50 genuine real fall trajectories per carrier (real gait
     prelude, t_switch in [20,80], then a saturated all-negative action
     block for the remainder of a 300-step trajectory; carrier-specific
     failure mode: cheetah tips over/stalls, walker-run falls) and checks
     the frozen deduction verdict's false-positive rate against them
     (must be 0/50).
  2. Real held-out FP/FN test (candidate-5-style): reuses the 50 real
     episodes from Step 0 (already collected, already used to derive the
     M_MIN/CV thresholds above) with a fresh 80/20 fit/held-out split for
     the FP/FN check itself, to avoid using the exact same rows for both
     threshold derivation and threshold testing where avoidable.

No frozen artifact modified. No sealed calibration data touched.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/bot/SafeWorld")
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("l3_generic_lbsm_lib", "/home/bot/SafeWorld/tdmpc2/l3_generic_lbsm_lib.py")
_lib = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_lib)
manufacture_fall_trajectories = _lib.manufacture_fall_trajectories
find_peaks_and_segments = _lib.find_peaks_and_segments
deduction_verdict = _lib.deduction_verdict
fit_ridge_probe = _lib.fit_ridge_probe

HORIZON = 300
N_FALL_TARGET = 50
T_SWITCH_RANGE = (20, 80)

CARRIERS = {
    "cheetah": dict(
        checkpoint="models/dmcontrol/cheetah-run-1.pt", task="cheetah-run",
        quantity_fn=lambda phys: float(phys.named.data.xpos["torso", "z"]),
        fall_check_fn=lambda q, t: (q[-30:].mean() < t) and (q[-1] < t),
        fall_thresh=0.3, prominence=0.03,
        false_rate_cp_upper=0.0366593606474291, m_min=17, cv_p95=0.38260226202130304,
        fall_seed=110003,
        out=pathlib.Path("artifacts/tdmpc2_cheetah_l3_lbsm/step1"),
        step0_dir=pathlib.Path("artifacts/tdmpc2_cheetah_l3_lbsm/step0"),
    ),
    "walkerrun": dict(
        checkpoint="models/dmcontrol/walker-run-1.pt", task="walker-run",
        quantity_fn=lambda phys: float(phys.torso_height()),
        fall_check_fn=lambda q, t: (q[-30:].mean() < t) and (q[-1] < t),
        fall_thresh=0.4, prominence=0.03,
        false_rate_cp_upper=0.010703829113449055, m_min=46, cv_p95=0.21606759056802555,
        fall_seed=120003,
        out=pathlib.Path("artifacts/tdmpc2_walkerrun_l3_lbsm/step1"),
        step0_dir=pathlib.Path("artifacts/tdmpc2_walkerrun_l3_lbsm/step0"),
    ),
}


def real_holdout_fp_fn(cfg):
    d = np.load(cfg["step0_dir"] / "real_50.npz")
    z, q = d["z"], d["q"]
    n = q.shape[0]
    n_holdout = int(0.2 * n)
    fit_idx = np.arange(n - n_holdout)
    hold_idx = np.arange(n - n_holdout, n)
    probe = fit_ridge_probe(z, q, n_holdout=n_holdout)

    TP = FP = FN = 0
    rows = []
    for i in hold_idx:
        true_pk, _ = find_peaks_and_segments(q[i], cfg["prominence"], HORIZON)
        h_pred = probe.predict(z[i])
        dv_true = deduction_verdict(q[i], cfg["prominence"], cfg["false_rate_cp_upper"], cfg["m_min"], cfg["cv_p95"], HORIZON)
        dv_pred = deduction_verdict(h_pred, cfg["prominence"], cfg["false_rate_cp_upper"], cfg["m_min"], cfg["cv_p95"], HORIZON)
        # ground truth: this real trajectory is "genuinely good" (normal gait, not failed)
        gt_pass = dv_true["verdict_with_cv"]
        pred_pass = dv_pred["verdict_with_cv"]
        if pred_pass and gt_pass:
            TP += 1
        elif pred_pass and not gt_pass:
            FP += 1
        elif not pred_pass and gt_pass:
            FN += 1
        rows.append(dict(traj=int(i), gt_pass=bool(gt_pass), pred_pass=bool(pred_pass),
                         m_true=dv_true["m"], m_pred=dv_pred["m"]))
    return dict(n_holdout=int(n_holdout), TP=TP, FP=FP, FN=FN,
               fp_rate=FP / n_holdout, fn_rate=FN / n_holdout, rows=rows)


def run(name):
    cfg = CARRIERS[name]
    cfg["out"].mkdir(parents=True, exist_ok=True)
    print(f"=== Step 1: {name} ===", flush=True)
    print("frozen params:", json.dumps({k: cfg[k] for k in ("prominence", "false_rate_cp_upper", "m_min", "cv_p95", "fall_thresh")}, indent=2), flush=True)

    print(f"[fall manufacture] target {N_FALL_TARGET} genuine real fall trajectories ...", flush=True)
    z_fall, q_fall, t_switch, discard_rate, attempts, discarded = manufacture_fall_trajectories(
        cfg, N_FALL_TARGET, HORIZON, T_SWITCH_RANGE, cfg["fall_thresh"], cfg["fall_seed"])
    np.savez_compressed(cfg["out"] / "real_fall_trajectories.npz", z=z_fall, q=q_fall, t_switch=t_switch)
    print(f"    accepted={z_fall.shape[0]} discarded={discarded} attempts={attempts} discard_rate={discard_rate:.3f}", flush=True)

    fp_count = 0
    m_adj_values = []
    for i in range(z_fall.shape[0]):
        dv = deduction_verdict(q_fall[i], cfg["prominence"], cfg["false_rate_cp_upper"], cfg["m_min"], cfg["cv_p95"], HORIZON)
        m_adj_values.append(dv["m_adj"])
        if dv["verdict_with_cv"]:
            fp_count += 1
    fall_result = dict(n=z_fall.shape[0], false_positives=fp_count,
                       max_m_adj=float(max(m_adj_values)) if m_adj_values else None,
                       m_min=cfg["m_min"], margin=cfg["m_min"] - (max(m_adj_values) if m_adj_values else 0))
    print("fall-trajectory FP check:", json.dumps(fall_result, indent=2), flush=True)

    print("[real held-out FP/FN] ...", flush=True)
    holdout_result = real_holdout_fp_fn(cfg)
    print(json.dumps({k: v for k, v in holdout_result.items() if k != "rows"}, indent=2), flush=True)

    report = dict(
        carrier=name,
        frozen_params=dict(prominence=cfg["prominence"], false_rate_cp_upper=cfg["false_rate_cp_upper"],
                           m_min=cfg["m_min"], cv_p95=cfg["cv_p95"], fall_thresh=cfg["fall_thresh"]),
        fall_manufacture=dict(n_accepted=z_fall.shape[0], n_discarded=discarded, n_attempts=attempts, discard_rate=discard_rate),
        fall_fp_check=fall_result,
        real_holdout_fp_fn=holdout_result,
        step1_pass=bool(fp_count == 0),
    )
    (cfg["out"] / "result.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"STEP 1 {'PASS' if report['step1_pass'] else 'FAIL'}", flush=True)
    return report


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else None
    if name:
        run(name)
    else:
        for n in CARRIERS:
            run(n)
