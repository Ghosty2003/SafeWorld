"""L3 rescue, Candidate 2: waveform (peak-detection) gait-event redefinition.

Reframing: instead of "height enters a fixed absolute-value band" (candidate
so far killed by decode/transfer error consuming the whole band width),
redefine the recurrence event as "the decoded height signal completes one
full oscillation" -- a local maximum with a genuine prominence drop on both
sides (a real gait peak, not a noise wiggle). This changes what the AP
ratio's DENOMINATOR is sensitive to: peak TIMING is a function of local
signal SHAPE, which a constant or slowly-varying decode bias does not
disturb, unlike absolute-value band membership.

Critical test (this candidate's actual pass/fail criterion, exactly as
specified): on held-out REAL data, do decoded-height peaks (from the
existing Ridge probe applied to REAL posterior z) line up with
ground-truth-height peaks within +/-2 steps? This is a pure decode-precision
question (C0-type: probe accuracy on real latents), deliberately NOT the
real-vs-imagined transfer-divergence question that killed Candidate 1 --
because a self-consistent Recurrence claim about the MODEL's own imagined
process only needs the probe to decode faithfully from whatever z it is
given (real or imagined), not for the imagined trajectory to match an
external real replay in absolute terms.

Reuses the existing height probe / real posterior data (tdmpc2_walker_lppm_latents)
and, if PASS, the already-generated 20 Phase-A imagination rollouts
(item3_imagination_latents.npz) for the bonus structure-confirmation report.
No frozen artifact modified; no sealed calibration data used.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import Ridge

LATENTS_DIR = pathlib.Path("artifacts/tdmpc2_walker_lppm_latents")
PHASE_A_DIR = pathlib.Path("artifacts/tdmpc2_walker_l3_statistical_lbsm")
OUT = pathlib.Path("artifacts/l3_rescue_screen/candidate2_waveform_event")
PROMINENCE = 0.03   # m, genuine gait-cycle peak vs noise wiggle
TOL_STEPS = 2
MISS_FALSE_THRESHOLD = 0.05


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    seed_files = sorted(LATENTS_DIR.glob("seed_*_latents.npz"))
    all_height, all_z = [], []
    for f in seed_files:
        d = np.load(f)
        all_height.append(d["height_m"])
        all_z.append(d["posterior_z"])
    height = np.concatenate(all_height, axis=0)  # (1500,501)
    z = np.concatenate(all_z, axis=0)
    n = height.shape[0]

    n_holdout = int(0.2 * n)
    fit_idx = np.arange(n - n_holdout)
    hold_idx = np.arange(n - n_holdout, n)

    Zfit = z[fit_idx].reshape(-1, 512)
    Hfit = height[fit_idx].reshape(-1)
    hprobe = Ridge(alpha=10.0).fit(Zfit, Hfit)
    print(f"probe fit on {len(fit_idx)} rollouts, evaluating peak-matching on {len(hold_idx)} held-out rollouts ...", flush=True)

    n_true_total, n_matched, n_missed, n_false = 0, 0, 0, 0
    interval_list_true, interval_list_pred = [], []
    per_traj_true_counts, per_traj_pred_counts = [], []
    for i in hold_idx:
        true_peaks, _ = find_peaks(height[i], prominence=PROMINENCE)
        h_pred = hprobe.predict(z[i])
        pred_peaks, _ = find_peaks(h_pred, prominence=PROMINENCE)

        per_traj_true_counts.append(len(true_peaks))
        per_traj_pred_counts.append(len(pred_peaks))
        if len(true_peaks) > 1:
            interval_list_true.extend(np.diff(true_peaks).tolist())
        if len(pred_peaks) > 1:
            interval_list_pred.extend(np.diff(pred_peaks).tolist())

        n_true_total += len(true_peaks)
        matched_pred = np.zeros(len(pred_peaks), dtype=bool)
        for tp in true_peaks:
            hit = np.abs(pred_peaks - tp) <= TOL_STEPS if len(pred_peaks) else np.array([])
            avail = hit & ~matched_pred if len(pred_peaks) else np.array([])
            if avail.any():
                matched_pred[np.argmax(avail)] = True
                n_matched += 1
            else:
                n_missed += 1
        n_false += int((~matched_pred).sum())

    miss_rate = n_missed / n_true_total if n_true_total else None
    false_rate = n_false / (n_matched + n_false) if (n_matched + n_false) else None
    passed = (miss_rate is not None and miss_rate < MISS_FALSE_THRESHOLD and
              false_rate is not None and false_rate < MISS_FALSE_THRESHOLD)

    real_interval_stats = dict(
        mean=float(np.mean(interval_list_true)) if interval_list_true else None,
        std=float(np.std(interval_list_true)) if interval_list_true else None,
        n=len(interval_list_true),
    )
    mean_true_per_traj = float(np.mean(per_traj_true_counts))
    mean_pred_per_traj = float(np.mean(per_traj_pred_counts))

    report = dict(
        n_holdout_rollouts=int(n_holdout), prominence_threshold_m=PROMINENCE, tolerance_steps=TOL_STEPS,
        n_true_peaks_total=n_true_total, n_matched=n_matched, n_missed=n_missed, n_false=n_false,
        miss_rate=miss_rate, false_detection_rate=false_rate,
        pass_threshold=MISS_FALSE_THRESHOLD,
        verdict="PASS" if passed else "FAIL",
        mean_ground_truth_peaks_per_300_window=mean_true_per_traj,
        mean_decoded_peaks_per_300_window=mean_pred_per_traj,
        ground_truth_peak_interval_stats=real_interval_stats,
        interpretation=(
            "Peak-timing decode is a shape/local-derivative question, not an "
            "absolute-level question -- if miss/false rates are both <5%, the "
            "probe's DECODE PRECISION (already known to be excellent on real "
            "latents, C0 MAE~0.003-0.013m) is sufficient to identify individual "
            "gait cycles reliably, in a way the absolute-value band predicate "
            "(candidate so far killed by conformal-margin consumption) was not."
            if passed else
            "Even the shape/timing-based redefinition does not clear the 5% "
            "miss/false bar on real held-out data -- the failure is not specific "
            "to absolute-value band membership."
        ),
        calibration_touched=False, track_b_touched=False, sealed_data_touched=False,
    )
    (OUT / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)

    if passed:
        print("PASS -- running bonus item 4: imagination peak-recurrence structure confirmation ...", flush=True)
        imag = np.load(PHASE_A_DIR / "item3_imagination_latents.npz")
        height_imag = imag["height_imag"]
        n_imag = height_imag.shape[0]
        imag_counts, imag_intervals = [], []
        for i in range(n_imag):
            pk, _ = find_peaks(height_imag[i], prominence=PROMINENCE)
            imag_counts.append(len(pk))
            if len(pk) > 1:
                imag_intervals.extend(np.diff(pk).tolist())
        imag_report = dict(
            n_imagination_rollouts=n_imag,
            mean_peaks_per_300_window=float(np.mean(imag_counts)),
            interval_mean=float(np.mean(imag_intervals)) if imag_intervals else None,
            interval_std=float(np.std(imag_intervals)) if imag_intervals else None,
            comparison_to_real=dict(real_mean_per_window=mean_true_per_traj,
                                    real_interval_mean=real_interval_stats["mean"],
                                    real_interval_std=real_interval_stats["std"]),
        )
        (OUT / "item4_imagination_structure.json").write_text(json.dumps(imag_report, indent=2) + "\n")
        print(json.dumps(imag_report, indent=2), flush=True)

    return report


if __name__ == "__main__":
    run()
