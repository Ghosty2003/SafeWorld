"""L3 rescue, Candidate 3: hopper-hop quick screen -- does a fundamentally
more strongly-periodic dm_control task give better "eyes" (AP ratio) than
walker-walk's subtle gait wobble?

50 real episodes (env.reset() + mpc_plan, horizon=300), extracting
physics.height() (hopper's torso-above-foot height, a large-amplitude
bounce, unlike walker's ~0.07-0.15m gait wobble) and real posterior z.
Quick Ridge probe fit + held-out error, candidate recurrence event
(peak-detection, reusing Candidate 2's prominence methodology, generalized;
and absolute-band, reusing Phase A's methodology) + ratio/match-rate report.

No frozen artifact touched; no sealed calibration data used.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from scipy.signal import find_peaks
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from wrappers.tdmpc2_wrapper import TDMPC2Wrapper

OUT = pathlib.Path("artifacts/l3_rescue_screen/candidate3_hopper_screen")
CHECKPOINT = "models/dmcontrol/hopper-hop-1.pt"
TASK = "hopper-hop"
N_EPISODES = 50
HORIZON = 300
SEED_BASE = 60000


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED_BASE)
    np.random.seed(SEED_BASE)

    wrapper = TDMPC2Wrapper()
    wrapper.load(checkpoint=CHECKPOINT, task=TASK, seed=1)
    if str(wrapper._agent.device).split(":")[0] != "cuda":
        wrapper.close()
        raise RuntimeError("CUDA required")

    print(f"collecting {N_EPISODES} real hopper-hop episodes, horizon={HORIZON} ...", flush=True)
    all_z, all_h = [], []
    try:
        for i in range(N_EPISODES):
            obs = wrapper._env.reset()
            z = wrapper.encode(obs)
            z_seq, h_seq = [], [float(wrapper._physics.height())]
            with torch.no_grad():
                for step in range(HORIZON):
                    a = wrapper.plan_action(z, t0=(step == 0), eval_mode=True)
                    obs, _r, done, _info = wrapper._env.step(a[0].detach().cpu())
                    z = wrapper.encode(obs)
                    z_seq.append(z[0].detach().cpu().numpy().astype(np.float64, copy=True))
                    h_seq.append(float(wrapper._physics.height()))
                    if done:
                        break
            if len(z_seq) == HORIZON:
                all_z.append(np.stack(z_seq))
                all_h.append(np.array(h_seq[:HORIZON]))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{N_EPISODES}", flush=True)
    finally:
        wrapper.close()

    z = np.stack(all_z)   # (n,300,512)
    h = np.stack(all_h)
    n = z.shape[0]
    print(f"collected {n} complete episodes. height range: min={h.min():.3f} max={h.max():.3f} mean={h.mean():.3f}", flush=True)

    # candidate structure survey: peak detection on GROUND TRUTH (real) height
    amp = h.max() - h.min()
    prominence = 0.1 * amp  # scale to this task's own amplitude, unlike walker's fixed 0.03m
    counts, intervals = [], []
    for i in range(n):
        pk, _ = find_peaks(h[i], prominence=prominence)
        counts.append(len(pk))
        if len(pk) > 1:
            intervals.extend(np.diff(pk).tolist())
    structure = dict(
        height_range_m=[float(h.min()), float(h.max())], amplitude_m=float(amp),
        prominence_used_m=float(prominence),
        mean_peaks_per_300_window=float(np.mean(counts)),
        interval_mean=float(np.mean(intervals)) if intervals else None,
        interval_std=float(np.std(intervals)) if intervals else None,
    )
    print("structure:", json.dumps(structure, indent=2), flush=True)

    # AP judgeability: Ridge probe, held-out error
    Zflat, Hflat = z.reshape(-1, 512), h.reshape(-1)
    Ztr, Zte, Htr, Hte = train_test_split(Zflat, Hflat, test_size=0.2, random_state=0)
    probe = Ridge(alpha=10.0).fit(Ztr, Htr)
    pred = probe.predict(Zte)
    err = np.abs(pred - Hte)
    probe_stats = dict(mae=float(err.mean()), p50=float(np.percentile(err, 50)),
                       p90=float(np.percentile(err, 90)), p95=float(np.percentile(err, 95)),
                       max=float(err.max()))
    print("probe (real, held-out, C0-style):", json.dumps(probe_stats, indent=2), flush=True)

    # absolute-band ratio (Phase A style): use a band around the LOW point of the bounce
    band_lo, band_hi = float(np.percentile(h, 20)), float(np.percentile(h, 40))
    band_width = band_hi - band_lo
    ratio_band_C0 = band_width / probe_stats["p95"] if probe_stats["p95"] > 0 else None

    # peak-matching, held-out split by episode (not by step, avoid leakage across a bounce)
    n_test_eps = int(0.3 * n)
    test_eps = np.arange(n - n_test_eps, n)
    probe_eptrain = Ridge(alpha=10.0).fit(z[:n - n_test_eps].reshape(-1, 512), h[:n - n_test_eps].reshape(-1))
    n_true_total = n_matched = n_missed = n_false = 0
    for i in test_eps:
        true_pk, _ = find_peaks(h[i], prominence=prominence)
        h_pred = probe_eptrain.predict(z[i])
        pred_pk, _ = find_peaks(h_pred, prominence=prominence)
        n_true_total += len(true_pk)
        matched = np.zeros(len(pred_pk), dtype=bool)
        for tp in true_pk:
            hit = np.abs(pred_pk - tp) <= 2 if len(pred_pk) else np.array([])
            avail = hit & ~matched if len(pred_pk) else np.array([])
            if avail.any():
                matched[np.argmax(avail)] = True
                n_matched += 1
            else:
                n_missed += 1
        n_false += int((~matched).sum())
    miss_rate = n_missed / n_true_total if n_true_total else None
    false_rate = n_false / (n_matched + n_false) if (n_matched + n_false) else None

    report = dict(
        checkpoint=CHECKPOINT, task=TASK, n_episodes=n, horizon=HORIZON,
        structure=structure,
        probe_C0_real=probe_stats,
        absolute_band=dict(band_lo=band_lo, band_hi=band_hi, band_width_m=band_width,
                           ratio_vs_C0=ratio_band_C0),
        peak_matching_holdout=dict(n_test_episodes=int(n_test_eps), tolerance_steps=2,
                                   prominence_m=float(prominence),
                                   n_true_peaks=n_true_total, n_matched=n_matched,
                                   n_missed=n_missed, n_false=n_false,
                                   miss_rate=miss_rate, false_detection_rate=false_rate),
        structure_gate_pass=bool(structure["mean_peaks_per_300_window"] >= 10),
        ap_gate_pass_band=bool(ratio_band_C0 is not None and ratio_band_C0 >= 3.0),
        ap_gate_pass_peak=bool(miss_rate is not None and miss_rate < 0.05 and false_rate is not None and false_rate < 0.05),
        eyes_better_than_walker=bool(probe_stats["p95"] < 0.064 * (amp / 0.07) if amp > 0 else False),
        interpretation=(
            "hopper-hop's own C0 (real, held-out) probe error and band-width/amplitude "
            "are reported at native scale (this task's own bounce amplitude, not walker's "
            "0.07m gait band) -- 'eyes better than walker' compares whether the SAME kind "
            "of decode precision scales proportionally with this task's own larger physical "
            "signal, not an absolute-unit comparison."
        ),
        calibration_touched=False, track_b_touched=False, sealed_data_touched=False,
    )
    (OUT / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run()
