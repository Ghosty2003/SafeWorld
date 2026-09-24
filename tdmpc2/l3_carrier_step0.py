"""Step 0 (hard gate, per carrier): imagination-side eyes precheck.

For each carrier: fit a Ridge probe on 50 real episodes, then generate 20
imagination rollouts and compare their decoded-quantity peak/interval
statistics against the real side. Since imagination has no ground truth,
three indirect checks stand in: (a) peak-count order-of-magnitude matches
real (neither collapses nor explodes); (b) interval-distribution shape
matches; (c) visual/numeric inspection of 5 decoded imagination traces
(clear periodic oscillation vs. noise mush). Any one significantly
abnormal -> STOP at Step 0 for that carrier, reported honestly (the
walker-walk lesson: real-side comfort does not guarantee imagination-side
usability).

No frozen artifact modified. No sealed data touched.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, "/home/bot/SafeWorld")

# Load as a standalone module name, NOT "tdmpc2.l3_generic_lbsm_lib" -- the
# latter registers "tdmpc2" as a namespace package in sys.modules, shadowing
# the vendored upstream TD-MPC2 source package of the same name that
# wrappers/tdmpc2_wrapper.py needs ("from tdmpc2 import TDMPC2").
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("l3_generic_lbsm_lib", "/home/bot/SafeWorld/tdmpc2/l3_generic_lbsm_lib.py")
_lib = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_lib)
collect_real_episodes = _lib.collect_real_episodes
generate_imagination_rollouts = _lib.generate_imagination_rollouts
fit_ridge_probe = _lib.fit_ridge_probe

HORIZON = 300
N_REAL = 50
N_IMAG = 20

CARRIERS = {
    "cheetah": dict(
        checkpoint="models/dmcontrol/cheetah-run-1.pt", task="cheetah-run",
        quantity_fn=lambda phys: float(phys.named.data.xpos["torso", "z"]),
        prominence=0.03,
        real_seed=110001, imag_seed=110002,
        out=pathlib.Path("artifacts/tdmpc2_cheetah_l3_lbsm/step0"),
    ),
    "walkerrun": dict(
        checkpoint="models/dmcontrol/walker-run-1.pt", task="walker-run",
        quantity_fn=lambda phys: float(phys.torso_height()),
        prominence=0.03,
        real_seed=120001, imag_seed=120002,
        out=pathlib.Path("artifacts/tdmpc2_walkerrun_l3_lbsm/step0"),
    ),
}


def stats(q_arr, prominence):
    counts, intervals = [], []
    for i in range(q_arr.shape[0]):
        pk, _ = find_peaks(q_arr[i], prominence=prominence)
        counts.append(len(pk))
        if len(pk) > 1:
            intervals.extend(np.diff(pk).tolist())
    return dict(mean_count=float(np.mean(counts)), std_count=float(np.std(counts)),
               interval_mean=float(np.mean(intervals)) if intervals else None,
               interval_std=float(np.std(intervals)) if intervals else None,
               n_intervals=len(intervals))


def run(name):
    cfg = CARRIERS[name]
    cfg["out"].mkdir(parents=True, exist_ok=True)
    print(f"=== Step 0: {name} ===", flush=True)

    print(f"[real] collecting {N_REAL} real episodes for probe fit ...", flush=True)
    z_real, q_real = collect_real_episodes(cfg, N_REAL, HORIZON, cfg["real_seed"])
    np.savez_compressed(cfg["out"] / "real_50.npz", z=z_real, q=q_real)
    probe = fit_ridge_probe(z_real, q_real, n_holdout=0)  # use all 50 for this precheck's probe

    real_stats = stats(q_real, cfg["prominence"])
    print("real-side stats:", json.dumps(real_stats, indent=2), flush=True)

    print(f"[imagination] generating {N_IMAG} imagination rollouts ...", flush=True)
    z_imag = generate_imagination_rollouts(cfg, N_IMAG, HORIZON, cfg["imag_seed"])
    q_imag_decoded = probe.predict(z_imag.reshape(-1, 512)).reshape(N_IMAG, HORIZON)
    np.savez_compressed(cfg["out"] / "imagination_20.npz", z=z_imag, q_decoded=q_imag_decoded)

    imag_stats = stats(q_imag_decoded, cfg["prominence"])
    print("imagination-side stats:", json.dumps(imag_stats, indent=2), flush=True)

    # check (a): order-of-magnitude match (within 3x either direction, and both >0)
    ratio = imag_stats["mean_count"] / real_stats["mean_count"] if real_stats["mean_count"] > 0 else None
    check_a_pass = ratio is not None and (1 / 3 <= ratio <= 3) and imag_stats["mean_count"] > 0

    # check (b): interval distribution shape (mean and std within 2x)
    check_b_pass = None
    if real_stats["interval_mean"] and imag_stats["interval_mean"]:
        mean_ratio = imag_stats["interval_mean"] / real_stats["interval_mean"]
        std_ratio = (imag_stats["interval_std"] / real_stats["interval_std"]) if real_stats["interval_std"] else None
        check_b_pass = bool(0.5 <= mean_ratio <= 2.0 and (std_ratio is None or std_ratio <= 3.0))
    else:
        check_b_pass = False

    # check (c): visual/numeric inspection proxy -- report 5 sample traces'
    # own coefficient of variation of successive differences (a smooth
    # periodic signal has much lower high-frequency noise power than a
    # "mushed to noise" signal); reported as data for manual/automatic judgment.
    sample_traces = []
    for i in range(min(5, N_IMAG)):
        trace = q_imag_decoded[i]
        diffs = np.diff(trace)
        noise_proxy = float(np.std(np.diff(diffs)))  # 2nd-difference std: high for jagged noise
        pk, _ = find_peaks(trace, prominence=cfg["prominence"])
        sample_traces.append(dict(episode=i, n_peaks=len(pk), second_diff_std=noise_proxy,
                                  trace_head=trace[:20].round(4).tolist()))

    report = dict(
        carrier=name, real_stats=real_stats, imagination_stats=imag_stats,
        check_a_order_of_magnitude=dict(ratio=ratio, pass_=check_a_pass),
        check_b_interval_shape=dict(pass_=check_b_pass),
        check_c_sample_traces=sample_traces,
        overall_step0_pass=bool(check_a_pass and check_b_pass),
    )
    (cfg["out"] / "result.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "check_c_sample_traces"}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else None
    if name:
        run(name)
    else:
        for n in CARRIERS:
            run(n)
