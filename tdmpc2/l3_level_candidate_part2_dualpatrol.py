"""L3 Level-mapped candidate screen, Part 2: L5 DualPatrol instance
GF(left_step) & GF(right_step), walker-walk only, read-only/screening-level
precheck (50 real episodes, 50 imagination rollouts).

Event definition (same family as gait_cycle's judgment layer): a footstep
(touchdown) is a LOCAL MINIMUM of that foot's z-height trace
(physics.named.data.xpos['left_foot'/'right_foot','z']), detected via
scipy.signal.find_peaks on the NEGATED trace with prominence=0.3 -- chosen
by direct trace inspection (a full-swing excursion is ~1.0-1.3m, stance-
phase micro-wobble is ~0.05-0.15m; prominence=0.3 sits cleanly between the
two and gave a stable 9-10 events/300-step plateau across prominence in
[0.2,0.5] on a calibration trace, checked before committing to this value).

No frozen artifact modified. No sealed calibration data touched. New data
collected: 50 real episodes + 50 imagination rollouts (screening-level,
per instruction).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from scipy.signal import find_peaks
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/home/bot/SafeWorld")
from wrappers.tdmpc2_wrapper import TDMPC2Wrapper  # noqa: E402

OUT = pathlib.Path("artifacts/l3_level_mapped_candidates/part2_dualpatrol")
CHECKPOINT = "models/walker-walk-3.pt"
TASK = "walker-walk"
N_REAL = 50
N_IMAG = 50
HORIZON = 300
PROMINENCE = 0.3
REAL_SEED = 150001
IMAG_SEED = 150002


def collect_real():
    wrapper = TDMPC2Wrapper()
    wrapper.load(checkpoint=CHECKPOINT, task=TASK, seed=1)
    torch.manual_seed(REAL_SEED); np.random.seed(REAL_SEED)
    all_z, all_lf, all_rf = [], [], []
    try:
        for i in range(N_REAL):
            obs = wrapper._env.reset()
            z = wrapper.encode(obs)
            z_seq, lf_seq, rf_seq = [], [], []
            with torch.no_grad():
                for step in range(HORIZON):
                    a = wrapper.plan_action(z, t0=(step == 0), eval_mode=True)
                    obs, _r, done, _info = wrapper._env.step(a[0].detach().cpu())
                    z = wrapper.encode(obs)
                    z_seq.append(z[0].detach().cpu().numpy().astype(np.float64, copy=True))
                    lf_seq.append(float(wrapper._physics.named.data.xpos["left_foot", "z"]))
                    rf_seq.append(float(wrapper._physics.named.data.xpos["right_foot", "z"]))
            all_z.append(np.stack(z_seq)); all_lf.append(np.array(lf_seq)); all_rf.append(np.array(rf_seq))
            if (i + 1) % 10 == 0:
                print(f"    real episode {i + 1}/{N_REAL}", flush=True)
    finally:
        wrapper.close()
    return np.stack(all_z), np.stack(all_lf), np.stack(all_rf)


def generate_imagination():
    wrapper = TDMPC2Wrapper()
    wrapper.load(checkpoint=CHECKPOINT, task=TASK, seed=1)
    torch.manual_seed(IMAG_SEED); np.random.seed(IMAG_SEED)
    all_z = []
    try:
        for i in range(N_IMAG):
            obs = wrapper._env.reset()
            z = wrapper.encode(obs)
            z_seq = []
            with torch.no_grad():
                for step in range(HORIZON):
                    a = wrapper.plan_action(z, t0=(step == 0), eval_mode=True)
                    z = wrapper.next(z, a)
                    z_seq.append(z[0].detach().cpu().numpy().astype(np.float64, copy=True))
            all_z.append(np.stack(z_seq))
            if (i + 1) % 10 == 0:
                print(f"    imagined {i + 1}/{N_IMAG}", flush=True)
    finally:
        wrapper.close()
    return np.stack(all_z)


def event_stats(trace_batch, prominence):
    counts, intervals = [], []
    events_per_traj = []
    for i in range(trace_batch.shape[0]):
        pk, _ = find_peaks(-trace_batch[i], prominence=prominence)
        events_per_traj.append(pk)
        counts.append(len(pk))
        if len(pk) > 1:
            intervals.extend(np.diff(pk).tolist())
    return dict(mean_count=float(np.mean(counts)), std_count=float(np.std(counts)),
               interval_mean=float(np.mean(intervals)) if intervals else None,
               interval_std=float(np.std(intervals)) if intervals else None), events_per_traj


def alternation_rate(left_events_list, right_events_list, horizon):
    """Merge left/right event times per trajectory, check what fraction of
    consecutive event pairs alternate feet (a genuine walking gait should
    alternate almost always; same-foot-twice-in-a-row would indicate a
    stumble/hop or a detector artifact)."""
    rates = []
    for lev, rev in zip(left_events_list, right_events_list):
        tagged = [(t, "L") for t in lev] + [(t, "R") for t in rev]
        tagged.sort()
        if len(tagged) < 2:
            continue
        alternations = sum(1 for a, b in zip(tagged[:-1], tagged[1:]) if a[1] != b[1])
        rates.append(alternations / (len(tagged) - 1))
    return dict(mean_alternation_rate=float(np.mean(rates)) if rates else None, n_trajectories_with_ge2_events=len(rates))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1] collecting 50 real episodes (left_foot/right_foot height + z) ...", flush=True)
    z_real, lf_real, rf_real = collect_real()
    np.savez_compressed(OUT / "real_50.npz", z=z_real, lf=lf_real, rf=rf_real)

    print("[2] real-side event structure ...", flush=True)
    lf_stats, lf_events = event_stats(lf_real, PROMINENCE)
    rf_stats, rf_events = event_stats(rf_real, PROMINENCE)
    alt = alternation_rate(lf_events, rf_events, HORIZON)
    print("  left_step real:", json.dumps(lf_stats, indent=2), flush=True)
    print("  right_step real:", json.dumps(rf_stats, indent=2), flush=True)
    print("  alternation:", json.dumps(alt, indent=2), flush=True)

    print("[3] eyes precheck: fit Ridge probes for left/right foot height ...", flush=True)
    n = z_real.shape[0]
    Zflat = z_real.reshape(-1, 512)
    probes = {}
    probe_errs = {}
    for name, trace in (("left_foot", lf_real), ("right_foot", rf_real)):
        Qflat = trace.reshape(-1)
        Ztr, Zte, Qtr, Qte = train_test_split(Zflat, Qflat, test_size=0.2, random_state=0)
        probe = Ridge(alpha=10.0).fit(Ztr, Qtr)
        err = np.abs(probe.predict(Zte) - Qte)
        probe_errs[name] = dict(mae=float(err.mean()), p90=float(np.percentile(err, 90)), p95=float(np.percentile(err, 95)))
        probes[name] = Ridge(alpha=10.0).fit(Zflat, Qflat)  # refit on all 50 for imagination decoding
        print(f"  {name} probe C0: {probe_errs[name]}", flush=True)

    print("[4] generating 50 imagination rollouts ...", flush=True)
    z_imag = generate_imagination()
    lf_imag = probes["left_foot"].predict(z_imag.reshape(-1, 512)).reshape(N_IMAG, HORIZON)
    rf_imag = probes["right_foot"].predict(z_imag.reshape(-1, 512)).reshape(N_IMAG, HORIZON)
    np.savez_compressed(OUT / "imagination_50.npz", z=z_imag, lf=lf_imag, rf=rf_imag)

    lf_imag_stats, lf_imag_events = event_stats(lf_imag, PROMINENCE)
    rf_imag_stats, rf_imag_events = event_stats(rf_imag, PROMINENCE)
    imag_alt = alternation_rate(lf_imag_events, rf_imag_events, HORIZON)
    print("  left_step imagination:", json.dumps(lf_imag_stats, indent=2), flush=True)
    print("  right_step imagination:", json.dumps(rf_imag_stats, indent=2), flush=True)
    print("  imagination alternation:", json.dumps(imag_alt, indent=2), flush=True)

    left_ratio = lf_imag_stats["mean_count"] / lf_stats["mean_count"] if lf_stats["mean_count"] else None
    right_ratio = rf_imag_stats["mean_count"] / rf_stats["mean_count"] if rf_stats["mean_count"] else None

    CLIFF_LOW, CLIFF_HIGH = 0.7, 3.0  # this project's own established "boundary zone" bar
    def cliff_flag(r):
        if r is None:
            return "N/A"
        return "IN_BOUNDARY_ZONE_LIKE_WALKER_WALK_GAIT" if CLIFF_LOW <= r <= CLIFF_HIGH and abs(r - 1.0) > 0.05 else ("CLEAN" if 0.9 <= r <= 1.15 else "OUT_OF_RANGE")

    report = dict(
        prominence=PROMINENCE, horizon=HORIZON, n_real=N_REAL, n_imagination=N_IMAG,
        real=dict(left_step=lf_stats, right_step=rf_stats, alternation=alt),
        imagination=dict(left_step=lf_imag_stats, right_step=rf_imag_stats, alternation=imag_alt),
        probe_C0_error=probe_errs,
        imagination_real_ratio=dict(left_step=left_ratio, right_step=right_ratio),
        ratio_flag=dict(left_step=cliff_flag(left_ratio), right_step=cliff_flag(right_ratio)),
        calibration_touched=False, track_b_touched=False, sealed_data_touched=False,
    )
    (OUT / "result.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ()}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run()
