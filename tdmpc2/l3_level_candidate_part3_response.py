"""L3 Level-mapped candidate screen, Part 3: L4 Response instance
G(stumble -> F(gait_resumed)), cheap "autopsy" using the ALREADY-SAVED
walker-walk Test1 1000-rollout imagination batch. Zero new rollouts.

stumble := an inter-gait-peak gap exceeding 2x the established real-data
interval mean (13.48 steps, from Phase A's own real-side survey) -- i.e.
gaps > ~27 steps, rounded to a clean 30-step threshold. Uses the SAME
frozen prominence=0.03 peak detector as gait_cycle itself (no new event
definition invented).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy.signal import find_peaks

OUT = pathlib.Path("artifacts/l3_level_mapped_candidates/part3_response")
DATA = "artifacts/tdmpc2_walker_l3_statistical_lbsm/phase_b/test1/fresh_1000_data.npz"
PROMINENCE = 0.03
STUMBLE_THRESHOLD = 30  # ~2x the established real interval mean (13.48)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA)
    h = d["h"]  # (1000, 300)
    n, T = h.shape

    n_stumbles_per_traj = []
    all_gaps = []
    trajectories_with_stumble = 0
    resumed_within_30 = 0
    for i in range(n):
        pk, _ = find_peaks(h[i], prominence=PROMINENCE)
        if len(pk) < 2:
            n_stumbles_per_traj.append(0)
            continue
        gaps = np.diff(pk)
        all_gaps.extend(gaps.tolist())
        stumbles = gaps[gaps > STUMBLE_THRESHOLD]
        n_stumbles_per_traj.append(len(stumbles))
        if len(stumbles) > 0:
            trajectories_with_stumble += 1
            # "resumed": does a gait_cycle peak eventually occur again within the trajectory? by
            # construction it does (that's how we detected the gap), so this is trivially checkable;
            # here we specifically check whether resumption happens within 30 steps of the stumble start.
            resumed_within_30 += int((stumbles <= 60).sum())  # generous secondary window, informational

    total_stumbles = int(sum(n_stumbles_per_traj))
    all_gaps = np.array(all_gaps)

    report = dict(
        n_trajectories=n, horizon=T, prominence=PROMINENCE, stumble_threshold_steps=STUMBLE_THRESHOLD,
        gap_distribution=dict(mean=float(all_gaps.mean()), p90=float(np.percentile(all_gaps, 90)),
                              p99=float(np.percentile(all_gaps, 99)), max=float(all_gaps.max())),
        total_stumble_events=total_stumbles,
        mean_stumbles_per_trajectory=float(np.mean(n_stumbles_per_traj)),
        n_trajectories_with_ge1_stumble=trajectories_with_stumble,
        fraction_trajectories_with_stumble=trajectories_with_stumble / n,
        verdict=(
            "TRIGGER_SCARCE_CANDIDATE_CLOSED" if total_stumbles < 10 else
            "TRIGGER_PRESENT_WORTH_FURTHER_LOOK"
        ),
    )
    (OUT / "result.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run()
