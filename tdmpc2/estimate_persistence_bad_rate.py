"""Step 1 (sizing estimate only, no training): reuse the existing 200x500
pure-planner model-imagination batch (collect_walker_imagination_rollouts.py,
already generated -- same anchors/mechanism the user is asking for: encode()
anchor -> loop(plan_action(mpc_plan) -> next(z,a)), no real MuJoCo physics
after the anchor restore, no forced/adversarial actions) and label it with
the existing real-contact classifier to estimate the NON-ABSORBING bad-rate
relevant to this task's spec.

IMPORTANT correction vs. all prior G(safe) scripts in this repo: those used
trap_trace() (np.maximum.accumulate), an ABSORBING "once contact, forever
trap" automaton appropriate for a Safety-class G(safe) certificate. This
task's spec is FG(safe) (Persistence, ⋄□safe) with an explicitly RECOVERABLE
two-state automaton per the user's instructions: q(t) = 1 iff contact(t)
holds AT THAT STEP (no memory). This is a materially different, much
smaller "bad-source transition" count than the sticky version, because a
recovered trajectory's bad-source transitions are just its raw isolated
contact steps, not the entire remaining tail after first contact.

Does not touch calibration data, Track B, or any real MuJoCo rollout. No
training is done here -- estimate only.
"""
from __future__ import annotations

import json
import pathlib

import joblib
import numpy as np

DATA = pathlib.Path("artifacts/tdmpc2_walker_imagination_rollouts/imagination_latents.npz")
CONTACT_CLF = pathlib.Path("artifacts/tdmpc2_walker_contact_classifier/contact_classifier.joblib")
OUT = pathlib.Path("artifacts/tdmpc2_walker_persistence_sizing")
TARGET_BAD_TRANSITIONS = 1000
# From collect_walker_imagination_rollouts.py's own result.json:
# 1128.707 seconds for 200 anchors x 500-step imagination.
SECONDS_PER_TRAJECTORY = 1128.7074949741364 / 200.0


def run():
    d = np.load(DATA)
    z = d["z"].astype(np.float64)  # (200, 500, 512) -- imagined states t=1..T
    n, horizon, _ = z.shape

    clf = joblib.load(CONTACT_CLF)
    contact = np.stack([clf.predict(z[i]).astype(bool) for i in range(n)])  # (200, 500)

    # Bad-source transitions: transitions (t -> t+1) whose SOURCE state t is
    # bad, under the non-sticky (recoverable) two-state automaton q(t)=contact(t).
    # We have 500 imagined states per trajectory -> 499 transitions each.
    q = contact  # q(t) = contact at imagined step t (no memory / no accumulate)
    bad_source = q[:, :-1]  # (200, 499) -- source-state bad flags for each transition
    n_transitions_total = bad_source.size
    n_bad_transitions = int(bad_source.sum())
    bad_rate = n_bad_transitions / n_transitions_total

    per_traj_bad_count = bad_source.sum(axis=1)  # (200,)
    traj_has_bad = per_traj_bad_count > 0
    n_traj_with_bad = int(traj_has_bad.sum())
    mean_bad_per_bad_traj = (
        float(per_traj_bad_count[traj_has_bad].mean()) if n_traj_with_bad else None
    )

    # Raw per-step contact rate (any imagined step, not just as a transition source)
    raw_contact_rate = float(contact.mean())
    n_traj_ever_contact = int(contact.any(axis=1).sum())

    # Extrapolation to reach TARGET_BAD_TRANSITIONS bad-source transitions.
    if n_bad_transitions > 0:
        transitions_needed = TARGET_BAD_TRANSITIONS / bad_rate
        trajectories_needed = transitions_needed / (horizon - 1)
        est_seconds = trajectories_needed * SECONDS_PER_TRAJECTORY
    else:
        transitions_needed = trajectories_needed = est_seconds = None

    report = dict(
        status="COMPLETE",
        experiment="tdmpc2_walker_persistence_sizing_estimate",
        purpose="Step 1 sizing-only estimate for FG(safe) Persistence (recoverable, non-absorbing) bad rate under pure model-imagination; NO training performed here.",
        data_source=str(DATA),
        data_provenance="pre-existing batch from collect_walker_imagination_rollouts.py: 200 real MuJoCo anchors (artifacts/tdmpc2_forward_speed/c1_exact_mpc_anchors.npz) restored once, then 500 steps of PURE model imagination via wrapper.encode()->plan_action(mpc_plan)->next(z,a); planner's own MPPI/CEM action only, zero forced/adversarial actions, zero real MuJoCo physics beyond the anchor restore.",
        contact_classifier=str(CONTACT_CLF),
        automaton_definition="NON-STICKY / recoverable: q(t) = contact(t) (current step only), NOT the absorbing trap_trace() used by prior G(safe) scripts in this repo -- matches this task's FG(safe) Persistence two-state spec.",
        n_trajectories=n, horizon=horizon,
        n_transitions_total=n_transitions_total,
        raw_contact_rate_per_imagined_state=raw_contact_rate,
        n_bad_source_transitions=n_bad_transitions,
        bad_source_transition_rate=bad_rate,
        n_trajectories_with_any_bad_transition=n_traj_with_bad,
        n_trajectories_with_any_raw_contact=n_traj_ever_contact,
        mean_bad_transitions_per_bad_containing_trajectory=mean_bad_per_bad_traj,
        extrapolation_to_target=dict(
            target_bad_source_transitions=TARGET_BAD_TRANSITIONS,
            transitions_needed_estimate=transitions_needed,
            trajectories_needed_estimate=trajectories_needed,
            seconds_per_trajectory_observed=SECONDS_PER_TRAJECTORY,
            estimated_generation_seconds=est_seconds,
            estimated_generation_hours=(est_seconds / 3600.0) if est_seconds else None,
        ),
        calibration_touched=False, track_b_touched=False, real_mujoco_used=False,
        forced_or_adversarial_actions_used=False,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
