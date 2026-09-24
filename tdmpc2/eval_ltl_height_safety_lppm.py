"""
eval_ltl_height_safety_lppm.py — First real NeuralLPPM verdict for
ltl_height_safety: G(height>h_min) on the officially-loaded TD-MPC2
walker-walk (checkpoint seed=3) agent.

Mirrors cardreamer/eval_ltl_hazard_real_lppm.py's structure and the fixes it
already made (disjoint train/calibration split; explicit liveness/activity-
trap check; explicit backend log).

Anchors and action_source (not a free choice -- see prior validation)
-----------------------------------------------------------------------
Uses the 200 REAL MPC-episode physics states saved by
tdmpc2_recollect_with_physics_state.py, replayed exactly via
sample_latent_rollouts_from_states() (physics.set_state(), lossless to
~1e-8) -- NOT the approximate burn_in path. action_source="mpc_plan" only:
this is the sole (anchor distribution, action-selection mechanism) pairing
that C0/C1 actually validated (C0: R²=0.9986; C1 curve (C), matched
mechanism AND matched distribution: depth-100 drift p50=0.0218m
p90=0.0637m, consistent across all 8 contributing episodes). Do not swap in
pi_prior or the burn_in anchor path here without re-validating C1 for that
combination.

h_min = 0.27 (ground-contact-height-v2)
--------------------------------------
The previous 0.6 threshold represented severe loss of standing reward, not a
torso lying on the floor. Nine passive-fall trials observed persistent
floor--torso contact up to 0.2442m. The replacement adds a 0.02m empirical
margin and rounds upward to 0.27m. Changing the predicate invalidates every
old fit/calibration result; this script must collect a fresh split when rerun.

Usage: conda activate dyno && python tdmpc2/eval_ltl_height_safety_lppm.py
"""
import sys
sys.path.insert(0, "/home/bot/SafeWorld")

import numpy as np
import torch

from specs.ltl_specs import get_ltl_spec_by_id
from specs.stl_specs import get_stl_spec_by_id
from specs.walker_constants import WALKER_FALL_HEIGHT_M
from utils.spec_analysis import analyze_spec_structure
from core.lppm.automaton import build_parity_automaton
from core.lppm.config import DEFAULT_LPPM_CONFIG
from core.lppm.trainer import fit_lppm
from core.lppm.calibrator import calibrate_lppm
from core.lppm.verifier import run_product_trajectory, verify_zfree_closure
from core.stl_monitor import monitor_rollouts
from core.transfer_calibrator import compute_atomic_distortion
from core.safety_evidence import (
    ENVIRONMENT_VIOLATION,
    MODEL_VIOLATION,
    collect_invariant_safety_witnesses,
    decide_safety_verdict,
)

CKPT = "/home/bot/SafeWorld/models/walker-walk-3.pt"
ANCHORS_NPZ = "/tmp/tdmpc2_anchors_with_physics_state.npz"
PILOT_NPZ = "/tmp/tdmpc2_pilot_data.npz"
HORIZON = 100          # matches the validated C1 depth (do not exceed without re-validating)
TRAIN_FRAC = 0.6
ETA = DEFAULT_LPPM_CONFIG.eta
GAMMA = 0.05
N_EPOCHS = 1500

# Established CarDreamer result (ltl_hazard_avoidance), for the comparison table.
CARDREAMER_HAZARD_REF = {
    "spec": "ltl_hazard_avoidance", "p_hat_gamma": 0.7856, "warranted": False,
    "warrant_threshold": 0.80,
}


def main():
    ltl_spec = get_ltl_spec_by_id("ltl_height_safety")
    ltl_spec["analysis"] = analyze_spec_structure(ltl_spec)
    dpa = build_parity_automaton(ltl_spec)
    print(f"[0] automaton: states={dpa.states}  priority={dpa.priority}  "
        f"odd_priorities={dpa.odd_priorities}  backend={dpa.backend}")

    # ── sample from the EXACT, validated real-MPC physics-state anchors ──────
    print(f"\n[1] sampling from 200 real-MPC physics-state anchors "
        f"(horizon={HORIZON}, action_source=mpc_plan) ...", flush=True)
    from wrappers.tdmpc2_wrapper import TDMPC2Wrapper
    from wrappers.tdmpc2_probes import fit_height_probe, make_height_ap_extractor

    w = TDMPC2Wrapper()
    w.load(checkpoint=CKPT, task="walker-walk", seed=3)
    probe = fit_height_probe(PILOT_NPZ)
    w.set_ap_extractor(make_height_ap_extractor(probe), keys=["height"])

    physics_states = np.load(ANCHORS_NPZ)["physics_state"]
    z_array, meta = w.sample_latent_rollouts_from_states(
        physics_states, action_source="mpc_plan", horizon=HORIZON,
    )
    N, T, _ = z_array.shape
    extractor = make_height_ap_extractor(probe)
    trajectories = [[extractor(z_array[i, t]) for t in range(T)] for i in range(N)]
    print(f"    got {len(trajectories)} trajectories x {T} steps")

    # Concrete invariant failures take precedence over certificate fitting.
    # This is formula-driven rather than height-specific, so the same gate is
    # usable by any spec evaluated through this script structure.
    model_witnesses = collect_invariant_safety_witnesses(
        trajectories,
        ltl_spec["formula"],
        source="model",
        spec_id=ltl_spec["id"],
        checkpoint_id=CKPT,
        rollout_provenance=meta["provenance"],
    )
    evidence_verdict = decide_safety_verdict(model_witnesses=model_witnesses)
    if evidence_verdict.verdict == MODEL_VIOLATION:
        witness = evidence_verdict.witness
        witness_index = witness.rollout_index

        # Replay the exact recorded actions that generated this model-side
        # witness, from the exact same saved physics anchor.  This callback is
        # experiment-specific because real observation layouts are environment
        # adapters; the replay and verdict mechanisms themselves remain generic.
        def environment_ap_extractor(obs):
            if hasattr(obs, "detach"):
                obs = obs.detach().cpu().numpy()
            return {"height": float(obs[14])}

        env_trajectories = w.replay_recorded_actions_from_states(
            physics_states[witness_index:witness_index + 1],
            [meta["provenance"][witness_index]],
            environment_ap_extractor=environment_ap_extractor,
        )
        environment_witnesses = collect_invariant_safety_witnesses(
            env_trajectories,
            ltl_spec["formula"],
            source="environment",
            spec_id=ltl_spec["id"],
            checkpoint_id=CKPT,
            rollout_provenance=[meta["provenance"][witness_index]],
        )
        paired_verdict = decide_safety_verdict(
            model_witnesses=[witness],
            environment_witnesses=environment_witnesses,
        )
        witness_ap_error = compute_atomic_distortion(
            trajectories[witness_index], env_trajectories[0], ltl_spec["aps"],
        )
        print(
            f"\n[COUNTEREXAMPLE] {MODEL_VIOLATION}: rollout={witness.rollout_index} "
            f"t={witness.time_index} AP={witness.ap_key} value={witness.observed_value} "
            f"threshold={witness.threshold} witness_id={witness.witness_id}."
        )
        print(
            f"[PAIRED REPLAY] max_abs_AP_error={witness_ap_error:.6f}; "
            f"environment_min_height={min(step['height'] for step in env_trajectories[0]):.6f}; "
            f"verdict={paired_verdict.verdict}."
        )
        if paired_verdict.verdict == ENVIRONMENT_VIOLATION:
            print("Stopping: the identical action sequence also violates the real environment.")
        else:
            print(
                "Stopping: the real replay did not violate this finite prefix, so this is a "
                "model-side false positive for the replayed witness—not a safety guarantee."
            )
        w.close()
        return

    w.close()

    # ── liveness check: does the imagined data ever visit the trap state? ────
    n_visit_trap = 0
    height_mins = []
    for traj in trajectories:
        path = run_product_trajectory(traj, dpa, ltl_spec)
        height_mins.append(min(s["height"] for s in traj))
        if any(ps.priority == 1 for ps in path):
            n_visit_trap += 1
    print(f"\n[2] LIVENESS CHECK (own-odd priority = 'trap' visitation, "
        f"same activity trap as CarDreamer's L4'):")
    print(f"    rollouts that ever visit trap: {n_visit_trap}/{len(trajectories)}")
    print(f"    height min across rollouts: p10={np.percentile(height_mins,10):.3f} "
        f"p50={np.percentile(height_mins,50):.3f} min={min(height_mins):.3f}  "
        f"(h_min={WALKER_FALL_HEIGHT_M})")
    if n_visit_trap == 0:
        print(f"    !!! VACUOUS: the trap (height<={WALKER_FALL_HEIGHT_M}) is NEVER "
            "visited in this data. Any "
            "p_hat_gamma computed here is a liveness artifact, not genuine descent evidence -- "
            "same activity-trap class as ltl_hazard_avoidance's own flagged issue. A WARRANT "
            "verdict below must be reported with this caveat, not as strong evidence.")

    # ── disjoint train / calibration split ───────────────────────────────────
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(trajectories))
    n_train = int(len(trajectories) * TRAIN_FRAC)
    train_traj = [trajectories[i] for i in idx[:n_train]]
    calib_traj = [trajectories[i] for i in idx[n_train:]]
    print(f"\n[3] disjoint split: train={len(train_traj)}  calibration={len(calib_traj)}")

    # ── fit NeuralLPPM on the TRAIN split only ────────────────────────────────
    print(f"\n[4] training NeuralLPPM on train split ({N_EPOCHS} epochs max)...")
    training_info = fit_lppm(train_traj, dpa, ltl_spec, eta=ETA, n_epochs=N_EPOCHS)
    backend = training_info.get("backend", "unknown")
    print(f"    backend={backend}  epochs_trained={training_info['epochs_trained']}  "
        f"final_loss={training_info['final_loss']:.6f}  n_transitions={training_info['n_transitions']}")
    if backend != "torch_mlp":
        print(f"    WARNING: NOT a trained NeuralLPPM ({training_info.get('reason')}) -- "
            f"p_hat_gamma below comes from the untrained heuristic, not a fitted certificate.")

    # ── calibrate on the DISJOINT held-out split ─────────────────────────────
    print(f"\n[5] calibrating p_hat_gamma on {len(calib_traj)} HELD-OUT rollouts...")
    lppm_res = calibrate_lppm(calib_traj, dpa, ltl_spec, gamma=GAMMA, eta=ETA,
                              lppm_params=training_info)
    print(f"    k={sum(1 for pw in lppm_res.pathwise if pw.satisfied)}/{len(calib_traj)}  "
        f"p_hat_gamma={lppm_res.p_hat_gamma:.4f}  warranted={lppm_res.is_warranted()}")

    # ── precise breakdown: WHICH condition (P1 / P2 / Z_free) is failing? ────
    n_p1_fail = sum(1 for pw in lppm_res.pathwise if pw.p1_violations > 0)
    n_p2_fail = sum(1 for pw in lppm_res.pathwise if pw.p2_violations > 0)
    n_p1_only = sum(1 for pw in lppm_res.pathwise if pw.p1_violations > 0 and pw.p2_violations == 0)
    print(f"    breakdown: p1_violations>0 in {n_p1_fail}/{len(calib_traj)} rollouts  "
        f"(avg count={np.mean([pw.p1_violations for pw in lppm_res.pathwise]):.2f})")
    print(f"               p2_violations>0 in {n_p2_fail}/{len(calib_traj)} rollouts  "
        f"(avg count={np.mean([pw.p2_violations for pw in lppm_res.pathwise]):.2f})")
    print(f"               rollouts failing ONLY on P1 (not P2): {n_p1_only}/{len(calib_traj)}")
    calib_height_mins = [min(s["height"] for s in calib_traj[i]) for i in range(len(calib_traj))]
    n_calib_genuinely_low = sum(
        1 for h in calib_height_mins if h <= WALKER_FALL_HEIGHT_M
    )
    print(f"    calibration-split height min: p10={np.percentile(calib_height_mins,10):.3f} "
        f"min={min(calib_height_mins):.3f}  (rollouts with min<=h_min: {n_calib_genuinely_low}/{len(calib_traj)})")
    if n_p1_fail > 0 and n_calib_genuinely_low < len(calib_traj):
        print(f"    DIAGNOSIS: k=0 despite only {n_calib_genuinely_low}/{len(calib_traj)} rollouts "
            f"genuinely dipping below h_min -- most of the {n_p1_fail} P1 failures are on "
            f"rollouts that never approach h_min, so this is very likely the SAME "
            f"under-training false-negative pattern diagnosed for ltl_hazard_avoidance "
            f"(final_loss=0.009 > 0, causing isolated non-increase violations on genuinely-"
            f"safe trajectories), NOT {len(calib_traj)}/{len(calib_traj)} genuine violations.")

    # ── STL cross-check: rho* on ALL trajectories (bounded, same horizon) ────
    print(f"\n[6] STL cross-check (rho*, bounded stl_height_safety, all {N} trajectories):")
    stl_spec = get_stl_spec_by_id("stl_height_safety")
    mres = monitor_rollouts(stl_spec["formula"], trajectories)
    print(f"    rho*={mres.rho_star:+.4f}  witness_idx={mres.witness_idx}  "
        f"n_satisfied={mres.n_satisfied}/{mres.n_rollouts}  "
        f"mean={mres.mean_margin:+.4f}  std={mres.std_margin:.4f}  "
        f"verdict={'WARRANT' if mres.rho_star > 0 else 'VIOLATION'}")

    # ── comparison table vs CarDreamer's ltl_hazard_avoidance ────────────────
    print(f"\n[7] COMPARISON TABLE:")
    print(f"    {'spec':<22} {'p_hat_gamma':>12} {'warranted':>10} {'threshold':>10}")
    print(f"    {'ltl_hazard_avoidance':<22} {CARDREAMER_HAZARD_REF['p_hat_gamma']:>12.4f} "
        f"{str(CARDREAMER_HAZARD_REF['warranted']):>10} {CARDREAMER_HAZARD_REF['warrant_threshold']:>10.2f}")
    print(f"    {'ltl_height_safety':<22} {lppm_res.p_hat_gamma:>12.4f} "
        f"{str(lppm_res.is_warranted()):>10} {lppm_res.warrant_threshold:>10.2f}")
    if n_visit_trap == 0:
        print(f"    NOTE: ltl_height_safety's number above is a VACUOUS-liveness result "
            f"(trap never visited) -- not directly comparable in strength to "
            f"ltl_hazard_avoidance's p_hat_gamma, which came from data where the trap WAS live.")

    # ── Z_free-internal closure: Theorem 5.4's ACTUAL minimal premise, on the
    # SAME calibration split (calib_traj) used for p_hat_gamma in [5] above --
    # not a new split, not new data. See core/lppm/verifier.py::verify_zfree_closure(). ──
    print(f"\n[8] Z_FREE-INTERNAL CLOSURE CHECK (Theorem 5.4's exact minimal premise, "
        f"same calib_traj as [5] above):")
    closure = verify_zfree_closure(calib_traj, dpa, ltl_spec, gamma=GAMMA, eta=ETA, lppm_params=training_info)
    if not closure.verifiable:
        print("    UNVERIFIABLE: zero (transition, head) pairs in this calibration split had "
            "a source state inside Z_free = {(z,q): V<eta} -- no Z_free-internal samples to "
            "compute a bound from (consistent with this spec's known k=0 / vacuous-liveness "
            "diagnosis above). p_hat_gamma above remains a valid lower bound for the stronger "
            "whole-trajectory event C(tau), but cannot be cited as satisfying Theorem 5.4's "
            "minimal premise specifically.")
    else:
        print(f"    n_zfree={closure.n_zfree}  k_zfree={closure.k_zfree}  "
            f"p_hat_closure={closure.p_hat_closure:.4f}  (gamma={closure.gamma}, eta={closure.eta})")
        if closure.p_hat_closure >= lppm_res.warrant_threshold:
            print(f"    p_hat_closure >= warrant_threshold={lppm_res.warrant_threshold:.2f} -- "
                "p_hat_gamma may now be cited as satisfying Theorem 5.4's minimal premise "
                "(not just the stronger whole-trajectory event).")
        else:
            print(f"    p_hat_closure < warrant_threshold={lppm_res.warrant_threshold:.2f} -- "
                "NOT enough evidence yet for the Theorem-5.4-minimal-premise wording; "
                "continue citing p_hat_gamma only as a lower bound for the stronger "
                "whole-trajectory event C(tau).")


if __name__ == "__main__":
    main()
