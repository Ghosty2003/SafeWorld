"""Candidate 5b: the missing anti-fraud test for the deduction scheme --
does it produce false positives on GENUINE real fall trajectories (real
observation-encoded posterior z, matched with real ground-truth height)?

This is the prerequisite Candidate 5 flagged as untested: its own real
held-out set (tdmpc2_walker_lppm_latents) contained zero ground-truth
failures by construction, so FP=0 there was vacuous. This script closes
that gap with genuinely manufactured real failures.

PRE-REGISTRATION (fixed before any number is computed):

1. Fall-trajectory manufacture: from a real anchor (c1_exact_mpc_anchors.npz,
   sampled without replacement), run REAL CLOSED-LOOP mpc_plan for a random
   t_switch in [20,80] steps (uniform, seeded), giving a genuine real gait
   segment; then switch to the SAME constant all-negative saturated action
   block that refuted the odd-budget candidate elsewhere in this project
   (action = -ones(action_dim), from search_walker_action_block_counterexample.py),
   applied for the REMAINDER of a fixed 300-step trajectory. At every step,
   record the REAL observation's encoded posterior z (wrapper.encode(obs)
   on the real post-step observation -- NOT imagined) and ground-truth
   torso height (physics.torso_height()).

   Ground-truth FAILURE label (fixed before checking any trajectory):
   mean(height[-30:]) < 0.4m AND height[-1] < 0.4m (comfortably above
   WALKER_FALL_HEIGHT_M=0.27m, i.e. genuinely down and not recovering).
   Trajectories that recover (fail this check) are discarded and replaced
   with a fresh anchor/t_switch draw until 50 genuine failures are
   collected; the discard rate is reported, not hidden.

2. Deduction-scheme evaluation: IDENTICAL pre-registered parameters as
   Candidate 5 -- prominence=0.03, tol=2 (unused here, no matching needed,
   kept for record), false-rate CP-upper deduction (0.07807546497166352,
   cited verbatim from Candidate 5's result.json, not recomputed), M_MIN=15,
   CV<=0.446. The SAME Ridge probe (fit on the first 1200 of the 1500
   tdmpc2_walker_lppm_latents rollouts) is reused, reproduced deterministically
   (Ridge has no randomness), not retrained.

   FALSE POSITIVE (the critical number): a real ground-truth FAILURE
   trajectory that the deduction verdict calls "recurrence maintained".

3. Lying-down-segment diagnostic: for each trajectory, restricted to steps
   [t_switch+30, 300) (a 30-step grace period after the action switch, to
   let the fall complete before diagnosing the settled lying-down segment),
   compute the decoded-peak rate and per-trajectory decoded-interval CV,
   to check whether the "fake peaks in a fallen segment are irregular"
   design assumption behind the CV gate actually holds on real failure data.

4. Mixed-scenario/front-loading check: for each trajectory, count decoded
   peaks whose index < t_switch (front-segment, genuine-gait-eligible)
   separately from peaks at index >= t_switch (back-segment, lying-down).
   If any false positive arises from front-segment peaks alone nearly
   reaching M_MIN and back-segment noise pushing it over, this is reported
   explicitly as the mechanism, and a PATCHED sliding-window M_MIN rule is
   tried and re-reported as a clearly-labeled post-hoc, stricter correction
   (any contiguous 150-step window must itself contain >= M_MIN/2 = 7 or 8
   decoded, deduction-adjusted cycles) -- flagged as needing fresh-data
   re-validation, not accepted as proven by this same generation run.

No frozen artifact modified. No sealed/consumed calibration data touched.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from scipy.signal import find_peaks
from sklearn.linear_model import Ridge

from wrappers.tdmpc2_wrapper import TDMPC2Wrapper

OUT = pathlib.Path("artifacts/l3_rescue_screen/candidate5_deduction_scheme")
LATENTS_DIR = pathlib.Path("artifacts/tdmpc2_walker_lppm_latents")
ANCHORS = pathlib.Path("artifacts/tdmpc2_forward_speed/c1_exact_mpc_anchors.npz")
CAND5_RESULT = OUT / "result.json"

N_TARGET = 50
HORIZON = 300
T_SWITCH_LOW, T_SWITCH_HIGH = 20, 80
FALL_HEIGHT_THRESH = 0.4
SEED_BASE = 70000
PROMINENCE = 0.03
GRACE_STEPS = 30
SLIDING_WINDOW = 150
SLIDING_MIN_CYCLES = 8  # ceil(M_MIN/2) = ceil(15/2)


def cp_upper_from_cand5():
    r = json.loads(CAND5_RESULT.read_text())
    return r["upper_bounds"]["false_rate_cp_upper_95"], r["M_MIN"], r["CV_p95_threshold"]


def per_traj_intervals(seq_1d, prominence):
    pk, _ = find_peaks(seq_1d, prominence=prominence)
    if len(pk) > 1:
        return pk, np.diff(pk).astype(np.float64)
    return pk, np.array([])


def fit_probe():
    seed_files = sorted(LATENTS_DIR.glob("seed_*_latents.npz"))
    all_h, all_z = [], []
    for f in seed_files:
        d = np.load(f)
        all_h.append(d["height_m"])
        all_z.append(d["posterior_z"])
    h = np.concatenate(all_h, axis=0)
    z = np.concatenate(all_z, axis=0)
    n = h.shape[0]
    n_holdout = 300
    fit_idx = np.arange(n - n_holdout)
    return Ridge(alpha=10.0).fit(z[fit_idx].reshape(-1, 512), h[fit_idx].reshape(-1))


def generate_fall_trajectories():
    print(f"[1] manufacturing {N_TARGET} genuine real fall trajectories ...", flush=True)
    rng = np.random.default_rng(SEED_BASE)
    anchors = np.load(ANCHORS)
    physics_states = anchors["physics_state"]
    n_anchors = len(physics_states)

    wrapper = TDMPC2Wrapper()
    wrapper.load(checkpoint="models/walker-walk-3.pt", task="walker-walk", seed=1)
    if str(wrapper._agent.device).split(":")[0] != "cuda":
        wrapper.close()
        raise RuntimeError("CUDA required")

    action_dim = wrapper._agent.cfg.action_dim
    all_negative = torch.tensor(-np.ones(action_dim, dtype=np.float32))
    print(f"    action_dim={action_dim}, saturated action=all_negative", flush=True)

    accepted, discarded = [], 0
    anchor_order = rng.permutation(n_anchors)
    anchor_cursor = 0
    attempts = 0
    try:
        while len(accepted) < N_TARGET:
            attempts += 1
            if anchor_cursor >= n_anchors:
                anchor_order = rng.permutation(n_anchors)
                anchor_cursor = 0
            anchor_idx = anchor_order[anchor_cursor]
            anchor_cursor += 1
            t_switch = int(rng.integers(T_SWITCH_LOW, T_SWITCH_HIGH + 1))

            obs = wrapper._obs_from_physics_state(physics_states[anchor_idx])
            z = wrapper.encode(obs)
            z_seq, h_seq = [], []
            with torch.no_grad():
                for step in range(HORIZON):
                    if step < t_switch:
                        a = wrapper.plan_action(z, t0=(step == 0), eval_mode=True)
                        a_apply = a[0].detach().cpu()
                    else:
                        a_apply = all_negative
                    obs, _r, done, _info = wrapper._env.step(a_apply)
                    z = wrapper.encode(obs)
                    z_seq.append(z[0].detach().cpu().numpy().astype(np.float64, copy=True))
                    h_seq.append(float(wrapper._physics.torso_height()))
                    if done:
                        break
            if len(h_seq) < HORIZON:
                discarded += 1
                continue  # episode terminated early, doesn't give a clean 300-step record

            h_arr = np.array(h_seq)
            is_failure = (np.mean(h_arr[-30:]) < FALL_HEIGHT_THRESH) and (h_arr[-1] < FALL_HEIGHT_THRESH)
            if not is_failure:
                discarded += 1
                continue

            accepted.append(dict(z=np.stack(z_seq), h=h_arr, t_switch=t_switch, anchor_idx=int(anchor_idx)))
            if len(accepted) % 10 == 0:
                print(f"    accepted {len(accepted)}/{N_TARGET} (discarded so far: {discarded})", flush=True)
    finally:
        wrapper.close()

    discard_rate = discarded / attempts if attempts else 0.0
    print(f"    done: {len(accepted)} accepted, {discarded} discarded, {attempts} attempts, discard_rate={discard_rate:.3f}", flush=True)

    z_all = np.stack([a["z"] for a in accepted])
    h_all = np.stack([a["h"] for a in accepted])
    t_switch_all = np.array([a["t_switch"] for a in accepted])
    anchor_idx_all = np.array([a["anchor_idx"] for a in accepted])
    np.savez_compressed(OUT / "real_fall_trajectories.npz", z=z_all, h=h_all,
                        t_switch=t_switch_all, anchor_idx=anchor_idx_all)
    return z_all, h_all, t_switch_all, discard_rate, attempts, discarded


def run():
    false_rate_cp_upper, m_min, cv_p95 = cp_upper_from_cand5()
    print(f"cited from Candidate 5: false_rate_cp_upper={false_rate_cp_upper}, M_MIN={m_min}, CV_p95={cv_p95}", flush=True)

    z_fall, h_fall, t_switch, discard_rate, attempts, discarded = generate_fall_trajectories()
    n = z_fall.shape[0]

    probe = fit_probe()

    rows = []
    for i in range(n):
        h_pred = probe.predict(z_fall[i])
        pred_pk, pred_iv = per_traj_intervals(h_pred, PROMINENCE)
        m = len(pred_pk)
        m_adj = m * (1 - false_rate_cp_upper)
        cv = (pred_iv.std() / pred_iv.mean()) if len(pred_iv) >= 2 and pred_iv.mean() > 0 else np.inf
        verdict_no_cv = m_adj >= m_min
        verdict_with_cv = verdict_no_cv and (cv <= cv_p95)

        # item 3: front (genuine-gait-eligible) vs back (lying-down) peak split
        n_front_peaks = int((pred_pk < t_switch[i]).sum())
        n_back_peaks = int((pred_pk >= t_switch[i]).sum())

        # item 2c: lying-down-only diagnostic (grace period after switch)
        grace_start = t_switch[i] + GRACE_STEPS
        if grace_start < HORIZON - 2:
            lying_pred = h_pred[grace_start:]
            lying_pk, lying_iv = per_traj_intervals(lying_pred, PROMINENCE)
            lying_false_peak_rate = len(lying_pk) / len(lying_pred)
            lying_cv = (lying_iv.std() / lying_iv.mean()) if len(lying_iv) >= 2 and lying_iv.mean() > 0 else None
        else:
            lying_false_peak_rate, lying_cv = None, None

        # sliding-window patch (item 3): does EVERY contiguous 150-step window
        # independently clear a proportional cycle bar?
        sliding_ok = True
        for w0 in range(0, HORIZON - SLIDING_WINDOW + 1, 10):
            seg = h_pred[w0:w0 + SLIDING_WINDOW]
            seg_pk, _ = per_traj_intervals(seg, PROMINENCE)
            seg_m_adj = len(seg_pk) * (1 - false_rate_cp_upper)
            if seg_m_adj < SLIDING_MIN_CYCLES:
                sliding_ok = False
                break

        rows.append(dict(
            traj=i, t_switch=int(t_switch[i]), m=m, m_adj=m_adj,
            cv=float(cv) if np.isfinite(cv) else None,
            verdict_no_cv=bool(verdict_no_cv), verdict_with_cv=bool(verdict_with_cv),
            verdict_sliding_patched=bool(sliding_ok),
            n_front_peaks=n_front_peaks, n_back_peaks=n_back_peaks,
            lying_down_false_peak_rate=lying_false_peak_rate, lying_down_cv=lying_cv,
            ground_truth_label="FAIL",  # by construction, all accepted trajectories
        ))

    fp_no_cv = sum(1 for r in rows if r["verdict_no_cv"])
    fp_with_cv = sum(1 for r in rows if r["verdict_with_cv"])
    fp_sliding = sum(1 for r in rows if r["verdict_sliding_patched"])

    lying_rates = [r["lying_down_false_peak_rate"] for r in rows if r["lying_down_false_peak_rate"] is not None]
    lying_cvs = [r["lying_down_cv"] for r in rows if r["lying_down_cv"] is not None]

    front_loading_cases = [r for r in rows if r["verdict_with_cv"] and r["n_front_peaks"] > 0 and r["n_back_peaks"] > 0]

    if fp_with_cv == 0 and fp_sliding == 0:
        final_verdict = "ANTI_FRAUD_CHECK_PASSED_FP_ZERO"
    else:
        final_verdict = "SOUNDNESS_REFUTED_BY_REAL_FALL_DATA"

    report = dict(
        pre_registration=dict(
            fall_height_thresh=FALL_HEIGHT_THRESH, t_switch_range=[T_SWITCH_LOW, T_SWITCH_HIGH],
            horizon=HORIZON, false_rate_cp_upper_cited_from_candidate5=false_rate_cp_upper,
            M_MIN=m_min, CV_p95=cv_p95, sliding_window=SLIDING_WINDOW, sliding_min_cycles=SLIDING_MIN_CYCLES,
        ),
        generation=dict(n_accepted=n, n_discarded=discarded, n_attempts=attempts, discard_rate=discard_rate,
                        note="discarded trajectories either terminated early (dm_control episode limit) or recovered (did not stay down) before completing 300 steps"),
        n_ground_truth_failures=n,
        false_positives=dict(
            without_CV_gate=fp_no_cv, with_CV_gate=fp_with_cv, with_sliding_window_patch=fp_sliding,
        ),
        lying_down_diagnostic=dict(
            mean_false_peak_rate=float(np.mean(lying_rates)) if lying_rates else None,
            mean_cv=float(np.mean([c for c in lying_cvs if np.isfinite(c)])) if lying_cvs else None,
            n_trajectories_with_valid_lying_segment=len(lying_rates),
            cv_gate_design_assumption_holds=bool(
                lying_cvs and np.mean([c for c in lying_cvs if np.isfinite(c)]) > cv_p95
            ) if lying_cvs else None,
        ),
        front_loading_mechanism=dict(
            n_trajectories_with_both_front_and_back_peaks_among_CV_passers=len(front_loading_cases),
            detail_sample=front_loading_cases[:10],
        ),
        final_verdict=final_verdict,
        calibration_touched=False, track_b_touched=False, sealed_data_touched=False,
    )
    (OUT / "candidate5b_result.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "candidate5b_rows.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "front_loading_mechanism"}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run()
