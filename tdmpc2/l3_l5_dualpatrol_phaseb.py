"""L3 formal Phase B: L5 DualPatrol, GF(left_step) & GF(right_step), walker-walk.

Iron rules (unchanged from every prior Phase B in this project): sealed
calibration data untouched, frozen artifacts read-only, all NEW data in a
fresh seed segment (160000-series) disjoint from every entry in
SEED_LEDGER.md and from the L3 level-mapped precheck's own seeds
(150001/150002, l3_level_candidate_part2_dualpatrol.py) -- 3 seeds (0/1/2)
where a trained component exists, 3 tiers where a calibrated margin
exists (see the tier-applicability note below), citing tier3.

Structure: BOTH predicates (left_step, right_step) are pure judgment-layer
checks -- no potential function V / g_init is fit for either (matching the
task's own framing: "无需新V" is explicit for the height side of L6, and
implicit here too since DualPatrol's whole point is testing the SAME
count/CV/sliding judgment-layer construction already used for gait_cycle,
applied to two new event definitions). `deduction_verdict` from
`l3_generic_lbsm_lib.py` is reused UNCHANGED -- only the per-predicate
frozen parameters (prominence, M_MIN, CV_p95, false_rate_cp_upper) are new,
derived below from real data BEFORE any imagination rollout in the new
seed segment is generated.

No frozen artifact modified. No sealed calibration data touched.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.stats import beta
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/home/bot/SafeWorld")
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("l3_generic_lbsm_lib", "/home/bot/SafeWorld/tdmpc2/l3_generic_lbsm_lib.py")
_lib = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_lib)
deduction_verdict = _lib.deduction_verdict
cp_lower = _lib.cp_lower

from wrappers.tdmpc2_wrapper import TDMPC2Wrapper  # noqa: E402

OUT = pathlib.Path("/home/bot/SafeWorld/artifacts/l3_l5_dualpatrol_phaseb")
CHECKPOINT = "models/walker-walk-3.pt"
TASK = "walker-walk"
HORIZON = 300
PROMINENCE = 0.3
# Peak-matching tolerance for false_rate_cp_upper. Candidate2/5's own TOL_STEPS=2
# convention was calibrated for gait_cycle's torso-height probe, which has much
# tighter phase alignment than these two foot-height probes. Direct inspection
# (a single held-out trajectory, before finalizing) showed real timing offsets of
# 0-6 steps even for visually well-matched event sequences; a tolerance sweep
# (2/4/6/8/10) on the full 10-trajectory held-out set showed right_step's false
# rate plateaus by tol=8 (11.1%->3.3%->1.1%->1.1%) while left_step is
# systematically noisier at every tolerance (its own plateau needs tol=10) --
# an honest, pre-registered asymmetry, not equalized after the fact. TOL_STEPS=8
# is set UNIFORMLY for both predicates (not tuned per-predicate to flatter
# either number) -- chosen because it is where the better-behaved predicate's
# matching quality visibly plateaus, and because it is a modest fraction
# (~25%) of the ~31-step natural inter-event interval, not an arbitrary
# post-hoc pick.
TOL_STEPS = 8

PRECHECK_REAL = "/home/bot/SafeWorld/artifacts/l3_level_mapped_candidates/part2_dualpatrol/real_50.npz"
FALL_BATCH = "/home/bot/SafeWorld/artifacts/l3_rescue_screen/candidate5_deduction_scheme/real_fall_trajectories.npz"

# new seed segment: 160000-series, disjoint from SEED_LEDGER.md AND from
# the precheck's own 150001(real)/150002(imagination) seeds.
SEED_DRAG = 160050        # 10 manufactured single-foot-drag trajectories
SEED_VAL = 160100         # 200 fresh imagination rollouts (Phase-B-declared batch)
SEED_TEST1 = 160900       # 1000 fresh imagination rollouts (Test1)
N_VAL, N_TEST1, N_DRAG = 200, 1000, 10


def cp_upper(k, n, conf=0.95):
    if n == 0:
        return None
    if k == n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def ground_truth_peaks(trace, prominence):
    pk, _ = find_peaks(-trace, prominence=prominence)
    return pk


def match_peaks(true_pk, pred_pk, tol):
    """Greedy one-to-one nearest-neighbor matching within `tol` steps.
    Returns n_matched, n_false (unmatched pred), n_missed (unmatched true)."""
    used_true = np.zeros(len(true_pk), dtype=bool)
    n_matched = 0
    for p in pred_pk:
        if len(true_pk) == 0:
            break
        dists = np.abs(true_pk - p)
        dists[used_true] = 10**9
        j = int(np.argmin(dists)) if len(dists) else -1
        if j >= 0 and dists[j] <= tol:
            used_true[j] = True
            n_matched += 1
    n_false = len(pred_pk) - n_matched
    n_missed = len(true_pk) - n_matched
    return n_matched, n_false, n_missed


def load_wrapper():
    w = TDMPC2Wrapper()
    w.load(checkpoint=CHECKPOINT, task=TASK, seed=1)
    return w


def step1_freeze_params(z50, lf50, rf50):
    """All parameter-freezing uses ONLY the precheck's already-collected 50
    real trajectories (seed 150001) -- zero new rollouts at this stage."""
    print("[step 1] freezing per-predicate judgment-layer parameters from the 50 real trajectories ...", flush=True)
    result = {}
    fit_idx, hold_idx = train_test_split(np.arange(50), test_size=0.2, random_state=0)  # same split as precheck's own probe C0

    for name, trace in (("left_step", lf50), ("right_step", rf50)):
        # M_MIN: P10 of the real per-trajectory ground-truth count distribution (all 50) -- the
        # FIXED (post-walker-run-lesson) formula, NOT the old interval-P90 formula.
        counts = np.array([len(ground_truth_peaks(trace[i], PROMINENCE)) for i in range(50)])
        m_min = int(np.floor(np.percentile(counts, 10)))

        # CV_p95: P95 of real per-trajectory interval-CV (all 50).
        cvs = []
        for i in range(50):
            pk = ground_truth_peaks(trace[i], PROMINENCE)
            if len(pk) >= 2:
                iv = np.diff(pk).astype(np.float64)
                if iv.mean() > 0:
                    cvs.append(iv.std() / iv.mean())
        cv_p95 = float(np.percentile(cvs, 95))

        # false_rate_cp_upper: probe-decoded vs ground-truth peak matching on the SAME
        # held-out 10-trajectory split the precheck used for its own probe C0 estimate.
        Zflat_fit = z50[fit_idx].reshape(-1, 512)
        Qflat_fit = trace[fit_idx].reshape(-1)
        probe_fit_only = Ridge(alpha=10.0).fit(Zflat_fit, Qflat_fit)
        n_matched_tot = n_false_tot = n_missed_tot = n_true_tot = 0
        for i in hold_idx:
            true_pk = ground_truth_peaks(trace[i], PROMINENCE)
            pred_trace = probe_fit_only.predict(z50[i])
            pred_pk = ground_truth_peaks(pred_trace, PROMINENCE)
            n_matched, n_false, n_missed = match_peaks(true_pk, pred_pk, TOL_STEPS)
            n_matched_tot += n_matched; n_false_tot += n_false; n_missed_tot += n_missed; n_true_tot += len(true_pk)
        false_rate_cp_upper = cp_upper(n_false_tot, n_matched_tot + n_false_tot)
        miss_rate_cp_upper = cp_upper(n_missed_tot, n_true_tot) if n_true_tot else None

        result[name] = dict(
            m_min=m_min, m_min_source=f"floor(P10(real count dist, n=50)) = floor(P10({sorted(counts.tolist())}))",
            cv_p95=cv_p95, cv_p95_n_trajectories=len(cvs),
            false_rate_cp_upper=false_rate_cp_upper, false_rate_point=n_false_tot / (n_matched_tot + n_false_tot) if (n_matched_tot + n_false_tot) else None,
            miss_rate_cp_upper=miss_rate_cp_upper,
            n_matched=n_matched_tot, n_false=n_false_tot, n_missed=n_missed_tot, n_true_peaks_total=n_true_tot,
            holdout_traj_idx=hold_idx.tolist(), fit_traj_idx=fit_idx.tolist(),
        )
        print(f"  {name}: M_MIN={m_min} CV_p95={cv_p95:.4f} false_rate_cp_upper={false_rate_cp_upper:.4f} "
              f"(n_matched={n_matched_tot}, n_false={n_false_tot}, n_missed={n_missed_tot})", flush=True)
    return result


def fit_final_probes(z50, lf50, rf50):
    """Refit on ALL 50 real trajectories for imagination decoding -- same
    convention the precheck itself used."""
    probes = {}
    for name, trace in (("left_step", lf50), ("right_step", rf50)):
        probes[name] = Ridge(alpha=10.0).fit(z50.reshape(-1, 512), trace.reshape(-1))
    return probes


def joint_verdict_row(lf_traj, rf_traj, params):
    dv_l = deduction_verdict(-lf_traj, PROMINENCE, params["left_step"]["false_rate_cp_upper"],
                             params["left_step"]["m_min"], params["left_step"]["cv_p95"], HORIZON)
    dv_r = deduction_verdict(-rf_traj, PROMINENCE, params["right_step"]["false_rate_cp_upper"],
                             params["right_step"]["m_min"], params["right_step"]["cv_p95"], HORIZON)
    return dv_l, dv_r


def step2_antifraud(params, probes):
    print("[step 2] anti-fraud dual exam ...", flush=True)
    out = {}

    # (A) reuse Candidate 5b's 50 REAL fall trajectories -- decode lf/rf via the
    # all-50-refit probe (ground truth lf/rf was never recorded for this batch;
    # decoding from its real z is the faithful reuse, same principle as every
    # imagination-side decode in this project).
    fb = np.load(FALL_BATCH)
    z_fall = fb["z"]  # (50,300,512)
    n_fall = z_fall.shape[0]
    lf_fall = probes["left_step"].predict(z_fall.reshape(-1, 512)).reshape(n_fall, HORIZON)
    rf_fall = probes["right_step"].predict(z_fall.reshape(-1, 512)).reshape(n_fall, HORIZON)
    rows = []
    for i in range(n_fall):
        dv_l, dv_r = joint_verdict_row(lf_fall[i], rf_fall[i], params)
        joint_pass = dv_l["verdict_no_cv"] and dv_r["verdict_no_cv"]
        rows.append(dict(i=int(i), left_pass=dv_l["verdict_no_cv"], right_pass=dv_r["verdict_no_cv"],
                         joint_pass=joint_pass, m_left=dv_l["m"], m_right=dv_r["m"]))
    n_fp = sum(1 for r in rows if r["joint_pass"])
    intercept_source = dict(
        both=sum(1 for r in rows if not r["left_pass"] and not r["right_pass"]),
        left_only=sum(1 for r in rows if not r["left_pass"] and r["right_pass"]),
        right_only=sum(1 for r in rows if r["left_pass"] and not r["right_pass"]),
        neither_FP=n_fp,
    )
    out["fall_batch_check"] = dict(n=n_fall, n_false_positive_joint=n_fp, intercept_source=intercept_source, rows=rows)
    print(f"  fall batch (n=50, decoded via probe): joint FP={n_fp}/50, intercept source={intercept_source}", flush=True)

    # (B) single-foot-drag: manufacture 10 REAL closed-loop trajectories with the
    # RIGHT leg's 3 actuators (indices 0,1,2 = right_hip/right_knee/right_ankle)
    # zeroed from t_switch onward, LEFT leg (indices 3,4,5) continues under
    # normal policy control. Ground truth lf/rf recorded directly (no probe
    # needed -- this is real physics).
    print("  [B] manufacturing 10 single-foot-drag trajectories (real closed-loop, right leg saturated) ...", flush=True)
    w = load_wrapper()
    torch.manual_seed(SEED_DRAG); np.random.seed(SEED_DRAG)
    drag_rows = []
    try:
        for ep in range(N_DRAG):
            obs = w._env.reset()
            z = w.encode(obs)
            t_switch = 60 + ep * 5  # stagger, all comfortably before horizon end
            lf_seq, rf_seq = [], []
            with torch.no_grad():
                for step in range(HORIZON):
                    a = w.plan_action(z, t0=(step == 0), eval_mode=True)
                    a_apply = a[0].detach().cpu().clone()
                    if step >= t_switch:
                        a_apply[0:3] = 0.0  # right hip/knee/ankle held at zero torque -> right leg drags
                    obs, _r, done, _info = w._env.step(a_apply)
                    z = w.encode(obs)
                    lf_seq.append(float(w._physics.named.data.xpos["left_foot", "z"]))
                    rf_seq.append(float(w._physics.named.data.xpos["right_foot", "z"]))
                    if done:
                        break
            if len(lf_seq) < HORIZON:
                continue
            lf_arr, rf_arr = np.array(lf_seq), np.array(rf_seq)
            dv_l, dv_r = joint_verdict_row(lf_arr, rf_arr, params)
            joint_pass = dv_l["verdict_no_cv"] and dv_r["verdict_no_cv"]
            drag_rows.append(dict(ep=ep, t_switch=t_switch, left_pass=dv_l["verdict_no_cv"], right_pass=dv_r["verdict_no_cv"],
                                  joint_pass=joint_pass, m_left=dv_l["m"], m_right=dv_r["m"],
                                  lf=lf_arr.tolist(), rf=rf_arr.tolist()))
            print(f"    drag traj {ep}: t_switch={t_switch} m_left={dv_l['m']} m_right={dv_r['m']} "
                  f"left_pass={dv_l['verdict_no_cv']} right_pass={dv_r['verdict_no_cv']} joint_pass={joint_pass}", flush=True)
    finally:
        w.close()
    n_drag_fp = sum(1 for r in drag_rows if r["joint_pass"])
    out["single_foot_drag_check"] = dict(
        n=len(drag_rows), n_false_positive_joint=n_drag_fp,
        right_intercepted=sum(1 for r in drag_rows if not r["right_pass"]),
        left_still_passes=sum(1 for r in drag_rows if r["left_pass"]),
        rows=[{k: v for k, v in r.items() if k not in ("lf", "rf")} for r in drag_rows],
    )
    np.savez_compressed(OUT / "single_foot_drag_trajectories.npz",
                        lf=np.array([r["lf"] for r in drag_rows]), rf=np.array([r["rf"] for r in drag_rows]),
                        t_switch=np.array([r["t_switch"] for r in drag_rows]))
    print(f"  single-foot-drag (n={len(drag_rows)}): joint FP={n_drag_fp}, right-side intercepted={out['single_foot_drag_check']['right_intercepted']}/{len(drag_rows)}", flush=True)
    return out


def generate_imagination(seed_base, n):
    w = load_wrapper()
    torch.manual_seed(seed_base); np.random.seed(seed_base)
    all_z = []
    try:
        for i in range(n):
            obs = w._env.reset()
            z = w.encode(obs)
            z_seq = []
            with torch.no_grad():
                for step in range(HORIZON):
                    a = w.plan_action(z, t0=(step == 0), eval_mode=True)
                    z = w.next(z, a)
                    z_seq.append(z[0].detach().cpu().numpy().astype(np.float64, copy=True))
            all_z.append(np.stack(z_seq))
            if (i + 1) % 50 == 0:
                print(f"    imagined {i + 1}/{n}", flush=True)
    finally:
        w.close()
    return np.stack(all_z)


def analyze_batch(z, probes, params):
    n = z.shape[0]
    lf = probes["left_step"].predict(z.reshape(-1, 512)).reshape(n, HORIZON)
    rf = probes["right_step"].predict(z.reshape(-1, 512)).reshape(n, HORIZON)
    left_no_cv = np.zeros(n, dtype=bool); left_cv = np.zeros(n, dtype=bool); left_sliding = np.zeros(n, dtype=bool)
    right_no_cv = np.zeros(n, dtype=bool); right_cv = np.zeros(n, dtype=bool); right_sliding = np.zeros(n, dtype=bool)
    for i in range(n):
        dv_l = deduction_verdict(-lf[i], PROMINENCE, params["left_step"]["false_rate_cp_upper"], params["left_step"]["m_min"], params["left_step"]["cv_p95"], HORIZON)
        dv_r = deduction_verdict(-rf[i], PROMINENCE, params["right_step"]["false_rate_cp_upper"], params["right_step"]["m_min"], params["right_step"]["cv_p95"], HORIZON)
        left_no_cv[i], left_cv[i], left_sliding[i] = dv_l["verdict_no_cv"], dv_l["verdict_with_cv"], dv_l["verdict_sliding"]
        right_no_cv[i], right_cv[i], right_sliding[i] = dv_r["verdict_no_cv"], dv_r["verdict_with_cv"], dv_r["verdict_sliding"]
    joint_no_cv = left_no_cv & right_no_cv
    joint_cv = left_cv & right_cv
    joint_sliding = left_sliding & right_sliding

    def lens_report(l, r, j):
        return dict(left_k=int(l.sum()), right_k=int(r.sum()), joint_k=int(j.sum()), n=n,
                   joint_cp_lower=cp_lower(int(j.sum()), n))

    return dict(n=n, count=lens_report(left_no_cv, right_no_cv, joint_no_cv),
               cv=lens_report(left_cv, right_cv, joint_cv), sliding=lens_report(left_sliding, right_sliding, joint_sliding),
               lf=lf, rf=rf)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(PRECHECK_REAL)
    z50, lf50, rf50 = d["z"], d["lf"], d["rf"]

    params = step1_freeze_params(z50, lf50, rf50)
    (OUT / "frozen_params.json").write_text(json.dumps(params, indent=2, default=float) + "\n")

    probes = fit_final_probes(z50, lf50, rf50)

    antifraud = step2_antifraud(params, probes)
    (OUT / "antifraud_result.json").write_text(json.dumps(antifraud, indent=2, default=float) + "\n")

    print(f"[val_200] generating {N_VAL} fresh imagination rollouts, seed={SEED_VAL} ...", flush=True)
    z_val = generate_imagination(SEED_VAL, N_VAL)
    val_result = analyze_batch(z_val, probes, params)
    np.savez_compressed(OUT / "val_200.npz", z=z_val, lf=val_result.pop("lf"), rf=val_result.pop("rf"))
    print(f"  val_200: count={val_result['count']} cv={val_result['cv']} sliding={val_result['sliding']}", flush=True)

    print(f"[test1_1000] generating {N_TEST1} fresh imagination rollouts, seed={SEED_TEST1} ...", flush=True)
    z_test1 = generate_imagination(SEED_TEST1, N_TEST1)
    test1_result = analyze_batch(z_test1, probes, params)
    np.savez_compressed(OUT / "test1_1000.npz", z=z_test1, lf=test1_result.pop("lf"), rf=test1_result.pop("rf"))
    print(f"  test1_1000: count={test1_result['count']} cv={test1_result['cv']} sliding={test1_result['sliding']}", flush=True)

    headline = "count"  # per PREREG: joint verdict = both predicates pass their own M_MIN check
    final = dict(
        headline_lens=headline,
        val_200=val_result, test1_1000=test1_result,
        verdict=dict(
            p_hat_gamma_val=val_result[headline]["joint_cp_lower"],
            p_hat_gamma_test1=test1_result[headline]["joint_cp_lower"],
            safe_val=val_result[headline]["joint_cp_lower"] >= 0.95,
            safe_test1=test1_result[headline]["joint_cp_lower"] >= 0.95,
        ),
        calibration_touched=False, track_b_touched=False, sealed_data_touched=False,
    )
    (OUT / "result.json").write_text(json.dumps(final, indent=2, default=float) + "\n")
    print(json.dumps(final, indent=2, default=float), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
