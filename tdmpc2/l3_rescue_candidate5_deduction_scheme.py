"""L3 rescue, Candidate 5: deduction-scheme trial for the waveform event.

Pure recount on already-existing data/probe (Candidate 2's), no new
training, no new real rollouts. One new imagination batch is generated
(200 rollouts, minute-scale, pure model.next() loop, no real physics) only
because none already exist at horizon 300 in this project's saved
artifacts (tdmpc2_walker_lppm_latents holds REAL rollouts only). No frozen
artifact modified.

PRE-REGISTRATION (written before any number below is computed):

1. Upper bounds: standard one-sided 95% Clopper-Pearson upper bound,
   beta.ppf(0.95, k+1, n-k), applied to false-detections/total-detections
   (k=n_false=817, n=n_matched+n_false=11062) and to
   misses/true-peaks (k=n_missed=560, n=n_true_peaks_total=10805) --
   the exact convention already used throughout this project's cp_upper().

2. Per-trajectory deduction rule, on the first 300 steps of each rollout
   (Candidate 2 itself measured on the full ~501-step real trajectories
   mislabeled "per 300-step window" in its own report -- corrected here to
   an actual 300-step slice, noted explicitly as a provenance fix, not a
   new finding):
     m         = number of probe-decoded peaks in steps [0,300)
     m_adj     = m * (1 - false_rate_cp_upper)          [conservative down-adjustment]
     (miss rate is NOT compensated for -- under-counting true peaks is the
      CONSERVATIVE direction for a "recurrence still holds" claim: it can
      only make the verdict harder to pass, never easier. Compensating it
      would inflate m_adj in the unsafe direction. This asymmetry is a
      deliberate design choice, not an oversight.)
     verdict   = "recurrence maintained" iff m_adj >= M_MIN

   M_MIN is fixed from the REAL interval distribution's own P90 (slow-gait
   tail), BEFORE any deduction-scheme number is computed: M_MIN =
   floor(300 / interval_p90_real). This is computed in step 0 below and
   printed before proceeding, exactly as pre-registered.

3. Additional structure check (guards against "uniform false peaks posing
   as gait"): per-trajectory CV = std(intervals)/mean(intervals) of the
   DECODED peak sequence; require CV <= P95 of the REAL per-trajectory CV
   distribution (genuine gait intervals are regular; pure decode-noise
   peaks are not). Reported both WITH and WITHOUT this extra gate, as
   required.

4. Two recounts:
   4a. Real held-out 300 rollouts (Candidate 2's own held-out split,
       ground truth known): deduction-verdict pass rate + confusion matrix
       against a ground-truth "still normal gait" label, itself defined
       BEFORE computing it: a real trajectory's ground truth label is PASS
       iff its OWN ground-truth peak count in [0,300) is >= M_MIN (i.e.,
       real gait genuinely completes enough real cycles), else FAIL. The
       critical number is FALSE POSITIVES: true-FAIL trajectories the
       deduction verdict lets through -- must be ~=0 for soundness.
   4b. 200 fresh imagination rollouts (horizon 300, generated fresh since
       none exist at this horizon in saved artifacts): deduction-verdict
       pass-rate distribution only (no ground truth exists for a purely
       imagined counterfactual trajectory).

No calibration/sealed data touched.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.stats import beta
from sklearn.linear_model import Ridge

from wrappers.tdmpc2_wrapper import TDMPC2Wrapper

LATENTS_DIR = pathlib.Path("artifacts/tdmpc2_walker_lppm_latents")
CAND2_DIR = pathlib.Path("artifacts/l3_rescue_screen/candidate2_waveform_event")
OUT = pathlib.Path("artifacts/l3_rescue_screen/candidate5_deduction_scheme")
PROMINENCE = 0.03
TOL_STEPS = 2
WINDOW = 300
ANCHORS = pathlib.Path("artifacts/tdmpc2_forward_speed/c1_exact_mpc_anchors.npz")
N_IMAG = 200


def cp_upper(k, n, conf=0.95):
    if n == 0:
        return None
    if k == n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def load_real():
    seed_files = sorted(LATENTS_DIR.glob("seed_*_latents.npz"))
    all_h, all_z = [], []
    for f in seed_files:
        d = np.load(f)
        all_h.append(d["height_m"])
        all_z.append(d["posterior_z"])
    return np.concatenate(all_h, axis=0), np.concatenate(all_z, axis=0)


def fit_probe(z, h, n_holdout):
    n = h.shape[0]
    fit_idx = np.arange(n - n_holdout)
    Zfit = z[fit_idx].reshape(-1, 512)
    Hfit = h[fit_idx].reshape(-1)
    return Ridge(alpha=10.0).fit(Zfit, Hfit)


def per_traj_intervals(seq_1d, prominence):
    pk, _ = find_peaks(seq_1d, prominence=prominence)
    if len(pk) > 1:
        return pk, np.diff(pk).astype(np.float64)
    return pk, np.array([])


def step0_upper_bounds():
    print("[step 0] Clopper-Pearson upper bounds from Candidate 2's numbers ...", flush=True)
    cand2 = json.loads((CAND2_DIR / "result.json").read_text())
    n_matched, n_false, n_missed, n_true = (
        cand2["n_matched"], cand2["n_false"], cand2["n_missed"], cand2["n_true_peaks_total"])
    false_rate_point = n_false / (n_matched + n_false)
    false_rate_cp_upper = cp_upper(n_false, n_matched + n_false)
    miss_rate_point = n_missed / n_true
    miss_rate_cp_upper = cp_upper(n_missed, n_true)
    result = dict(
        false_rate_point=false_rate_point, false_rate_cp_upper_95=false_rate_cp_upper,
        miss_rate_point=miss_rate_point, miss_rate_cp_upper_95=miss_rate_cp_upper,
        n_false=n_false, n_matched=n_matched, n_missed=n_missed, n_true_peaks_total=n_true,
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def step0b_m_min(h_real, n_holdout):
    print("[step 0b] M_MIN from real interval P90 (pre-registered formula) ...", flush=True)
    n = h_real.shape[0]
    hold_idx = np.arange(n - n_holdout, n)
    all_intervals = []
    for i in hold_idx:
        _, iv = per_traj_intervals(h_real[i, :WINDOW], PROMINENCE)
        all_intervals.extend(iv.tolist())
    all_intervals = np.array(all_intervals)
    p90 = float(np.percentile(all_intervals, 90))
    m_min = int(np.floor(WINDOW / p90))
    result = dict(interval_p90_real=p90, window=WINDOW, M_MIN=m_min,
                 rationale=f"floor({WINDOW}/{p90:.3f}) -- minimum complete cycles a slow (p90-interval) real gait would still complete in a 300-step window")
    print(json.dumps(result, indent=2), flush=True)
    return m_min, all_intervals


def step0c_cv_threshold(h_real, n_holdout):
    print("[step 0c] CV(interval) P95 threshold from real held-out data ...", flush=True)
    n = h_real.shape[0]
    hold_idx = np.arange(n - n_holdout, n)
    cvs = []
    for i in hold_idx:
        _, iv = per_traj_intervals(h_real[i, :WINDOW], PROMINENCE)
        if len(iv) >= 2 and iv.mean() > 0:
            cvs.append(iv.std() / iv.mean())
    cvs = np.array(cvs)
    cv_p95 = float(np.percentile(cvs, 95))
    print(f"    n_trajectories_with_cv={len(cvs)}, CV_p95={cv_p95:.4f}", flush=True)
    return cv_p95


def step4a_real_holdout(h_real, z_real, probe, false_rate_cp_upper, m_min, cv_p95, n_holdout):
    print("[step 4a] real held-out recount ...", flush=True)
    n = h_real.shape[0]
    hold_idx = np.arange(n - n_holdout, n)
    rows = []
    for i in hold_idx:
        true_pk, true_iv = per_traj_intervals(h_real[i, :WINDOW], PROMINENCE)
        h_pred = probe.predict(z_real[i, :WINDOW])
        pred_pk, pred_iv = per_traj_intervals(h_pred, PROMINENCE)
        m = len(pred_pk)
        m_adj = m * (1 - false_rate_cp_upper)
        verdict_no_cv = m_adj >= m_min
        cv = (pred_iv.std() / pred_iv.mean()) if len(pred_iv) >= 2 and pred_iv.mean() > 0 else np.inf
        verdict_with_cv = verdict_no_cv and (cv <= cv_p95)
        gt_label_pass = len(true_pk) >= m_min
        rows.append(dict(traj=int(i), m=m, m_adj=m_adj, cv=float(cv) if np.isfinite(cv) else None,
                         verdict_no_cv=bool(verdict_no_cv), verdict_with_cv=bool(verdict_with_cv),
                         gt_label_pass=bool(gt_label_pass), gt_true_peaks=int(len(true_pk))))

    def confusion(verdict_key):
        TP = sum(1 for r in rows if r[verdict_key] and r["gt_label_pass"])
        FP = sum(1 for r in rows if r[verdict_key] and not r["gt_label_pass"])
        FN = sum(1 for r in rows if not r[verdict_key] and r["gt_label_pass"])
        TN = sum(1 for r in rows if not r[verdict_key] and not r["gt_label_pass"])
        pass_rate = (TP + FP) / len(rows)
        false_positive_cases = [r for r in rows if r[verdict_key] and not r["gt_label_pass"]]
        return dict(TP=TP, FP=FP, FN=FN, TN=TN, pass_rate=pass_rate,
                    false_positive_rate_among_true_fail=(FP / (FP + TN) if (FP + TN) > 0 else None),
                    false_positive_case_details=false_positive_cases[:10])

    result = dict(
        n_holdout=len(rows),
        n_ground_truth_fail=sum(1 for r in rows if not r["gt_label_pass"]),
        n_ground_truth_pass=sum(1 for r in rows if r["gt_label_pass"]),
        without_CV_gate=confusion("verdict_no_cv"),
        with_CV_gate=confusion("verdict_with_cv"),
    )
    print(json.dumps({k: v for k, v in result.items()}, indent=2), flush=True)
    return result, rows


def generate_imagination():
    print(f"[step 4b] generating {N_IMAG} fresh imagination rollouts, horizon={WINDOW} (none exist at this horizon in saved artifacts) ...", flush=True)
    anchors = np.load(ANCHORS)
    physics_states = anchors["physics_state"][:N_IMAG]
    wrapper = TDMPC2Wrapper()
    wrapper.load(checkpoint="models/walker-walk-3.pt", task="walker-walk", seed=1)
    if str(wrapper._agent.device).split(":")[0] != "cuda":
        wrapper.close()
        raise RuntimeError("CUDA required")
    all_z = []
    try:
        for i in range(len(physics_states)):
            obs = wrapper._obs_from_physics_state(physics_states[i])
            z = wrapper.encode(obs)
            z_seq = []
            with torch.no_grad():
                for step in range(WINDOW):
                    a = wrapper.plan_action(z, t0=(step == 0), eval_mode=True)
                    z = wrapper.next(z, a)
                    z_seq.append(z[0].detach().cpu().numpy().astype(np.float64, copy=True))
            all_z.append(np.stack(z_seq))
            if (i + 1) % 50 == 0:
                print(f"    imagined {i + 1}/{len(physics_states)}", flush=True)
    finally:
        wrapper.close()
    z_array = np.stack(all_z)
    np.savez_compressed(OUT / "imagination_200_latents.npz", z=z_array)
    return z_array


def step4b_imagination(z_imag, probe, false_rate_cp_upper, m_min, cv_p95):
    print("[step 4b] imagination recount ...", flush=True)
    n = z_imag.shape[0]
    height_imag = probe.predict(z_imag.reshape(-1, 512)).reshape(n, WINDOW)
    rows = []
    for i in range(n):
        pred_pk, pred_iv = per_traj_intervals(height_imag[i], PROMINENCE)
        m = len(pred_pk)
        m_adj = m * (1 - false_rate_cp_upper)
        verdict_no_cv = m_adj >= m_min
        cv = (pred_iv.std() / pred_iv.mean()) if len(pred_iv) >= 2 and pred_iv.mean() > 0 else np.inf
        verdict_with_cv = verdict_no_cv and (cv <= cv_p95)
        rows.append(dict(traj=int(i), m=m, m_adj=m_adj, verdict_no_cv=bool(verdict_no_cv), verdict_with_cv=bool(verdict_with_cv)))
    pass_rate_no_cv = float(np.mean([r["verdict_no_cv"] for r in rows]))
    pass_rate_with_cv = float(np.mean([r["verdict_with_cv"] for r in rows]))
    result = dict(n_imagination_rollouts=n, pass_rate_without_CV_gate=pass_rate_no_cv,
                 pass_rate_with_CV_gate=pass_rate_with_cv,
                 mean_m=float(np.mean([r["m"] for r in rows])),
                 mean_m_adj=float(np.mean([r["m_adj"] for r in rows])))
    print(json.dumps(result, indent=2), flush=True)
    return result


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    bounds = step0_upper_bounds()
    h_real, z_real = load_real()
    n_holdout = 300
    m_min, real_intervals = step0b_m_min(h_real, n_holdout)
    cv_p95 = step0c_cv_threshold(h_real, n_holdout)
    probe = fit_probe(z_real, h_real, n_holdout)

    real_result, real_rows = step4a_real_holdout(
        h_real, z_real, probe, bounds["false_rate_cp_upper_95"], m_min, cv_p95, n_holdout)

    z_imag = generate_imagination()
    imag_result = step4b_imagination(z_imag, probe, bounds["false_rate_cp_upper_95"], m_min, cv_p95)

    fp_no_cv = real_result["without_CV_gate"]["FP"]
    fp_with_cv = real_result["with_CV_gate"]["FP"]
    pass_rate_imag_no_cv = imag_result["pass_rate_without_CV_gate"]
    pass_rate_imag_with_cv = imag_result["pass_rate_with_CV_gate"]

    if fp_with_cv == 0 and pass_rate_imag_with_cv > 0.70:
        final_verdict = "DEDUCTION_SCHEME_USABLE"
    elif fp_with_cv > 0:
        final_verdict = "NO_SOUND_DEDUCTION_SCHEME_AT_THIS_ERROR_MAGNITUDE"
    else:
        final_verdict = "SOUND_BUT_UNINFORMATIVE"

    report = dict(
        pre_registration=dict(
            false_rate_cp_upper_formula="Clopper-Pearson 95% upper, beta.ppf(0.95,k+1,n-k)",
            M_MIN_formula="floor(300 / P90(real peak intervals))",
            CV_gate_formula="per-traj CV(decoded intervals) <= P95(real per-traj CV)",
            miss_rate_compensation="NONE -- conservative by construction",
        ),
        upper_bounds=bounds,
        M_MIN=m_min,
        CV_p95_threshold=cv_p95,
        real_holdout_4a=real_result,
        imagination_4b=imag_result,
        final_verdict=final_verdict,
        calibration_touched=False, track_b_touched=False, sealed_data_touched=False,
    )
    (OUT / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "real_holdout_rows.json").write_text(json.dumps(real_rows, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "real_holdout_4a"}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run()
