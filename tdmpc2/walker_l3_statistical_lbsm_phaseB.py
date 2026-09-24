"""Phase B, Correction 1 hardening gate (+ Correction 2 boundary-ratio
pipeline-behavior measurement) for the candidate GF(height-band) event
identified in Phase A ([1.20,1.27] m, imagination-side ratio 1.10).

Correction 1 sequence, applied in the pre-registered order, stopping at the
first failing step per the task's own explicit rule:
  1. conformal-tighten the band by a P95 probe-error margin; if the
     resulting effective width is <=0, Phase B dies at the AP layer here.
  2. (only if step 1 passes) K-consecutive-step confirmation, K chosen from
     the real band-dwell-time distribution.
  3. (only if step 1 passes) FP/TP on real held-out data.
  4. (only if step 1 passes) re-test the 20 Phase-A imagination rollouts'
     recurrence statistics with the hardened detector.

Step 1's margin: no clean (non-catastrophic-divergence) depth~300
model-vs-real height-residual dataset exists in this project's saved
artifacts to compute an empirical P95 directly -- the only paired dataset
found (`tdmpc2_walker_imagination_real_divergence/paired_traces.npz`, 15
anchors) is dominated by real trajectories that have already fallen by
depth ~50-300 (0-1 "clean, still-walking" anchors remain at those depths),
making its raw percentiles reflect catastrophic real-model divergence, not
probe/decode noise -- an entirely different, already-separately-documented
failure mode. This is reported explicitly rather than silently used.
Instead, two independent extrapolations from the ALREADY-VALIDATED clean
depth-100 numbers (p50=0.0218m, p90=0.0637m, max=0.1123m, 8 contributing
episodes, `wrappers/tdmpc2_probes.py`) are used and cross-checked.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

LATENTS_DIR = pathlib.Path("artifacts/tdmpc2_walker_lppm_latents")
PHASE_A_DIR = pathlib.Path("artifacts/tdmpc2_walker_l3_statistical_lbsm")
OUT = PHASE_A_DIR
HEIGHT_LO, HEIGHT_HI = 1.20, 1.27
NOMINAL_WIDTH = HEIGHT_HI - HEIGHT_LO
WINDOW = 300

# published clean C1 depth-100 numbers (cited, not re-derived)
P50, P90, MAXV = 0.0218, 0.0637, 0.1123


def band_entries(x, lo, hi):
    inside = (x >= lo) & (x <= hi)
    prev = np.concatenate([[False], inside[:-1]])
    return np.where(inside & ~prev)[0]


def band_entries_with_K(x, lo, hi, K):
    """Entrance confirmed only after K consecutive in-band steps."""
    inside = (x >= lo) & (x <= hi)
    events = []
    run = 0
    armed = True  # allow next confirmation once we've left the band (or at start)
    for t, v in enumerate(inside):
        if v:
            run += 1
            if run == K and armed:
                events.append(t - K + 1)
                armed = False
        else:
            run = 0
            armed = True
    return np.array(events, dtype=np.int64)


def dwell_lengths(x, lo, hi):
    inside = (x >= lo) & (x <= hi)
    lengths = []
    run = 0
    for v in inside:
        if v:
            run += 1
        else:
            if run > 0:
                lengths.append(run)
            run = 0
    if run > 0:
        lengths.append(run)
    return lengths


def step1_margin():
    print("[step 1] conformal band tightening ...", flush=True)
    lam = np.log(10) / P90
    p95_exp = np.log(20) / lam
    margin_exp = p95_exp
    margin_max = MAXV
    eff_exp = NOMINAL_WIDTH - margin_exp
    eff_max = NOMINAL_WIDTH - margin_max
    result = dict(
        nominal_width_m=NOMINAL_WIDTH,
        clean_depth100_reference=dict(p50=P50, p90=P90, max=MAXV, source="wrappers/tdmpc2_probes.py, 8 contributing episodes"),
        contamination_note=(
            "The only saved paired real/imagined height dataset at depth ~300 "
            "(tdmpc2_walker_imagination_real_divergence/paired_traces.npz, 15 "
            "anchors) has 0-1 anchors where the REAL trajectory is still "
            "genuinely walking (real height>0.8m) by depth>=100 -- the rest "
            "have already fallen in reality by then, so raw percentiles from "
            "it reflect catastrophic model/real divergence (a different, "
            "already-documented failure mode), not probe/decode noise. Not "
            "used for this margin; the published clean depth-100 numbers are "
            "used instead via two independent extrapolations."
        ),
        margin_estimate_exponential_tail=dict(
            method="fit lambda from p90 (exponential tail assumption), extrapolate to p95",
            p95_estimate_m=margin_exp,
            effective_width_m=eff_exp,
            positive=bool(eff_exp > 0),
        ),
        margin_estimate_max_as_conservative_p95_proxy=dict(
            method="use published max (n=8) as a conservative upper-bound substitute for an empirically-unobtainable p95 from only 8 samples",
            margin_m=margin_max,
            effective_width_m=eff_max,
            positive=bool(eff_max > 0),
        ),
        both_methods_agree_negative=bool(eff_exp <= 0 and eff_max <= 0),
        verdict=(
            "PHASE B DIES AT THE AP LAYER for the height-band GF(event) construction: "
            "both independent P95 margin estimates (0.0829m exponential-tail, 0.1123m "
            "max-as-proxy) exceed the nominal band width (0.07m), giving a negative "
            "effective width under either method. Per the pre-registered stopping "
            "rule, Phase B does not proceed to the full 300 fit/cal + 200 val "
            "imagination generation, drift-V fit, gate, multi-seed, or real-rollout "
            "control for this event -- there is no sound band left to build a "
            "witness on."
        ),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phaseB_step1_margin.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def supplementary_diagnostics():
    """INFORMATIONAL ONLY, not part of a passing pipeline (step 1 already
    failed): K-selection from dwell times, FP/TP on the NOMINAL (un-shrunk)
    band, and imagination re-test with K-persistence on the nominal band.
    Run anyway to give Correction 2's pipeline-behavior section concrete
    numbers, and to fully document what the rest of the hardening procedure
    would have measured."""
    print("[supplementary] K-selection from real dwell-time distribution ...", flush=True)
    seed_files = sorted(LATENTS_DIR.glob("seed_*_latents.npz"))
    all_height, all_z = [], []
    for f in seed_files:
        d = np.load(f)
        all_height.append(d["height_m"])
        all_z.append(d["posterior_z"])
    height = np.concatenate(all_height, axis=0)  # (1500,501)
    z = np.concatenate(all_z, axis=0)
    n = height.shape[0]

    all_dwells = []
    for i in range(n):
        all_dwells.extend(dwell_lengths(height[i], HEIGHT_LO, HEIGHT_HI))
    all_dwells = np.array(all_dwells)
    dwell_pcts = {p: float(np.percentile(all_dwells, p)) for p in (10, 20, 25, 50, 75, 90)}
    K_CHOSEN = max(1, int(np.percentile(all_dwells, 20)))  # 20th pct: most real entrances survive
    print(f"    dwell-length percentiles: {dwell_pcts}, K_chosen={K_CHOSEN}", flush=True)

    print("[supplementary] FP/TP on real held-out (nominal band, K-persistence) ...", flush=True)
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    n_holdout = int(0.2 * n)
    fit_idx = np.arange(n - n_holdout)
    hold_idx = np.arange(n - n_holdout, n)

    Zfit = z[fit_idx].reshape(-1, 512)
    Hfit = height[fit_idx].reshape(-1)
    hprobe = Ridge(alpha=10.0).fit(Zfit, Hfit)

    TP, FP, FN = 0, 0, 0
    TOL = 3  # steps, matched-event tolerance window
    for i in hold_idx:
        true_events = band_entries_with_K(height[i], HEIGHT_LO, HEIGHT_HI, K_CHOSEN)
        h_probe = hprobe.predict(z[i])
        det_events = band_entries_with_K(h_probe, HEIGHT_LO, HEIGHT_HI, K_CHOSEN)
        matched_true = np.zeros(len(true_events), dtype=bool)
        for de in det_events:
            hit = np.abs(true_events - de) <= TOL if len(true_events) else np.array([])
            if hit.any() and not matched_true[np.argmax(hit)]:
                matched_true[np.argmax(hit)] = True
                TP += 1
            else:
                FP += 1
        FN += int((~matched_true).sum())

    fp_rate = FP / (TP + FP) if (TP + FP) > 0 else None

    print("[supplementary] re-test Phase-A imagination rollouts with K-persistence (nominal band) ...", flush=True)
    imag = np.load(PHASE_A_DIR / "item3_imagination_latents.npz")
    height_imag = imag["height_imag"]
    n_imag = height_imag.shape[0]

    raw_events_imag = [band_entries(height_imag[i], HEIGHT_LO, HEIGHT_HI) for i in range(n_imag)]
    k1_events_imag = [band_entries_with_K(height_imag[i], HEIGHT_LO, HEIGHT_HI, K_CHOSEN) for i in range(n_imag)]

    mean_raw = float(np.mean([len(e) for e in raw_events_imag]))
    mean_k = float(np.mean([len(e) for e in k1_events_imag]))
    collapse_fraction = 1.0 - (mean_k / mean_raw if mean_raw > 0 else 0.0)

    result = dict(
        K_selection=dict(dwell_length_percentiles_steps=dwell_pcts, K_chosen=K_CHOSEN,
                         rationale="20th percentile of real band-dwell-run-lengths: most genuine real entrances (80%) survive this persistence filter, while single-frame noise blips (typically dwell=1-2 steps) are rejected"),
        real_holdout_FP_TP=dict(
            n_holdout_rollouts=int(n_holdout), tolerance_steps=TOL,
            TP=TP, FP=FP, FN=FN, fp_rate=fp_rate,
            fp_near_zero=bool(fp_rate is not None and fp_rate < 0.05),
            note="INFORMATIONAL: computed on the NOMINAL (un-shrunk) band since step 1 already found no positive-width conformal-tightened band exists; not part of a passing pipeline.",
        ),
        imagination_recollapse_check=dict(
            n_imagination_rollouts=n_imag,
            mean_events_per_rollout_raw_K1=mean_raw,
            mean_events_per_rollout_hardened_K=mean_k,
            detection_rate_collapse_fraction=collapse_fraction,
            structure_survives=bool(mean_k >= 10),
            note="INFORMATIONAL, nominal band + K-persistence only (no conformal shrink, since shrink already kills the band).",
        ),
    )
    (OUT / "phaseB_supplementary_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def correction2_boundary_ratio_table(margin_result, supp_result):
    print("[Correction 2] boundary-ratio pipeline-behavior comparison ...", flush=True)
    rows = dict(
        door_close=dict(
            tolerance_or_width_m=0.08, probe_p95_error_m=0.0107, ratio=7.48,
            margin_consumption_fraction=0.0107 / 0.08,
        ),
        cardreamer_roundabout=dict(
            tolerance_or_width_m=8.0, probe_p90_error_m=2.62, ratio=3.05,
            margin_consumption_fraction=2.62 / 8.0,
        ),
        walker_height_band_real_latents=dict(
            tolerance_or_width_m=NOMINAL_WIDTH, probe_C0_error_m=0.0032, ratio=NOMINAL_WIDTH / 0.0032,
            margin_consumption_fraction=0.0032 / NOMINAL_WIDTH,
            note="AP-judgeable comfortably (real/replayed latents only, not the imagination-transfer regime Phase B actually needs)",
        ),
        walker_height_band_imagination_P90=dict(
            tolerance_or_width_m=NOMINAL_WIDTH, probe_p90_error_m=P90, ratio=NOMINAL_WIDTH / P90,
            margin_consumption_fraction=P90 / NOMINAL_WIDTH,
            note="this is Phase A's reported 'marginal' ratio (1.10)",
        ),
        walker_height_band_imagination_P95_hardened=dict(
            tolerance_or_width_m=NOMINAL_WIDTH,
            probe_p95_error_m_exp_tail=margin_result["margin_estimate_exponential_tail"]["p95_estimate_m"],
            probe_p95_error_m_max_proxy=margin_result["margin_estimate_max_as_conservative_p95_proxy"]["margin_m"],
            margin_consumption_fraction_exp_tail=margin_result["margin_estimate_exponential_tail"]["p95_estimate_m"] / NOMINAL_WIDTH,
            margin_consumption_fraction_max_proxy=margin_result["margin_estimate_max_as_conservative_p95_proxy"]["margin_m"] / NOMINAL_WIDTH,
            note="exceeds 100% under both estimates -- this is the boundary-ratio case's operative, hardened number",
        ),
    )
    gate_interception = dict(
        ap_width_gate="INTERCEPTED (only gate that fired): effective width negative under both margin estimates, stopped before any rollout/drift-fit gate could run",
        supplementary_real_holdout_fp_rate=supp_result["real_holdout_FP_TP"]["fp_rate"],
        supplementary_imagination_detection_collapse_fraction=supp_result["imagination_recollapse_check"]["detection_rate_collapse_fraction"],
    )
    report = dict(
        title="Boundary-ratio pipeline behavior (independent of the Recurrence verdict)",
        margin_consumption_fraction_table=rows,
        one_sentence_pattern=(
            "Margin consumption (probe-error / tolerance-or-band-width) rises steeply as "
            "ratio falls: ~13% at door-close's ratio 7.48, ~33% at CarDreamer's ratio "
            "3.05, ~91% at the walker height-band's own P90 ratio 1.10, and >100% "
            "(both P95 estimates) once the same conformal-tightening discipline used "
            "throughout this project is applied at this boundary ratio -- this is the "
            "first internal measurement of what happens when the ratio approaches 1 "
            "rather than sitting comfortably above 3 or collapsing below 0.7."
        ),
        gate_interception_statistics=gate_interception,
    )
    (OUT / "correction2_boundary_ratio_pipeline_behavior.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def run():
    margin_result = step1_margin()
    supp_result = supplementary_diagnostics()
    c2 = correction2_boundary_ratio_table(margin_result, supp_result)
    return dict(margin=margin_result, supplementary=supp_result, correction2=c2)


if __name__ == "__main__":
    run()
