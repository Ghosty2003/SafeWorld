"""Track 1 tolerance-band control experiment.

Diagnosis (l2_efficacy_diagnose_zero_phat.py) found that under strict
p1_tol=0.0, ALL 300 inspected held-out pathwise failures across 5 systems
were either (a) P1 "violations" of 1e-4 to 1e-3 magnitude between
structurally-equivalent safe states (free-MLP training noise), or (b) P2
"violations" where V descended in the correct direction but fell short of
the exact eta=0.01 margin by a similar order of magnitude -- never a
genuine large jump near a bad state. This script asks: does adding a small
numeric tolerance band, matched to that diagnosed noise scale, restore
discriminative power to the calibrated route?

Tolerance bands: 5e-4, 1e-3 (primary, matches the diagnosed noise
ceiling), 5e-3 -- chosen from the diagnosis BEFORE running this sweep, not
tuned afterward.

Reuses the EXACT SAME 215 systems and retrains via the EXACT SAME
(seed, torch_seed, rng-call-order) sequence as the original pipeline run,
so every trained V is bit-identical to the one behind
track1_full_result.json -- this is a re-evaluation at a different
tolerance, not a retrain. The witnessed-violation route is untouched by
tolerance (it never used any numeric threshold).

Does not touch calibration data or Track B. Does not modify the baseline
(p1_tol=0.0) result file.
"""
from __future__ import annotations

import json
import pathlib
import random

import numpy as np
import torch

from core.lppm.calibrator import _clopper_pearson_lower
from tdmpc2.l2_efficacy_run_pipeline import (
    sample_rollouts, witnessed_violation, train_v,
    N_TRAIN_ROLLOUTS, N_HELD_OUT_ROLLOUTS, ROLLOUT_LEN, ETA, GAMMA, SAFE_THRESHOLD,
)

SYSTEMS_DIR = pathlib.Path("artifacts/tdmpc2_l2_efficacy_study/systems")
OUT = pathlib.Path("artifacts/tdmpc2_l2_efficacy_study")
TOL_BANDS = [5e-4, 1e-3, 5e-3]


def pathwise_pass_tol(model, seq, state_to_idx, bad_set, tol, device="cpu"):
    idx = torch.tensor([state_to_idx[s] for s in seq], device=device)
    with torch.no_grad():
        v = model(idx).cpu().numpy()
    for i in range(len(seq) - 1):
        delta = v[i + 1] - v[i]
        if delta > tol:
            return False
        if seq[i] in bad_set and delta > -(ETA - tol):
            return False
    return True


def run():
    files = sorted(SYSTEMS_DIR.glob("*.json"))
    rng = random.Random(42)

    per_band_records = {tol: [] for tol in TOL_BANDS}

    for i, path in enumerate(files):
        system = json.loads(path.read_text())
        transitions = {k: tuple(v) for k, v in system["transitions"].items()}
        initial_states = tuple(system["initial_states"])
        bad_set = set(system["bad_states"])
        all_states = sorted(transitions.keys())
        state_to_idx = {s: idx for idx, s in enumerate(all_states)}

        torch.manual_seed(42 * 100000 + i)
        train_rollouts = sample_rollouts(rng, transitions, initial_states, N_TRAIN_ROLLOUTS, ROLLOUT_LEN)
        witness = witnessed_violation(train_rollouts, bad_set)
        if witness is not None:
            for tol in TOL_BANDS:
                per_band_records[tol].append(dict(
                    system_id=system["system_id"], kind=system["kind"],
                    ground_truth=system["ground_truth"], l2_verdict="UNSAFE",
                    route="witnessed_violation", p_hat_gamma=None, k=None, n=None,
                ))
            continue

        model = train_v(train_rollouts, state_to_idx, bad_set)
        held_out_rollouts = sample_rollouts(rng, transitions, initial_states, N_HELD_OUT_ROLLOUTS, ROLLOUT_LEN)

        for tol in TOL_BANDS:
            passes = [pathwise_pass_tol(model, seq, state_to_idx, bad_set, tol) for seq in held_out_rollouts]
            k = sum(passes); n = len(passes)
            p_hat_gamma = _clopper_pearson_lower(k, n, GAMMA)
            verdict = "SAFE" if p_hat_gamma >= SAFE_THRESHOLD else "ABSTAIN"
            per_band_records[tol].append(dict(
                system_id=system["system_id"], kind=system["kind"],
                ground_truth=system["ground_truth"], l2_verdict=verdict,
                route="calibrated", p_hat_gamma=p_hat_gamma, k=k, n=n,
            ))
        if (i + 1) % 40 == 0:
            print(f"processed {i+1}/{len(files)}", flush=True)

    return per_band_records


def cp_upper(k, n, conf=0.95):
    from scipy.stats import beta
    if n == 0:
        return None
    if k == n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def summarize_band(records):
    unsafe = [r for r in records if r["ground_truth"] == "VIOLATION"]
    safe = [r for r in records if r["ground_truth"] == "SAFE"]
    calib_unsafe = [r for r in unsafe if r["route"] == "calibrated"]
    calib_safe = [r for r in safe if r["route"] == "calibrated"]

    n_false_safe = sum(1 for r in unsafe if r["l2_verdict"] == "SAFE")
    n_true_safe = sum(1 for r in safe if r["l2_verdict"] == "SAFE")
    false_safe_examples = [r for r in unsafe if r["l2_verdict"] == "SAFE"]

    def phat_stats(rs):
        vals = [r["p_hat_gamma"] for r in rs if r["p_hat_gamma"] is not None]
        if not vals:
            return dict(n=0)
        vals_sorted = sorted(vals)
        return dict(
            n=len(vals), min=vals_sorted[0], median=vals_sorted[len(vals_sorted) // 2],
            max=vals_sorted[-1], mean=sum(vals) / len(vals),
            n_ge_0_95=sum(1 for v in vals if v >= 0.95),
            n_zero=sum(1 for v in vals if v == 0.0),
        )

    return dict(
        n_unsafe=len(unsafe), n_safe=len(safe),
        n_calib_unsafe=len(calib_unsafe), n_calib_safe=len(calib_safe),
        false_safe_rate=n_false_safe / len(unsafe) if unsafe else None,
        false_safe_rate_cp_upper_all=cp_upper(n_false_safe, len(unsafe)),
        false_safe_rate_cp_upper_calibrated_only=cp_upper(
            sum(1 for r in calib_unsafe if r["l2_verdict"] == "SAFE"), len(calib_unsafe)
        ),
        true_safe_rate=n_true_safe / len(safe) if safe else None,
        n_false_safe=n_false_safe,
        false_safe_examples=[r["system_id"] for r in false_safe_examples],
        p_hat_gamma_on_safe_systems=phat_stats(calib_safe),
        p_hat_gamma_on_calibrated_unsafe_systems=phat_stats(calib_unsafe),
    )


if __name__ == "__main__":
    per_band = run()
    report = {}
    for tol, records in per_band.items():
        summary = summarize_band(records)
        report[str(tol)] = dict(tolerance=tol, summary=summary, records=records)
        print(f"\n=== tolerance={tol} ===")
        print(json.dumps(summary, indent=2))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tolerance_sweep_result.json").write_text(json.dumps(report, indent=2) + "\n")
