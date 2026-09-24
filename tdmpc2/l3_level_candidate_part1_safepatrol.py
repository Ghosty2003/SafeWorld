"""L3 Level-mapped candidate screen, Part 1: L6 SafePatrol instance
GF(gait_cycle) & G(height>h_min), read-only precheck.

Reuses ALREADY-SAVED decoded height arrays from Phase B / Step2 (200-roll
val) and Test1 (1000-roll) for all three carriers -- zero new rollouts,
zero training. Applies the SAME frozen judgment-layer parameters already
established (prominence/M_MIN/CV_p95/false_rate_cp_upper, cited verbatim)
to recompute per-trajectory gait-cycle pass/fail, and separately checks
G(height>h_min) per trajectory from the identical decoded height array.

h_min: 0.27 (WALKER_FALL_HEIGHT_M) for walker-walk/walker-run (same body);
0.3 for cheetah-run, substituting its own already-established fall_thresh
from tdmpc2/l3_carrier_step1.py (cheetah has no dm_control-standard 0.27
convention -- different body/XML).

All combined k/N numbers here are explicitly NEGATIVE-PRECHECK / informal
-- no certificate, no calibration launched, no frozen artifact modified.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/bot/SafeWorld")
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("l3_generic_lbsm_lib", "/home/bot/SafeWorld/tdmpc2/l3_generic_lbsm_lib.py")
_lib = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_lib)
deduction_verdict = _lib.deduction_verdict
cp_lower = _lib.cp_lower

OUT = pathlib.Path("artifacts/l3_level_mapped_candidates/part1_safepatrol")
HORIZON = 300

CARRIERS = {
    "walker-walk": dict(
        val="artifacts/tdmpc2_walker_l3_statistical_lbsm/phase_b/val_data.npz",
        test1="artifacts/tdmpc2_walker_l3_statistical_lbsm/phase_b/test1/fresh_1000_data.npz",
        h_key="h",
        h_min=0.27, h_min_source="WALKER_FALL_HEIGHT_M (specs/walker_constants.py), same body as walker-run",
        probe_p95=0.0132, probe_p95_source="published C0 (wrappers/tdmpc2_probes.py); sanity-refit on this pool gave 0.0032",
        prominence=0.03, false_rate_cp_upper=0.07807546497166352, m_min=15, cv_p95=0.4460545042500288,
    ),
    "walker-run": dict(
        val="artifacts/tdmpc2_walkerrun_l3_lbsm/step2/val_data.npz",
        test1="artifacts/tdmpc2_walkerrun_l3_lbsm/step3/fresh_1000_data.npz",
        h_key="q",
        h_min=0.27, h_min_source="WALKER_FALL_HEIGHT_M, same body as walker-walk",
        probe_p95=0.01733200426772711, probe_p95_source="l3_multimodel_screen Group1 C0 (real held-out)",
        prominence=0.03, false_rate_cp_upper=0.010703829113449055, m_min=46, cv_p95=0.21606759056802555,
    ),
    "cheetah-run": dict(
        val="artifacts/tdmpc2_cheetah_l3_lbsm/step2/val_data.npz",
        test1="artifacts/tdmpc2_cheetah_l3_lbsm/step3/fresh_1000_data.npz",
        h_key="q",
        h_min=0.3, h_min_source="SUBSTITUTED: cheetah's own established fall_thresh (tdmpc2/l3_carrier_step1.py CARRIERS['cheetah']) -- no dm_control-standard 0.27 convention exists for this body",
        probe_p95=0.009170136116078316, probe_p95_source="l3_multimodel_screen Group1 C0 (real held-out)",
        prominence=0.03, false_rate_cp_upper=0.0366593606474291, m_min=17, cv_p95=0.38260226202130304,
    ),
}


def analyze_batch(h, cfg):
    n = h.shape[0]
    # G(height>h_min): whole-trajectory pass
    g_height_pass = (h > cfg["h_min"]).all(axis=1)
    min_height_per_traj = h.min(axis=1)

    gait_pass = np.zeros(n, dtype=bool)
    for i in range(n):
        dv = deduction_verdict(h[i], cfg["prominence"], cfg["false_rate_cp_upper"], cfg["m_min"], cfg["cv_p95"], HORIZON)
        gait_pass[i] = dv["verdict_with_cv"]

    joint_pass = g_height_pass & gait_pass
    k_joint = int(joint_pass.sum())
    k_gait = int(gait_pass.sum())
    k_height = int(g_height_pass.sum())

    # bottleneck: which single conjunct, if any, most limits the joint result
    only_gait_fails = int((~gait_pass & g_height_pass).sum())
    only_height_fails = int((gait_pass & ~g_height_pass).sum())
    both_fail = int((~gait_pass & ~g_height_pass).sum())

    return dict(
        n=n,
        height_conjunct=dict(k=k_height, rate=k_height / n, min_height_observed=float(min_height_per_traj.min()),
                             margin_to_h_min=float(min_height_per_traj.min() - cfg["h_min"])),
        gait_conjunct=dict(k=k_gait, rate=k_gait / n),
        joint=dict(k=k_joint, n=n, rate=k_joint / n, cp_lower_NEGATIVE_PRECHECK=cp_lower(k_joint, n)),
        failure_breakdown=dict(only_gait_fails=only_gait_fails, only_height_fails=only_height_fails,
                               both_fail=both_fail, both_pass=k_joint),
        bottleneck="height" if only_height_fails > only_gait_fails else ("gait" if only_gait_fails > only_height_fails else "tied/neither"),
    )


def run_carrier(name, cfg):
    print(f"=== {name} ===", flush=True)
    ap_margin = cfg["h_min"] - cfg["probe_p95"]
    ap_flag = "AP_INSUFFICIENT" if ap_margin < 3 * cfg["probe_p95"] else "AP_sufficient"
    print(f"  probe p95 error on height dim: {cfg['probe_p95']} ({cfg['probe_p95_source']})", flush=True)
    print(f"  h_min={cfg['h_min']} ({cfg['h_min_source']}), margin-to-p95 ratio={cfg['h_min']/cfg['probe_p95']:.2f} -> {ap_flag}", flush=True)

    results = {}
    for batch_name, path_key in (("val_200", "val"), ("test1_1000", "test1")):
        d = np.load(cfg[path_key])
        h = d[cfg["h_key"]]
        r = analyze_batch(h, cfg)
        results[batch_name] = r
        print(f"  [{batch_name}] n={r['n']} height_k={r['height_conjunct']['k']} gait_k={r['gait_conjunct']['k']} "
              f"joint_k={r['joint']['k']} cp_lower={r['joint']['cp_lower_NEGATIVE_PRECHECK']:.4f} "
              f"bottleneck={r['bottleneck']} (only_gait_fails={r['failure_breakdown']['only_gait_fails']}, "
              f"only_height_fails={r['failure_breakdown']['only_height_fails']})", flush=True)

    report = dict(
        carrier=name,
        probe_p95_height=cfg["probe_p95"], probe_p95_source=cfg["probe_p95_source"],
        h_min=cfg["h_min"], h_min_source=cfg["h_min_source"],
        ap_ratio=cfg["h_min"] / cfg["probe_p95"], ap_flag=ap_flag,
        batches=results,
        note="joint k/N and CP-lower are NEGATIVE-PRECHECK / informal -- no certificate, no calibration launched",
    )
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for name, cfg in CARRIERS.items():
        all_results[name] = run_carrier(name, cfg)
    (OUT / "result.json").write_text(json.dumps(all_results, indent=2, default=float) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
