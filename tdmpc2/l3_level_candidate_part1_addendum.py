"""L3 level-mapped candidates, Part 1 ADDENDUM (read-only follow-up query).

Answers three things about the original Part 1 SafePatrol joint k/N, using
the exact same already-saved decoded height arrays -- zero new rollouts,
zero training, zero touching of frozen artifacts or sealed calibration
data:

(1) which lens/tier the original gait_pass actually was (identify by exact
    k-count match against each carrier's own frozen result.json numbers);
(2) recompute the joint using each carrier's own HEADLINE lens instead
    (walker-walk: count/no_cv tier3; walker-run: sliding tier3;
    cheetah-run: count/no_cv tier3) -- same batches, same height conjunct
    as the original;
(3) on the height conjunct, separate out violations whose depth is smaller
    than the probe's own C0 p95 error ("inside probe-noise, not counted")
    and report the noise-adjusted height pass rate + exclusion count.
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
        h_key="h", h_min=0.27, probe_p95=0.0132,
        prominence=0.03, false_rate_cp_upper=0.07807546497166352, m_min=15, cv_p95=0.4460545042500288,
        headline_lens="count",
        headline_source="FIVE_TABLES.md row16: count lens, anti-fraud validated on count lens in Candidate 5/5b",
    ),
    "walker-run": dict(
        val="artifacts/tdmpc2_walkerrun_l3_lbsm/step2/val_data.npz",
        test1="artifacts/tdmpc2_walkerrun_l3_lbsm/step3/fresh_1000_data.npz",
        h_key="q", h_min=0.27, probe_p95=0.01733200426772711,
        prominence=0.03, false_rate_cp_upper=0.010703829113449055, m_min=46, cv_p95=0.21606759056802555,
        headline_lens="sliding",
        headline_source="FIVE_TABLES.md row16 / L3_DUAL_CARRIER_LAUNCH_MASTER_SUMMARY.md addendum: sliding lens, count lens capped by carrier-specific M_MIN-tightness artifact",
    ),
    "cheetah-run": dict(
        val="artifacts/tdmpc2_cheetah_l3_lbsm/step2/val_data.npz",
        test1="artifacts/tdmpc2_cheetah_l3_lbsm/step3/fresh_1000_data.npz",
        h_key="q", h_min=0.3, probe_p95=0.009170136116078316,
        prominence=0.03, false_rate_cp_upper=0.0366593606474291, m_min=17, cv_p95=0.38260226202130304,
        headline_lens="count",
        headline_source="FIVE_TABLES.md row16: count lens, Step 1 anti-fraud validated on count",
    ),
}

# (1) Frozen reference numbers to confirm the ORIGINAL Part 1 gait_pass lens identity.
FROZEN_REFERENCE = {
    "walker-walk": dict(val_with_cv=188, test1_with_cv=920, source_val="phase_b/SUMMARY.md L59 ('With CV gate' 188/200)", source_test1="phase_b/test1/result.json per_seed_tier_results.seed0.*.combined_with_cv.k=920 (identical all 3 tiers)"),
    "walker-run": dict(val_with_cv=181, test1_with_cv=908, source_val="STEP2_SUMMARY.md L11 (CV-lens k/N column, tier3=181/200)", source_test1="step3/result.json per_seed_tier_results.seed0.tier3.combined_with_cv.k=908"),
    "cheetah-run": dict(val_with_cv=200, test1_with_cv=1000, source_val="STEP2_SUMMARY.md L35 ('CV and sliding lenses (tier2/tier3): 200/200')", source_test1="step3/result.json per_seed_tier_results.seed0.tier3.combined_with_cv.k=1000"),
}
ORIGINAL_GAIT_K = {
    "walker-walk": dict(val=188, test1=920),
    "walker-run": dict(val=181, test1=908),
    "cheetah-run": dict(val=200, test1=1000),
}


def per_trajectory_lens_verdicts(h, cfg):
    """Return dict of lens_name -> bool array (n,), all from the SAME
    deduction_verdict call (no re-derivation, no new formula)."""
    n = h.shape[0]
    no_cv = np.zeros(n, dtype=bool)
    with_cv = np.zeros(n, dtype=bool)
    sliding = np.zeros(n, dtype=bool)
    for i in range(n):
        dv = deduction_verdict(h[i], cfg["prominence"], cfg["false_rate_cp_upper"], cfg["m_min"], cfg["cv_p95"], HORIZON)
        no_cv[i] = dv["verdict_no_cv"]
        with_cv[i] = dv["verdict_with_cv"]
        sliding[i] = dv["verdict_sliding"]
    return dict(count=no_cv, cv=with_cv, sliding=sliding)


def height_analysis(h, cfg):
    """Per-trajectory min height, raw pass/fail, and noise-zone exclusion."""
    min_h = h.min(axis=1)
    raw_pass = min_h > cfg["h_min"]
    violation_depth = np.maximum(cfg["h_min"] - min_h, 0.0)  # 0 if passing, >0 if violating
    in_noise_zone = (~raw_pass) & (violation_depth < cfg["probe_p95"])
    adjusted_pass = raw_pass | in_noise_zone  # noise-zone violations reclassified as "not counted"
    return dict(
        min_h=min_h, raw_pass=raw_pass, violation_depth=violation_depth,
        in_noise_zone=in_noise_zone, adjusted_pass=adjusted_pass,
    )


def bottleneck(gait_pass, height_pass):
    only_gait_fails = int((~gait_pass & height_pass).sum())
    only_height_fails = int((gait_pass & ~height_pass).sum())
    both_fail = int((~gait_pass & ~height_pass).sum())
    label = "height" if only_height_fails > only_gait_fails else ("gait" if only_gait_fails > only_height_fails else "tied/neither")
    return dict(only_gait_fails=only_gait_fails, only_height_fails=only_height_fails, both_fail=both_fail, bottleneck=label)


def run_carrier(name, cfg):
    print(f"=== {name} ===", flush=True)
    out = dict(carrier=name, headline_lens=cfg["headline_lens"], headline_source=cfg["headline_source"])
    batches = {}
    for batch_name, path_key in (("val_200", "val"), ("test1_1000", "test1")):
        d = np.load(cfg[path_key])
        h = d[cfg["h_key"]]
        n = h.shape[0]

        # (1) confirm original gait_pass identity
        lenses = per_trajectory_lens_verdicts(h, cfg)
        confirm = dict(
            recomputed_count_k=int(lenses["count"].sum()),
            recomputed_cv_k=int(lenses["cv"].sum()),
            recomputed_sliding_k=int(lenses["sliding"].sum()),
            original_part1_gait_k=ORIGINAL_GAIT_K[name][batch_name.split("_")[0]],
            matches_cv_lens=int(lenses["cv"].sum()) == ORIGINAL_GAIT_K[name][batch_name.split("_")[0]],
        )

        # height, raw + noise-adjusted
        hb = height_analysis(h, cfg)

        # (2) joint with HEADLINE lens, ORIGINAL (raw) height
        headline_pass = lenses[{"count": "count", "sliding": "sliding", "cv": "cv"}[cfg["headline_lens"]]]
        joint_headline_raw_height = headline_pass & hb["raw_pass"]
        bn_headline = bottleneck(headline_pass, hb["raw_pass"])

        # (3) height noise-zone exclusion, standalone
        n_excluded = int(hb["in_noise_zone"].sum())
        height_adjusted_k = int(hb["adjusted_pass"].sum())

        # bonus: joint with BOTH headline lens AND noise-adjusted height (not explicitly requested, labeled as such)
        joint_headline_adjusted_height = headline_pass & hb["adjusted_pass"]
        bn_headline_adjusted = bottleneck(headline_pass, hb["adjusted_pass"])

        # also keep the ORIGINAL (CV-lens) joint for direct side-by-side comparison
        joint_original_cv = lenses["cv"] & hb["raw_pass"]

        batches[batch_name] = dict(
            n=n,
            lens_identity_check=confirm,
            original_joint_CV_lens=dict(k=int(joint_original_cv.sum()), n=n),
            headline_joint_raw_height=dict(
                lens=cfg["headline_lens"], k=int(joint_headline_raw_height.sum()), n=n,
                cp_lower_NEGATIVE_PRECHECK=cp_lower(int(joint_headline_raw_height.sum()), n),
                height_k_raw=int(hb["raw_pass"].sum()), gait_k_headline=int(headline_pass.sum()),
                **bn_headline,
            ),
            height_noise_zone_exclusion=dict(
                probe_p95=cfg["probe_p95"], h_min=cfg["h_min"],
                n_raw_violations=int((~hb["raw_pass"]).sum()),
                n_excluded_in_noise_zone=n_excluded,
                n_genuine_violations_after_exclusion=int((~hb["adjusted_pass"]).sum()),
                height_k_raw=int(hb["raw_pass"].sum()),
                height_k_noise_adjusted=height_adjusted_k,
                height_rate_raw=float(hb["raw_pass"].sum() / n),
                height_rate_noise_adjusted=float(height_adjusted_k / n),
                excluded_trajectory_depths=[float(x) for x in hb["violation_depth"][hb["in_noise_zone"]]],
            ),
            bonus_joint_headline_lens_AND_noise_adjusted_height=dict(
                k=int(joint_headline_adjusted_height.sum()), n=n,
                cp_lower_NEGATIVE_PRECHECK=cp_lower(int(joint_headline_adjusted_height.sum()), n),
                **bn_headline_adjusted,
            ),
        )
        print(f"  [{batch_name}] n={n} | lens-ID: recomputed_cv={confirm['recomputed_cv_k']} vs original_gait_k={confirm['original_part1_gait_k']} match={confirm['matches_cv_lens']}", flush=True)
        print(f"    headline({cfg['headline_lens']}) joint(raw height): k={joint_headline_raw_height.sum()}/{n} bottleneck={bn_headline['bottleneck']} (only_gait={bn_headline['only_gait_fails']}, only_height={bn_headline['only_height_fails']})", flush=True)
        print(f"    height noise-zone: raw_violations={int((~hb['raw_pass']).sum())} excluded={n_excluded} genuine_after_exclusion={int((~hb['adjusted_pass']).sum())} height_rate_adjusted={height_adjusted_k/n:.4f}", flush=True)

    out["batches"] = batches
    return out


def main():
    all_results = {}
    for name, cfg in CARRIERS.items():
        all_results[name] = run_carrier(name, cfg)
    (OUT / "addendum_lens_recheck.json").write_text(json.dumps(all_results, indent=2, default=float) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
