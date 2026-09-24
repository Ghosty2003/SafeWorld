"""L3 formal Phase B: L6 SafePatrol, GF(gait_cycle) & G(height>h_min),
three carriers (walker-walk, walker-run, cheetah-run).

Iron rules: sealed calibration data untouched, frozen artifacts read-only
(g_init weights are REGENERATED bit-exactly from each carrier's own
already-frozen fit/cal data + documented training seed -- an established,
previously-used technique in this project, verified bit-exact below, never
written into any existing frozen output directory), all NEW data (val_200,
test1_1000) in a fresh 150000-series seed segment disjoint from every
SEED_LEDGER.md entry and from the L3 level-mapped precheck's own
150001/150002 seeds. 3 seeds (0/1/2) for g_init, 3 tiers (cal-max,
cal-max*1.5, cal-max+0.5) all run, tier3 cited (per this project's own
8-times-confirmed tier-fragility regularity).

Gait side: EACH CARRIER'S OWN frozen headline lens (walker-walk/cheetah-run:
count/no_cv; walker-run: sliding), combined with the tier-based full LBSM
pathwise-soundness check exactly as the original certificates do
(`combined_pass = lbsm_sound & dedu_pass[headline]`) -- zero changes to
prominence/M_MIN/CV_p95/false_rate_cp_upper.

Height side: pure monitor, no V/g_init. G(height>h_min) per trajectory,
with the noise-zone exclusion rule pre-registered in PREREG.md (violation
depth < probe C0 P95 -> not counted).

Joint = gait combined_pass[headline] AND height_pass. Reported at BOTH the
raw and noise-adjusted height definition; PREREG.md's declared rule uses
the noise-adjusted definition.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from sklearn.linear_model import Ridge

sys.path.insert(0, "/home/bot/SafeWorld")
import importlib.util as _ilu  # noqa: E402


def _load(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_lib = _load("l3_generic_lbsm_lib", "/home/bot/SafeWorld/tdmpc2/l3_generic_lbsm_lib.py")
generate_imagination_rollouts = _lib.generate_imagination_rollouts
build_training_pairs = _lib.build_training_pairs
train_g_init = _lib.train_g_init
evaluate_trajectory = _lib.evaluate_trajectory
deduction_verdict = _lib.deduction_verdict
fit_ridge_probe = _lib.fit_ridge_probe
cp_lower = _lib.cp_lower
cp_upper = _lib.cp_upper

OUT_ROOT = pathlib.Path("/home/bot/SafeWorld/artifacts/l3_l6_safepatrol_phaseb")
HORIZON = 300
N_FIT_CAL = 300
N_VAL = 200
N_TEST1 = 1000

CARRIERS = {
    "walker-walk": dict(
        checkpoint="models/walker-walk-3.pt", task="walker-walk", h_key="h",
        prominence=0.03, false_rate_cp_upper=0.07807546497166352, m_min=15, cv_p95=0.4460545042500288,
        headline_lens="count", h_min=0.27, probe_p95=0.0132,
        orig_seed_base=90000,
        orig_fitcal_npz="artifacts/tdmpc2_walker_l3_statistical_lbsm/phase_b/fitcal_data.npz",
        orig_result_json="artifacts/tdmpc2_walker_l3_statistical_lbsm/phase_b/result.json",
        probe_kind="latents_pool",
        latents_dir="artifacts/tdmpc2_walker_lppm_latents",
        fall_batch="artifacts/l3_rescue_screen/candidate5_deduction_scheme/real_fall_trajectories.npz",
        real50_for_edge="artifacts/l3_rescue_screen/candidate5_deduction_scheme/real_fall_trajectories.npz",  # has no clean gait pool; edge check uses latents pool instead (see below)
        val_seed=150100, test1_seed=150900,
    ),
    "walker-run": dict(
        checkpoint="models/dmcontrol/walker-run-1.pt", task="walker-run", h_key="q",
        prominence=0.03, false_rate_cp_upper=0.010703829113449055, m_min=46, cv_p95=0.21606759056802555,
        headline_lens="sliding", h_min=0.27, probe_p95=0.01733200426772711,
        orig_seed_base=140000,
        orig_fitcal_npz="artifacts/tdmpc2_walkerrun_l3_lbsm/step2/fitcal_data.npz",
        orig_result_json="artifacts/tdmpc2_walkerrun_l3_lbsm/step2/result.json",
        probe_kind="step0_real50",
        step0_real50="artifacts/tdmpc2_walkerrun_l3_lbsm/step0/real_50.npz",
        fall_batch="artifacts/tdmpc2_walkerrun_l3_lbsm/step1/real_fall_trajectories.npz",
        val_seed=151100, test1_seed=151900,
    ),
    "cheetah-run": dict(
        checkpoint="models/dmcontrol/cheetah-run-1.pt", task="cheetah-run", h_key="q",
        prominence=0.03, false_rate_cp_upper=0.0366593606474291, m_min=17, cv_p95=0.38260226202130304,
        headline_lens="count", h_min=0.3, probe_p95=0.009170136116078316,
        orig_seed_base=130000,
        orig_fitcal_npz="artifacts/tdmpc2_cheetah_l3_lbsm/step2/fitcal_data.npz",
        orig_result_json="artifacts/tdmpc2_cheetah_l3_lbsm/step2/result.json",
        probe_kind="step0_real50",
        step0_real50="artifacts/tdmpc2_cheetah_l3_lbsm/step0/real_50.npz",
        fall_batch="artifacts/tdmpc2_cheetah_l3_lbsm/step1/real_fall_trajectories.npz",
        val_seed=152100, test1_seed=152900,
    ),
}


def fit_probe(cfg):
    if cfg["probe_kind"] == "latents_pool":
        seed_files = sorted(pathlib.Path(cfg["latents_dir"]).glob("seed_*_latents.npz"))
        all_h, all_z = [], []
        for f in seed_files:
            d = np.load(f)
            all_h.append(d["height_m"]); all_z.append(d["posterior_z"])
        h = np.concatenate(all_h, axis=0); z = np.concatenate(all_z, axis=0)
        n = h.shape[0]; n_holdout = 300
        fit_idx = np.arange(n - n_holdout)
        return Ridge(alpha=10.0).fit(z[fit_idx].reshape(-1, 512), h[fit_idx].reshape(-1)), (z, h)
    else:
        d0 = np.load(cfg["step0_real50"])
        probe = fit_ridge_probe(d0["z"], d0["q"], n_holdout=0)
        return probe, (d0["z"], d0["q"])


def regenerate_models(cfg, probe):
    print(f"  [regenerate] reproducing original fit/cal rollouts (seed_base={cfg['orig_seed_base']}) + g_init training ...", flush=True)
    z_fitcal = generate_imagination_rollouts(cfg, N_FIT_CAL, HORIZON, seed_base=cfg["orig_seed_base"])
    h_fitcal = probe.predict(z_fitcal.reshape(-1, 512)).reshape(N_FIT_CAL, HORIZON)

    orig = np.load(cfg["orig_fitcal_npz"])
    z_match = bool(np.array_equal(z_fitcal, orig["z"]))
    h_match = bool(np.array_equal(h_fitcal, orig[cfg["h_key"]]))
    print(f"    bit-exact match vs original saved fit/cal data: z={z_match}, {cfg['h_key']}={h_match}", flush=True)

    rng = np.random.default_rng(0)
    idx = rng.permutation(N_FIT_CAL)
    n_cal = int(N_FIT_CAL * 0.33)
    cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]
    z_fit_pairs, y_fit_pairs = build_training_pairs(z_fitcal[fit_idx], h_fitcal[fit_idx], cfg["prominence"], HORIZON)
    z_cal_pairs, y_cal_pairs = build_training_pairs(z_fitcal[cal_idx], h_fitcal[cal_idx], cfg["prominence"], HORIZON)

    orig_result = json.loads(pathlib.Path(cfg["orig_result_json"]).read_text())
    orig_seeds = orig_result.get("seeds", orig_result.get("per_seed_tier_results"))
    models, verification = {}, {}
    for seed in (0, 1, 2):
        g_init, train_report = train_g_init(z_fit_pairs, y_fit_pairs, z_cal_pairs, y_cal_pairs, seed=seed)
        cal_max = train_report["calibration_residual_distribution"]["max"]
        deltas = {"tier1_max": cal_max, "tier2_max_x1_5": cal_max * 1.5, "tier3_max_plus_0_5": cal_max + 0.5}
        exact_match = None
        try:
            pb_seed = orig_result["seeds"][f"seed{seed}"]
            pb_train = pb_seed["training_report"]
            exact_match = (train_report["selected_epoch"] == pb_train["selected_epoch"] and
                          abs(train_report["selected_cal_p95"] - pb_train["selected_cal_p95"]) < 1e-9)
        except Exception:
            pass
        verification[f"seed{seed}"] = dict(exact_match=exact_match, cal_max=cal_max)
        print(f"    seed{seed}: cal_max={cal_max:.6f} exact_match_vs_original={exact_match}", flush=True)
        models[seed] = (g_init, deltas)
    return models, dict(z_fitcal_bit_exact=z_match, h_fitcal_bit_exact=h_match, per_seed=verification)


def generate_new_batch(cfg, seed_base, n):
    z = generate_imagination_rollouts(cfg, n, HORIZON, seed_base=seed_base)
    return z


def height_analysis(h, h_min, probe_p95):
    min_h = h.min(axis=1)
    raw_pass = min_h > h_min
    violation_depth = np.maximum(h_min - min_h, 0.0)
    in_noise_zone = (~raw_pass) & (violation_depth < probe_p95)
    adjusted_pass = raw_pass | in_noise_zone
    return dict(min_h=min_h, raw_pass=raw_pass, adjusted_pass=adjusted_pass,
               n_raw_violations=int((~raw_pass).sum()), n_excluded=int(in_noise_zone.sum()),
               n_genuine=int((~adjusted_pass).sum()))


def analyze_new_batch(z, h, cfg, models):
    n = h.shape[0]
    hb = height_analysis(h, cfg["h_min"], cfg["probe_p95"])
    results = {}
    for seed, (g_init, deltas) in models.items():
        tier_results = {}
        for tier_name, delta in deltas.items():
            lbsm_rows, dedu_rows = [], []
            for i in range(n):
                lb = evaluate_trajectory(g_init, delta, z[i], h[i], cfg["prominence"], HORIZON)
                dv = deduction_verdict(h[i], cfg["prominence"], cfg["false_rate_cp_upper"], cfg["m_min"], cfg["cv_p95"], HORIZON)
                lbsm_rows.append(lb); dedu_rows.append(dv)
            lbsm_sound = np.array([r["lbsm_pathwise_sound"] for r in lbsm_rows])
            zfree_gate_violated = any(r["n_zfree_source"] > 0 and r["k_zfree_passing"] < r["n_zfree_source"] for r in lbsm_rows)
            key = {"count": "verdict_no_cv", "cv": "verdict_with_cv", "sliding": "verdict_sliding"}[cfg["headline_lens"]]
            dedu_pass = np.array([r[key] for r in dedu_rows])
            gait_combined = lbsm_sound & dedu_pass

            joint_raw = gait_combined & hb["raw_pass"]
            joint_adj = gait_combined & hb["adjusted_pass"]
            tier_results[tier_name] = dict(
                delta=float(delta), lbsm_own_soundness_rate=float(lbsm_sound.mean()), zfree_gate_violated=bool(zfree_gate_violated),
                gait_combined_k=int(gait_combined.sum()),
                joint_raw_height=dict(k=int(joint_raw.sum()), n=n, cp_lower=cp_lower(int(joint_raw.sum()), n)),
                joint_noise_adjusted_height=dict(k=int(joint_adj.sum()), n=n, cp_lower=cp_lower(int(joint_adj.sum()), n)),
            )
        results[f"seed{seed}"] = tier_results
    return dict(n=n, height=dict(n_raw_violations=hb["n_raw_violations"], n_excluded=hb["n_excluded"], n_genuine=hb["n_genuine"],
                                 height_rate_raw=float(hb["raw_pass"].sum() / n), height_rate_adjusted=float(hb["adjusted_pass"].sum() / n)),
               per_seed_tier=results)


def antifraud_check(cfg, probe, models):
    print("  [antifraud A] fall batch joint check ...", flush=True)
    fb = np.load(cfg["fall_batch"])
    z_fall = fb["z"]; h_fall_key = "h" if "h" in fb.files else "q"
    h_fall = fb[h_fall_key]
    n = z_fall.shape[0]
    key = {"count": "verdict_no_cv", "cv": "verdict_with_cv", "sliding": "verdict_sliding"}[cfg["headline_lens"]]
    gait_pass = np.zeros(n, dtype=bool)
    for i in range(n):
        dv = deduction_verdict(h_fall[i], cfg["prominence"], cfg["false_rate_cp_upper"], cfg["m_min"], cfg["cv_p95"], HORIZON)
        gait_pass[i] = dv[key]
    height_pass = h_fall.min(axis=1) > cfg["h_min"]
    joint_pass = gait_pass & height_pass
    n_fp = int(joint_pass.sum())
    intercept = dict(both=int((~gait_pass & ~height_pass).sum()), gait_only=int((~gait_pass & height_pass).sum()),
                     height_only=int((gait_pass & ~height_pass).sum()), neither_FP=n_fp)
    print(f"    fall batch n={n}: joint FP={n_fp}, intercept source={intercept}", flush=True)

    print("  [antifraud B] height-edge check (10 lowest-height real trajectories) ...", flush=True)
    if cfg["probe_kind"] == "latents_pool":
        _, (z_pool, h_pool) = fit_probe(cfg)
    else:
        d0 = np.load(cfg["step0_real50"]); h_pool = d0["q"]
    min_per_traj = h_pool.min(axis=1)
    lowest10_idx = np.argsort(min_per_traj)[:10]
    lowest10_vals = min_per_traj[lowest10_idx]
    lowest10_pass = lowest10_vals > cfg["h_min"]
    print(f"    10 lowest real min-heights: {lowest10_vals.tolist()}, h_min={cfg['h_min']}, all_pass={bool(lowest10_pass.all())}", flush=True)

    return dict(fall_batch=dict(n=n, n_false_positive_joint=n_fp, intercept_source=intercept),
               height_edge=dict(lowest10_values=lowest10_vals.tolist(), h_min=cfg["h_min"], all_pass_direction_correct=bool(lowest10_pass.all()),
                                per_traj_pass=lowest10_pass.tolist()))


def run_carrier(name, cfg):
    print(f"=== {name} ===", flush=True)
    out = OUT_ROOT / name.replace("-", "_")
    out.mkdir(parents=True, exist_ok=True)

    probe, _ = fit_probe(cfg)
    models, regen_report = regenerate_models(cfg, probe)
    (out / "regeneration_verification.json").write_text(json.dumps(regen_report, indent=2, default=float) + "\n")

    antifraud = antifraud_check(cfg, probe, models)
    (out / "antifraud_result.json").write_text(json.dumps(antifraud, indent=2, default=float) + "\n")

    print(f"  [val_200] generating {N_VAL} fresh imagination rollouts, seed={cfg['val_seed']} ...", flush=True)
    z_val = generate_new_batch(cfg, cfg["val_seed"], N_VAL)
    h_val = probe.predict(z_val.reshape(-1, 512)).reshape(N_VAL, HORIZON)
    val_result = analyze_new_batch(z_val, h_val, cfg, models)
    np.savez_compressed(out / "val_200.npz", z=z_val, h=h_val)
    r = val_result["per_seed_tier"]["seed0"]["tier3_max_plus_0_5"]
    print(f"    val_200 tier3/seed0: gait_k={r['gait_combined_k']}/{N_VAL} joint_raw={r['joint_raw_height']['k']}/{N_VAL} joint_adj={r['joint_noise_adjusted_height']['k']}/{N_VAL}", flush=True)

    print(f"  [test1_1000] generating {N_TEST1} fresh imagination rollouts, seed={cfg['test1_seed']} ...", flush=True)
    z_test1 = generate_new_batch(cfg, cfg["test1_seed"], N_TEST1)
    h_test1 = probe.predict(z_test1.reshape(-1, 512)).reshape(N_TEST1, HORIZON)
    test1_result = analyze_new_batch(z_test1, h_test1, cfg, models)
    np.savez_compressed(out / "test1_1000.npz", z=z_test1, h=h_test1)
    for seed in ("seed0", "seed1", "seed2"):
        r = test1_result["per_seed_tier"][seed]["tier3_max_plus_0_5"]
        print(f"    test1_1000 tier3/{seed}: gait_k={r['gait_combined_k']}/{N_TEST1} joint_raw={r['joint_raw_height']['k']}/{N_TEST1} joint_adj={r['joint_noise_adjusted_height']['k']}/{N_TEST1} (cp_lower_adj={r['joint_noise_adjusted_height']['cp_lower']:.4f})", flush=True)

    cited = test1_result["per_seed_tier"]["seed0"]["tier3_max_plus_0_5"]["joint_noise_adjusted_height"]
    verdict = "SAFE" if cited["cp_lower"] >= 0.95 else "ABSTAIN"
    report = dict(
        carrier=name, headline_lens=cfg["headline_lens"], h_min=cfg["h_min"], probe_p95=cfg["probe_p95"],
        regeneration_verification=regen_report, antifraud=antifraud,
        val_200=val_result, test1_1000=test1_result,
        cited_tier="tier3_max_plus_0_5", cited_seed="seed0",
        verdict=verdict, p_hat_gamma=cited["cp_lower"],
        calibration_touched=False, track_b_touched=False, sealed_data_touched=False,
    )
    (out / "result.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"  VERDICT[{name}]: {verdict}, p_hat_gamma(tier3,seed0,noise-adjusted)={cited['cp_lower']:.4f}", flush=True)
    return report


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] if len(sys.argv) > 1 else list(CARRIERS.keys())
    for name in names:
        run_carrier(name, CARRIERS[name])
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
