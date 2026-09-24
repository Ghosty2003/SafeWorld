"""Step 3 (per carrier, only for Step-2-completed carriers): 1000-rollout
Test 1 honesty check. Regenerates Step 2's g_init weights (never saved to
disk, same gap as before) from the identical fit/cal data and seeds,
verifies bit-exactness, then judges 1000 FRESH rollouts (new seed, disjoint
from every seed used anywhere in this carrier's pipeline) across all 3
tiers x 3 seeds x 3 lenses.

No frozen artifact modified. No sealed calibration data touched.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/bot/SafeWorld")
import importlib.util as _ilu  # noqa: E402


def _load(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_lib = _load("l3_generic_lbsm_lib", "/home/bot/SafeWorld/tdmpc2/l3_generic_lbsm_lib.py")
_s2 = _load("l3_carrier_step2", "/home/bot/SafeWorld/tdmpc2/l3_carrier_step2.py")

generate_imagination_rollouts = _lib.generate_imagination_rollouts
build_training_pairs = _lib.build_training_pairs
train_g_init = _lib.train_g_init
evaluate_trajectory = _lib.evaluate_trajectory
deduction_verdict = _lib.deduction_verdict
fit_ridge_probe = _lib.fit_ridge_probe
cp_lower = _lib.cp_lower
cp_upper = _lib.cp_upper

HORIZON = 300
N_FIT_CAL = 300
N_NEW = 1000

CARRIERS = {
    "cheetah": dict(
        checkpoint="models/dmcontrol/cheetah-run-1.pt", task="cheetah-run",
        prominence=0.03, false_rate_cp_upper=0.0366593606474291, m_min=17, cv_p95=0.38260226202130304,
        step2_seed_base=130000, new_seed_base=135000,
        out=pathlib.Path("artifacts/tdmpc2_cheetah_l3_lbsm/step3"),
        step2_dir=pathlib.Path("artifacts/tdmpc2_cheetah_l3_lbsm/step2"),
        step0_dir=pathlib.Path("artifacts/tdmpc2_cheetah_l3_lbsm/step0"),
    ),
    "walkerrun": dict(
        checkpoint="models/dmcontrol/walker-run-1.pt", task="walker-run",
        prominence=0.03, false_rate_cp_upper=0.010703829113449055, m_min=46, cv_p95=0.21606759056802555,
        step2_seed_base=140000, new_seed_base=145000,
        out=pathlib.Path("artifacts/tdmpc2_walkerrun_l3_lbsm/step3"),
        step2_dir=pathlib.Path("artifacts/tdmpc2_walkerrun_l3_lbsm/step2"),
        step0_dir=pathlib.Path("artifacts/tdmpc2_walkerrun_l3_lbsm/step0"),
    ),
}


def regenerate(cfg, probe):
    print("[regenerate] reproducing Step 2's exact fit/cal rollouts + g_init training ...", flush=True)
    z_fitcal = generate_imagination_rollouts(cfg, N_FIT_CAL, HORIZON, seed_base=cfg["step2_seed_base"])
    q_fitcal = probe.predict(z_fitcal.reshape(-1, 512)).reshape(N_FIT_CAL, HORIZON)

    pb_fitcal = np.load(cfg["step2_dir"] / "fitcal_data.npz")
    z_match = bool(np.array_equal(z_fitcal, pb_fitcal["z"]))
    q_match = bool(np.array_equal(q_fitcal, pb_fitcal["q"]))
    print(f"    bit-exact match vs Step 2's saved data: z={z_match}, q={q_match}", flush=True)

    rng = np.random.default_rng(0)
    idx = rng.permutation(N_FIT_CAL)
    n_cal = int(N_FIT_CAL * 0.33)
    cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]
    z_fit_pairs, y_fit_pairs = build_training_pairs(z_fitcal[fit_idx], q_fitcal[fit_idx], cfg["prominence"], HORIZON)
    z_cal_pairs, y_cal_pairs = build_training_pairs(z_fitcal[cal_idx], q_fitcal[cal_idx], cfg["prominence"], HORIZON)

    pb_result = json.loads((cfg["step2_dir"] / "result.json").read_text())
    models, verification = {}, {}
    for seed in (0, 1, 2):
        g_init, train_report = train_g_init(z_fit_pairs, y_fit_pairs, z_cal_pairs, y_cal_pairs, seed=seed)
        cal_max = train_report["calibration_residual_distribution"]["max"]
        deltas = {"tier1_max": cal_max, "tier2_max_x1_5": cal_max * 1.5, "tier3_max_plus_0_5": cal_max + 0.5}
        pb_seed = pb_result["seeds"][f"seed{seed}"]
        pb_train = pb_seed["training_report"]
        exact_match = (
            train_report["selected_epoch"] == pb_train["selected_epoch"] and
            abs(train_report["selected_cal_p95"] - pb_train["selected_cal_p95"]) < 1e-9 and
            abs(cal_max - pb_seed["deltas"]["tier1_max"]) < 1e-9
        )
        verification[f"seed{seed}"] = dict(exact_match=exact_match)
        print(f"    seed{seed}: exact_match={exact_match}", flush=True)
        models[seed] = (g_init, deltas)
    return models, dict(z_fitcal_bit_exact=z_match, q_fitcal_bit_exact=q_match, per_seed=verification)


def run(name):
    cfg = CARRIERS[name]
    cfg["out"].mkdir(parents=True, exist_ok=True)
    print(f"=== Step 3: {name} ===", flush=True)

    d0 = np.load(cfg["step0_dir"] / "real_50.npz")
    probe = fit_ridge_probe(d0["z"], d0["q"], n_holdout=0)

    models, regen_report = regenerate(cfg, probe)
    (cfg["out"] / "regeneration_verification.json").write_text(json.dumps(regen_report, indent=2) + "\n")

    print(f"[generate] {N_NEW} FRESH rollouts, seed_base={cfg['new_seed_base']} ...", flush=True)
    z_new = generate_imagination_rollouts(cfg, N_NEW, HORIZON, seed_base=cfg["new_seed_base"])
    q_new = probe.predict(z_new.reshape(-1, 512)).reshape(N_NEW, HORIZON)
    np.savez_compressed(cfg["out"] / "fresh_1000_data.npz", z=z_new, q=q_new)

    results = {}
    for seed, (g_init, deltas) in models.items():
        tier_results = {}
        for tier_name, delta in deltas.items():
            lbsm_rows, dedu_rows = [], []
            for i in range(N_NEW):
                lb = evaluate_trajectory(g_init, delta, z_new[i], q_new[i], cfg["prominence"], HORIZON)
                dv = deduction_verdict(q_new[i], cfg["prominence"], cfg["false_rate_cp_upper"], cfg["m_min"], cfg["cv_p95"], HORIZON)
                lbsm_rows.append(lb)
                dedu_rows.append(dv)
            n_consistent = sum(1 for lb, dv in zip(lbsm_rows, dedu_rows) if lb["m_raw_peaks"] == dv["m"])
            lbsm_sound = np.array([r["lbsm_pathwise_sound"] for r in lbsm_rows])
            zfree_gate_violated = any(r["n_zfree_source"] > 0 and r["k_zfree_passing"] < r["n_zfree_source"] for r in lbsm_rows)

            def combined(key):
                dedu_pass = np.array([r[key] for r in dedu_rows])
                combined_pass = lbsm_sound & dedu_pass
                k = int(combined_pass.sum())
                return dict(k=k, n=N_NEW, empirical_rate=k / N_NEW, cp_95_lower=cp_lower(k, N_NEW), cp_95_upper=cp_upper(k, N_NEW))

            tier_results[tier_name] = dict(
                delta=delta, lbsm_own_soundness_rate=float(lbsm_sound.mean()),
                zfree_gate_violated=bool(zfree_gate_violated),
                event_source_consistency=dict(n_consistent=n_consistent, n_total=N_NEW, rate=n_consistent / N_NEW),
                combined_no_cv=combined("verdict_no_cv"), combined_with_cv=combined("verdict_with_cv"),
                combined_sliding=combined("verdict_sliding"),
            )
            r = tier_results[tier_name]
            print(f"  [seed{seed}] {tier_name}: sound={r['lbsm_own_soundness_rate']:.3f} zfree_violated={zfree_gate_violated} "
                  f"consistency={n_consistent}/{N_NEW} "
                  f"no_cv={r['combined_no_cv']['k']}/{N_NEW} lower={r['combined_no_cv']['cp_95_lower']:.4f} "
                  f"with_cv={r['combined_with_cv']['k']}/{N_NEW} lower={r['combined_with_cv']['cp_95_lower']:.4f} "
                  f"sliding={r['combined_sliding']['k']}/{N_NEW} lower={r['combined_sliding']['cp_95_lower']:.4f}",
                  flush=True)
        results[f"seed{seed}"] = tier_results

    pb_result = json.loads((cfg["step2_dir"] / "result.json").read_text())
    pb_declared = pb_result["seeds"]["seed0"]["tier_results"]["tier1_max"]
    declared = dict(no_cv=pb_declared["combined_no_cv"]["p_hat_gamma"],
                    with_cv=pb_declared["combined_with_cv"]["p_hat_gamma"],
                    sliding=pb_declared["combined_sliding"]["p_hat_gamma"])

    def judge(empirical, declared_val):
        if empirical >= declared_val:
            return "CONFIRMED"
        elif empirical >= declared_val - 0.02:
            return "BORDERLINE"
        return "CONCERNING"

    comparison = {}
    for lens, key in (("no_cv", "combined_no_cv"), ("with_cv", "combined_with_cv"), ("sliding", "combined_sliding")):
        rates = [results[f"seed{s}"]["tier1_max"][key]["empirical_rate"] for s in (0, 1, 2)]
        comparison[lens] = dict(declared_p_hat_gamma=declared[lens],
                                empirical_rates_by_seed=dict(zip(("seed0", "seed1", "seed2"), rates)),
                                judgments=[judge(r, declared[lens]) for r in rates])
    print(json.dumps(comparison, indent=2), flush=True)

    tier1_zfree_violations = {s: results[f"seed{s}"]["tier1_max"]["zfree_gate_violated"] for s in (0, 1, 2)}
    report = dict(
        status="COMPLETE", experiment=f"tdmpc2_{name}_l3_test1",
        n_new_rollouts=N_NEW, horizon=HORIZON, new_seed_base=cfg["new_seed_base"],
        regeneration_verification=regen_report,
        per_seed_tier_results=results,
        comparison_vs_step2_declared=comparison,
        tier1_zfree_violations=tier1_zfree_violations,
        tier1_gate_violation_recurred=any(tier1_zfree_violations.values()),
        calibration_touched=False, track_b_touched=False, sealed_data_touched=False,
    )
    (cfg["out"] / "result.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print("DONE", flush=True)
    return report


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else None
    if name:
        run(name)
    else:
        for n in CARRIERS:
            run(n)
