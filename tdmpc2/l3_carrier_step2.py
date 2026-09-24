"""Step 2 (per carrier, only for Step-1-passing carriers): LBSM
construction. 300 fit/cal + 200 val imagination rollouts, recurring
fixed-eta-decrement V (g_init trained on every peak-to-peak segment, not
just t=0), conformal Delta 3 tiers, event-source-consistency gate, 3
seeds. Three judgment lenses (count/CV/sliding) all reported; headline
verdict uses the count lens per instruction (its anti-fraud soundness was
already independently validated in Step 1).

Uses the FROZEN parameters from Step 1 (prominence, false_rate_cp_upper,
m_min, cv_p95) -- never re-tuned here.

No frozen artifact modified. No sealed calibration data touched.
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
generate_imagination_rollouts = _lib.generate_imagination_rollouts
build_training_pairs = _lib.build_training_pairs
train_g_init = _lib.train_g_init
evaluate_trajectory = _lib.evaluate_trajectory
deduction_verdict = _lib.deduction_verdict
fit_ridge_probe = _lib.fit_ridge_probe
cp_lower = _lib.cp_lower
SAFE_THRESHOLD = _lib.SAFE_THRESHOLD

HORIZON = 300
N_FIT_CAL = 300
N_VAL = 200

CARRIERS = {
    "cheetah": dict(
        checkpoint="models/dmcontrol/cheetah-run-1.pt", task="cheetah-run",
        prominence=0.03, false_rate_cp_upper=0.0366593606474291, m_min=17, cv_p95=0.38260226202130304,
        seed_base=130000,
        out=pathlib.Path("artifacts/tdmpc2_cheetah_l3_lbsm/step2"),
        step0_dir=pathlib.Path("artifacts/tdmpc2_cheetah_l3_lbsm/step0"),
    ),
    "walkerrun": dict(
        checkpoint="models/dmcontrol/walker-run-1.pt", task="walker-run",
        prominence=0.03, false_rate_cp_upper=0.010703829113449055, m_min=46, cv_p95=0.21606759056802555,
        seed_base=140000,
        out=pathlib.Path("artifacts/tdmpc2_walkerrun_l3_lbsm/step2"),
        step0_dir=pathlib.Path("artifacts/tdmpc2_walkerrun_l3_lbsm/step0"),
    ),
}


def run_one_seed(cfg, z_fit_pairs, y_fit_pairs, z_cal_pairs, y_cal_pairs, z_val, q_val, probe, seed, tag):
    g_init, train_report = train_g_init(z_fit_pairs, y_fit_pairs, z_cal_pairs, y_cal_pairs, seed=seed)
    cal_max = train_report["calibration_residual_distribution"]["max"]
    deltas = {"tier1_max": cal_max, "tier2_max_x1_5": cal_max * 1.5, "tier3_max_plus_0_5": cal_max + 0.5}

    tier_results = {}
    n = z_val.shape[0]
    for tier_name, delta in deltas.items():
        lbsm_rows, dedu_rows = [], []
        for i in range(n):
            lb = evaluate_trajectory(g_init, delta, z_val[i], q_val[i], cfg["prominence"], HORIZON)
            dv = deduction_verdict(q_val[i], cfg["prominence"], cfg["false_rate_cp_upper"], cfg["m_min"], cfg["cv_p95"], HORIZON)
            lbsm_rows.append(lb)
            dedu_rows.append(dv)

        n_consistent = sum(1 for lb, dv in zip(lbsm_rows, dedu_rows) if lb["m_raw_peaks"] == dv["m"])
        lbsm_sound = np.array([r["lbsm_pathwise_sound"] for r in lbsm_rows])
        zfree_gate_violated = any(r["n_zfree_source"] > 0 and r["k_zfree_passing"] < r["n_zfree_source"] for r in lbsm_rows)

        def combined(key):
            dedu_pass = np.array([r[key] for r in dedu_rows])
            combined_pass = lbsm_sound & dedu_pass
            k = int(combined_pass.sum())
            p_hat_gamma = cp_lower(k, n)
            return dict(k=k, n=n, rate=k / n, p_hat_gamma=p_hat_gamma,
                       verdict="SAFE" if p_hat_gamma >= SAFE_THRESHOLD else "ABSTAIN")

        tier_results[tier_name] = dict(
            delta=delta, lbsm_own_soundness_rate=float(lbsm_sound.mean()),
            n_closed_form_mismatch_total=int(sum(r["n_closed_form_mismatch"] for r in lbsm_rows)),
            zfree_gate_violated=bool(zfree_gate_violated),
            event_source_consistency=dict(n_consistent=n_consistent, n_total=n, rate=n_consistent / n),
            combined_no_cv=combined("verdict_no_cv"),
            combined_with_cv=combined("verdict_with_cv"),
            combined_sliding=combined("verdict_sliding"),
        )
        r = tier_results[tier_name]
        print(f"  [{tag}] {tier_name}: Delta={delta:.4f} sound={r['lbsm_own_soundness_rate']:.3f} "
              f"consistency={n_consistent}/{n} zfree_violated={zfree_gate_violated} "
              f"no_cv={r['combined_no_cv']['k']}/{n} p_hat={r['combined_no_cv']['p_hat_gamma']:.4f} "
              f"with_cv={r['combined_with_cv']['k']}/{n} p_hat={r['combined_with_cv']['p_hat_gamma']:.4f} "
              f"sliding={r['combined_sliding']['k']}/{n} p_hat={r['combined_sliding']['p_hat_gamma']:.4f}",
              flush=True)

    return dict(seed=seed, n_fit_pairs=len(y_fit_pairs), n_cal_pairs=len(y_cal_pairs),
               training_report=train_report, deltas=deltas, tier_results=tier_results)


def run(name):
    cfg = CARRIERS[name]
    cfg["out"].mkdir(parents=True, exist_ok=True)
    print(f"=== Step 2: {name} ===", flush=True)

    d0 = np.load(cfg["step0_dir"] / "real_50.npz")
    probe = fit_ridge_probe(d0["z"], d0["q"], n_holdout=0)

    print(f"[1] generating {N_FIT_CAL} fit/cal rollouts ...", flush=True)
    z_fitcal = generate_imagination_rollouts(cfg, N_FIT_CAL, HORIZON, seed_base=cfg["seed_base"])
    print(f"[2] generating {N_VAL} validation rollouts ...", flush=True)
    z_val = generate_imagination_rollouts(cfg, N_VAL, HORIZON, seed_base=cfg["seed_base"] + 1)

    q_fitcal = probe.predict(z_fitcal.reshape(-1, 512)).reshape(N_FIT_CAL, HORIZON)
    q_val = probe.predict(z_val.reshape(-1, 512)).reshape(N_VAL, HORIZON)
    np.savez_compressed(cfg["out"] / "fitcal_data.npz", z=z_fitcal, q=q_fitcal)
    np.savez_compressed(cfg["out"] / "val_data.npz", z=z_val, q=q_val)

    print("[3] anchor-disjoint fit/cal split (67/33) ...", flush=True)
    rng = np.random.default_rng(0)
    idx = rng.permutation(N_FIT_CAL)
    n_cal = int(N_FIT_CAL * 0.33)
    cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]

    print("[4] building recurring-segment training pairs ...", flush=True)
    z_fit_pairs, y_fit_pairs = build_training_pairs(z_fitcal[fit_idx], q_fitcal[fit_idx], cfg["prominence"], HORIZON)
    z_cal_pairs, y_cal_pairs = build_training_pairs(z_fitcal[cal_idx], q_fitcal[cal_idx], cfg["prominence"], HORIZON)
    print(f"    fit pairs={len(y_fit_pairs)}, cal pairs={len(y_cal_pairs)}", flush=True)

    results = {}
    for seed in (0, 1, 2):
        print(f"=== {name} seed {seed} ===", flush=True)
        results[f"seed{seed}"] = run_one_seed(cfg, z_fit_pairs, y_fit_pairs, z_cal_pairs, y_cal_pairs,
                                              z_val, q_val, probe, seed, tag=f"{name}-seed{seed}")

    report = dict(
        status="COMPLETE", experiment=f"tdmpc2_{name}_l3_phase_b_statistical_lbsm",
        frozen_params=dict(prominence=cfg["prominence"], false_rate_cp_upper=cfg["false_rate_cp_upper"],
                           m_min=cfg["m_min"], cv_p95=cfg["cv_p95"]),
        n_fit_cal_rollouts=N_FIT_CAL, n_val_rollouts=N_VAL, horizon=HORIZON,
        n_fit_pairs=len(y_fit_pairs), n_cal_pairs=len(y_cal_pairs),
        seeds=results,
        calibration_touched=False, track_b_touched=False, sealed_data_touched=False,
    )
    (cfg["out"] / "result.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"DONE: {name}", flush=True)
    return report


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else None
    if name:
        run(name)
    else:
        for n in CARRIERS:
            run(n)
