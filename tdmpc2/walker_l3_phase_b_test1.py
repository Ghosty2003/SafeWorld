"""L3 Recurrence Test 1 (CarDreamer step10 protocol, third transplant):
independent 1000-rollout honesty check of Phase B's statistical LBSM.

Phase B's g_init weights were never saved to disk (the same oversight the
door-close line hit before) -- this script first REGENERATES them exactly
(same seed_base=90000/offset=0 fit/cal rollouts, same anchor-disjoint split
seed=0, same train_g_init seeds 0/1/2) and verifies the regeneration against
Phase B's own saved result.json training_report numbers before treating the
weights as "frozen." Any mismatch is reported, not hidden (mirroring the
door-close regenerate precedent, where seed0 reproduced exactly and
seed1/seed2 did not, due to CPU/GPU non-determinism in SGD -- documented,
not silently accepted).

Then: 1000 FRESH imagination rollouts, seed_base=100000 (disjoint from
every seed used anywhere in this project's walker line: 2000/5000/6000/
9000/20000/50000/60000/70000/90000, and the fixed c1_exact_mpc_anchors.npz
pool). Same protocol as Phase B (horizon=300, mpc_plan, pure imagination).
Judged by the frozen (regenerated) g_init/Delta for all 3 seeds x 3 tiers,
combined with the frozen deduction-scheme detector's three lenses (count
only, +CV, +sliding-window) -- exactly Phase B's own combined-verdict
construction, reused, not reinvented.

No frozen artifact modified (Phase B's own result.json/npz are read-only
inputs). No sealed/consumed calibration data touched.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/bot/SafeWorld")

# NOTE: cannot import this sibling module as "tdmpc2.walker_l3_phase_b_statistical_lbsm"
# -- doing so registers "tdmpc2" as a (namespace) package in sys.modules, which then
# shadows the vendored upstream TD-MPC2 source package of the SAME name that
# wrappers/tdmpc2_wrapper.py needs ("from tdmpc2 import TDMPC2"), breaking wrapper.load().
# Load the sibling file directly by path instead, under an unrelated module name.
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "walker_l3_phase_b_statistical_lbsm",
    "/home/bot/SafeWorld/tdmpc2/walker_l3_phase_b_statistical_lbsm.py",
)
_pb = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pb)

BaseG = _pb.BaseG
ETA = _pb.ETA
HORIZON = _pb.HORIZON
PROMINENCE = _pb.PROMINENCE
PB_SEED_BASE = _pb.SEED_BASE
N_FIT_CAL = _pb.N_FIT_CAL
cp_lower = _pb.cp_lower
cp_upper = _pb.cp_upper
evaluate_trajectory = _pb.evaluate_trajectory
deduction_verdict = _pb.deduction_verdict
find_peaks_and_segments = _pb.find_peaks_and_segments
build_training_pairs = _pb.build_training_pairs
train_g_init = _pb.train_g_init
generate_rollouts = _pb.generate_rollouts
fit_frozen_height_probe = _pb.fit_frozen_height_probe
load_frozen_deduction_scheme = _pb.load_frozen_deduction_scheme

PB_DIR = pathlib.Path("artifacts/tdmpc2_walker_l3_statistical_lbsm/phase_b")
OUT = pathlib.Path("artifacts/tdmpc2_walker_l3_statistical_lbsm/phase_b/test1")
N_NEW = 1000
NEW_SEED_BASE = 100000
SAFE_THRESHOLD = 0.95


def regenerate_frozen_models(params, probe):
    print("[regenerate] reproducing Phase B's exact fit/cal rollouts + g_init training ...", flush=True)
    z_fitcal = generate_rollouts(N_FIT_CAL, seed_offset=0)  # bit-identical seed to Phase B's own call
    h_fitcal = probe.predict(z_fitcal.reshape(-1, 512)).reshape(N_FIT_CAL, HORIZON)

    # cross-check against Phase B's own saved fitcal_data.npz
    pb_fitcal = np.load(PB_DIR / "fitcal_data.npz")
    z_match = bool(np.array_equal(z_fitcal, pb_fitcal["z"]))
    h_match = bool(np.array_equal(h_fitcal, pb_fitcal["h"]))
    print(f"    z_fitcal bit-exact match vs Phase B's saved data: {z_match}, h match: {h_match}", flush=True)

    rng = np.random.default_rng(0)
    idx = rng.permutation(N_FIT_CAL)
    n_cal = int(N_FIT_CAL * 0.33)
    cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]
    z_fit_pairs, y_fit_pairs = build_training_pairs(z_fitcal[fit_idx], h_fitcal[fit_idx], PROMINENCE)
    z_cal_pairs, y_cal_pairs = build_training_pairs(z_fitcal[cal_idx], h_fitcal[cal_idx], PROMINENCE)

    pb_result = json.loads((PB_DIR / "result.json").read_text())
    models = {}
    verification = {}
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
        verification[f"seed{seed}"] = dict(
            exact_match=exact_match,
            regenerated=dict(selected_epoch=train_report["selected_epoch"],
                             selected_cal_p95=train_report["selected_cal_p95"], cal_max=cal_max),
            phase_b_original=dict(selected_epoch=pb_train["selected_epoch"],
                                  selected_cal_p95=pb_train["selected_cal_p95"], cal_max=pb_seed["deltas"]["tier1_max"]),
        )
        print(f"    seed{seed}: exact_match={exact_match} (regenerated cal_max={cal_max:.6f} vs "
              f"Phase B's {pb_seed['deltas']['tier1_max']:.6f})", flush=True)
        torch.save(g_init.state_dict(), OUT / f"g_init_seed{seed}.pt")
        models[seed] = (g_init, deltas)

    return models, dict(z_fitcal_bit_exact=z_match, h_fitcal_bit_exact=h_match, per_seed=verification)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    params = load_frozen_deduction_scheme()
    probe = fit_frozen_height_probe()
    print("frozen deduction-scheme params:", json.dumps(params, indent=2), flush=True)

    models, regen_report = regenerate_frozen_models(params, probe)
    (OUT / "regeneration_verification.json").write_text(json.dumps(regen_report, indent=2) + "\n")

    print(f"[generate] {N_NEW} FRESH imagination rollouts, seed_base={NEW_SEED_BASE} ...", flush=True)
    z_new = generate_rollouts(N_NEW, seed_offset=NEW_SEED_BASE - PB_SEED_BASE)
    h_new = probe.predict(z_new.reshape(-1, 512)).reshape(N_NEW, HORIZON)
    np.savez_compressed(OUT / "fresh_1000_data.npz", z=z_new, h=h_new)

    results = {}
    for seed, (g_init, deltas) in models.items():
        tier_results = {}
        for tier_name, delta in deltas.items():
            lbsm_rows, dedu_rows = [], []
            for i in range(N_NEW):
                lb = evaluate_trajectory(g_init, delta, z_new[i], h_new[i], PROMINENCE,
                                         params["false_rate_cp_upper"], params["m_min"], params["cv_p95"])
                dv = deduction_verdict(h_new[i], z_new[i], probe, PROMINENCE,
                                       params["false_rate_cp_upper"], params["m_min"], params["cv_p95"])
                lbsm_rows.append(lb)
                dedu_rows.append(dv)

            n_consistent = sum(1 for lb, dv in zip(lbsm_rows, dedu_rows) if lb["m_raw_peaks"] == dv["m"])
            lbsm_sound = np.array([r["lbsm_pathwise_sound"] for r in lbsm_rows])
            zfree_gate_violated = any(r["n_zfree_source"] > 0 and r["k_zfree_passing"] < r["n_zfree_source"] for r in lbsm_rows)

            def combined(key):
                dedu_pass = np.array([r[key] for r in dedu_rows])
                combined_pass = lbsm_sound & dedu_pass
                k = int(combined_pass.sum())
                lower = cp_lower(k, N_NEW)
                upper = cp_upper(k, N_NEW)
                return dict(k=k, n=N_NEW, empirical_rate=k / N_NEW, cp_95_lower=lower, cp_95_upper=upper)

            tier_results[tier_name] = dict(
                delta=delta, lbsm_own_soundness_rate=float(lbsm_sound.mean()),
                n_closed_form_mismatch_total=int(sum(r["n_closed_form_mismatch"] for r in lbsm_rows)),
                zfree_gate_violated=bool(zfree_gate_violated),
                event_source_consistency=dict(n_consistent=n_consistent, n_total=N_NEW, rate=n_consistent / N_NEW),
                combined_no_cv=combined("verdict_no_cv"),
                combined_with_cv=combined("verdict_with_cv"),
                combined_sliding=combined("verdict_sliding"),
            )
            r = tier_results[tier_name]
            print(f"  [seed{seed}] {tier_name}: sound={r['lbsm_own_soundness_rate']:.3f} "
                  f"consistency={n_consistent}/{N_NEW} "
                  f"no_cv={r['combined_no_cv']['k']}/{N_NEW} lower={r['combined_no_cv']['cp_95_lower']:.4f} "
                  f"with_cv={r['combined_with_cv']['k']}/{N_NEW} lower={r['combined_with_cv']['cp_95_lower']:.4f} "
                  f"sliding={r['combined_sliding']['k']}/{N_NEW} lower={r['combined_sliding']['cp_95_lower']:.4f}",
                  flush=True)
        results[f"seed{seed}"] = tier_results

    # comparison vs Phase B's own declared numbers (200-rollout, tier1, since tier-invariant there)
    pb_result = json.loads((PB_DIR / "result.json").read_text())
    pb_declared = pb_result["seeds"]["seed0"]["tier_results"]["tier1_max"]
    declared = dict(
        no_cv=pb_declared["combined_no_cv"]["p_hat_gamma"],
        with_cv=pb_declared["combined_with_cv"]["p_hat_gamma"],
        sliding=pb_declared["combined_sliding"]["p_hat_gamma"],
    )

    def judge(empirical, declared_val):
        if empirical >= declared_val:
            return "CONFIRMED"
        elif empirical >= declared_val - 0.02:
            return "BORDERLINE"
        else:
            return "CONCERNING -- fell below Phase B's own number"

    comparison = {}
    for lens, key in (("no_cv", "combined_no_cv"), ("with_cv", "combined_with_cv"), ("sliding", "combined_sliding")):
        rates = [results[f"seed{s}"]["tier1_max"][key]["empirical_rate"] for s in (0, 1, 2)]
        comparison[lens] = dict(
            declared_p_hat_gamma=declared[lens],
            empirical_rates_by_seed=dict(zip(("seed0", "seed1", "seed2"), rates)),
            judgments=[judge(r, declared[lens]) for r in rates],
        )
    print(json.dumps(comparison, indent=2), flush=True)

    report = dict(
        status="COMPLETE", experiment="tdmpc2_walker_l3_phase_b_test1",
        n_new_rollouts=N_NEW, horizon=HORIZON, new_seed_base=NEW_SEED_BASE,
        regeneration_verification=regen_report,
        per_seed_tier_results=results,
        comparison_vs_phase_b_declared=comparison,
        calibration_touched=False, track_b_touched=False, sealed_data_touched=False,
    )
    (OUT / "result.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print("DONE", flush=True)
    return report


if __name__ == "__main__":
    run()
