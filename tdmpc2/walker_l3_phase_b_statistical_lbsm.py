"""L3 Phase B: statistical LBSM for GF(gait_cycle) on TD-MPC2 walker-walk.

Judgment layer: the deduction-scheme detector frozen exactly as validated in
Candidate 5/5b (prominence=0.03, false-rate CP-upper=0.07807546497166352,
M_MIN=15, CV_p95=0.4460545042500288, sliding-window=150/8) -- cited
verbatim from artifacts/l3_rescue_screen/candidate5_deduction_scheme/result.json,
NOT recomputed. The height probe (Ridge, real posterior z -> height) is
also reused frozen from that same pipeline (refit deterministically from
the identical tdmpc2_walker_lppm_latents fit split -- Ridge has no
randomness, so this reproduces bit-identically without needing a saved
weight file).

V construction (the genuinely NEW piece this round): a recurring
generalization of the CarDreamer Step7/8 / door-close Phase1-2 fixed-eta-
decrement recipe. Because GF is a RECURRING property (unlike a single
Guarantee-class goal-reach), g_init is trained on (z_t, y_t) pairs pooled
from EVERY peak-to-peak segment of every fit trajectory (not just t=0),
giving a genuine state-dependent potential evaluable at any point along a
trajectory -- this generalizes, rather than reuses verbatim, the established
single-reset math. y_t = ETA*(next_peak_time - t) within a segment, reset
at each detected peak (raw, prominence=0.03 -- the SAME peak-detection call
that also powers the deduction verdict, guaranteeing the "event source
consistency" reconciliation in item 3 holds by construction, checked and
reported explicitly rather than merely assumed). The trailing segment after
a trajectory's LAST peak has no known next-peak time within the 300-step
horizon; it is excluded from TRAINING targets (no ground truth) but its
transitions are still checked for soundness (V must not go negative) during
validation, exactly mirroring how prior work in this project has handled
"failed/incomplete" trajectory tails honestly rather than silently.

Two independent statistical corrections are combined, not conflated:
  (a) the V-regression's OWN conformal margin (Delta, 3 tiers: cal-max,
      cal-max*1.5, cal-max+0.5) -- fit fresh this round from this Phase's
      own fit/cal split, exactly like every prior Guarantee construction;
  (b) the deduction-scheme's pre-established false-detection-rate CP-upper
      discount on raw decoded PEAK COUNTS -- cited, not refit.
These answer different questions ((a): is V's prediction of segment length
calibrated; (b): how many of the raw decoded peaks are trustworthy at all)
and are applied at different layers (a: per-transition V soundness; b:
per-trajectory event-count admissibility).

Data: 500 FRESH real env.reset() anchors (NOT the c1_exact_mpc_anchors.npz
pool candidates 1/5/5b already used), seed_base=90000 -- disjoint from
every seed/anchor-pool used anywhere else in this project's TD-MPC2 walker
line (2000/5000/6000/9000/20000/50000/60000/70000, and the fixed 200-anchor
c1_exact_mpc_anchors.npz pool itself). 300 for fit/cal (anchor-disjoint
split within), 200 for validation. Horizon 300 throughout. Pure imagination
after the initial real anchor (wrapper.encode(obs) once, then
wrapper.next(z,a) only) -- no real MuJoCo physics during rollout generation.

No frozen artifact modified. No sealed/consumed calibration data touched.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from torch.nn import functional as F
from scipy.signal import find_peaks
from scipy.stats import beta
from sklearn.linear_model import Ridge

from wrappers.tdmpc2_wrapper import TDMPC2Wrapper

CAND5_DIR = pathlib.Path("artifacts/l3_rescue_screen/candidate5_deduction_scheme")
LATENTS_DIR = pathlib.Path("artifacts/tdmpc2_walker_lppm_latents")
OUT = pathlib.Path("artifacts/tdmpc2_walker_l3_statistical_lbsm/phase_b")
CHECKPOINT = "models/walker-walk-3.pt"
TASK = "walker-walk"
HORIZON = 300
N_FIT_CAL = 300
N_VAL = 200
SEED_BASE = 90000
PROMINENCE = 0.03
ETA = 0.01
GAMMA = 0.05
SAFE_THRESHOLD = 0.95
MAX_EPOCHS, CHECK_EVERY, BATCH_SIZE = 300, 10, 256


def cp_lower(k, n, conf=1 - GAMMA):
    if k == 0:
        return 0.0
    return float(beta.ppf(1 - conf, k, n - k + 1))


def cp_upper(k, n, conf=0.95):
    if n == 0:
        return None
    if k == n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


class BaseG(torch.nn.Module):
    def __init__(self, dim, mean, scale, hidden=(256, 256, 128)):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        layers, in_dim = [], dim
        for h in hidden:
            layers += [torch.nn.Linear(in_dim, h), torch.nn.ReLU()]
            in_dim = h
        self.mlp = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(in_dim, 1)

    def forward(self, x):
        xn = (x - self.mean) / self.scale
        return F.softplus(self.head(self.mlp(xn))).squeeze(-1)


def load_frozen_deduction_scheme():
    r = json.loads((CAND5_DIR / "result.json").read_text())
    return dict(false_rate_cp_upper=r["upper_bounds"]["false_rate_cp_upper_95"],
                m_min=r["M_MIN"], cv_p95=r["CV_p95_threshold"])


def fit_frozen_height_probe():
    seed_files = sorted(LATENTS_DIR.glob("seed_*_latents.npz"))
    all_h, all_z = [], []
    for f in seed_files:
        d = np.load(f)
        all_h.append(d["height_m"]); all_z.append(d["posterior_z"])
    h = np.concatenate(all_h, axis=0)
    z = np.concatenate(all_z, axis=0)
    n = h.shape[0]
    n_holdout = 300
    fit_idx = np.arange(n - n_holdout)
    return Ridge(alpha=10.0).fit(z[fit_idx].reshape(-1, 512), h[fit_idx].reshape(-1))


def find_peaks_and_segments(height_seq, prominence, horizon):
    pk, _ = find_peaks(height_seq, prominence=prominence)
    segments = []
    start = 0
    for p in pk:
        segments.append(dict(start=start, end=int(p), is_tail=False))
        start = int(p)
    segments.append(dict(start=start, end=horizon, is_tail=True))
    return pk, segments


def generate_rollouts(n_rollouts, seed_offset):
    wrapper = TDMPC2Wrapper()
    wrapper.load(checkpoint=CHECKPOINT, task=TASK, seed=1)
    if str(wrapper._agent.device).split(":")[0] != "cuda":
        wrapper.close()
        raise RuntimeError("CUDA required")
    torch.manual_seed(SEED_BASE + seed_offset)
    np.random.seed(SEED_BASE + seed_offset)
    all_z = []
    try:
        for i in range(n_rollouts):
            obs = wrapper._env.reset()
            z = wrapper.encode(obs)
            z_seq = []
            with torch.no_grad():
                for step in range(HORIZON):
                    a = wrapper.plan_action(z, t0=(step == 0), eval_mode=True)
                    z = wrapper.next(z, a)
                    z_seq.append(z[0].detach().cpu().numpy().astype(np.float64, copy=True))
            all_z.append(np.stack(z_seq))
            if (i + 1) % 50 == 0:
                print(f"    generated {i + 1}/{n_rollouts}", flush=True)
    finally:
        wrapper.close()
    return np.stack(all_z)


def build_training_pairs(z_arr, height_arr, prominence):
    Xs, Ys = [], []
    n = z_arr.shape[0]
    for i in range(n):
        pk, segs = find_peaks_and_segments(height_arr[i], prominence, HORIZON)
        for seg in segs:
            if seg["is_tail"]:
                continue
            for t in range(seg["start"], seg["end"]):
                Xs.append(z_arr[i, t])
                Ys.append(ETA * (seg["end"] - t))
    return np.stack(Xs).astype(np.float32), np.array(Ys, dtype=np.float32)


def train_g_init(z_fit, y_fit, z_cal, y_cal, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = z_fit.shape[-1]
    mean, scale = z_fit.mean(0), np.maximum(z_fit.std(0), 0.01)
    model = BaseG(dim, mean, scale).to(device)
    xt, yt = torch.tensor(z_fit, dtype=torch.float32, device=device), torch.tensor(y_fit, dtype=torch.float32, device=device)
    xc = torch.tensor(z_cal, dtype=torch.float32, device=device)
    n = len(xt)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_p95, best_state, best_epoch = np.inf, None, -1
    checkpoints = []
    for epoch in range(MAX_EPOCHS):
        order = torch.randperm(n)
        for s in range(0, n, BATCH_SIZE):
            idx = order[s:s + BATCH_SIZE]
            opt.zero_grad()
            loss = F.mse_loss(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        if (epoch + 1) % CHECK_EVERY == 0:
            model.eval()
            with torch.no_grad():
                pred_cal = model(xc).cpu().numpy()
            resid = y_cal - pred_cal
            p95 = float(np.percentile(resid, 95))
            checkpoints.append({"epoch": epoch + 1, "cal_p95": p95, "cal_max": float(resid.max())})
            if p95 < best_p95:
                best_p95, best_state, best_epoch = p95, {k: v.clone() for k, v in model.state_dict().items()}, epoch + 1
            model.train()
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        resid_final = y_cal - model(xc).cpu().numpy()
    return model, {
        "selected_epoch": best_epoch, "selected_cal_p95": best_p95,
        "calibration_residual_distribution": {
            "median": float(np.median(resid_final)), "p95": float(np.percentile(resid_final, 95)),
            "max": float(resid_final.max()), "min": float(resid_final.min()),
        },
        "n_calibration": len(y_cal), "checkpoints_tail": checkpoints[-5:],
    }


def evaluate_trajectory(g_init, delta, z_traj, height_traj, prominence,
                        false_rate_cp_upper, m_min, cv_p95):
    pk, segs = find_peaks_and_segments(height_traj, prominence, HORIZON)
    v = np.zeros(HORIZON)
    q = np.ones(HORIZON, dtype=np.int64)  # 1=bad/pending, 0=just-reached (only at peak indices)
    seg_reports = []
    with torch.no_grad():
        for seg in segs:
            s, e = seg["start"], seg["end"]
            if s >= HORIZON:
                continue
            z0 = torch.tensor(z_traj[s], dtype=torch.float32, device=next(g_init.parameters()).device).unsqueeze(0)
            v0 = float(g_init(z0).item())
            v0_delta = v0 + delta
            length = e - s
            t_idx = np.arange(s, min(e, HORIZON))
            v_seg = v0_delta - ETA * (t_idx - s)
            v[s:min(e, HORIZON)] = v_seg
            if not seg["is_tail"]:
                q[e - 1] = 1  # last pending step before reset
                if e < HORIZON:
                    q[e] = 0  # the reset instant itself
                zero_cross = bool((v_seg < -1e-9).any())
                closed_form_ok = v0_delta >= ETA * length
                full_sim_ok = not zero_cross
                seg_reports.append(dict(start=s, end=e, length=length, v0=v0, v0_delta=v0_delta,
                                        zero_cross=zero_cross, closed_form_ok=closed_form_ok,
                                        mismatch=(closed_form_ok != full_sim_ok)))
            else:
                seg_reports.append(dict(start=s, end=e, length=length, v0=v0, v0_delta=v0_delta,
                                        zero_cross=bool((v_seg < -1e-9).any()), is_tail=True))

    delta_v = np.diff(v)
    bad = q[:-1] == 1
    p1_ok = delta_v <= 1e-9
    p2_ok = delta_v <= -ETA + 1e-9
    trans_ok = np.where(bad, p2_ok, p1_ok)

    non_tail_segs = [s for s in seg_reports if not s.get("is_tail")]
    tail_seg = next((s for s in seg_reports if s.get("is_tail")), None)

    lbsm_pathwise_sound = (
        all(not s["zero_cross"] for s in non_tail_segs) and
        all(s["closed_form_ok"] for s in non_tail_segs) and
        not (tail_seg is not None and tail_seg["zero_cross"])
    )
    n_mismatch = sum(1 for s in non_tail_segs if s["mismatch"])

    source_in_zfree = v[:-1] < ETA
    n_zfree = int(source_in_zfree.sum())
    k_zfree = int(trans_ok[source_in_zfree].sum()) if n_zfree > 0 else 0

    return dict(
        m_raw_peaks=len(pk), n_segments_non_tail=len(non_tail_segs),
        lbsm_pathwise_sound=lbsm_pathwise_sound, n_closed_form_mismatch=n_mismatch,
        n_zfree_source=n_zfree, k_zfree_passing=k_zfree,
        p1_pass_rate=float(p1_ok[~bad].mean()) if (~bad).any() else None,
        p2_pass_rate=float(p2_ok[bad].mean()) if bad.any() else None,
        v_final=float(v[-1]),
    )


def deduction_verdict(height_traj, z_traj, probe, prominence, false_rate_cp_upper, m_min, cv_p95):
    h_pred = probe.predict(z_traj)
    pk, _ = find_peaks(h_pred, prominence=prominence)
    m = len(pk)
    m_adj = m * (1 - false_rate_cp_upper)
    iv = np.diff(pk).astype(np.float64) if len(pk) > 1 else np.array([])
    cv = (iv.std() / iv.mean()) if len(iv) >= 2 and iv.mean() > 0 else np.inf
    verdict_no_cv = m_adj >= m_min
    verdict_with_cv = verdict_no_cv and (cv <= cv_p95)
    sliding_ok = True
    for w0 in range(0, HORIZON - 150 + 1, 10):
        seg_pk, _ = find_peaks(h_pred[w0:w0 + 150], prominence=prominence)
        if len(seg_pk) * (1 - false_rate_cp_upper) < 8:
            sliding_ok = False
            break
    return dict(m=m, m_adj=m_adj, cv=float(cv) if np.isfinite(cv) else None,
               verdict_no_cv=bool(verdict_no_cv), verdict_with_cv=bool(verdict_with_cv),
               verdict_sliding=bool(sliding_ok), decoded_peaks=pk)


def run_one_seed(z_fit, y_fit, z_cal, y_cal, z_val, height_val, probe, params, seed, tag):
    g_init, train_report = train_g_init(z_fit, y_fit, z_cal, y_cal, seed=seed)
    cal_max = train_report["calibration_residual_distribution"]["max"]
    deltas = {"tier1_max": cal_max, "tier2_max_x1_5": cal_max * 1.5, "tier3_max_plus_0_5": cal_max + 0.5}

    tier_results = {}
    n = z_val.shape[0]
    for tier_name, delta in deltas.items():
        lbsm_rows, dedu_rows = [], []
        for i in range(n):
            lb = evaluate_trajectory(g_init, delta, z_val[i], height_val[i], PROMINENCE,
                                     params["false_rate_cp_upper"], params["m_min"], params["cv_p95"])
            dv = deduction_verdict(height_val[i], z_val[i], probe, PROMINENCE,
                                   params["false_rate_cp_upper"], params["m_min"], params["cv_p95"])
            lbsm_rows.append(lb)
            dedu_rows.append(dv)

        # event-source consistency: LBSM segment builder uses probe-DECODED height (via
        # the SAME height_val array used for deduction_verdict's own probe.predict call)
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
            delta=delta,
            lbsm_own_soundness_rate=float(lbsm_sound.mean()),
            n_closed_form_mismatch_total=int(sum(r["n_closed_form_mismatch"] for r in lbsm_rows)),
            zfree_gate_violated=bool(zfree_gate_violated),
            event_source_consistency=dict(n_consistent=n_consistent, n_total=n, rate=n_consistent / n),
            combined_no_cv=combined("verdict_no_cv"),
            combined_with_cv=combined("verdict_with_cv"),
            combined_sliding=combined("verdict_sliding"),
        )
        print(f"  [{tag}] {tier_name}: Delta={delta:.4f} lbsm_sound_rate={tier_results[tier_name]['lbsm_own_soundness_rate']:.3f} "
              f"event_src_consistency={n_consistent}/{n} "
              f"no_cv={tier_results[tier_name]['combined_no_cv']['k']}/{n} p_hat={tier_results[tier_name]['combined_no_cv']['p_hat_gamma']:.4f} "
              f"with_cv={tier_results[tier_name]['combined_with_cv']['k']}/{n} p_hat={tier_results[tier_name]['combined_with_cv']['p_hat_gamma']:.4f} "
              f"sliding={tier_results[tier_name]['combined_sliding']['k']}/{n} p_hat={tier_results[tier_name]['combined_sliding']['p_hat_gamma']:.4f}",
              flush=True)

    return dict(seed=seed, n_fit_pairs=len(y_fit), n_cal_pairs=len(y_cal),
               training_report=train_report, deltas=deltas, tier_results=tier_results)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    params = load_frozen_deduction_scheme()
    print("frozen deduction-scheme params (cited from Candidate 5, not recomputed):", json.dumps(params, indent=2), flush=True)
    probe = fit_frozen_height_probe()

    print(f"[1] generating {N_FIT_CAL} fit/cal rollouts ...", flush=True)
    z_fitcal = generate_rollouts(N_FIT_CAL, seed_offset=0)
    print(f"[2] generating {N_VAL} validation rollouts ...", flush=True)
    z_val = generate_rollouts(N_VAL, seed_offset=1)

    h_fitcal = probe.predict(z_fitcal.reshape(-1, 512)).reshape(N_FIT_CAL, HORIZON)
    h_val = probe.predict(z_val.reshape(-1, 512)).reshape(N_VAL, HORIZON)

    np.savez_compressed(OUT / "fitcal_data.npz", z=z_fitcal, h=h_fitcal)
    np.savez_compressed(OUT / "val_data.npz", z=z_val, h=h_val)

    print("[3] anchor-disjoint fit/cal split (67/33) ...", flush=True)
    rng = np.random.default_rng(0)
    idx = rng.permutation(N_FIT_CAL)
    n_cal = int(N_FIT_CAL * 0.33)
    cal_idx, fit_idx = idx[:n_cal], idx[n_cal:]

    print("[4] building recurring-segment training pairs from FIT trajectories ...", flush=True)
    z_fit_pairs, y_fit_pairs = build_training_pairs(z_fitcal[fit_idx], h_fitcal[fit_idx], PROMINENCE)
    z_cal_pairs, y_cal_pairs = build_training_pairs(z_fitcal[cal_idx], h_fitcal[cal_idx], PROMINENCE)
    print(f"    fit pairs={len(y_fit_pairs)}, cal pairs={len(y_cal_pairs)}", flush=True)

    results = {}
    for seed in (0, 1, 2):
        print(f"=== seed {seed} ===", flush=True)
        results[f"seed{seed}"] = run_one_seed(z_fit_pairs, y_fit_pairs, z_cal_pairs, y_cal_pairs,
                                              z_val, h_val, probe, params, seed, tag=f"seed{seed}")

    report = dict(
        status="COMPLETE", experiment="tdmpc2_walker_l3_phase_b_statistical_lbsm",
        frozen_deduction_scheme_params=params,
        n_fit_cal_rollouts=N_FIT_CAL, n_val_rollouts=N_VAL, horizon=HORIZON,
        seed_base=SEED_BASE, anchor_note="fresh env.reset() anchors, disjoint from every prior seed/anchor-pool in this project's walker line",
        n_fit_pairs=len(y_fit_pairs), n_cal_pairs=len(y_cal_pairs),
        seeds=results,
        calibration_touched=False, track_b_touched=False, sealed_data_touched=False,
    )
    (OUT / "result.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print("DONE", flush=True)
    return report


if __name__ == "__main__":
    main()
