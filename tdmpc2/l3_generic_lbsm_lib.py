"""Shared library for L3 statistical-LBSM pipelines across carriers
(walker-walk precedent, generalized to cheetah-run / walker-run). All
functions are parameterized by a CARRIER config dict:

    dict(checkpoint=str, task=str, quantity_fn=callable(physics)->float,
         action_dim=int or None (looked up at runtime if None))

No frozen artifact modified anywhere. No sealed calibration data touched.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from scipy.signal import find_peaks
from scipy.stats import beta
from sklearn.linear_model import Ridge

from wrappers.tdmpc2_wrapper import TDMPC2Wrapper

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


def load_wrapper(checkpoint, task, seed=1):
    wrapper = TDMPC2Wrapper()
    wrapper.load(checkpoint=checkpoint, task=task, seed=seed)
    if str(wrapper._agent.device).split(":")[0] != "cuda":
        wrapper.close()
        raise RuntimeError("CUDA required")
    return wrapper


def collect_real_episodes(cfg, n_episodes, horizon, seed_base):
    """Real closed-loop episodes: encode(obs)->plan_action->env.step->encode(new obs).
    Returns z (n,T,512), q (n,T) ground-truth quantity."""
    wrapper = load_wrapper(cfg["checkpoint"], cfg["task"])
    torch.manual_seed(seed_base); np.random.seed(seed_base)
    all_z, all_q = [], []
    try:
        for i in range(n_episodes):
            obs = wrapper._env.reset()
            z = wrapper.encode(obs)
            z_seq, q_seq = [], []
            with torch.no_grad():
                for step in range(horizon):
                    a = wrapper.plan_action(z, t0=(step == 0), eval_mode=True)
                    obs, _r, done, _info = wrapper._env.step(a[0].detach().cpu())
                    z = wrapper.encode(obs)
                    z_seq.append(z[0].detach().cpu().numpy().astype(np.float64, copy=True))
                    q_seq.append(cfg["quantity_fn"](wrapper._physics))
                    if done:
                        break
            if len(z_seq) == horizon:
                all_z.append(np.stack(z_seq))
                all_q.append(np.array(q_seq))
            if (i + 1) % 10 == 0:
                print(f"    real episode {i + 1}/{n_episodes}", flush=True)
    finally:
        wrapper.close()
    return np.stack(all_z), np.stack(all_q)


def generate_imagination_rollouts(cfg, n_rollouts, horizon, seed_base):
    """Pure imagination: real env.reset() anchor, then encode once + next(z,a) only."""
    wrapper = load_wrapper(cfg["checkpoint"], cfg["task"])
    torch.manual_seed(seed_base); np.random.seed(seed_base)
    all_z = []
    try:
        for i in range(n_rollouts):
            obs = wrapper._env.reset()
            z = wrapper.encode(obs)
            z_seq = []
            with torch.no_grad():
                for step in range(horizon):
                    a = wrapper.plan_action(z, t0=(step == 0), eval_mode=True)
                    z = wrapper.next(z, a)
                    z_seq.append(z[0].detach().cpu().numpy().astype(np.float64, copy=True))
            all_z.append(np.stack(z_seq))
            if (i + 1) % 50 == 0:
                print(f"    imagined {i + 1}/{n_rollouts}", flush=True)
    finally:
        wrapper.close()
    return np.stack(all_z)


def manufacture_fall_trajectories(cfg, n_target, horizon, t_switch_range, fall_thresh, seed_base, max_attempts=20000):
    """Real closed-loop gait prelude -> saturated all-negative action ->
    ground-truth 'fallen and stayed down' check. Returns z, q, t_switch,
    discard_rate."""
    wrapper = load_wrapper(cfg["checkpoint"], cfg["task"])
    action_dim = cfg.get("action_dim") or wrapper._agent.cfg.action_dim
    all_negative = torch.tensor(-np.ones(action_dim, dtype=np.float32))
    rng = np.random.default_rng(seed_base)
    accepted, discarded, attempts = [], 0, 0
    try:
        while len(accepted) < n_target and attempts < max_attempts:
            attempts += 1
            t_switch = int(rng.integers(t_switch_range[0], t_switch_range[1] + 1))
            obs = wrapper._env.reset()
            z = wrapper.encode(obs)
            z_seq, q_seq = [], []
            with torch.no_grad():
                for step in range(horizon):
                    if step < t_switch:
                        a = wrapper.plan_action(z, t0=(step == 0), eval_mode=True)
                        a_apply = a[0].detach().cpu()
                    else:
                        a_apply = all_negative
                    obs, _r, done, _info = wrapper._env.step(a_apply)
                    z = wrapper.encode(obs)
                    z_seq.append(z[0].detach().cpu().numpy().astype(np.float64, copy=True))
                    q_seq.append(cfg["quantity_fn"](wrapper._physics))
                    if done:
                        break
            if len(q_seq) < horizon:
                discarded += 1
                continue
            q_arr = np.array(q_seq)
            is_failure = cfg["fall_check_fn"](q_arr, fall_thresh)
            if not is_failure:
                discarded += 1
                continue
            accepted.append(dict(z=np.stack(z_seq), q=q_arr, t_switch=t_switch))
            if len(accepted) % 10 == 0:
                print(f"    accepted {len(accepted)}/{n_target} (discarded {discarded}, attempts {attempts})", flush=True)
    finally:
        wrapper.close()
    discard_rate = discarded / attempts if attempts else 0.0
    z_all = np.stack([a["z"] for a in accepted])
    q_all = np.stack([a["q"] for a in accepted])
    t_switch_all = np.array([a["t_switch"] for a in accepted])
    return z_all, q_all, t_switch_all, discard_rate, attempts, discarded


def find_peaks_and_segments(q_seq, prominence, horizon):
    pk, _ = find_peaks(q_seq, prominence=prominence)
    segments = []
    start = 0
    for p in pk:
        segments.append(dict(start=start, end=int(p), is_tail=False))
        start = int(p)
    segments.append(dict(start=start, end=horizon, is_tail=True))
    return pk, segments


def build_training_pairs(z_arr, q_arr, prominence, horizon):
    Xs, Ys = [], []
    n = z_arr.shape[0]
    for i in range(n):
        pk, segs = find_peaks_and_segments(q_arr[i], prominence, horizon)
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
    xt = torch.tensor(z_fit, dtype=torch.float32, device=device)
    yt = torch.tensor(y_fit, dtype=torch.float32, device=device)
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


def evaluate_trajectory(g_init, delta, z_traj, q_traj, prominence, horizon):
    pk, segs = find_peaks_and_segments(q_traj, prominence, horizon)
    v = np.zeros(horizon)
    q = np.ones(horizon, dtype=np.int64)
    seg_reports = []
    device = next(g_init.parameters()).device
    with torch.no_grad():
        for seg in segs:
            s, e = seg["start"], seg["end"]
            if s >= horizon:
                continue
            z0 = torch.tensor(z_traj[s], dtype=torch.float32, device=device).unsqueeze(0)
            v0 = float(g_init(z0).item())
            v0_delta = v0 + delta
            length = e - s
            t_idx = np.arange(s, min(e, horizon))
            v_seg = v0_delta - ETA * (t_idx - s)
            v[s:min(e, horizon)] = v_seg
            if not seg["is_tail"]:
                if e < horizon:
                    q[e] = 0
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
        v_final=float(v[-1]),
    )


def deduction_verdict(q_traj, prominence, false_rate_cp_upper, m_min, cv_p95, horizon,
                      sliding_window=150, sliding_min_cycles=8):
    pk, _ = find_peaks(q_traj, prominence=prominence)
    m = len(pk)
    m_adj = m * (1 - false_rate_cp_upper)
    iv = np.diff(pk).astype(np.float64) if len(pk) > 1 else np.array([])
    cv = (iv.std() / iv.mean()) if len(iv) >= 2 and iv.mean() > 0 else np.inf
    verdict_no_cv = m_adj >= m_min
    verdict_with_cv = verdict_no_cv and (cv <= cv_p95)
    sliding_ok = True
    for w0 in range(0, horizon - sliding_window + 1, 10):
        seg_pk, _ = find_peaks(q_traj[w0:w0 + sliding_window], prominence=prominence)
        if len(seg_pk) * (1 - false_rate_cp_upper) < sliding_min_cycles:
            sliding_ok = False
            break
    return dict(m=m, m_adj=m_adj, cv=float(cv) if np.isfinite(cv) else None,
               verdict_no_cv=bool(verdict_no_cv), verdict_with_cv=bool(verdict_with_cv),
               verdict_sliding=bool(sliding_ok))


def fit_ridge_probe(z, q, n_holdout):
    n = q.shape[0]
    fit_idx = np.arange(n - n_holdout)
    return Ridge(alpha=10.0).fit(z[fit_idx].reshape(-1, z.shape[-1]), q[fit_idx].reshape(-1))
