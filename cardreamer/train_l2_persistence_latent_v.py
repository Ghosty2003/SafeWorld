"""Train a full-RSSM LPPM for ``F G(hazard_dist > 0)``.

This is a training/feasibility experiment, not formal calibration. It reads
only the biased, ``calibration_eligible=false`` latent pool and makes a
fingerprint-disjoint stratified split. The default ``paper_mean`` objective
implements Appendix C.1's P1/P2 hinge losses and latent-gradient penalty.
Finite-prefix shaping and CVaR remain available only as explicitly labelled
non-paper ablations.

Raw data and ``*.pt`` model files stay local and Git-ignored.  The JSON report
is intentionally versionable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from core.lppm.automaton import build_parity_automaton
from core.lppm.model import NeuralLPPM
from specs.ltl_specs import get_ltl_spec_by_id
from utils.spec_analysis import analyze_spec_structure


ETA = 0.01
SPEC_ID = "ltl_eventual_hazard_stability"
SCHEMA_VERSION = 1
STRATA = ("trap", "near_1m", "near_3m", "safe_control")


@dataclass(frozen=True)
class LatentPool:
    hazard: np.ndarray
    deter: np.ndarray
    stoch: np.ndarray
    fingerprints: np.ndarray
    manifest: dict[str, Any]
    manifest_path: pathlib.Path

    def __len__(self) -> int:
        return len(self.hazard)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pool(
    root: pathlib.Path, required_calibration_eligible: bool | None = False
) -> LatentPool:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        required_calibration_eligible is not None
        and manifest.get("calibration_eligible") is not required_calibration_eligible
    ):
        role = "calibration" if required_calibration_eligible else "training-only"
        raise ValueError(f"refusing data not explicitly marked {role}")
    if manifest.get("spec") != SPEC_ID:
        raise ValueError(f"expected {SPEC_ID}, got {manifest.get('spec')!r}")

    hazards, deters, stochs, fingerprints = [], [], [], []
    for entry in manifest["batches"]:
        path = root / entry["name"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"batch hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            hazards.append(data["hazard_dist"].astype(np.float32, copy=False))
            deters.append(data["deter"].astype(np.float16, copy=False))
            stochs.append(data["stoch"].astype(np.float16, copy=False))
            fingerprints.append(data["fingerprints"].astype("U64"))
    hazard = np.concatenate(hazards)
    deter = np.concatenate(deters)
    stoch = np.concatenate(stochs)
    fingerprint = np.concatenate(fingerprints)
    unique_idx = unique_first_indices(fingerprint)
    pool = LatentPool(
        hazard=hazard[unique_idx],
        deter=deter[unique_idx],
        stoch=stoch[unique_idx],
        fingerprints=fingerprint[unique_idx],
        manifest=manifest,
        manifest_path=manifest_path,
    )
    if pool.hazard.shape[:2] != pool.deter.shape[:2] or pool.hazard.shape[:2] != pool.stoch.shape[:2]:
        raise ValueError("hazard/deter/stoch axes are not aligned")
    return pool


def unique_first_indices(fingerprints: np.ndarray) -> np.ndarray:
    """Keep the first occurrence of every trajectory fingerprint in order."""
    seen: set[str] = set()
    indices = []
    for i, value in enumerate(fingerprints.tolist()):
        if value not in seen:
            seen.add(value)
            indices.append(i)
    return np.asarray(indices, dtype=np.int64)


def hazard_stratum(hazard: np.ndarray) -> str:
    minimum = float(np.min(hazard))
    if minimum <= 0.0:
        return "trap"
    if minimum <= 1.0:
        return "near_1m"
    if minimum <= 3.0:
        return "near_3m"
    return "safe_control"


def stratified_split(
    hazard: np.ndarray, fingerprints: np.ndarray, seed: int, diagnostic_fraction: float
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, int]]]:
    rng = np.random.default_rng(seed)
    train, diagnostic = [], []
    counts: dict[str, dict[str, int]] = {}
    labels = np.asarray([hazard_stratum(row) for row in hazard])
    for stratum in STRATA:
        indices = np.flatnonzero(labels == stratum)
        indices = indices[rng.permutation(len(indices))]
        if len(indices) < 2:
            raise ValueError(f"stratum {stratum!r} needs at least two trajectories")
        n_diagnostic = min(len(indices) - 1, max(1, int(round(len(indices) * diagnostic_fraction))))
        diagnostic.extend(indices[:n_diagnostic].tolist())
        train.extend(indices[n_diagnostic:].tolist())
        counts[stratum] = {
            "total": int(len(indices)),
            "train": int(len(indices) - n_diagnostic),
            "diagnostic": int(n_diagnostic),
        }
    train_idx = np.asarray(train, dtype=np.int64)
    diagnostic_idx = np.asarray(diagnostic, dtype=np.int64)
    if set(fingerprints[train_idx].tolist()) & set(fingerprints[diagnostic_idx].tolist()):
        raise AssertionError("fingerprint leakage between training and diagnostic splits")
    return train_idx, diagnostic_idx, counts


def stabilization_targets(hazard: np.ndarray, eta: float = ETA) -> np.ndarray:
    """Empirical remaining-bad-visit target for optional shaping ablations.

    Appendix C.7's automaton starts in even ``p``. A violation observed at
    time t places the product state at time t+1 in odd ``not_p``; P2 therefore
    charges that next transition. The target counts those future charged
    states. The paper-faithful ``paper_mean`` objective does not use it.
    """
    hazard = np.asarray(hazard)
    if hazard.ndim != 2:
        raise ValueError("hazard must have shape [trajectory, time]")
    n, horizon = hazard.shape
    targets = np.zeros((n, horizon), dtype=np.float32)
    charged = np.zeros((n, horizon), dtype=np.float32)
    charged[:, 1:] = hazard[:, :-1] <= 0.0
    targets = eta * np.flip(
        np.cumsum(np.flip(charged, axis=1), axis=1), axis=1
    )
    return targets


def automaton_trace(hazard: np.ndarray, dpa) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_to_idx = {state: i for i, state in enumerate(dpa.states)}
    n, horizon = hazard.shape
    q_curr = np.empty((n, horizon), dtype=np.int64)
    q_after = np.empty((n, horizon), dtype=np.int64)
    priorities = np.empty((n, horizon), dtype=np.int64)
    for i in range(n):
        q = dpa.initial
        for t in range(horizon):
            q_curr[i, t] = state_to_idx[q]
            active = frozenset({"hazard_dist"}) if hazard[i, t] > 0.0 else frozenset()
            q_next, priority = dpa.step_with_priority(q, active)
            q_after[i, t] = state_to_idx[q_next]
            priorities[i, t] = priority
            q = q_next
    return q_curr, q_after, priorities


def assemble_features(pool: LatentPool, include_hazard: bool = False) -> np.ndarray:
    n, horizon = pool.hazard.shape
    deter = pool.deter.reshape(n, horizon, -1).astype(np.float32)
    stoch = pool.stoch.reshape(n, horizon, -1).astype(np.float32)
    parts = (deter, stoch, pool.hazard[..., None]) if include_hazard else (deter, stoch)
    return np.concatenate(parts, axis=-1)


def fit_normalizer(features: np.ndarray, train_idx: np.ndarray, min_scale: float):
    selected = features[train_idx].reshape(-1, features.shape[-1])
    mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = selected.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, min_scale)
    return mean, scale


def normalize_features(features: np.ndarray, mean: np.ndarray, scale: np.ndarray, clip: float):
    normalized = (features - mean[None, None, :]) / scale[None, None, :]
    return np.clip(normalized, -clip, clip).astype(np.float32, copy=False)


def transition_arrays(indices: np.ndarray, q_curr: np.ndarray, priorities: np.ndarray):
    horizon = q_curr.shape[1]
    trajectory = np.repeat(indices, horizon - 1)
    time_idx = np.tile(np.arange(horizon - 1), len(indices))
    return (
        trajectory,
        time_idx,
        q_curr[trajectory, time_idx],
        q_curr[trajectory, time_idx + 1],
        priorities[trajectory, time_idx],
    )


def _rates(vc: np.ndarray, vn: np.ndarray, priorities: np.ndarray, eta: float):
    # Definition 5.2 P1 applies to every admissible product transition.
    # P2 adds the eta decrease on transitions sourced at odd priority.
    p1 = np.ones_like(priorities, dtype=bool)
    p2 = priorities == 1
    p1_pass = (vn[p1] <= vc[p1]) if np.any(p1) else np.asarray([], dtype=bool)
    p2_pass = (vc[p2] - vn[p2] >= eta) if np.any(p2) else np.asarray([], dtype=bool)
    return p1, p2, p1_pass, p2_pass


def _predict_states(model, features, q, indices, batch_size):
    values = []
    flat_x = features[indices].reshape(-1, features.shape[-1])
    flat_q = q[indices].reshape(-1)
    with torch.no_grad():
        for start in range(0, len(flat_x), batch_size):
            x = torch.from_numpy(flat_x[start : start + batch_size])
            qi = torch.from_numpy(flat_q[start : start + batch_size])
            values.append(model(x, qi)[:, 0].cpu().numpy())
    return np.concatenate(values).reshape(len(indices), features.shape[1])


def _predict_pairs(model, features: np.ndarray, q: np.ndarray, batch_size: int) -> np.ndarray:
    values = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            x = torch.from_numpy(features[start : start + batch_size])
            qi = torch.from_numpy(q[start : start + batch_size])
            values.append(model(x, qi)[:, 0].cpu().numpy())
    return np.concatenate(values)


def diagnose(model, features, targets, q_curr, q_after, priorities, indices, dpa, args):
    values = _predict_states(model, features, q_curr, indices, args.eval_batch_size)
    vc = values[:, :-1].reshape(-1)
    vn = values[:, 1:].reshape(-1)
    pr = priorities[indices, :-1].reshape(-1)
    p1, p2, p1_pass, p2_pass = _rates(vc, vn, pr, ETA)

    state_metrics = {}
    state_to_idx = {state: i for i, state in enumerate(dpa.states)}
    flat_values = values.reshape(-1)
    flat_q = q_curr[indices].reshape(-1)
    flat_targets = targets[indices].reshape(-1)
    for state, state_idx in state_to_idx.items():
        mask = flat_q == state_idx
        observed = flat_values[mask]
        target = flat_targets[mask]
        corr = None
        if len(observed) > 1 and np.std(observed) > 0 and np.std(target) > 0:
            corr = float(np.corrcoef(observed, target)[0, 1])
        state_metrics[state] = {
            "n": int(np.sum(mask)),
            "v_min": float(np.min(observed)),
            "v_max": float(np.max(observed)),
            "v_range": float(np.ptp(observed)),
            "v_std": float(np.std(observed)),
            "target_correlation": corr,
        }

    per_path_pass = []
    zfree_source_paths = 0
    zfree_closure_pass = 0
    for local_i, trajectory_idx in enumerate(indices):
        path_v = values[local_i]
        path_pr = priorities[trajectory_idx]
        p1_ok = np.all(path_v[1:] <= path_v[:-1])
        p2_ok = np.all(
            path_v[:-1][path_pr[:-1] == 1] - path_v[1:][path_pr[:-1] == 1] >= ETA
        )
        # The stored H latent states define H-1 real product transitions. The
        # paper's endpoint is therefore (z[H-1], q[H-1]); inventing a second
        # automaton transition on z[H-1] without a physical successor would
        # not be a product transition from the sampled rollout.
        final_zfree = path_v[-1] < ETA
        per_path_pass.append(bool(p1_ok and p2_ok and final_zfree))

        source_mask = path_v[:-1] < ETA
        if np.any(source_mask):
            zfree_source_paths += 1
            local_p1 = path_v[1:] <= path_v[:-1]
            local_p2 = np.logical_or(path_pr[:-1] != 1, path_v[:-1] - path_v[1:] >= ETA)
            if np.all(np.logical_or(~source_mask, local_p1 & local_p2)):
                zfree_closure_pass += 1

    minimum_range = min(item["v_range"] for item in state_metrics.values())
    minimum_std = min(item["v_std"] for item in state_metrics.values())
    noncollapsed = minimum_range >= ETA and minimum_std >= ETA / 10.0
    p1_rate = float(np.mean(p1_pass)) if len(p1_pass) else 1.0
    p2_rate = float(np.mean(p2_pass)) if len(p2_pass) else 1.0
    path_rate = float(np.mean(per_path_pass))
    positive_p1_delta = np.maximum(vn[p1] - vc[p1], 0.0)
    p2_shortfall = np.maximum(ETA - (vc[p2] - vn[p2]), 0.0)
    closure_rate = zfree_closure_pass / zfree_source_paths if zfree_source_paths else None
    gate = {
        "noncollapsed": noncollapsed,
        "p1": p1_rate >= args.gate_p1,
        "p2": p2_rate >= args.gate_p2,
        "pathwise": path_rate >= args.gate_pathwise,
        "zfree_coverage": zfree_source_paths > 0,
        "zfree_closure": closure_rate is not None and closure_rate >= args.gate_zfree_closure,
    }
    return {
        "n_trajectories": int(len(indices)),
        "n_transitions": int(len(vc)),
        "n_p1_transitions": int(np.sum(p1)),
        "n_p2_transitions": int(np.sum(p2)),
        "p1_transition_pass_rate": p1_rate,
        "p1_positive_delta_p50": float(np.percentile(positive_p1_delta, 50)),
        "p1_positive_delta_p90": float(np.percentile(positive_p1_delta, 90)),
        "p1_positive_delta_p99": float(np.percentile(positive_p1_delta, 99)),
        "p1_positive_delta_max": float(np.max(positive_p1_delta)),
        "p2_transition_pass_rate": p2_rate,
        "p2_shortfall_p90": float(np.percentile(p2_shortfall, 90)),
        "p2_shortfall_max": float(np.max(p2_shortfall)),
        "pathwise_pass_count": int(np.sum(per_path_pass)),
        "pathwise_pass_rate": path_rate,
        "zfree_source_trajectories": zfree_source_paths,
        "zfree_closure_pass_count": zfree_closure_pass,
        "zfree_closure_pass_rate": closure_rate,
        "target_mse": float(np.mean((values - targets[indices]) ** 2)),
        "state_metrics": state_metrics,
        "noncollapse_thresholds": {"min_state_range": ETA, "min_state_std": ETA / 10.0},
        "noncollapsed": noncollapsed,
        "gate": gate,
        "gate_passed": all(gate.values()),
    }


def train_one(seed, features, targets, q_curr, q_after, priorities, train_idx, diagnostic_idx, dpa, args):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = NeuralLPPM(
        features.shape[-1], len(dpa.states), 1,
        hidden_dim=args.hidden_dim, q_embed_dim=args.q_embed_dim,
    )
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.objective == "paper_mean"
        else torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    )
    trajectory, time_idx, qc, qn, pr = transition_arrays(train_idx, q_curr, priorities)
    target_curr = targets[trajectory, time_idx]
    target_next = targets[trajectory, time_idx + 1]
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(args.epochs):
        if args.objective == "trajectory_cvar":
            order = rng.permutation(train_idx)
            step_size = args.trajectory_batch_size
        else:
            order = rng.permutation(len(trajectory))
            step_size = args.batch_size
        totals = np.zeros(6, dtype=np.float64)
        batches = 0
        model.train()
        for start in range(0, len(order), step_size):
            take = order[start : start + step_size]
            if args.objective == "trajectory_cvar":
                tr = take
                batch_n = len(tr)
                horizon = features.shape[1]
                x = torch.from_numpy(features[tr].reshape(-1, features.shape[-1]))
                tq = torch.from_numpy(q_curr[tr].reshape(-1))
                values = model(x, tq)[:, 0].reshape(batch_n, horizon)
                vc = values[:, :-1]
                vn = values[:, 1:]
                tpr = torch.from_numpy(priorities[tr, :-1])
                p1_mask = tpr < 1
                p2_mask = tpr == 1
                p1_hinge = F.relu(vn - vc + args.p1_training_margin)
                p2_hinge = F.relu(vn - vc + ETA + args.p2_training_slack)
                loss_p1 = p1_hinge[p1_mask].mean()
                loss_p2 = p2_hinge[p2_mask].mean()
                constraint_hinge = torch.where(p1_mask, p1_hinge, p2_hinge)
                loss_cvar = trajectory_cvar(constraint_hinge, args.cvar_alpha)
                loss_shape = F.smooth_l1_loss(
                    values, torch.from_numpy(targets[tr])
                )
                loss_smooth = torch.zeros((), dtype=values.dtype, device=values.device)
            else:
                tr = trajectory[take]
                ti = time_idx[take]
                x_curr = torch.from_numpy(features[tr, ti])
                if args.objective == "paper_mean":
                    x_curr.requires_grad_(True)
                x_next = torch.from_numpy(features[tr, ti + 1])
                tqc = torch.from_numpy(qc[take])
                tqn = torch.from_numpy(qn[take])
                tpr = torch.from_numpy(pr[take])
                y_curr = torch.from_numpy(target_curr[take])
                y_next = torch.from_numpy(target_next[take])
                vc = model(x_curr, tqc)
                vn = model(x_next, tqn)
                p1_mask = tpr < 1
                p2_mask = tpr == 1
                if args.objective == "paper_mean":
                    # Appendix C.1, Eqs. (10)--(12). P1 is averaged over all
                    # product transitions. P2 carries a bad-state indicator,
                    # so it is also normalized by |E_phi| rather than only by
                    # the number of bad transitions.
                    delta = vn[:, 0] - vc[:, 0]
                    loss_p1 = F.relu(delta).mean()
                    loss_p2 = (
                        p2_mask.to(delta.dtype) * F.relu(delta + ETA)
                    ).mean()
                    gradient = torch.autograd.grad(
                        vc.sum(), x_curr, create_graph=True
                    )[0]
                    loss_smooth = gradient.pow(2).sum(dim=-1).mean()
                    loss_shape = torch.zeros((), dtype=vc.dtype, device=vc.device)
                else:
                    loss_p1 = F.relu(
                        vn[p1_mask, 0] - vc[p1_mask, 0] + args.p1_training_margin
                    ).mean()
                    loss_p2 = F.relu(
                        vn[p2_mask, 0] - vc[p2_mask, 0] + ETA + args.p2_training_slack
                    ).mean()
                    loss_shape = F.smooth_l1_loss(
                        vc[:, 0], y_curr
                    ) + F.smooth_l1_loss(vn[:, 0], y_next)
                    loss_smooth = torch.zeros((), dtype=vc.dtype, device=vc.device)
                loss_cvar = torch.zeros((), dtype=vc.dtype, device=vc.device)
            if args.objective == "paper_mean":
                loss = loss_p1 + loss_p2 + args.lambda_smooth * loss_smooth
            else:
                loss = (
                    args.lambda_p1 * loss_p1
                    + args.lambda_p2 * loss_p2
                    + args.lambda_shape * loss_shape
                    + args.lambda_cvar * loss_cvar
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            totals += [
                float(loss.detach()), float(loss_p1.detach()),
                float(loss_p2.detach()), float(loss_shape.detach()),
                float(loss_cvar.detach()), float(loss_smooth.detach()),
            ]
            batches += 1
        if epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch + 1 == args.epochs:
            history.append({
                "epoch": epoch + 1,
                "total": totals[0] / batches,
                "p1": totals[1] / batches,
                "p2": totals[2] / batches,
                "shape": totals[3] / batches,
                "cvar": totals[4] / batches,
                "smooth": totals[5] / batches,
            })
    model.eval()
    return {
        "seed": seed,
        "history": history,
        "training": diagnose(model, features, targets, q_curr, q_after, priorities, train_idx, dpa, args),
        "diagnostic": diagnose(model, features, targets, q_curr, q_after, priorities, diagnostic_idx, dpa, args),
        "model": model,
    }


def trajectory_cvar(losses: torch.Tensor, alpha: float) -> torch.Tensor:
    """Mean of each trajectory's worst alpha fraction of transition losses."""
    if losses.ndim != 2:
        raise ValueError("trajectory CVaR losses must have shape [trajectory, transition]")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("CVaR alpha must lie in (0, 1]")
    k = max(1, int(np.ceil(alpha * losses.shape[1])))
    return torch.topk(losses, k=k, dim=1, largest=True).values.mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="artifacts/l2_persistence_latents")
    parser.add_argument("--output-dir", default="artifacts/l2_persistence_training")
    parser.add_argument("--split-seed", type=int, default=20260907)
    parser.add_argument("--diagnostic-fraction", type=float, default=0.20)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--objective", choices=("paper_mean", "transition_mean", "trajectory_cvar"),
        default="paper_mean",
    )
    parser.add_argument(
        "--include-hazard-feature", action="store_true",
        help="non-paper ablation: append decoded hazard_dist to the RSSM latent",
    )
    parser.add_argument("--trajectory-batch-size", type=int, default=32)
    parser.add_argument("--cvar-alpha", type=float, default=0.10)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--q-embed-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-p2", type=float, default=2.0)
    parser.add_argument("--lambda-p1", type=float, default=1.0)
    parser.add_argument("--lambda-shape", type=float, default=5.0)
    parser.add_argument("--lambda-cvar", type=float, default=10.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.01)
    parser.add_argument("--p1-training-margin", type=float, default=0.001)
    parser.add_argument("--p2-training-slack", type=float, default=0.005)
    parser.add_argument("--shape-step", type=float, default=0.015)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--min-scale", type=float, default=0.01)
    parser.add_argument("--normalization-clip", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--gate-p1", type=float, default=0.99)
    parser.add_argument("--gate-p2", type=float, default=0.90)
    parser.add_argument("--gate-pathwise", type=float, default=0.80)
    parser.add_argument("--gate-zfree-closure", type=float, default=0.80)
    args = parser.parse_args()
    if not 0.0 < args.diagnostic_fraction < 1.0:
        raise ValueError("--diagnostic-fraction must lie in (0, 1)")
    if args.shape_step < ETA:
        raise ValueError("--shape-step must be at least eta")
    if not 0.0 < args.cvar_alpha <= 1.0:
        raise ValueError("--cvar-alpha must lie in (0, 1]")

    root = pathlib.Path(args.data_root)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = load_pool(root)
    train_idx, diagnostic_idx, split_counts = stratified_split(
        pool.hazard, pool.fingerprints, args.split_seed, args.diagnostic_fraction
    )
    spec = get_ltl_spec_by_id(SPEC_ID)
    if spec is None:
        raise RuntimeError(f"{SPEC_ID} is not registered")
    spec["analysis"] = analyze_spec_structure(spec)
    dpa = build_parity_automaton(spec)
    q_curr, q_after, priorities = automaton_trace(pool.hazard, dpa)
    targets = stabilization_targets(pool.hazard, eta=args.shape_step)
    raw_features = assemble_features(pool, include_hazard=args.include_hazard_feature)
    mean, scale = fit_normalizer(raw_features, train_idx, args.min_scale)
    features = normalize_features(raw_features, mean, scale, args.normalization_clip)
    del raw_features

    print(f"pool={len(pool)} train={len(train_idx)} diagnostic={len(diagnostic_idx)} dim={features.shape[-1]}")
    print(json.dumps(split_counts, indent=2), flush=True)
    started = time.time()
    results = []
    for seed in [int(item) for item in args.seeds.split(",") if item.strip()]:
        print(f"Training seed={seed}", flush=True)
        result = train_one(
            seed, features, targets, q_curr, q_after, priorities,
            train_idx, diagnostic_idx, dpa, args,
        )
        diag = result["diagnostic"]
        print(
            f"  noncollapsed={diag['noncollapsed']} P1={diag['p1_transition_pass_rate']:.4f} "
            f"P2={diag['p2_transition_pass_rate']:.4f} path={diag['pathwise_pass_rate']:.4f} "
            f"gate={diag['gate_passed']}", flush=True,
        )
        results.append(result)

    def score(item):
        diag = item["diagnostic"]
        return (int(diag["gate_passed"]), int(diag["noncollapsed"]), diag["pathwise_pass_rate"], diag["p2_transition_pass_rate"], diag["p1_transition_pass_rate"])

    best = max(results, key=score)
    model_path = output_dir / "best_lppm.pt"
    feature_layout = {
        "deter": int(np.prod(pool.deter.shape[2:])),
        "stoch_flat": int(np.prod(pool.stoch.shape[2:])),
    }
    if args.include_hazard_feature:
        feature_layout["hazard_dist"] = 1
    torch.save({
        "backend": "torch_mlp",
        "weights": {key: value.detach().cpu() for key, value in best["model"].state_dict().items()},
        "feature_layout": feature_layout,
        "feature_mean": torch.from_numpy(mean),
        "feature_scale": torch.from_numpy(scale),
        "normalization_clip": args.normalization_clip,
        "state_to_idx": {state: i for i, state in enumerate(dpa.states)},
        "odd_to_idx": {1: 0},
        "hidden_dim": args.hidden_dim,
        "q_embed_dim": args.q_embed_dim,
        "spec_id": SPEC_ID,
        "eta": ETA,
        "automaton_states": list(dpa.states),
        "automaton_initial": dpa.initial,
        "automaton_priority": dict(dpa.priority),
        "fit_fingerprints": pool.fingerprints[train_idx].tolist(),
        "diagnostic_fingerprints": pool.fingerprints[diagnostic_idx].tolist(),
        "training_data_manifest_sha256": sha256_file(pool.manifest_path),
        "training_only": True,
        "formal_calibration_completed": False,
    }, model_path)

    serializable = [{key: value for key, value in item.items() if key != "model"} for item in results]
    all_gated = all(item["diagnostic"]["gate_passed"] for item in results)
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "cardreamer_l2_persistence_full_rssm_training",
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "spec_id": SPEC_ID,
        "formula": "F G(hazard_dist > 0)",
        "eta": ETA,
        "automaton": {
            "states": list(dpa.states),
            "initial": dpa.initial,
            "priority": dict(dpa.priority),
            "semantics": (
                "safe label -> p; bad label -> not_p; neither state is absorbing"
            ),
        },
        "verdict": "READY_FOR_FRESH_CALIBRATION" if all_gated else "TRAINING_DIAGNOSTIC_ONLY",
        "formal_calibration_completed": False,
        "calibration_launched": False,
        "calibration_hold_reason": None if all_gated else "one or more predeclared diagnostic gates failed",
        "data_manifest": {"path": str(pool.manifest_path), "sha256": sha256_file(pool.manifest_path)},
        "feature_dim": int(features.shape[-1]),
        "feature_layout": feature_layout,
        "paper_state_input": "RSSM deter + flattened categorical stochastic state",
        "decoded_hazard_appended_to_v": args.include_hazard_feature,
        "normalization_fit": "training trajectories only",
        "split_seed": args.split_seed,
        "split_counts": split_counts,
        "train_size": int(len(train_idx)),
        "diagnostic_size": int(len(diagnostic_idx)),
        "gate_thresholds": {
            "p1": args.gate_p1, "p2": args.gate_p2,
            "pathwise": args.gate_pathwise,
            "zfree_closure": args.gate_zfree_closure,
        },
        "hyperparameters": {
            "objective": args.objective,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "trajectory_batch_size": args.trajectory_batch_size,
            "cvar_alpha": args.cvar_alpha,
            "hidden_dim": args.hidden_dim, "q_embed_dim": args.q_embed_dim,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "lambda_p1": args.lambda_p1, "lambda_p2": args.lambda_p2,
            "lambda_shape": args.lambda_shape, "lambda_cvar": args.lambda_cvar,
            "lambda_smooth": args.lambda_smooth,
            "p1_training_margin": args.p1_training_margin,
            "p2_training_slack": args.p2_training_slack,
            "shape_step": args.shape_step,
            "min_scale": args.min_scale, "normalization_clip": args.normalization_clip,
        },
        "transition_scope_note": "Training uses 49 real adjacent latent transitions; no synthetic terminal latent is invented.",
        "objective_scope_note": (
            "Appendix C.1 Eqs. (10)-(12) exactly: unweighted all-transition P1/P2 plus latent gradient penalty."
            if args.objective == "paper_mean"
            else "Non-paper stabilization/CVaR training ablation; P1/P2 remain explicit loss terms."
        ),
        "runs": serializable,
        "all_seeds_passed_gate": all_gated,
        "best_seed": best["seed"],
        "local_model": {"path": str(model_path.resolve()), "sha256": sha256_file(model_path), "git_ignored": True},
    }
    report_path = output_dir / "diagnostics.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": report["verdict"], "all_seeds_passed_gate": all_gated,
        "best_seed": best["seed"], "diagnostics": str(report_path),
        "local_model": str(model_path),
    }, indent=2))


if __name__ == "__main__":
    main()
