"""Retrain and diagnose CarDreamer's hazard-avoidance NeuralLPPM.

This is deliberately a training diagnostic, not Theorem-5.4 calibration. It
uses only the three ``calibration_eligible=false`` pools produced by
``collect_l2_hazard_training_data.py``. Trajectory fingerprints are deduplicated
globally, then every danger stratum is split into optimization and held-out
diagnostic subsets.

Only the AP declared by ``ltl_hazard_avoidance`` (``hazard_dist``) is fed to V;
the wrapper's unrelated sentinel-valued signals are excluded. Input
normalization is fit on optimization transitions only. Model weights are saved
as an ignored ``*.pt`` local artifact; JSON diagnostics are safe to version.
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

from core.lppm.automaton import build_parity_automaton
from core.lppm.loss import p1_loss, p2_loss, smoothness_penalty
from core.lppm.model import NeuralLPPM
from core.lppm.verifier import check_pathwise_conditions, iter_transitions, run_product_trajectory
from specs.ltl_specs import get_ltl_spec_by_id
from utils.spec_analysis import analyze_spec_structure


ETA = 0.01
FEATURE_KEYS = ("hazard_dist",)
DEFAULT_ROOTS = (
    "artifacts/l2_hazard_training_data",
    "artifacts/l2_hazard_training_data_exploration",
    "artifacts/l2_hazard_training_data_low_hazard",
)


@dataclass(frozen=True)
class Record:
    fingerprint: str
    trajectory: list[dict[str, float]]
    hazard_min: float
    source: str

    @property
    def stratum(self) -> str:
        if self.hazard_min <= 0.0:
            return "trap"
        if self.hazard_min <= 1.0:
            return "near_1m"
        if self.hazard_min <= 3.0:
            return "near_3m"
        return "safe_control"


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(roots: list[pathlib.Path]) -> tuple[list[Record], list[dict[str, Any]]]:
    records: dict[str, Record] = {}
    manifests: list[dict[str, Any]] = []
    for root in roots:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("calibration_eligible") is not False:
            raise ValueError(f"refusing pool not explicitly marked training-only: {root}")
        manifests.append(
            {
                "path": str(manifest_path),
                "sha256": _sha256_file(manifest_path),
                "action_source": manifest["action_source"],
                "anchor_strategy": manifest.get("replay_anchor_strategy", "uniform"),
            }
        )
        for batch in manifest["batches"]:
            path = root / batch["name"]
            if _sha256_file(path) != batch["sha256"]:
                raise ValueError(f"batch hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as data:
                keys = data["ap_keys"].astype("U32").tolist()
                hazard_idx = keys.index("hazard_dist")
                values = data["values"].astype(np.float32, copy=False)
                fingerprints = data["fingerprints"].astype("U64")
                for i, fingerprint in enumerate(fingerprints.tolist()):
                    if fingerprint in records:
                        continue
                    hazard = values[i, :, hazard_idx]
                    records[fingerprint] = Record(
                        fingerprint=fingerprint,
                        trajectory=[{"hazard_dist": float(value)} for value in hazard],
                        hazard_min=float(np.min(hazard)),
                        source=root.name,
                    )
    return list(records.values()), manifests


def stratified_split(
    records: list[Record], seed: int, diagnostic_fraction: float, max_per_safe_stratum: int
) -> tuple[list[Record], list[Record], dict[str, dict[str, int]]]:
    rng = np.random.default_rng(seed)
    train: list[Record] = []
    diagnostic: list[Record] = []
    counts: dict[str, dict[str, int]] = {}
    for stratum in ("trap", "near_1m", "near_3m", "safe_control"):
        items = [record for record in records if record.stratum == stratum]
        order = rng.permutation(len(items))
        items = [items[i] for i in order]
        if stratum in {"near_3m", "safe_control"}:
            items = items[:max_per_safe_stratum]
        n_diagnostic = max(1, int(round(len(items) * diagnostic_fraction)))
        diagnostic.extend(items[:n_diagnostic])
        train.extend(items[n_diagnostic:])
        counts[stratum] = {
            "available_unique": sum(record.stratum == stratum for record in records),
            "selected": len(items),
            "train": len(items) - n_diagnostic,
            "diagnostic": n_diagnostic,
        }
    if {record.fingerprint for record in train} & {
        record.fingerprint for record in diagnostic
    }:
        raise AssertionError("fingerprint leakage between training and diagnostic splits")
    return train, diagnostic, counts


def _transitions(records: list[Record], dpa, spec) -> list[tuple[Any, Any]]:
    result = []
    for record in records:
        result.extend(iter_transitions(run_product_trajectory(record.trajectory, dpa, spec), dpa))
    return result


def _tensorize(transitions, mean: float, scale: float, dpa):
    z_curr = torch.tensor(
        [[(float(curr.z["hazard_dist"]) - mean) / scale] for curr, _ in transitions],
        dtype=torch.float32,
    )
    z_next = torch.tensor(
        [[(float(nxt.z["hazard_dist"]) - mean) / scale] for _, nxt in transitions],
        dtype=torch.float32,
    )
    state_to_idx = {state: i for i, state in enumerate(dpa.states)}
    q_curr = torch.tensor([state_to_idx[curr.q] for curr, _ in transitions], dtype=torch.long)
    q_next = torch.tensor([state_to_idx[nxt.q] for _, nxt in transitions], dtype=torch.long)
    priorities = torch.tensor([curr.priority for curr, _ in transitions], dtype=torch.long)
    return z_curr, z_next, q_curr, q_next, priorities, state_to_idx


def _predict(model, hazards: np.ndarray, q: str, mean: float, scale: float, state_to_idx):
    z = torch.tensor(((hazards - mean) / scale)[:, None], dtype=torch.float32)
    q_idx = torch.full((len(z),), state_to_idx[q], dtype=torch.long)
    with torch.no_grad():
        return model(z, q_idx)[:, 0].cpu().numpy()


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def diagnose(model, records: list[Record], mean: float, scale: float, dpa, spec, state_to_idx):
    transitions = _transitions(records, dpa, spec)
    zc, zn, qc, qn, priorities, _ = _tensorize(transitions, mean, scale, dpa)
    with torch.no_grad():
        vc = model(zc, qc)[:, 0].cpu().numpy()
        vn = model(zn, qn)[:, 0].cpu().numpy()
    priority_np = priorities.numpy()
    p1_mask = priority_np < 1
    p2_mask = priority_np == 1
    p1_delta = vn[p1_mask] - vc[p1_mask]
    p2_margin = vc[p2_mask] - vn[p2_mask]

    params = {
        "backend": "torch_mlp",
        "weights": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "feature_keys": list(FEATURE_KEYS),
        "feature_mean": [mean],
        "feature_scale": [scale],
        "state_to_idx": state_to_idx,
        "odd_to_idx": {1: 0},
        "hidden_dim": 128,
        "q_embed_dim": 16,
    }
    pathwise = [
        check_pathwise_conditions(
            run_product_trajectory(record.trajectory, dpa, spec),
            dpa,
            spec,
            eta=ETA,
            lppm_params=params,
        )
        for record in records
    ]
    hazards = np.asarray(
        [step["hazard_dist"] for record in records for step in record.trajectory],
        dtype=np.float32,
    )
    grid = np.linspace(-0.25, 32.0, 257, dtype=np.float32)
    by_state = {}
    for q in dpa.states:
        grid_v = _predict(model, grid, q, mean, scale, state_to_idx)
        observed_v = _predict(model, hazards, q, mean, scale, state_to_idx)
        by_state[q] = {
            "grid_v_min": float(np.min(grid_v)),
            "grid_v_max": float(np.max(grid_v)),
            "grid_v_range": float(np.ptp(grid_v)),
            "grid_v_std": float(np.std(grid_v)),
            "grid_hazard_correlation": _pearson(grid, grid_v),
            "observed_v_std": float(np.std(observed_v)),
            "observed_hazard_correlation": _pearson(hazards, observed_v),
        }
    min_state_range = min(metrics["grid_v_range"] for metrics in by_state.values())
    min_state_std = min(metrics["grid_v_std"] for metrics in by_state.values())
    noncollapsed = min_state_range >= ETA and min_state_std >= ETA / 10.0
    return {
        "n_trajectories": len(records),
        "n_transitions": len(transitions),
        "n_p1_transitions": int(np.sum(p1_mask)),
        "n_p2_transitions": int(np.sum(p2_mask)),
        "p1_transition_pass_rate": float(np.mean(p1_delta <= 0.0)),
        "p1_positive_delta_p99": float(np.percentile(np.maximum(p1_delta, 0.0), 99)),
        "p2_transition_pass_rate": float(np.mean(p2_margin >= ETA)) if len(p2_margin) else None,
        "p2_margin_mean": float(np.mean(p2_margin)) if len(p2_margin) else None,
        "pathwise_pass": int(sum(result.satisfied for result in pathwise)),
        "state_metrics": by_state,
        "noncollapse_thresholds": {"min_grid_range": ETA, "min_grid_std": ETA / 10.0},
        "noncollapsed": noncollapsed,
    }


def train_one(seed: int, train_records: list[Record], diagnostic_records: list[Record], dpa, spec, args):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transitions = _transitions(train_records, dpa, spec)
    raw_curr = np.asarray([float(curr.z["hazard_dist"]) for curr, _ in transitions])
    mean = float(np.mean(raw_curr))
    scale = float(max(np.std(raw_curr), 1e-6))
    zc, zn, qc, qn, priorities, state_to_idx = _tensorize(
        transitions, mean, scale, dpa
    )
    model = NeuralLPPM(1, len(dpa.states), 1, hidden_dim=128, q_embed_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    odd_to_idx = {1: 0}
    history = []
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        zc_epoch = zc.clone().requires_grad_(True)
        vc = model(zc_epoch, qc)
        vn = model(zn, qn)
        loss_p1 = p1_loss(vc, vn, priorities, odd_to_idx)
        loss_p2 = p2_loss(vc, vn, priorities, odd_to_idx, ETA)
        loss_smooth = smoothness_penalty(vc, zc_epoch)
        loss = loss_p1 + args.lambda_p2 * loss_p2 + args.lambda_smooth * loss_smooth
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 100 == 0 or epoch + 1 == args.epochs:
            history.append(
                {
                    "epoch": epoch + 1,
                    "total": float(loss.detach()),
                    "p1": float(loss_p1.detach()),
                    "p2": float(loss_p2.detach()),
                    "smooth": float(loss_smooth.detach()),
                }
            )
    model.eval()
    return {
        "seed": seed,
        "feature_mean": mean,
        "feature_scale": scale,
        "history": history,
        "training": diagnose(model, train_records, mean, scale, dpa, spec, state_to_idx),
        "diagnostic": diagnose(
            model, diagnostic_records, mean, scale, dpa, spec, state_to_idx
        ),
        "model": model,
        "state_to_idx": state_to_idx,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", action="append", dest="data_roots")
    parser.add_argument("--output-dir", default="artifacts/l2_hazard_retraining")
    parser.add_argument("--split-seed", type=int, default=20260907)
    parser.add_argument("--diagnostic-fraction", type=float, default=0.20)
    parser.add_argument("--max-per-safe-stratum", type=int, default=200)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-p2", type=float, default=1.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.01)
    args = parser.parse_args()
    if not 0.0 < args.diagnostic_fraction < 1.0:
        raise ValueError("--diagnostic-fraction must lie in (0, 1)")
    roots = [pathlib.Path(path) for path in (args.data_roots or DEFAULT_ROOTS)]
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, manifests = load_records(roots)
    train_records, diagnostic_records, split_counts = stratified_split(
        records, args.split_seed, args.diagnostic_fraction, args.max_per_safe_stratum
    )
    spec = get_ltl_spec_by_id("ltl_hazard_avoidance")
    if spec is None:
        raise RuntimeError("ltl_hazard_avoidance is not registered")
    spec["analysis"] = analyze_spec_structure(spec)
    dpa = build_parity_automaton(spec)

    print(f"Unique pool={len(records)} train={len(train_records)} diagnostic={len(diagnostic_records)}")
    print(json.dumps(split_counts, indent=2))
    started = time.time()
    results = []
    for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
        print(f"Training seed={seed}", flush=True)
        result = train_one(seed, train_records, diagnostic_records, dpa, spec, args)
        print(
            f"  diagnostic noncollapsed={result['diagnostic']['noncollapsed']} "
            f"P1={result['diagnostic']['p1_transition_pass_rate']:.4f} "
            f"P2={result['diagnostic']['p2_transition_pass_rate']}",
            flush=True,
        )
        results.append(result)

    def score(result):
        diag = result["diagnostic"]
        return (
            int(diag["noncollapsed"]),
            diag["p1_transition_pass_rate"],
            diag["p2_transition_pass_rate"] or 0.0,
        )

    best = max(results, key=score)
    model_path = output_dir / "best_lppm.pt"
    torch.save(
        {
            "backend": "torch_mlp",
            "weights": {k: v.detach().cpu() for k, v in best["model"].state_dict().items()},
            "feature_keys": list(FEATURE_KEYS),
            "feature_mean": [best["feature_mean"]],
            "feature_scale": [best["feature_scale"]],
            "state_to_idx": best["state_to_idx"],
            "odd_to_idx": {1: 0},
            "hidden_dim": 128,
            "q_embed_dim": 16,
            "training_only": True,
            "formal_calibration_completed": False,
        },
        model_path,
    )
    serializable_results = []
    for result in results:
        serializable_results.append(
            {key: value for key, value in result.items() if key not in {"model", "state_to_idx"}}
        )
    report = {
        "experiment": "cardreamer_l2_hazard_v_retraining_diagnostic",
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "formal_calibration_completed": False,
        "verdict": "TRAINING_DIAGNOSTIC_ONLY",
        "data_manifests": manifests,
        "unique_pool_size": len(records),
        "train_size": len(train_records),
        "diagnostic_size": len(diagnostic_records),
        "split_counts": split_counts,
        "split_seed": args.split_seed,
        "eta": ETA,
        "feature_keys": list(FEATURE_KEYS),
        "hyperparameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "lambda_p2": args.lambda_p2,
            "lambda_smooth": args.lambda_smooth,
        },
        "runs": serializable_results,
        "all_seeds_noncollapsed": all(
            result["diagnostic"]["noncollapsed"] for result in results
        ),
        "best_seed": best["seed"],
        "local_model": {
            "path": str(model_path.resolve()),
            "sha256": _sha256_file(model_path),
            "git_ignored": True,
        },
    }
    report_path = output_dir / "diagnostics.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_seeds_noncollapsed": report["all_seeds_noncollapsed"],
        "best_seed": report["best_seed"],
        "diagnostics": str(report_path),
        "local_model": str(model_path),
    }, indent=2))


if __name__ == "__main__":
    main()
