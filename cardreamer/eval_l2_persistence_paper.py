"""Fail-closed paper-faithful evaluation for CarDreamer F G safe.

This runner keeps three facts separate: a fitted candidate LPPM, the sampled
finite-rollout indicator C(tau), and Theorem 5.4's support-wide P1/P2 premise
on every admissible transition sourced in Z_free. No warrant is emitted unless
the calibration manifest is explicitly eligible, calibration fingerprints are
disjoint from all fit/model-selection fingerprints, and an external support
verifier report is bound to the exact model SHA-256.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

import numpy as np
import torch

from cardreamer.train_l2_persistence_latent_v import (
    ETA,
    SPEC_ID,
    assemble_features,
    automaton_trace,
    load_pool,
    normalize_features,
    sha256_file,
)
from core.lppm.automaton import build_parity_automaton
from core.lppm.calibrator import _clopper_pearson_lower
from core.lppm.model import NeuralLPPM
from specs.ltl_specs import get_ltl_spec_by_id
from utils.spec_analysis import analyze_spec_structure


def rollout_indicators(
    values: np.ndarray, priorities: np.ndarray, eta: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate Eq. (4) on H-state/H-1-transition product rollouts."""
    values = np.asarray(values)
    priorities = np.asarray(priorities)
    if values.ndim != 2 or priorities.shape != values.shape:
        raise ValueError("values and priorities must have equal [trajectory, time] shape")
    vc, vn = values[:, :-1], values[:, 1:]
    source_priority = priorities[:, :-1]
    p1 = vn <= vc
    p2_relevant = source_priority == 1
    p2 = np.logical_or(~p2_relevant, vn <= vc - eta)
    endpoint_zfree = values[:, -1] < eta
    indicators = np.all(p1 & p2, axis=1) & endpoint_zfree

    zfree_source = vc < eta
    sampled_closure_trials = np.any(zfree_source, axis=1)
    sampled_closure_success = np.logical_or(
        ~sampled_closure_trials,
        np.all(np.logical_or(~zfree_source, p1 & p2), axis=1),
    )
    n_closure = int(np.sum(sampled_closure_trials))
    k_closure = int(np.sum(sampled_closure_success & sampled_closure_trials))
    return indicators, {
        "n_rollouts": int(len(values)),
        "p1_all_transition_paths": int(np.sum(np.all(p1, axis=1))),
        "p2_all_relevant_paths": int(np.sum(np.all(p2, axis=1))),
        "endpoint_zfree_paths": int(np.sum(endpoint_zfree)),
        "c_successes": int(np.sum(indicators)),
        "sampled_zfree_closure_n": n_closure,
        "sampled_zfree_closure_k": k_closure,
        "sampled_zfree_closure_rate": k_closure / n_closure if n_closure else None,
    }


def check_support_report(
    path: pathlib.Path | None, model_sha256: str, eta: float
) -> dict[str, Any]:
    if path is None:
        return {"verified": False, "reason": "no support-wide Z_free report supplied"}
    report = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "spec_id": SPEC_ID,
        "model_sha256": model_sha256,
        "eta": eta,
        "scope": "all_admissible_transitions_sourced_in_zfree",
        "verified": True,
    }
    mismatches = {
        key: {"expected": value, "observed": report.get(key)}
        for key, value in required.items()
        if report.get(key) != value
    }
    method = report.get("method")
    if not isinstance(method, str) or not method.strip():
        mismatches["method"] = {
            "expected": "non-empty verifier method",
            "observed": method,
        }
    return {
        "verified": not mismatches,
        "path": str(path),
        "sha256": sha256_file(path),
        "mismatches": mismatches,
        "report": report,
    }


def _load_model(model_path: pathlib.Path):
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if payload.get("spec_id") != SPEC_ID:
        raise ValueError("model checkpoint spec mismatch")
    if float(payload.get("eta", ETA)) != ETA:
        raise ValueError("model checkpoint eta mismatch")
    layout = payload["feature_layout"]
    model = NeuralLPPM(
        latent_dim=sum(int(value) for value in layout.values()),
        n_states=len(payload["state_to_idx"]),
        n_heads=1,
        hidden_dim=int(payload["hidden_dim"]),
        q_embed_dim=int(payload["q_embed_dim"]),
    )
    model.load_state_dict(payload["weights"])
    model.eval()
    return payload, model


def _predict(model, features, q_curr, batch_size):
    flat_x = features.reshape(-1, features.shape[-1])
    flat_q = q_curr.reshape(-1)
    output = []
    with torch.no_grad():
        for start in range(0, len(flat_x), batch_size):
            output.append(
                model(
                    torch.from_numpy(flat_x[start : start + batch_size]),
                    torch.from_numpy(flat_q[start : start + batch_size]),
                )[:, 0].numpy()
            )
    return np.concatenate(output).reshape(features.shape[:2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--support-wide-zfree-report")
    parser.add_argument(
        "--output", default="artifacts/l2_persistence_paper_evaluation.json"
    )
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--warrant-threshold", type=float, default=0.80)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    model_path = pathlib.Path(args.model)
    model_sha = sha256_file(model_path)
    payload, model = _load_model(model_path)
    pool = load_pool(
        pathlib.Path(args.calibration_root), required_calibration_eligible=True
    )
    manifest = pool.manifest
    if manifest.get("action_source") != "actor":
        raise ValueError("formal calibration requires deployment actor actions")
    if manifest.get("anchor_strategy") != "uniform":
        raise ValueError("formal calibration requires uniform replay anchors")
    touched = set(payload.get("fit_fingerprints", ())) | set(
        payload.get("diagnostic_fingerprints", ())
    )
    overlap = touched & set(pool.fingerprints.tolist())
    if overlap:
        raise ValueError(
            f"calibration overlaps {len(overlap)} model-selection fingerprints"
        )

    include_hazard = "hazard_dist" in payload["feature_layout"]
    raw = assemble_features(pool, include_hazard=include_hazard)
    mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    scale = np.asarray(payload["feature_scale"], dtype=np.float32)
    features = normalize_features(
        raw, mean, scale, float(payload["normalization_clip"])
    )
    spec = get_ltl_spec_by_id(SPEC_ID)
    if spec is None:
        raise RuntimeError(f"{SPEC_ID} is not registered")
    spec["analysis"] = analyze_spec_structure(spec)
    dpa = build_parity_automaton(spec)
    expected_states = {state: i for i, state in enumerate(dpa.states)}
    if (
        payload["state_to_idx"] != expected_states
        or payload["automaton_priority"] != dpa.priority
    ):
        raise ValueError(
            "model checkpoint was not trained against the current paper automaton"
        )
    q_curr, _q_after, priorities = automaton_trace(pool.hazard, dpa)
    values = _predict(model, features, q_curr, args.batch_size)
    indicators, diagnostics = rollout_indicators(values, priorities, ETA)
    n, k = len(indicators), int(np.sum(indicators))
    p_hat = _clopper_pearson_lower(k, n, args.gamma)
    support = check_support_report(
        pathlib.Path(args.support_wide_zfree_report)
        if args.support_wide_zfree_report
        else None,
        model_sha,
        ETA,
    )
    warranted = support["verified"] and p_hat >= args.warrant_threshold
    report = {
        "experiment": "cardreamer_l2_persistence_paper_evaluation",
        "created_at_unix": time.time(),
        "spec_id": SPEC_ID,
        "formula": "F G(hazard_dist > 0)",
        "verdict": "L2_CALIBRATED_MODEL_SCOPE" if warranted else "INCONCLUSIVE",
        "model": {"path": str(model_path), "sha256": model_sha},
        "calibration_manifest": {
            "path": str(pool.manifest_path),
            "sha256": sha256_file(pool.manifest_path),
        },
        "n": n,
        "k": k,
        "gamma": args.gamma,
        "p_hat_gamma": p_hat,
        "warrant_threshold": args.warrant_threshold,
        "support_wide_zfree": support,
        "sampled_diagnostics": diagnostics,
        "scope_note": (
            "Sampled closure never substitutes for Theorem 5.4's support-wide "
            "all-admissible-successor premise."
        ),
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
