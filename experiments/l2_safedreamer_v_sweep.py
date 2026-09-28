"""Sweep several LPPM candidates on one fixed SafeDreamer rollout split.

This is a diagnostic experiment, not a new warranting rule.  In particular,
an invariant-Safety trajectory whose product automaton ends in the absorbing
odd trap can never satisfy the certificate event, independently of the choice
of V.  The script reports that function-independent ceiling before comparing
candidate functions.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.settings import RolloutConfig
from core.lppm import build_parity_automaton, calibrate_lppm, fit_lppm
from core.lppm.calibrator import _clopper_pearson_lower
from core.lppm.model import NeuralLPPM, compute_lppm_value, infer_feature_keys
from core.lppm.verifier import iter_transitions, run_product_trajectory
from experiments.l2_safedreamer import build_extra, collect_model_rollouts
from specs import get_spec_by_id
from utils.spec_analysis import analyze_spec_structure
from wrappers.safedreamer_wrapper import SafeDreamerWrapper


def _load_or_collect(args, extra):
    cache = Path(args.rollout_cache)
    if args.reuse_rollouts:
        if not cache.exists():
            raise FileNotFoundError(f"--reuse-rollouts requested but cache is missing: {cache}")
        payload = json.loads(cache.read_text())
        metadata = payload["metadata"]
        expected = {
            "horizon": args.horizon,
            "n_train": args.n_train,
            "n_cal": args.n_cal,
            "seed": args.seed,
            "spec": args.spec,
        }
        if metadata != expected:
            raise ValueError(f"Rollout cache metadata mismatch: cached={metadata}, requested={expected}")
        print(f"Reusing fixed rollout cache: {cache}")
        return payload["train"], payload["calibration"]

    wrapper_config = RolloutConfig(
        horizon=args.horizon,
        n_rollouts=1,
        seed=args.seed,
        action_source="random",
        extra=dict(extra),
    )
    with SafeDreamerWrapper(wrapper_config) as wrapper:
        wrapper.load()
        print("Collecting one fixed LPPM training split...", flush=True)
        train = collect_model_rollouts(
            wrapper, horizon=args.horizon, count=args.n_train,
            seed=args.seed, extra=extra,
        )
        print("Collecting one fixed held-out calibration split...", flush=True)
        calibration = collect_model_rollouts(
            wrapper, horizon=args.horizon, count=args.n_cal,
            seed=args.seed + 1, extra=extra,
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "metadata": {
            "horizon": args.horizon,
            "n_train": args.n_train,
            "n_cal": args.n_cal,
            "seed": args.seed,
            "spec": args.spec,
        },
        "train": train,
        "calibration": calibration,
    }))
    print(f"Saved fixed rollout cache: {cache}")
    return train, calibration


def _constant_lppm_params(trajectories, dpa, value: float, hidden_dim: int = 8):
    """Build an exact positive constant using the normal NeuralLPPM backend."""
    if not 0.0 < value:
        raise ValueError("Softplus constant must be strictly positive.")
    feature_keys = infer_feature_keys(trajectories)
    state_to_idx = {state: idx for idx, state in enumerate(dpa.states)}
    odd_to_idx = {priority: idx for idx, priority in enumerate(dpa.odd_priorities)}
    model = NeuralLPPM(
        latent_dim=len(feature_keys),
        n_states=len(state_to_idx),
        n_heads=max(len(odd_to_idx), 1),
        hidden_dim=hidden_dim,
        q_embed_dim=16,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # softplus(log(exp(value)-1)) == value (up to float precision).
        model.head.bias.fill_(math.log(math.expm1(value)))
    return {
        "loss_history": [],
        "final_loss": float("nan"),
        "n_transitions": sum(len(t) for t in trajectories),
        "epochs_trained": 0,
        "odd_priorities": dpa.odd_priorities,
        "weights": {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
        "backend": "torch_mlp",
        "feature_keys": feature_keys,
        "state_to_idx": state_to_idx,
        "odd_to_idx": odd_to_idx,
        "hidden_dim": hidden_dim,
        "q_embed_dim": 16,
        "candidate_kind": "constructed_constant",
    }


def _transition_counts(trajectories, dpa, spec):
    p1_checked = 0
    bad_transitions = 0
    even_terminal = 0
    for trajectory in trajectories:
        transitions = iter_transitions(run_product_trajectory(trajectory, dpa, spec), dpa)
        terminal_q = transitions[-1][1].q
        if dpa.priority.get(terminal_q, 0) % 2 == 0:
            even_terminal += 1
        for curr, _ in transitions:
            for odd_priority in dpa.odd_priorities:
                if curr.priority == odd_priority:
                    bad_transitions += 1
                elif odd_priority > curr.priority:
                    p1_checked += 1
    return p1_checked, bad_transitions, even_terminal


def _value_stats(trajectories, dpa, spec, params):
    values = []
    for trajectory in trajectories:
        path = run_product_trajectory(trajectory, dpa, spec)
        horizon = len(path)
        for state in path:
            for odd_priority in dpa.odd_priorities:
                values.append(compute_lppm_value(
                    state.z, state.q, odd_priority, spec,
                    state.t, horizon, params, dpa,
                ))
    return float(np.min(values)), float(np.median(values)), float(np.max(values))


def _evaluate(name, params, train, calibration, dpa, spec, args, counts):
    result = calibrate_lppm(
        calibration,
        dpa,
        spec,
        gamma=args.gamma,
        eta=args.eta,
        warrant_threshold=args.warrant_threshold,
        lppm_params=params,
        train_trajectories=train,
    )
    p1_checked, bad_transitions, _ = counts
    p1_violations = sum(path.p1_violations for path in result.pathwise)
    p2_violations = sum(path.p2_violations for path in result.pathwise)
    successes = sum(path.satisfied for path in result.pathwise)
    v_min, v_median, v_max = _value_stats(calibration, dpa, spec, params)
    closure = result.zfree_closure
    closure_text = "UNVERIFIABLE"
    closure_pass = False
    if closure is not None and closure.verifiable:
        closure_text = (
            f"{closure.k_zfree}/{closure.n_zfree},"
            f" lower={closure.p_hat_closure:.4f}"
        )
        closure_pass = (
            closure.p_hat_closure is not None
            and closure.p_hat_closure >= args.warrant_threshold
        )
    ready = result.is_warranted() and closure_pass
    raw_loss = params.get("final_loss")
    loss = None if raw_loss is None or not math.isfinite(float(raw_loss)) else float(raw_loss)
    loss_text = "n/a" if loss is None else f"{loss:.5g}"
    print(
        f"{name:<22} loss={loss_text:>9}  "
        f"V=[{v_min:.5g},{v_median:.5g},{v_max:.5g}]  "
        f"P1={p1_violations}/{p1_checked}  P2={p2_violations}/{bad_transitions}  "
        f"C={successes}/{len(calibration)}  p_hat={result.p_hat_gamma:.4f}  "
        f"closure={closure_text}  ready={ready}"
    )
    return {
        "name": name,
        "loss": loss,
        "v_min": v_min,
        "v_median": v_median,
        "v_max": v_max,
        "p1_violations": p1_violations,
        "p1_checked": p1_checked,
        "p2_violations": p2_violations,
        "bad_transitions": bad_transitions,
        "certificate_successes": successes,
        "n_calibration": len(calibration),
        "p_hat_gamma": result.p_hat_gamma,
        "closure": closure_text,
        "calibrated_ready": ready,
    }


def main(args):
    spec = get_spec_by_id(args.spec)
    if spec is None:
        raise ValueError(f"Unknown specification: {args.spec}")
    spec = copy.deepcopy(spec)
    spec["analysis"] = analyze_spec_structure(spec)
    if spec["analysis"]["mp_class"] != "Safety":
        raise ValueError("This diagnostic currently targets invariant Safety specifications.")

    extra = build_extra(args)
    train, calibration = _load_or_collect(args, extra)
    dpa = build_parity_automaton(spec)
    counts = _transition_counts(calibration, dpa, spec)
    p1_checked, bad_transitions, even_terminal = counts
    ceiling = _clopper_pearson_lower(even_terminal, len(calibration), args.gamma)
    print("\nFunction-independent calibration ceiling")
    print(f"  even-terminal trajectories: {even_terminal}/{len(calibration)}")
    print(f"  odd-trap-terminal trajectories: {len(calibration) - even_terminal}/{len(calibration)}")
    print(f"  maximum possible p_hat_gamma: {ceiling:.6f}")
    print(f"  warrant threshold: {args.warrant_threshold:.6f}")
    print(f"  checked populations: P1={p1_checked}, P2/bad={bad_transitions}")

    rows = []
    print("\nCandidate sweep on the identical held-out split")
    for label, value in (("constant_eta_over_2", args.eta / 2), ("constant_2eta", args.eta * 2)):
        params = _constant_lppm_params(train, dpa, value)
        rows.append(_evaluate(label, params, train, calibration, dpa, spec, args, counts))

    variants = [(32, 0), (64, 0), (128, 0), (128, 1), (256, 0)]
    for hidden_dim, torch_seed in variants:
        candidate_spec = copy.deepcopy(spec)
        candidate_spec["lppm_hidden_dim"] = hidden_dim
        torch.manual_seed(torch_seed)
        params = fit_lppm(
            train,
            dpa,
            candidate_spec,
            eta=args.eta,
            n_epochs=args.epochs,
            calib_trajectories=calibration,
        )
        label = f"mlp_h{hidden_dim}_seed{torch_seed}"
        rows.append(_evaluate(
            label, params, train, calibration, dpa, candidate_spec, args, counts,
        ))

    report = {
        "configuration": {
            "spec": args.spec,
            "eta": args.eta,
            "gamma": args.gamma,
            "warrant_threshold": args.warrant_threshold,
            "epochs": args.epochs,
        },
        "function_independent_ceiling": {
            "even_terminal": even_terminal,
            "n_calibration": len(calibration),
            "maximum_p_hat_gamma": ceiling,
        },
        "candidates": rows,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"\nSaved sweep report: {report_path}")
    if ceiling < args.warrant_threshold:
        print(
            "Conclusion: no choice of V can reach the requested calibrated "
            "warrant on this fixed split because odd-trap terminal trajectories "
            "already force the function-independent ceiling below threshold."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="ltl_hazard_avoidance")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--n-train", type=int, default=60)
    parser.add_argument("--n-cal", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--warrant-threshold", type=float, default=0.80)
    parser.add_argument(
        "--rollout-cache",
        default="artifacts/safedreamer_l2_diagnostics/v_sweep_rollouts.json",
    )
    parser.add_argument("--reuse-rollouts", action="store_true")
    parser.add_argument(
        "--report",
        default="artifacts/safedreamer_l2_diagnostics/v_sweep_report.json",
    )
    main(parser.parse_args())
