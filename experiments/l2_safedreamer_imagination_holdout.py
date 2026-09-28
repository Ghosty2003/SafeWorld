"""Train, calibrate, and independently test a SafeDreamer LPPM.

This experiment is deliberately model-scope only: every trajectory comes
from SafeDreamer's imagination model.  The three splits are disjoint:

* train: fit V_phi;
* calibration: estimate reach/stay/certificate probabilities;
* test: evaluate the frozen V_phi and Z_free rule on fresh trajectories.

The finite rollout horizon is an observation horizon, not a replacement for
the unbounded LTL specification.  Therefore sampled stay-after-entry means
"stayed through the remainder of the observed horizon", not an infinite-time
deductive closure proof.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from statistics import mean
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.settings import RolloutConfig
from core.lppm import build_parity_automaton, fit_lppm
from core.lppm.calibrator import _clopper_pearson_lower
from core.lppm.model import compute_lppm_value
from core.lppm.verifier import (
    check_pathwise_conditions,
    iter_transitions,
    run_product_trajectory,
)
from experiments.l2_safedreamer import build_extra, collect_model_rollouts
from specs import get_spec_by_id
from utils.spec_analysis import analyze_spec_structure
from wrappers.safedreamer_wrapper import SafeDreamerWrapper


def _jsonable_training_info(training: dict[str, Any]) -> dict[str, Any]:
    """Keep audit metadata in JSON; model tensors are stored in a .pt file."""
    omitted = {"weights", "_model_cache"}
    result: dict[str, Any] = {}
    for key, value in training.items():
        if key in omitted:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, (list, dict)):
            result[key] = value
        else:
            result[key] = str(value)
    return result


def _is_zfree(state, *, horizon: int, dpa, spec, params, eta: float) -> bool:
    """Sampled Z_free membership used in this experiment.

    The learned value must be below eta for every odd-priority head, and the
    automaton state itself must not have odd priority.  The latter prevents an
    absorbing Safety trap from being mislabeled free merely because a learned
    network outputs a small value there.
    """
    if dpa.priority.get(state.q, state.priority) % 2 == 1:
        return False
    return all(
        compute_lppm_value(
            state.z,
            state.q,
            odd_priority,
            spec,
            state.t,
            horizon,
            params,
            dpa,
        )
        < eta
        for odd_priority in dpa.odd_priorities
    )


def _trajectory_result(trajectory, *, dpa, spec, params, eta: float) -> dict[str, Any]:
    product_path = run_product_trajectory(trajectory, dpa, spec)
    transitions = iter_transitions(product_path, dpa)
    horizon = len(product_path)
    checked = check_pathwise_conditions(
        product_path,
        dpa,
        spec,
        eta=eta,
        lppm_params=params,
    )

    # T real product states plus the synthetic post-final-transition state.
    states = [curr for curr, _ in transitions] + [transitions[-1][1]]
    membership = [
        _is_zfree(
            state,
            horizon=horizon,
            dpa=dpa,
            spec=spec,
            params=params,
            eta=eta,
        )
        for state in states
    ]
    hit_indices = [index for index, in_zfree in enumerate(membership) if in_zfree]
    reached = bool(hit_indices)
    first_hit = hit_indices[0] if reached else None
    stayed_after_hit = bool(reached and all(membership[first_hit:]))
    p1p2_hold = checked.p1_violations == 0 and checked.p2_violations == 0
    terminal_zfree = membership[-1]

    # paper_style_terminal_event matches the current core calibrator's C(tau):
    # all sampled P1/P2 checks and terminal membership.  requested_stay_event
    # is the user's stronger finite-observation diagnostic: after first entry,
    # every subsequent observed product state remains in Z_free.
    paper_style_terminal_event = p1p2_hold and terminal_zfree
    requested_stay_event = p1p2_hold and stayed_after_hit
    return {
        "p1_violations": checked.p1_violations,
        "p2_violations": checked.p2_violations,
        "total_transitions": checked.total_transitions,
        "p1p2_hold": p1p2_hold,
        "reached_zfree": reached,
        "first_hit": first_hit,
        "terminal_zfree": terminal_zfree,
        "stayed_after_first_hit": stayed_after_hit,
        "paper_style_terminal_event": paper_style_terminal_event,
        "requested_stay_event": requested_stay_event,
        "zfree_membership": membership,
    }


def _transition_populations(trajectories, *, dpa, spec) -> tuple[int, int]:
    p1_checked = 0
    p2_checked = 0
    for trajectory in trajectories:
        transitions = iter_transitions(run_product_trajectory(trajectory, dpa, spec), dpa)
        for curr, _ in transitions:
            for odd_priority in dpa.odd_priorities:
                if curr.priority == odd_priority:
                    p2_checked += 1
                elif odd_priority > curr.priority:
                    p1_checked += 1
    return p1_checked, p2_checked


def _rate(k: int, n: int, gamma: float) -> dict[str, Any]:
    return {
        "successes": k,
        "trials": n,
        "empirical_rate": (k / n) if n else None,
        "cp_lower_one_sided": _clopper_pearson_lower(k, n, gamma) if n else None,
        "confidence": 1.0 - gamma,
    }


def _evaluate_split(trajectories, *, dpa, spec, params, eta: float, gamma: float):
    rows = [
        _trajectory_result(t, dpa=dpa, spec=spec, params=params, eta=eta)
        for t in trajectories
    ]
    n = len(rows)
    n_reach = sum(row["reached_zfree"] for row in rows)
    n_stay = sum(row["stayed_after_first_hit"] for row in rows)
    n_p1p2 = sum(row["p1p2_hold"] for row in rows)
    n_terminal = sum(row["paper_style_terminal_event"] for row in rows)
    n_requested = sum(row["requested_stay_event"] for row in rows)
    p1_checked, p2_checked = _transition_populations(trajectories, dpa=dpa, spec=spec)
    hit_times = [row["first_hit"] for row in rows if row["first_hit"] is not None]
    summary = {
        "n_rollouts": n,
        "p1": {
            "violations": sum(row["p1_violations"] for row in rows),
            "checked": p1_checked,
        },
        "p2": {
            "violations": sum(row["p2_violations"] for row in rows),
            "checked": p2_checked,
        },
        "p1p2_trajectory_success": _rate(n_p1p2, n, gamma),
        "reach_zfree": _rate(n_reach, n, gamma),
        "stay_given_reach": _rate(n_stay, n_reach, gamma),
        "reach_and_stay": _rate(n_stay, n, gamma),
        "paper_style_terminal_certificate": _rate(n_terminal, n, gamma),
        "requested_p1p2_reach_stay_event": _rate(n_requested, n, gamma),
        "first_hit_time": {
            "count": len(hit_times),
            "min": min(hit_times) if hit_times else None,
            "mean": mean(hit_times) if hit_times else None,
            "max": max(hit_times) if hit_times else None,
        },
    }
    return rows, summary


def _print_split(
    name: str,
    summary: dict[str, Any],
    *,
    candidate_rejected: bool = False,
) -> None:
    def show(label: str, metric: dict[str, Any]) -> None:
        rate = metric["empirical_rate"]
        lower = metric["cp_lower_one_sided"]
        rate_text = "n/a" if rate is None else f"{rate:.4f}"
        lower_text = "n/a" if lower is None else f"{lower:.4f}"
        print(
            f"  {label:<30} {metric['successes']}/{metric['trials']}  "
            f"rate={rate_text}  95% lower={lower_text}"
        )

    print(f"\n{name}")
    if candidate_rejected:
        print("  RAW DEGENERATE-CANDIDATE DIAGNOSTICS (not an established Z_free)")
    print(
        f"  P1 violations: {summary['p1']['violations']}/{summary['p1']['checked']} | "
        f"P2 violations: {summary['p2']['violations']}/{summary['p2']['checked']}"
    )
    show("P1/P2 trajectory success", summary["p1p2_trajectory_success"])
    show("reached Z_free", summary["reach_zfree"])
    show("stayed | reached", summary["stay_given_reach"])
    show("reached and stayed", summary["reach_and_stay"])
    show("paper terminal C(tau)", summary["paper_style_terminal_certificate"])
    show("P1/P2 + reach + stay", summary["requested_p1p2_reach_stay_event"])


def _load_cached_splits(args: argparse.Namespace):
    path = Path(args.rollouts)
    if not path.exists():
        raise FileNotFoundError(f"--reuse-rollouts requested but cache is missing: {path}")
    payload = json.loads(path.read_text())
    metadata = payload.get("metadata", {})
    # Rollout generation is specification-independent: the cached AP traces
    # contain every grounded SafeDreamer AP.  This permits selecting a new
    # spec using the training split only while preserving the untouched
    # calibration/test traces.  All generation parameters must still match.
    expected_generation = {
        "horizon": args.horizon,
        "n_train": args.n_train,
        "n_cal": args.n_cal,
        "n_test": args.n_test,
        "seed_train": args.seed,
        "seed_calibration": args.seed + 1,
        "seed_test": args.seed + 2,
        "action_source": "random",
        "environment_rollouts": 0,
    }
    observed_generation = {
        key: metadata.get(key) for key in expected_generation
    }
    if observed_generation != expected_generation:
        raise ValueError(
            "Rollout cache generation metadata mismatch: "
            f"cached={observed_generation}, expected={expected_generation}"
        )
    print(f"Reusing preregistered imagination splits: {path}")
    if metadata.get("spec") != args.spec:
        print(
            "  Note: AP traces were originally collected for "
            f"{metadata.get('spec')!r}; generation is spec-independent and only "
            "the training split was used to select the new specification."
        )
    return payload["train"], payload["calibration"], payload["test"], metadata


def main(args: argparse.Namespace) -> None:
    spec = get_spec_by_id(args.spec)
    if spec is None:
        raise ValueError(f"Unknown specification: {args.spec}")
    spec = copy.deepcopy(spec)
    spec["analysis"] = analyze_spec_structure(spec)
    if not args.spec.startswith("ltl_"):
        raise ValueError("LPPM requires an LTL specification.")

    extra = build_extra(args)
    print("MODEL-ONLY SafeDreamer LPPM holdout experiment")
    print(f"  specification: {args.spec}")
    print(f"  checkpoint: {extra['checkpoint_path']}")
    print(f"  horizon: {args.horizon}")
    print(
        f"  disjoint splits: train={args.n_train} (seed {args.seed}), "
        f"calibration={args.n_cal} (seed {args.seed + 1}), "
        f"test={args.n_test} (seed {args.seed + 2})"
    )
    print("  environment rollouts: NONE")

    wrapper_config = RolloutConfig(
        horizon=args.horizon,
        n_rollouts=1,
        seed=args.seed,
        action_source="random",
        extra=dict(extra),
    )
    with ExitStack() as stack:
        wrapper = None
        rollout_cache_metadata = None
        if args.reuse_rollouts:
            train, calibration, test, rollout_cache_metadata = _load_cached_splits(args)
        else:
            wrapper = stack.enter_context(SafeDreamerWrapper(wrapper_config))
            wrapper.load()
            print("Collecting training imagination rollouts...", flush=True)
            train = collect_model_rollouts(
                wrapper,
                horizon=args.horizon,
                count=args.n_train,
                seed=args.seed,
                extra=extra,
            )
            print("Collecting held-out calibration imagination rollouts...", flush=True)
            calibration = collect_model_rollouts(
                wrapper,
                horizon=args.horizon,
                count=args.n_cal,
                seed=args.seed + 1,
                extra=extra,
            )

        dpa = build_parity_automaton(spec)
        torch.manual_seed(args.torch_seed)
        print(f"Training V_phi for {args.epochs} epochs...", flush=True)
        training = fit_lppm(
            train,
            dpa,
            spec,
            eta=args.eta,
            n_epochs=args.epochs,
            calib_trajectories=calibration,
        )
        if training.get("backend") != "torch_mlp":
            raise RuntimeError(f"Neural V_phi training unavailable: {training.get('reason')}")
        print(
            f"Frozen V_phi: backend={training['backend']} "
            f"epochs={training['epochs_trained']} loss={training['final_loss']:.6g}"
        )
        print(
            "Post-fit audit: "
            f"status={training['training_status']}  "
            f"P1={training['train_p1_violations']}/{training['train_p1_checked']}  "
            f"P2={training['train_p2_violations']}/{training['train_p2_checked']}  "
            f"V_span={training['train_value_span']:.6g}"
        )
        if training["collapse_detected"]:
            print(
                "REJECTING collapsed V_phi candidate: "
                f"{training['failure_reason']}"
            )

        print("Evaluating frozen V_phi on calibration split...", flush=True)
        calibration_rows, calibration_summary = _evaluate_split(
            calibration,
            dpa=dpa,
            spec=spec,
            params=training,
            eta=args.eta,
            gamma=args.gamma,
        )

        # In a fresh run the test split is collected only after V_phi and all
        # evaluation rules have been fixed.  In --reuse-rollouts mode it is
        # loaded from that already-frozen run and is still never passed to
        # fit_lppm().
        if not args.reuse_rollouts:
            print("Collecting independent test imagination rollouts...", flush=True)
            test = collect_model_rollouts(
                wrapper,
                horizon=args.horizon,
                count=args.n_test,
                seed=args.seed + 2,
                extra=extra,
            )
        print("Evaluating frozen V_phi on independent test split...", flush=True)
        test_rows, test_summary = _evaluate_split(
            test,
            dpa=dpa,
            spec=spec,
            params=training,
            eta=args.eta,
            gamma=args.gamma,
        )

    cal_metric = calibration_summary["requested_p1p2_reach_stay_event"]
    test_metric = test_summary["requested_p1p2_reach_stay_event"]
    prediction_error = abs(cal_metric["empirical_rate"] - test_metric["empirical_rate"])
    cal_lower = cal_metric["cp_lower_one_sided"]
    candidate_rejected = bool(training.get("collapse_detected", False))
    warranted = (
        not candidate_rejected
        and cal_lower is not None
        and cal_lower >= args.warrant_threshold
    )

    _print_split(
        "CALIBRATION (prediction)",
        calibration_summary,
        candidate_rejected=candidate_rejected,
    )
    _print_split(
        "INDEPENDENT TEST (observed)",
        test_summary,
        candidate_rejected=candidate_rejected,
    )
    print("\nFrozen prediction versus independent test")
    print(f"  calibration point estimate: {cal_metric['empirical_rate']:.4f}")
    print(f"  calibration 95% lower bound: {cal_lower:.4f}")
    print(f"  independent test rate: {test_metric['empirical_rate']:.4f}")
    print(f"  absolute point-estimate error: {prediction_error:.4f}")
    print(f"  warrant threshold: {args.warrant_threshold:.4f}")
    if candidate_rejected:
        print("  V_phi candidate status: REJECTED_COLLAPSED")
        print("  Z_free status: NOT_ESTABLISHED (raw membership above is diagnostic only)")
        result_label = "NO_WARRANT_INVALID_V"
    else:
        result_label = "WARRANT" if warranted else "NO_WARRANT"
    print(f"  sampled model-scope result: {result_label}")
    print(
        "  scope: fresh SafeDreamer imagination paths from the same random-action "
        "distribution; finite observed horizon only"
    )

    report_path = Path(args.report)
    rollouts_path = Path(args.rollouts)
    weights_path = Path(args.weights)
    for path in (report_path, rollouts_path, weights_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            key: value
            for key, value in training.items()
            if key != "_model_cache"
        },
        weights_path,
    )
    if not args.reuse_rollouts:
        rollouts_path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "spec": args.spec,
                        "horizon": args.horizon,
                        "n_train": args.n_train,
                        "n_cal": args.n_cal,
                        "n_test": args.n_test,
                        "seed_train": args.seed,
                        "seed_calibration": args.seed + 1,
                        "seed_test": args.seed + 2,
                        "action_source": "random",
                        "environment_rollouts": 0,
                    },
                    "train": train,
                    "calibration": calibration,
                    "test": test,
                }
            )
        )
    report = {
        "configuration": {
            "spec": args.spec,
            "horizon": args.horizon,
            "n_train": args.n_train,
            "n_cal": args.n_cal,
            "n_test": args.n_test,
            "seed_train": args.seed,
            "seed_calibration": args.seed + 1,
            "seed_test": args.seed + 2,
            "torch_seed": args.torch_seed,
            "epochs": args.epochs,
            "eta": args.eta,
            "gamma": args.gamma,
            "confidence": 1.0 - args.gamma,
            "warrant_threshold": args.warrant_threshold,
            "action_source": "random",
            "environment_rollouts": 0,
            "reused_rollout_cache": bool(args.reuse_rollouts),
            "rollout_cache_original_spec": (
                rollout_cache_metadata.get("spec")
                if rollout_cache_metadata is not None
                else args.spec
            ),
            "zfree_definition": "even automaton priority and V_phi(z,q,r) < eta for every odd head r",
            "stay_definition": "after first sampled Z_free entry, every remaining observed product state is in Z_free",
        },
        "training": _jsonable_training_info(training),
        "calibration": calibration_summary,
        "test": test_summary,
        "comparison": {
            "metric": "P1/P2 + reach Z_free + sampled stay after first hit",
            "calibration_point_estimate": cal_metric["empirical_rate"],
            "calibration_cp_lower": cal_lower,
            "test_observed_rate": test_metric["empirical_rate"],
            "absolute_point_estimate_error": prediction_error,
            "candidate_rejected": candidate_rejected,
            "candidate_rejection_reason": training.get("failure_reason") if candidate_rejected else None,
            "zfree_status": "NOT_ESTABLISHED" if candidate_rejected else "CANDIDATE_EVALUATED",
            "warranted": warranted,
            "result": result_label,
            "scope": "sampled model scope, same imagination/action distribution, finite observed horizon",
        },
        "per_trajectory": {
            "calibration": calibration_rows,
            "test": test_rows,
        },
        "artifacts": {
            "rollouts": str(rollouts_path),
            "weights": str(weights_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"\nSaved report: {report_path}")
    print(f"{'Reused' if args.reuse_rollouts else 'Saved'} rollouts: {rollouts_path}")
    print(f"Saved frozen V_phi: {weights_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="ltl_hazard_avoidance")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--n-train", type=int, default=60)
    parser.add_argument("--n-cal", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--torch-seed", type=int, default=0)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--warrant-threshold", type=float, default=0.80)
    parser.add_argument(
        "--report",
        default="artifacts/safedreamer_l2_diagnostics/imagination_holdout_report.json",
    )
    parser.add_argument(
        "--rollouts",
        default="artifacts/safedreamer_l2_diagnostics/imagination_holdout_rollouts.json",
    )
    parser.add_argument(
        "--weights",
        default="artifacts/safedreamer_l2_diagnostics/imagination_holdout_v_phi.pt",
    )
    parser.add_argument(
        "--reuse-rollouts",
        action="store_true",
        help="Reuse the saved disjoint train/calibration/test imagination splits.",
    )
    main(parser.parse_args())
