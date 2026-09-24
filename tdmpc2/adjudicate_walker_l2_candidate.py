"""Adjudicate the fitted walker LPPM separately from the LTL property.

A concrete transition can refute one candidate V without proving that the
underlying co-Buchi property is false.  This program records both verdicts so
that a candidate failure is never silently promoted to a model violation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any


# The inputs adjudicated by this script are frozen v1 artifacts. Their core
# and action witness were both constructed for the old 0.6 m predicate.
LEGACY_ARTIFACT_HEIGHT_MIN_M = 0.6


def _load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adjudicate(
    candidate_path: pathlib.Path,
    support_validation_path: pathlib.Path,
    action_counterexample_path: pathlib.Path,
) -> dict[str, Any]:
    candidate = _load(candidate_path)
    validation = _load(support_validation_path)
    action = _load(action_counterexample_path)
    contract_path = pathlib.Path(validation.get("contract", ""))
    contract = _load(contract_path) if contract_path.is_file() else {}
    hyper = candidate.get("hyperparameters", {})
    structured = candidate.get("structured_constraint", {})
    primary = contract.get("primary_scope", {})
    persistent = action.get("persistent_bad_candidates", [])
    cycle_searches = action.get("exact_bad_cycle_searches", {})
    searched_steps = max(
        (int(item.get("searched_steps", 0)) for item in cycle_searches.values()),
        default=0,
    )

    checks = {
        "support_contract_valid": validation.get("contract_valid") is True,
        "candidate_fit_gate_passed": candidate.get("all_seeds_passed_gate") is True,
        "structured_core_is_exact_zero": (
            hyper.get("structured_stable_core") is True
            and structured.get("value", structured.get("stable_core_value")) == 0.0
        ),
        "candidate_uses_odd_budget_positive_off_core": (
            hyper.get("odd_budget_envelope") is True
            and int(hyper.get("odd_budget_steps", 0)) > 0
        ),
        "same_150_step_core": (
            hyper.get("stable_core_dwell_steps") == 150
            and "150-step stable-core" in action.get("scope", "")
        ),
        "same_checkpoint": (
            action.get("checkpoint_sha256")
            == primary.get("checkpoint", {}).get("sha256")
        ),
        "same_tdmpc2_source": (
            action.get("tdmpc2_commit") == primary.get("tdmpc2_commit")
        ),
        "zero_force_nominal_scope": primary.get("external_xfrc_applied") == "identically zero",
        "finite_path_is_in_planner_support": (
            action.get("finite_action_block_in_declared_action_range") is True
            and action.get("planner_action_support_conditions_hold") is True
        ),
        "path_starts_inside_core": (
            action.get("core", {}).get("height_m", 0.0)
            > LEGACY_ARTIFACT_HEIGHT_MIN_M
            and action.get("core", {}).get("progress_m", 0.0) > 20.0
        ),
        "path_exits_core_into_bad": any(
            item.get("exits_core") is True
            and isinstance(item.get("first_bad_state_offset"), int)
            for item in persistent
        ),
        "no_infinite_bad_lasso_found": not any(
            item.get("found") is True
            for item in cycle_searches.values()
        ),
    }
    candidate_refuted = all(
        value for key, value in checks.items() if key != "no_infinite_bad_lasso_found"
    )
    property_violated = action.get("universal_support_fg_safe_violated") is True
    return {
        "schema_version": 1,
        "experiment": "tdmpc2_walker_l2_candidate_adjudication",
        "inputs": {
            "candidate": str(candidate_path.resolve()),
            "candidate_sha256": _sha256(candidate_path),
            "support_validation": str(support_validation_path.resolve()),
            "support_validation_sha256": _sha256(support_validation_path),
            "action_counterexample": str(action_counterexample_path.resolve()),
            "action_counterexample_sha256": _sha256(action_counterexample_path),
        },
        "checks": checks,
        "candidate_validity": "REFUTED" if candidate_refuted else "UNRESOLVED",
        "candidate_refutation": (
            "At the last safe state the 150-step core fraction is 1, so the "
            "structured V is exactly 0. At the next bad-height state the "
            "fraction resets below 1 and the odd-budget branch is strictly "
            "positive. Therefore V(next) > V(current), which refutes P1."
            if candidate_refuted else None
        ),
        "property_verdict": "VIOLATION" if property_violated else "ABSTAIN",
        "exact_cycle_search": {
            "maximum_tail_steps": searched_steps,
            "searches": cycle_searches,
        },
        "property_reason": (
            "A reachable infinite bad lasso was concretized."
            if property_violated
            else f"The core-exit path and {searched_steps:,}-step bad-tail search "
            "refute this V but do not prove infinitely many bad visits."
        ),
        "paper_l2_warrant_issued": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=pathlib.Path,
        default="artifacts/tdmpc2_walker_lppm_training_odd_budget_dwell150/diagnostics.json",
    )
    parser.add_argument(
        "--support-validation",
        type=pathlib.Path,
        default="artifacts/tdmpc2_walker_l2_support/support_validation.json",
    )
    parser.add_argument(
        "--action-counterexample",
        type=pathlib.Path,
        default="artifacts/tdmpc2_walker_action_support/result.json",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default="artifacts/tdmpc2_walker_l2_support/candidate_adjudication.json",
    )
    args = parser.parse_args()
    result = adjudicate(args.candidate, args.support_validation, args.action_counterexample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
