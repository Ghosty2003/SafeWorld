"""Fail-closed gate before TD-MPC2 walker L2 calibration.

Finite rollout rates are diagnostics.  This program only accepts a separate,
sound enclosure result that quantifies over every admissible successor in the
frozen support; in its absence the only valid formal result is ABSTAIN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify_support_wide_gate(
    candidate_path: pathlib.Path,
    support_validation_path: pathlib.Path,
    *,
    enclosure_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    candidate = load_json(candidate_path)
    support = load_json(support_validation_path)
    candidate_hash = sha256_file(candidate_path)
    support_hash = sha256_file(support_validation_path)
    best_seed = candidate.get("best_seed")
    runs = candidate.get("runs", [])
    best = next((run for run in runs if run.get("seed") == best_seed), None)
    best_diagnostic = (best or {}).get("diagnostic", {})
    hyperparameters = candidate.get("hyperparameters", {})

    fit_checks = {
        "candidate_diagnostic_gate": candidate.get("all_seeds_passed_gate") is True,
        "pathwise_gate": best_diagnostic.get("gates", {}).get("pathwise") is True,
        "all_automaton_states_observed": best_diagnostic.get("gates", {}).get(
            "all_automaton_states_observed"
        ) is True,
        "nominal_transition_training_scope": (
            (
                hyperparameters.get("nominal_only_stable_core") is True
                and hyperparameters.get("include_recovery_data") is False
            )
            or (
                hyperparameters.get("include_recovery_data") is True
                and hyperparameters.get("nominal_recovery_transitions_only") is True
            )
        ),
        "no_finite_deadline_countdown": (
            hyperparameters.get("bounded_countdown_envelope") is False
            and hyperparameters.get("global_countdown_envelope") is False
        ),
        "calibration_not_launched": candidate.get("calibration_launched") is False,
    }
    scope_checks = {
        "support_contract_valid": support.get("contract_valid") is True,
        "support_contract_issues_no_warrant": support.get("paper_warrant_issued") is False,
        "frozen_support_still_unverified": support.get("support_wide_verified") is False,
    }

    enclosure: dict[str, Any] = {}
    if enclosure_path is not None:
        enclosure = load_json(enclosure_path)
    enclosure_checks = {
        "sound_backend": enclosure.get("sound_backend") is True,
        "all_admissible_successors": enclosure.get("quantification")
        == "all_admissible_successors",
        "candidate_bound": enclosure.get("candidate_diagnostics_sha256")
        == candidate_hash,
        "support_bound": enclosure.get("support_validation_sha256") == support_hash,
        "p1_p2_verified": enclosure.get("p1_p2_verified") is True,
        "zfree_closure_verified": enclosure.get("zfree_closure_verified") is True,
        "uniform_odd_visit_bound_verified": (
            hyperparameters.get("odd_budget_envelope") is not True
            or enclosure.get("uniform_odd_visit_bound_verified") is True
        ),
    }

    fit_passed = all(fit_checks.values())
    support_passed = all(scope_checks.values()) and all(enclosure_checks.values())
    reasons: list[str] = []
    if not fit_checks["candidate_diagnostic_gate"]:
        reasons.append("CANDIDATE_DIAGNOSTIC_GATE_FAILED")
    if not fit_checks["pathwise_gate"]:
        reasons.append("EQ4_PATHWISE_INDICATOR_ZERO_ON_DIAGNOSTIC_PATHS")
    if not fit_checks["all_automaton_states_observed"]:
        reasons.append("ACHIEVED_BAD_AUTOMATON_STATE_UNOBSERVED")
    if (
        hyperparameters.get("odd_budget_envelope") is True
        and not enclosure_checks["uniform_odd_visit_bound_verified"]
    ):
        reasons.append("UNIFORM_ODD_VISIT_BOUND_UNVERIFIED")
    if not support_passed:
        reasons.append("NO_SOUND_SUPPORT_WIDE_SUCCESSOR_ENCLOSURE")
        reasons.append("ZFREE_CLOSURE_SUPPORT_WIDE_UNVERIFIED")

    return {
        "schema_version": 1,
        "experiment": "tdmpc2_walker_l2_support_wide_gate",
        "candidate": {
            "path": str(candidate_path.resolve()),
            "sha256": candidate_hash,
            "best_seed": best_seed,
            "sampled_p1_rate": best_diagnostic.get("p1_transition_pass_rate"),
            "sampled_p2_rate": best_diagnostic.get("p2_transition_pass_rate"),
            "sampled_pathwise_rate": best_diagnostic.get("pathwise_pass_rate"),
            "sampled_zfree_closure_rate": best_diagnostic.get(
                "sampled_zfree_closure_rate"
            ),
        },
        "support_validation": {
            "path": str(support_validation_path.resolve()),
            "sha256": support_hash,
        },
        "enclosure_evidence": (
            str(enclosure_path.resolve()) if enclosure_path is not None else None
        ),
        "fit_checks": fit_checks,
        "scope_checks": scope_checks,
        "enclosure_checks": enclosure_checks,
        "candidate_fit_passed": fit_passed,
        "support_wide_verified": support_passed,
        "fresh_calibration_allowed": fit_passed and support_passed,
        "fresh_calibration_launched": False,
        "reasons": reasons,
        "formal_infinite_horizon_verdict": "ABSTAIN",
        "paper_warrant_issued": False,
        "note": (
            "sampled rates, including sampled Z_free closure, cannot replace "
            "quantification over every admissible closed-loop successor"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-diagnostics", type=pathlib.Path, required=True)
    parser.add_argument(
        "--support-validation",
        type=pathlib.Path,
        default="artifacts/tdmpc2_walker_l2_support/support_validation.json",
    )
    parser.add_argument("--enclosure-result", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    result = verify_support_wide_gate(
        args.candidate_diagnostics,
        args.support_validation,
        enclosure_path=args.enclosure_result,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
