"""Validate the frozen TD-MPC2 walker L2 support contract.

This validator checks provenance and scope governance.  Passing it means that
later experiments are talking about one fixed transition system; it does not
prove P1/P2 or infinite-horizon satisfaction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
from typing import Any

from tdmpc2.calibration_governance import has_valid_read_only_marker


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_contract(
    contract_path: pathlib.Path,
    *,
    tdmpc2_source: pathlib.Path | None = None,
) -> dict[str, Any]:
    contract = _load_json(contract_path)
    errors: list[str] = []
    checks: dict[str, bool] = {}

    def check(name: str, condition: bool, message: str) -> None:
        checks[name] = bool(condition)
        if not condition:
            errors.append(message)

    primary = contract.get("primary_scope", {})
    robust = contract.get("robustness_scope", {})
    governance = contract.get("calibration_governance", {})

    check(
        "scopes_are_distinct",
        primary.get("name") != robust.get("name"),
        "primary and robustness scopes must have different names",
    )
    check(
        "primary_external_force_is_zero",
        primary.get("external_xfrc_applied") == "identically zero",
        "primary theorem scope must explicitly fix external xfrc_applied to zero",
    )
    check(
        "primary_is_fail_closed",
        primary.get("support_wide_verified") is False
        and primary.get("verdict") == "ABSTAIN",
        "an unverified primary support must remain ABSTAIN",
    )
    check(
        "no_paper_warrant",
        contract.get("paper_warrant_issued") is False,
        "scope freezing cannot issue a paper warrant",
    )
    state = primary.get("complete_product_state", [])
    check(
        "complete_state_declared",
        isinstance(state, list) and len(state) >= 5,
        "complete product state declaration is missing components",
    )

    checkpoint = pathlib.Path(primary.get("checkpoint", {}).get("path", ""))
    expected_checkpoint_hash = primary.get("checkpoint", {}).get("sha256")
    checkpoint_ok = (
        checkpoint.is_file()
        and bool(expected_checkpoint_hash)
        and sha256_file(checkpoint) == expected_checkpoint_hash
    )
    check(
        "checkpoint_hash",
        checkpoint_ok,
        f"checkpoint is missing or hash-mismatched: {checkpoint}",
    )

    expected_commit = contract.get("primary_scope", {}).get("tdmpc2_commit")
    if tdmpc2_source is not None:
        try:
            actual_commit = subprocess.run(
                ["git", "-C", str(tdmpc2_source), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            actual_commit = None
        check(
            "tdmpc2_source_commit",
            actual_commit == expected_commit,
            "TD-MPC2 source checkout is missing or at the wrong commit",
        )

    counterexample_path = pathlib.Path(robust.get("counterexample", ""))
    try:
        counterexample = _load_json(counterexample_path)
    except (OSError, ValueError, json.JSONDecodeError):
        counterexample = {}
    attacks = counterexample.get("attacks", [])
    replayable_attacks = bool(attacks) and all(
        attack.get("replayable_closure_counterexample") is True
        and len(attack.get("repetitions", [])) >= 2
        and all(
            repetition.get("closure_violated") is True
            and (
                index == 0
                or repetition.get("bit_exact_state_trace_vs_first") is True
            )
            for index, repetition in enumerate(attack.get("repetitions", []))
        )
        for attack in attacks
    )
    check(
        "robustness_counterexample",
        robust.get("stable_core_closure") is False
        and counterexample.get("verdict")
        == "CONCRETE_STABLE_CORE_CLOSURE_COUNTEREXAMPLE"
        and counterexample.get("support_wide_stable_core_closure_verified") is False
        and replayable_attacks,
        "robustness scope lacks the declared replayable closure counterexample",
    )

    prior_path = pathlib.Path(governance.get("prior_result", ""))
    try:
        prior = _load_json(prior_path)
    except (OSError, ValueError, json.JSONDecodeError):
        prior = {}
    check(
        "calibration_is_consumed",
        governance.get("prior_nominal_calibration_consumed") is True
        and governance.get("read_only") is True
        and governance.get("recalibration_forbidden") is True
        and governance.get("reusable_for_new_candidate") is False
        and len(prior.get("calibration_fingerprints", [])) > 0
        and has_valid_read_only_marker(prior)
        and prior.get("paper_warrant_issued") is False,
        "prior calibration provenance is missing, writable, or incorrectly reusable",
    )

    return {
        "schema_version": 1,
        "experiment": "tdmpc2_walker_l2_support_contract_validation",
        "contract": str(contract_path.resolve()),
        "checks": checks,
        "contract_valid": not errors,
        "errors": errors,
        "meaning": "scope/provenance validation only; not P1/P2 or L2 proof",
        "support_wide_verified": False,
        "paper_warrant_issued": False,
        "verdict": "SCOPE_FROZEN" if not errors else "INVALID_CONTRACT",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=pathlib.Path,
        default="artifacts/tdmpc2_walker_l2_support/support_contract.json",
    )
    parser.add_argument("--tdmpc2-source", type=pathlib.Path, default="/tmp/tdmpc2_src")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    result = validate_contract(args.contract, tdmpc2_source=args.tdmpc2_source)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if not result["contract_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
