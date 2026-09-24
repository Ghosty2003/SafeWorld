"""Reproduce exact binary L2 outcomes on explicitly finite support graphs."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

from core.lppm import verify_finite_cobuchi


def run_examples() -> dict:
    safe = verify_finite_cobuchi(
        {
            "waiting": ("waiting", "debt"),
            "debt": ("recovered",),
            "recovered": ("recovered",),
        },
        initial_states=("waiting",),
        bad_states=("debt",),
        eta=0.25,
    )
    violation = verify_finite_cobuchi(
        {
            "start": ("safe",),
            "safe": ("bad",),
            "bad": ("safe",),
        },
        initial_states=("start",),
        bad_states=("bad",),
        eta=0.25,
    )
    return {
        "schema_version": 1,
        "experiment": "exact_finite_support_l2_binary_verification",
        "semantics": "universal over every infinite path in each finite graph",
        "safe_case": dataclasses.asdict(safe),
        "violation_case": dataclasses.asdict(violation),
        "checks": {
            "safe_has_exact_p1_p2_certificate": (
                safe.verdict == "SAFE" and safe.p1_verified and safe.p2_verified
            ),
            "violation_has_reachable_bad_lasso": (
                violation.verdict == "VIOLATION"
                and violation.lasso_prefix is not None
                and violation.lasso_cycle is not None
                and violation.lasso_cycle[0] == violation.lasso_cycle[-1]
            ),
            "no_abstain_outcome": (
                {safe.verdict, violation.verdict} == {"SAFE", "VIOLATION"}
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default="artifacts/l2_exact_finite_graph/result.json",
    )
    args = parser.parse_args()
    result = run_examples()
    if not all(result["checks"].values()):
        raise RuntimeError("exact L2 binary self-check failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
