import hashlib
import json

from tdmpc2.validate_walker_l2_support_contract import validate_contract


def write_json(path, value):
    path.write_text(json.dumps(value))


def test_support_contract_validator_separates_scope_from_proof(tmp_path):
    checkpoint = tmp_path / "walker.pt"
    checkpoint.write_bytes(b"checkpoint")
    counterexample = tmp_path / "counterexample.json"
    write_json(counterexample, {
        "verdict": "CONCRETE_STABLE_CORE_CLOSURE_COUNTEREXAMPLE",
        "support_wide_stable_core_closure_verified": False,
        "attacks": [{
            "replayable_closure_counterexample": True,
            "repetitions": [
                {"closure_violated": True},
                {
                    "closure_violated": True,
                    "bit_exact_state_trace_vs_first": True,
                },
            ],
        }],
    })
    calibration = tmp_path / "calibration.json"
    write_json(calibration, {
        "calibration_fingerprints": ["freshness-witness"],
        "sealed_calibration_governance": {
            "status": "CONSUMED_READ_ONLY",
            "consumed": True,
            "fingerprint_count": 1,
            "read_only": True,
            "reusable": False,
            "recalibration_forbidden": True,
        },
        "paper_warrant_issued": False,
    })
    contract = tmp_path / "contract.json"
    write_json(contract, {
        "paper_warrant_issued": False,
        "primary_scope": {
            "name": "nominal",
            "external_xfrc_applied": "identically zero",
            "support_wide_verified": False,
            "verdict": "ABSTAIN",
            "tdmpc2_commit": "commit",
            "complete_product_state": ["a", "b", "c", "d", "e"],
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            },
        },
        "robustness_scope": {
            "name": "stress",
            "stable_core_closure": False,
            "counterexample": str(counterexample),
        },
        "calibration_governance": {
            "prior_nominal_calibration_consumed": True,
            "read_only": True,
            "recalibration_forbidden": True,
            "reusable_for_new_candidate": False,
            "prior_result": str(calibration),
        },
    })

    result = validate_contract(contract)
    assert result["contract_valid"]
    assert result["verdict"] == "SCOPE_FROZEN"
    assert not result["support_wide_verified"]
    assert not result["paper_warrant_issued"]
