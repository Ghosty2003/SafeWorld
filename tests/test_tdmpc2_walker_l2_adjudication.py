import json

from tdmpc2.adjudicate_walker_l2_candidate import adjudicate


def _write(path, value):
    path.write_text(json.dumps(value))


def test_finite_core_exit_refutes_candidate_but_not_fg_property(tmp_path):
    contract = tmp_path / "contract.json"
    validation = tmp_path / "validation.json"
    candidate = tmp_path / "candidate.json"
    action = tmp_path / "action.json"
    _write(contract, {
        "primary_scope": {
            "checkpoint": {"sha256": "checkpoint"},
            "tdmpc2_commit": "commit",
            "external_xfrc_applied": "identically zero",
        },
    })
    _write(validation, {"contract_valid": True, "contract": str(contract)})
    _write(candidate, {
        "all_seeds_passed_gate": True,
        "hyperparameters": {
            "structured_stable_core": True,
            "stable_core_dwell_steps": 150,
            "odd_budget_envelope": True,
            "odd_budget_steps": 1000,
        },
        "structured_constraint": {"stable_core_value": 0.0},
    })
    _write(action, {
        "scope": "real MuJoCo from a nominally reached 150-step stable-core state",
        "checkpoint_sha256": "checkpoint",
        "tdmpc2_commit": "commit",
        "finite_action_block_in_declared_action_range": True,
        "planner_action_support_conditions_hold": True,
        "core": {"height_m": 1.0, "progress_m": 21.0},
        "persistent_bad_candidates": [{"exits_core": True, "first_bad_state_offset": 6}],
        "exact_bad_cycle_searches": {"trial": {"found": False}},
        "universal_support_fg_safe_violated": False,
    })

    result = adjudicate(candidate, validation, action)
    assert result["candidate_validity"] == "REFUTED"
    assert result["property_verdict"] == "ABSTAIN"
    assert "refutes P1" in result["candidate_refutation"]
    assert "candidate_lppm_verdict" not in result
    assert "candidate_violation" not in result
