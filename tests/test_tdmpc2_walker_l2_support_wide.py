import json

from tdmpc2.verify_walker_l2_support_wide import verify_support_wide_gate


def write_json(path, value):
    path.write_text(json.dumps(value))


def test_sampled_closure_cannot_promote_without_sound_enclosure(tmp_path):
    candidate = tmp_path / "candidate.json"
    write_json(candidate, {
        "best_seed": 0,
        "all_seeds_passed_gate": True,
        "calibration_launched": False,
        "hyperparameters": {
            "nominal_only_stable_core": True,
            "include_recovery_data": False,
            "bounded_countdown_envelope": False,
            "global_countdown_envelope": False,
        },
        "runs": [{
            "seed": 0,
            "diagnostic": {
                "p1_transition_pass_rate": 1.0,
                "p2_transition_pass_rate": 1.0,
                "pathwise_pass_rate": 1.0,
                "sampled_zfree_closure_rate": 1.0,
                "gates": {
                    "pathwise": True,
                    "all_automaton_states_observed": True,
                },
            },
        }],
    })
    support = tmp_path / "support.json"
    write_json(support, {
        "contract_valid": True,
        "paper_warrant_issued": False,
        "support_wide_verified": False,
    })

    result = verify_support_wide_gate(candidate, support)
    assert result["candidate_fit_passed"]
    assert not result["support_wide_verified"]
    assert not result["fresh_calibration_allowed"]
    assert result["formal_infinite_horizon_verdict"] == "ABSTAIN"


def test_bound_sound_enclosure_can_open_fresh_calibration_gate(tmp_path):
    candidate = tmp_path / "candidate.json"
    support = tmp_path / "support.json"
    write_json(candidate, {
        "best_seed": 0,
        "all_seeds_passed_gate": True,
        "calibration_launched": False,
        "hyperparameters": {
            "nominal_only_stable_core": True,
            "include_recovery_data": False,
            "bounded_countdown_envelope": False,
            "global_countdown_envelope": False,
        },
        "runs": [{"seed": 0, "diagnostic": {
            "pathwise_pass_rate": 1.0,
            "sampled_zfree_closure_rate": 1.0,
            "gates": {"pathwise": True, "all_automaton_states_observed": True},
        }}],
    })
    write_json(support, {
        "contract_valid": True,
        "paper_warrant_issued": False,
        "support_wide_verified": False,
    })
    from tdmpc2.verify_walker_l2_support_wide import sha256_file
    enclosure = tmp_path / "enclosure.json"
    write_json(enclosure, {
        "sound_backend": True,
        "quantification": "all_admissible_successors",
        "candidate_diagnostics_sha256": sha256_file(candidate),
        "support_validation_sha256": sha256_file(support),
        "p1_p2_verified": True,
        "zfree_closure_verified": True,
    })

    result = verify_support_wide_gate(candidate, support, enclosure_path=enclosure)
    assert result["support_wide_verified"]
    assert result["fresh_calibration_allowed"]
    assert not result["paper_warrant_issued"]
