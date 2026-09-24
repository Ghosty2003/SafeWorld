from __future__ import annotations

import json

import numpy as np

from cardreamer.eval_l2_persistence_paper import (
    check_support_report,
    rollout_indicators,
)


def test_rollout_indicator_is_paper_p1_p2_and_endpoint_zfree():
    values = np.asarray(
        [
            [0.03, 0.02, 0.005],
            [0.03, 0.04, 0.005],
            [0.03, 0.025, 0.020],
        ],
        dtype=np.float32,
    )
    priorities = np.asarray([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
    indicator, report = rollout_indicators(values, priorities, eta=0.01)
    assert indicator.tolist() == [True, False, False]
    assert report["c_successes"] == 1


def test_support_report_is_bound_to_exact_model_and_scope(tmp_path):
    path = tmp_path / "support.json"
    path.write_text(
        json.dumps(
            {
                "spec_id": "ltl_eventual_hazard_stability",
                "model_sha256": "abc",
                "eta": 0.01,
                "scope": "all_admissible_transitions_sourced_in_zfree",
                "verified": True,
                "method": "sound interval verifier",
            }
        )
    )
    assert check_support_report(path, "abc", 0.01)["verified"] is True
    result = check_support_report(path, "different", 0.01)
    assert result["verified"] is False
    assert "model_sha256" in result["mismatches"]
