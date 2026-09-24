from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cardreamer.collect_l2_persistence_latents import (
    pack_latent_rollouts,
    recovered_after_failure,
    stopping_rule_met,
)


def test_pack_latent_rollouts_preserves_alignment_and_strict_boundary():
    rollouts = [
        {
            "aps": [{"hazard_dist": 1.0}, {"hazard_dist": 0.0}],
            "deter": np.ones((2, 3), dtype=np.float16),
            "stoch": np.ones((2, 2, 2), dtype=np.float16),
        },
        {
            "aps": [{"hazard_dist": 2.0}, {"hazard_dist": 1.0}],
            "deter": np.zeros((2, 3), dtype=np.float16),
            "stoch": np.zeros((2, 2, 2), dtype=np.float16),
        },
    ]
    hazard, deter, stoch, trap, fingerprints = pack_latent_rollouts(rollouts)
    assert hazard.shape == (2, 2)
    assert deter.shape == (2, 2, 3)
    assert stoch.shape == (2, 2, 2, 2)
    assert trap.tolist() == [True, False]
    assert len(set(fingerprints.tolist())) == 2


def test_recovered_after_failure_requires_a_long_enough_safe_suffix():
    hazard = np.asarray(
        [
            [2, -1, 2, 2, 2, 2, 2],
            [2, 2, 2, -1, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [-1, 2, -1, 2, 2, 2, 2],
        ], dtype=np.float32,
    )
    assert recovered_after_failure(hazard, min_recovery_suffix=5).tolist() == [True, False, False, False]


def test_calibration_collection_stops_only_at_fixed_rollout_budget():
    args = SimpleNamespace(
        calibration=True,
        max_rollouts=100,
        target_recovered_rollouts=1,
        target_trap_rollouts=1,
    )
    summary = {
        "n_rollouts": 99,
        "n_unique_recovered_rollouts": 50,
        "n_unique_trap_rollouts": 50,
    }
    assert stopping_rule_met(summary, args) is False
    summary["n_rollouts"] = 100
    assert stopping_rule_met(summary, args) is True
