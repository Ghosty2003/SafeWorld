from __future__ import annotations

import numpy as np

from cardreamer.collect_l2_hazard_training_data import (
    AP_KEYS,
    _pack_trajectories,
    _summarize,
)
from wrappers.cardreamer_wrapper import CarDreamerWrapper


def _step(hazard: float) -> dict[str, float]:
    return {
        "hazard_dist": hazard,
        "near_obstacle": hazard - 5.0,
        "goal_dist": 2.5,
        "velocity": 1e6,
    }


def test_pack_marks_strict_boundary_as_trap_and_is_deterministic():
    trajectories = [
        [_step(2.0), _step(1.0)],
        [_step(1.0), _step(0.0)],
        [_step(1.0), _step(-0.5)],
    ]
    values, trap, fingerprints = _pack_trajectories(trajectories)

    assert values.shape == (3, 2, len(AP_KEYS))
    assert trap.tolist() == [False, True, True]
    assert len(set(fingerprints.tolist())) == 3
    assert np.array_equal(fingerprints, _pack_trajectories(trajectories)[2])


def test_summary_counts_unique_trap_trajectories_and_transitions(tmp_path):
    trajectories = [
        [_step(2.0), _step(1.0)],
        [_step(0.0), _step(-0.5)],
        [_step(0.0), _step(-0.5)],
    ]
    values, trap, fingerprints = _pack_trajectories(trajectories)
    summary = _summarize(
        [
            {
                "path": tmp_path / "batch_0000.npz",
                "values": values,
                "trap": trap,
                "fingerprints": fingerprints,
                "seed": 1,
            }
        ]
    )

    assert summary["n_rollouts"] == 3
    assert summary["n_trap_rollouts"] == 2
    assert summary["n_unique_trajectories"] == 2
    assert summary["n_unique_trap_trajectories"] == 1
    assert summary["n_trap_transitions"] == 4


def test_epsilon_random_probability_is_validated_without_loading_jax():
    CarDreamerWrapper(epsilon_random=0.0)
    CarDreamerWrapper(epsilon_random=1.0)

    for invalid in (-0.01, 1.01):
        try:
            CarDreamerWrapper(epsilon_random=invalid)
        except ValueError as error:
            assert "epsilon_random" in str(error)
        else:
            raise AssertionError("invalid epsilon_random should be rejected")


def test_replay_anchor_selection_configuration_is_validated():
    CarDreamerWrapper(replay_anchor_strategy="uniform")
    CarDreamerWrapper(
        replay_anchor_strategy="low_hazard", replay_low_hazard_fraction=0.25
    )

    for kwargs in (
        {"replay_anchor_strategy": "unknown"},
        {"replay_low_hazard_fraction": 0.0},
        {"replay_low_hazard_fraction": 1.01},
    ):
        try:
            CarDreamerWrapper(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid anchor configuration accepted: {kwargs}")
