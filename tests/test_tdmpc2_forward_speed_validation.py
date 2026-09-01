from __future__ import annotations

import numpy as np

from tdmpc2.validate_forward_speed_probe import (
    _c1_cache_matches_probe,
    validate_c1,
)


def test_c1_reports_longest_consecutive_valid_prefix(tmp_path):
    real = np.full((6, 20), 1.2, dtype=np.float32)
    model = real.copy()
    model[:, 3:] = 0.2  # first failure is depth 4: large error and wrong side of 1m/s
    path = tmp_path / "paired.npz"
    np.savez(
        path,
        model_forward_speed=model,
        real_forward_speed=real,
        episode_id=np.arange(6),
        anchor_step=np.zeros(6),
    )

    result = validate_c1(path)
    assert result["max_prefix_depth_meeting_fixed_c1_criteria"] == 3
    assert not result["passed"]


def test_c1_cache_is_bound_to_exact_probe_artifact(tmp_path):
    path = tmp_path / "paired.npz"
    np.savez(path, probe_sha256=np.asarray("abc"))
    assert _c1_cache_matches_probe(path, "abc")
    assert not _c1_cache_matches_probe(path, "different")
