from __future__ import annotations

import numpy as np
import pytest

from wrappers.tdmpc2_probes import (
    fit_forward_speed_probe,
    make_forward_speed_ap_extractor,
    walker_forward_speed_from_physics,
)


def test_forward_speed_probe_fits_and_emits_scalar(tmp_path):
    rng = np.random.default_rng(7)
    z = rng.normal(size=(200, 4)).astype(np.float32)
    speed = (1.5 * z[:, 0] - 0.25 * z[:, 2] + 0.1).astype(np.float32)
    path = tmp_path / "speed.npz"
    np.savez(path, z=z, forward_speed=speed)

    probe = fit_forward_speed_probe(str(path), alpha=1e-4)
    extractor = make_forward_speed_ap_extractor(probe)
    prediction = extractor(z[10])["forward_speed"]
    assert prediction == pytest.approx(float(speed[10]), abs=1e-3)


def test_forward_speed_probe_rejects_missing_or_nonfinite_data(tmp_path):
    missing = tmp_path / "missing.npz"
    np.savez(missing, z=np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="forward_speed"):
        fit_forward_speed_probe(str(missing))

    nonfinite = tmp_path / "nonfinite.npz"
    np.savez(
        nonfinite,
        z=np.ones((2, 3), dtype=np.float32),
        forward_speed=np.asarray([0.0, np.nan], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="finite"):
        fit_forward_speed_probe(str(nonfinite))


def test_forward_speed_uses_task_physics_not_observation_index():
    class DummyPhysics:
        def horizontal_velocity(self):
            return -1.25

    assert walker_forward_speed_from_physics(DummyPhysics()) == -1.25
