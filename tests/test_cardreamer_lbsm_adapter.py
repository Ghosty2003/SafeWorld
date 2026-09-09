from __future__ import annotations

import numpy as np
import pytest

from cardreamer.lbsm_adapter import (
    CarDreamerProductLayout,
    encode_product_states,
    make_roundabout_region_predicates,
)
from cardreamer.probe_common import RING_C


def test_product_encoding_keeps_full_rssm_state_and_auditable_region_fields():
    layout = CarDreamerProductLayout(deter_dim=3, stoch_size=4, position_scale=64.0)
    deter = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    stoch = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    xy = np.asarray([[4.0, -8.0]], dtype=np.float32)
    encoded = encode_product_states(deter, stoch, xy, np.asarray([True]), layout)
    assert encoded.shape == (1, 10)
    assert np.array_equal(encoded[0, :3], deter[0])
    assert np.array_equal(encoded[0, 3:7], stoch.reshape(1, -1)[0])
    assert encoded[0, layout.q_index] == 1.0
    assert layout.position(encoded[0]) == pytest.approx(xy[0])


def test_roundabout_f_is_strictly_inside_retention_i():
    layout = CarDreamerProductLayout(deter_dim=1, stoch_size=1)
    is_accepting, in_retention = make_roundabout_region_predicates(
        layout, accepting_radius=24.9, retention_radius=30.0,
    )

    def state(radius, q):
        x = np.zeros(layout.state_dim, dtype=np.float32)
        x[layout.q_index] = q
        x[layout.xy_slice] = (RING_C + np.asarray([radius, 0.0])) / layout.position_scale
        return x

    assert is_accepting(state(20.0, 1.0))
    assert in_retention(state(20.0, 1.0))
    assert not is_accepting(state(27.0, 0.0))
    assert in_retention(state(27.0, 0.0))
    assert not in_retention(state(31.0, 0.0))

    with pytest.raises(ValueError, match="strictly smaller"):
        make_roundabout_region_predicates(
            layout, accepting_radius=30.0, retention_radius=30.0,
        )
