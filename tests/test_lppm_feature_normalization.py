from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from core.lppm.model import NeuralLPPM, predict_learned_lppm_value


def _params():
    torch.manual_seed(0)
    model = NeuralLPPM(1, 1, 1, hidden_dim=4, q_embed_dim=2)
    return {
        "backend": "torch_mlp",
        "weights": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "feature_keys": ["hazard_dist"],
        "feature_mean": [10.0],
        "feature_scale": [2.0],
        "state_to_idx": {"ok": 0},
        "odd_to_idx": {1: 0},
        "hidden_dim": 4,
        "q_embed_dim": 2,
    }, model


def test_predict_applies_saved_feature_normalization():
    params, model = _params()
    with torch.no_grad():
        expected = float(model(torch.tensor([[1.0]]), torch.tensor([0]))[0, 0])
    actual = predict_learned_lppm_value({"hazard_dist": 12.0}, "ok", 1, params)
    assert actual == pytest.approx(expected)


def test_predict_rejects_partial_or_wrong_length_preprocessing():
    params, _ = _params()
    del params["feature_scale"]
    with pytest.raises(ValueError, match="requires both"):
        predict_learned_lppm_value({"hazard_dist": 12.0}, "ok", 1, params)

    params, _ = _params()
    params["feature_scale"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="length"):
        predict_learned_lppm_value({"hazard_dist": 12.0}, "ok", 1, params)
