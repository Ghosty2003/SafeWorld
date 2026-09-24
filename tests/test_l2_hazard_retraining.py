from __future__ import annotations

from cardreamer.retrain_l2_hazard_v import Record, stratified_split


def _record(index: int, hazard_min: float) -> Record:
    return Record(
        fingerprint=f"fingerprint-{index}",
        trajectory=[{"hazard_dist": hazard_min}],
        hazard_min=hazard_min,
        source="test",
    )


def test_record_strata_use_strict_safety_boundary():
    assert _record(0, 0.0).stratum == "trap"
    assert _record(1, -0.1).stratum == "trap"
    assert _record(2, 0.1).stratum == "near_1m"
    assert _record(3, 1.0).stratum == "near_1m"
    assert _record(4, 1.1).stratum == "near_3m"
    assert _record(5, 3.1).stratum == "safe_control"


def test_stratified_split_is_disjoint_and_caps_abundant_safe_strata():
    hazards = [-0.1] * 5 + [0.5] * 5 + [2.0] * 10 + [10.0] * 10
    records = [_record(index, hazard) for index, hazard in enumerate(hazards)]
    train, diagnostic, counts = stratified_split(
        records, seed=7, diagnostic_fraction=0.20, max_per_safe_stratum=5
    )

    train_ids = {record.fingerprint for record in train}
    diagnostic_ids = {record.fingerprint for record in diagnostic}
    assert train_ids.isdisjoint(diagnostic_ids)
    assert counts["trap"] == {
        "available_unique": 5,
        "selected": 5,
        "train": 4,
        "diagnostic": 1,
    }
    assert counts["near_3m"]["available_unique"] == 10
    assert counts["near_3m"]["selected"] == 5
    assert counts["safe_control"]["selected"] == 5
