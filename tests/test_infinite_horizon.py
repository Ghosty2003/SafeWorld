from __future__ import annotations

import numpy as np

from core.infinite_horizon import (
    ABSTAIN,
    SAFE,
    VIOLATE,
    aggregate_exact_run_verdict,
    build_sampled_partition_graph,
    detect_exact_lasso,
    linear_probe_simnorm_bounds,
    replay_validate_exact_lasso,
    sampled_bad_scc_candidates,
)


def test_linear_probe_bounds_are_exact_on_product_of_simplexes():
    coefficients = np.asarray([1.0, -2.0, 4.0, 3.0])
    lower, upper = linear_probe_simnorm_bounds(coefficients, 0.5, group_dim=2)
    assert lower == 1.5  # 0.5 + min(1,-2) + min(4,3)
    assert upper == 5.5  # 0.5 + max(1,-2) + max(4,3)


def test_exact_lasso_with_bad_in_cycle_is_violation():
    lasso = detect_exact_lasso(["a", "b", "c", "b"], [False, False, True, False])
    assert lasso is not None
    assert lasso.prefix_length == 1
    assert lasso.cycle_length == 2
    assert lasso.bad_cycle_offsets == (1,)
    assert lasso.verdict == VIOLATE


def test_exact_lasso_with_safe_cycle_is_safe_for_that_execution():
    lasso = detect_exact_lasso(["a", "b", "c", "b"], [True, False, False, False])
    assert lasso is not None
    assert lasso.verdict == SAFE


def test_replay_validation_requires_every_edge_and_exact_closure():
    states = [0, 1, 2, 1]
    lasso = detect_exact_lasso([str(x) for x in states], [False] * 4)
    assert lasso is not None
    transitions = {1: 2, 2: 1}
    result = replay_validate_exact_lasso(
        states,
        lasso,
        fingerprint=lambda value: str(value),
        step=lambda value: transitions[value],
    )
    assert result["verified"] is True
    assert result["verdict"] == SAFE

    bad_result = replay_validate_exact_lasso(
        states,
        lasso,
        fingerprint=lambda value: str(value),
        step=lambda value: 99,
    )
    assert bad_result["verified"] is False
    assert bad_result["verdict"] == ABSTAIN


def test_sampled_bad_scc_is_never_promoted_to_violation():
    features = np.asarray([[[0.1], [1.1], [0.2], [1.2]]])
    bad = np.asarray([[False, True, False, True]])
    graph, initial = build_sampled_partition_graph(
        features, bad, cell_widths=np.asarray([1.0])
    )
    candidates = sampled_bad_scc_candidates(graph, initial)
    assert graph.graph["sound_for_verdict"] is False
    assert len(candidates) == 1
    assert candidates[0]["verdict"] == ABSTAIN


def test_exact_run_aggregation_is_fail_closed():
    safe = detect_exact_lasso(["a", "b", "b"], [False, False, False])
    violate = detect_exact_lasso(["a", "b", "b"], [False, True, True])
    assert aggregate_exact_run_verdict([safe, safe]) == SAFE
    assert aggregate_exact_run_verdict([safe, violate]) == VIOLATE
    assert aggregate_exact_run_verdict([safe, None]) == ABSTAIN
