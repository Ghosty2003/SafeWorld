from __future__ import annotations

import numpy as np
import torch

from cardreamer.train_l2_persistence_latent_v import (
    _rates,
    automaton_trace,
    stabilization_targets,
    stratified_split,
    trajectory_cvar,
    unique_first_indices,
)
from core.lppm.automaton import build_parity_automaton
from specs.ltl_specs import get_ltl_spec_by_id
from utils.spec_analysis import analyze_spec_structure


def test_stabilization_target_counts_down_through_recovery_step():
    hazard = np.asarray(
        [
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [2.0, -0.1, 2.0, 2.0, 2.0],
            [-0.1, -0.2, 2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    target = stabilization_targets(hazard, eta=0.01)
    np.testing.assert_allclose(target[0], [0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(target[1], [0.01, 0.01, 0.01, 0.0, 0.0])
    np.testing.assert_allclose(target[2], [0.02, 0.02, 0.01, 0.0, 0.0])


def test_stabilization_target_can_add_training_slack_above_formal_eta():
    target = stabilization_targets(np.asarray([[2.0, -1.0, 2.0, 2.0]]), eta=0.015)
    np.testing.assert_allclose(target[0], [0.015, 0.015, 0.015, 0.0])


def test_stratified_split_has_every_stratum_and_no_fingerprint_leakage():
    hazard = np.asarray(
        [
            [-1, 2], [0, 2],
            [0.5, 2], [0.7, 2],
            [2, 2], [2.5, 2],
            [4, 4], [5, 5],
        ], dtype=np.float32,
    )
    fingerprints = np.asarray([f"fp-{i}" for i in range(len(hazard))])
    train, diagnostic, counts = stratified_split(hazard, fingerprints, 7, 0.5)
    assert len(train) == len(diagnostic) == 4
    assert all(item == {"total": 2, "train": 1, "diagnostic": 1} for item in counts.values())
    assert set(fingerprints[train]) & set(fingerprints[diagnostic]) == set()


def test_trajectory_cvar_averages_each_paths_worst_tail():
    losses = torch.tensor([[0.0, 1.0, 2.0, 3.0], [4.0, 0.0, 0.0, 2.0]])
    # Worst 50%: (3+2)/2 and (4+2)/2, then average paths.
    assert float(trajectory_cvar(losses, 0.5)) == 2.75


def test_unique_first_indices_deduplicates_without_reordering():
    fingerprints = np.asarray(["a", "b", "a", "c", "b"])
    assert unique_first_indices(fingerprints).tolist() == [0, 1, 3]


def test_paper_p1_uses_every_transition_and_p2_only_odd_sources():
    vc = np.asarray([1.0, 1.0, 1.0])
    vn = np.asarray([0.9, 1.1, 0.98])
    priorities = np.asarray([0, 0, 1])
    p1, p2, p1_pass, p2_pass = _rates(vc, vn, priorities, eta=0.01)
    assert p1.tolist() == [True, True, True]
    assert p2.tolist() == [False, False, True]
    assert p1_pass.tolist() == [True, False, True]
    assert p2_pass.tolist() == [True]


def test_persistence_trace_priority_belongs_to_source_product_state():
    spec = get_ltl_spec_by_id("ltl_eventual_hazard_stability")
    assert spec is not None
    spec["analysis"] = analyze_spec_structure(spec)
    dpa = build_parity_automaton(spec)
    q_curr, q_after, priorities = automaton_trace(
        np.asarray([[2.0, -1.0, 2.0]], dtype=np.float32), dpa
    )
    assert q_curr.tolist() == [[0, 0, 1]]
    assert q_after.tolist() == [[0, 1, 0]]
    assert priorities.tolist() == [[0, 0, 1]]
