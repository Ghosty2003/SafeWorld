import numpy as np
import torch

from tdmpc2.train_walker_goal_stability_lppm import (
    ETA,
    STATE_TO_IDX,
    StableCoreAnchoredLPPM,
    automaton_trace,
    cumulative_odd_visits,
    remaining_bad_budget_target,
    recovery_training_enabled,
    rowwise_masked_cvar,
    stable_core_mask,
    transition_index,
    upper_tail_mean,
)


def test_nominal_only_stable_core_does_not_enable_recovery_data():
    assert not recovery_training_enabled(
        structured_stable_core=True,
        include_recovery_data=False,
        nominal_only_stable_core=True,
    )
    assert recovery_training_enabled(
        structured_stable_core=True,
        include_recovery_data=False,
        nominal_only_stable_core=False,
    )


def test_nominal_only_stable_core_rejects_ambiguous_modes():
    for structured, include_recovery in ((False, False), (True, True)):
        try:
            recovery_training_enabled(
                structured_stable_core=structured,
                include_recovery_data=include_recovery,
                nominal_only_stable_core=True,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous nominal-only mode was accepted")


def test_transition_index_excludes_masked_external_steps():
    valid = np.asarray([
        [True, False, True],
        [False, True, True],
    ])
    trajectory, time_index = transition_index(
        np.asarray([0, 1]), horizon=4, transition_valid=valid
    )
    assert list(zip(trajectory.tolist(), time_index.tolist())) == [
        (0, 0), (0, 2), (1, 1), (1, 2),
    ]


def structured_context(
    core, *, progress=None, step=None, bad=None, safe=None, odd=None
):
    zeros = torch.zeros_like(core)
    return torch.stack(
        (
            core,
            zeros if progress is None else progress,
            zeros if step is None else step,
            zeros if bad is None else bad,
            zeros if safe is None else safe,
            zeros if odd is None else odd,
        ),
        dim=-1,
    )


def test_cumulative_odd_visits_increments_only_after_odd_sources():
    priorities = np.asarray([[1, 0, 1, 0]], dtype=np.int64)
    counts = cumulative_odd_visits(
        priorities, np.asarray([4.0], dtype=np.float32)
    )
    assert counts.tolist() == [[4.0, 5.0, 5.0, 6.0]]


def test_joint_automaton_waits_for_goal_and_recovers_height():
    progress = np.asarray([[0.0, 21.0, 22.0, 23.0, 24.0]])
    height = np.asarray([[1.2, 1.2, 0.2, 1.2, 1.2]])
    q, priorities = automaton_trace(progress, height)
    assert q.tolist() == [[
        STATE_TO_IDX["waiting_safe"],
        STATE_TO_IDX["waiting_safe"],
        STATE_TO_IDX["achieved_safe"],
        STATE_TO_IDX["achieved_bad"],
        STATE_TO_IDX["achieved_safe"],
    ]]
    assert priorities.tolist() == [[1, 1, 0, 1, 0]]


def test_remaining_bad_budget_is_nonincreasing_and_drops_on_odd_sources():
    priorities = np.asarray([[1, 1, 0, 1, 0]], dtype=np.int64)
    target = remaining_bad_budget_target(priorities)
    assert np.all(target[:, 1:] <= target[:, :-1])
    assert np.allclose(target[0], [3 * ETA, 2 * ETA, ETA, ETA, 0.0])


def test_remaining_bad_budget_ignores_masked_intervention_transition():
    priorities = np.asarray([[1, 1, 1, 0]], dtype=np.int64)
    valid = np.asarray([[True, False, True]])
    target = remaining_bad_budget_target(
        priorities, transition_valid=valid
    )
    assert np.allclose(target[0], [2 * ETA, ETA, ETA, 0.0])


def test_structured_lppm_anchors_only_stable_achieved_safe_states():
    q = np.asarray([[
        STATE_TO_IDX["achieved_safe"],
        STATE_TO_IDX["achieved_safe"],
        STATE_TO_IDX["achieved_bad"],
    ]])
    dwell = np.asarray([[0.99, 1.0, 1.0]], dtype=np.float32)
    core = stable_core_mask(q, dwell)
    assert core.tolist() == [[False, True, False]]

    torch.manual_seed(0)
    model = StableCoreAnchoredLPPM(2, len(STATE_TO_IDX), 1, hidden_dim=4)
    values = model(
        torch.zeros((1, 3, 2)),
        torch.from_numpy(q),
        structured_context(torch.from_numpy(core).float()),
    )[..., 0]
    assert values[0, 0] > 0
    assert values[0, 1] == 0
    assert values[0, 2] > 0


def test_trajectory_cvar_selects_hard_masked_transitions_and_trajectories():
    violations = torch.tensor([[0.0, 1.0, 3.0, 2.0], [4.0, 0.0, 2.0, 1.0]])
    mask = torch.tensor([[False, True, True, True], [True, False, True, False]])
    per_trajectory = rowwise_masked_cvar(
        violations, mask, tail_fraction=0.5
    )
    assert torch.allclose(per_trajectory, torch.tensor([2.5, 4.0]))
    assert upper_tail_mean(per_trajectory, 0.5) == 4.0


def test_safe_dwell_envelope_is_monotone_and_reaches_zero():
    model = StableCoreAnchoredLPPM(
        2, len(STATE_TO_IDX), 1, hidden_dim=4, safe_dwell_envelope=True
    )
    q = torch.full((1, 3), STATE_TO_IDX["achieved_safe"], dtype=torch.long)
    fraction = torch.tensor([[0.1, 0.2, 1.0]])
    context = structured_context(fraction, safe=fraction)
    values = model(torch.zeros((1, 3, 2)), q, context)[0, :, 0]
    assert values[0] > values[1] > values[2]
    assert values[2] == 0.0


def test_waiting_progress_envelope_is_nonincreasing_with_running_max_progress():
    model = StableCoreAnchoredLPPM(
        2, len(STATE_TO_IDX), 1,
        hidden_dim=4, waiting_progress_envelope=True,
    )
    q = torch.full((1, 3), STATE_TO_IDX["waiting_safe"], dtype=torch.long)
    fraction = torch.zeros((1, 3))
    max_progress = torch.tensor([[1.0, 1.0, 2.0]])
    context = structured_context(fraction, progress=max_progress)
    values = model(torch.randn((1, 3, 2)), q, context)[0, :, 0]
    assert values[0] == values[1]
    assert values[1] > values[2]


def test_bounded_countdown_strictly_decreases_on_waiting_plateau():
    model = StableCoreAnchoredLPPM(
        2, len(STATE_TO_IDX), 1,
        hidden_dim=4,
        waiting_progress_envelope=True,
        bounded_countdown_envelope=True,
        countdown_step=0.03,
        goal_deadline_steps=400,
    )
    q = torch.full((1, 3), STATE_TO_IDX["waiting_safe"], dtype=torch.long)
    zeros = torch.zeros((1, 3))
    steps = torch.tensor([[10.0, 11.0, 12.0]])
    context = structured_context(zeros, progress=zeros, step=steps)
    values = model(torch.randn((1, 3, 2)), q, context)[0, :, 0]
    assert torch.allclose(
        values[:-1] - values[1:], torch.tensor([0.03, 0.03]), atol=1e-5
    )


def test_global_countdown_crosses_automaton_states_without_resetting():
    model = StableCoreAnchoredLPPM(
        2, len(STATE_TO_IDX), 1,
        hidden_dim=4,
        global_countdown_envelope=True,
        countdown_step=0.03,
        global_deadline_steps=600,
    )
    q = torch.tensor([[
        STATE_TO_IDX["waiting_safe"],
        STATE_TO_IDX["achieved_safe"],
        STATE_TO_IDX["achieved_bad"],
    ]])
    zeros = torch.zeros((1, 3))
    steps = torch.tensor([[250.0, 251.0, 252.0]])
    context = structured_context(zeros, step=steps)
    values = model(torch.randn((1, 3, 2)), q, context)[0, :, 0]
    assert torch.allclose(
        values[:-1] - values[1:], torch.tensor([0.03, 0.03]), atol=1e-5
    )


def test_odd_budget_drops_only_after_odd_priority_sources():
    model = StableCoreAnchoredLPPM(
        2, len(STATE_TO_IDX), 1,
        hidden_dim=4,
        odd_budget_envelope=True,
        odd_budget_steps=20,
        countdown_step=0.03,
    )
    q = torch.tensor([[
        STATE_TO_IDX["waiting_safe"],
        STATE_TO_IDX["achieved_safe"],
        STATE_TO_IDX["achieved_bad"],
        STATE_TO_IDX["achieved_safe"],
    ]])
    zeros = torch.zeros((1, 4))
    odd = torch.tensor([[4.0, 5.0, 5.0, 6.0]])
    context = structured_context(zeros, odd=odd)
    values = model(torch.randn((1, 4, 2)), q, context)[0, :, 0]
    assert torch.allclose(values[0] - values[1], torch.tensor(0.03), atol=1e-5)
    assert values[1] == values[2]
    assert torch.allclose(values[2] - values[3], torch.tensor(0.03), atol=1e-5)
