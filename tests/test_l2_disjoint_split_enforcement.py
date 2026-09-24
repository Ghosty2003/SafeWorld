"""
tests/test_l2_disjoint_split_enforcement.py

External-review finding #3: core/lppm/trainer.py::fit_lppm() and
core/lppm/calibrator.py::calibrate_lppm()'s opt-in disjointness check
(`warn_if_overlap=True`) only issued a Python `warnings.warn()` on overlap
-- never a hard failure -- despite main.py's own comment at the one real
call site claiming "guarded here so it fails loudly rather than silently".
That comment was simply wrong: passing `warn_if_overlap=True` cannot raise
anything, by construction. Worse, main.py::verify()'s fit_lppm_params=True
branch passed the SAME `trajectories` list as both the training set and
`calib_trajectories`, guaranteeing 100% overlap every time that branch runs
-- previously silently downgraded to a warning.

Fixed: overlap now raises by default (once the caller opts in by passing
calib_trajectories/train_trajectories at all), unless the caller explicitly
passes allow_overlap=True with a non-empty overlap_reason (forced,
auditable opt-out). main.py::verify()'s fit_lppm_params=True branch now
requires a separately-supplied VerifyConfig.lppm_train_trajectories --
reusing `trajectories` for both roles is no longer possible even
accidentally.
"""

from __future__ import annotations

import warnings

import pytest

from core.lppm.automaton import build_parity_automaton
from core.lppm.calibrator import calibrate_lppm
from core.lppm.trainer import fit_lppm


def _safety_spec(ap: str = "hazard") -> dict:
    return {
        "id": "test_disjoint", "formula": {
            "type": "always", "a": 0, "b": 100000,
            "child": {"type": "atom", "dim": ap, "threshold": 0.5, "op": "<"},
        },
        "aps": [ap],
    }


def _persistence_spec(ap: str = "hazard") -> dict:
    """Recoverable co-Buchi spec eligible for main.verify's LPPM route."""
    return {
        "id": "test_disjoint_persistence",
        "formula": {
            "type": "eventually", "a": 0, "b": 100000,
            "child": {
                "type": "always", "a": 0, "b": 100000,
                "child": {"type": "atom", "dim": ap, "threshold": 0.5, "op": "<"},
            },
        },
        "aps": [ap],
    }


def _trajectories(n: int = 3, t: int = 4) -> list[list[dict]]:
    return [[{"hazard": 0.0} for _ in range(t)] for _ in range(n)]


# ─── fit_lppm() ──────────────────────────────────────────────────────────

def test_fit_lppm_raises_on_overlap_by_default():
    spec = _safety_spec()
    dpa = build_parity_automaton(spec)
    shared = _trajectories()
    with pytest.raises(ValueError, match="(?i)overlap|disjoint"):
        fit_lppm(shared, dpa, spec, n_epochs=1, calib_trajectories=shared)


def test_fit_lppm_allow_overlap_without_reason_raises():
    spec = _safety_spec()
    dpa = build_parity_automaton(spec)
    shared = _trajectories()
    with pytest.raises(ValueError, match="(?i)reason"):
        fit_lppm(shared, dpa, spec, n_epochs=1, calib_trajectories=shared, allow_overlap=True)


def test_fit_lppm_allow_overlap_with_reason_warns_and_proceeds():
    spec = _safety_spec()
    dpa = build_parity_automaton(spec)
    shared = _trajectories()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_lppm(
            shared, dpa, spec, n_epochs=1, calib_trajectories=shared,
            allow_overlap=True, overlap_reason="deliberate ablation test",
        )
    assert any("overlap" in str(w.message).lower() for w in caught)
    assert result["n_transitions"] > 0


def test_fit_lppm_disjoint_trajectories_do_not_raise_or_warn():
    spec = _safety_spec()
    dpa = build_parity_automaton(spec)
    train = _trajectories()
    calib = _trajectories()  # distinct list objects, no shared identity
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_lppm(train, dpa, spec, n_epochs=1, calib_trajectories=calib)
    assert not any("overlap" in str(w.message).lower() for w in caught)


# ─── calibrate_lppm() ────────────────────────────────────────────────────

def test_calibrate_lppm_raises_on_overlap_by_default():
    spec = _safety_spec()
    dpa = build_parity_automaton(spec)
    shared = _trajectories()
    with pytest.raises(ValueError, match="(?i)overlap|disjoint"):
        calibrate_lppm(shared, dpa, spec, train_trajectories=shared)


def test_calibrate_lppm_allow_overlap_with_reason_proceeds():
    spec = _safety_spec()
    dpa = build_parity_automaton(spec)
    shared = _trajectories()
    result = calibrate_lppm(
        shared, dpa, spec, train_trajectories=shared,
        allow_overlap=True, overlap_reason="deliberate ablation test",
    )
    assert result.p_hat_gamma >= 0.0


# ─── main.py::verify()'s fit_lppm_params branch ─────────────────────────

def test_verify_fit_lppm_params_without_train_split_raises():
    """
    Previously this branch silently reused `trajectories` for both training
    and calibration (100% overlap, only ever warned). Now it must require a
    separately-supplied disjoint training split and refuse to proceed
    without one.
    """
    from main import VerifyConfig, verify

    # Strict Safety is intentionally routed away from LPPM because its
    # absorbing odd trap makes positive-eta P2 infeasible. Persistence keeps
    # this test focused on the disjoint-split branch it is meant to exercise.
    spec = _persistence_spec()
    trajectory = [{"hazard": 0.0}, {"hazard": 0.0}, {"hazard": 0.0}]
    cfg = VerifyConfig(verbose=False, fit_lppm_params=True, lppm_epochs=1)
    with pytest.raises(ValueError, match="(?i)lppm_train_trajectories|disjoint|training split"):
        verify([trajectory, trajectory], spec, cfg)


def test_verify_fit_lppm_params_with_disjoint_split_succeeds():
    from main import VerifyConfig, verify

    spec = _persistence_spec()
    calib_trajectory = [{"hazard": 0.0}, {"hazard": 0.0}, {"hazard": 0.0}]
    train_trajectory = [{"hazard": 0.0}, {"hazard": 0.0}, {"hazard": 0.0}]
    cfg = VerifyConfig(
        verbose=False, fit_lppm_params=True, lppm_epochs=1,
        lppm_train_trajectories=[train_trajectory, train_trajectory],
    )
    result = verify([calib_trajectory, calib_trajectory], spec, cfg)
    assert result.lppm is not None
