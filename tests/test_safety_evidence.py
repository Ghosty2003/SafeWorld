"""Regression tests for generic counterexample-first safety verdicts."""

from __future__ import annotations

from core.safety_evidence import (
    ENVIRONMENT_VIOLATION,
    INCONCLUSIVE,
    L2_CALIBRATED_MODEL_SCOPE,
    MODEL_VIOLATION,
    collect_invariant_safety_witnesses,
    decide_safety_verdict,
)


def _strict_height_formula() -> dict:
    return {
        "type": "always", "a": 0, "b": 100,
        "child": {"type": "atom", "dim": "x", "op": ">", "threshold": 0.0},
    }


def test_middle_step_strict_invariant_violation_has_replayable_model_witness():
    trace = [{"x": 1.0}, {"x": -0.2}, {"x": 1.0}]
    witnesses = collect_invariant_safety_witnesses(
        [trace], _strict_height_formula(), source="model", spec_id="any_spec",
        checkpoint_id="checkpoint-A", rollout_provenance=[{"anchor_id": "a0", "actions": [[0.1]] * 3}],
    )
    assert len(witnesses) == 1
    witness = witnesses[0]
    assert witness.source == "model"
    assert witness.time_index == 1
    assert witness.ap_key == "x"
    assert witness.observed_value == -0.2
    assert witness.action_hash
    assert witness.anchor_id == "a0"
    assert witness.to_dict()["witness_id"] == witness.witness_id


def test_final_step_and_equality_are_both_strict_safety_violations():
    final_bad = [{"x": 1.0}, {"x": 0.0}]
    equality_bad = [{"x": 0.0}]
    final_witness = collect_invariant_safety_witnesses(
        [final_bad], _strict_height_formula(), source="model", spec_id="generic",
    )[0]
    equality_witness = collect_invariant_safety_witnesses(
        [equality_bad], _strict_height_formula(), source="model", spec_id="generic",
    )[0]
    assert final_witness.time_index == 1
    assert equality_witness.time_index == 0
    assert equality_witness.observed_value == equality_witness.threshold == 0.0


def test_environment_counterexample_wins_over_model_counterexample_and_l2_statistic():
    formula = _strict_height_formula()
    model = collect_invariant_safety_witnesses(
        [[{"x": -1.0}]], formula, source="model", spec_id="generic",
    )
    environment = collect_invariant_safety_witnesses(
        [[{"x": -2.0}]], formula, source="environment", spec_id="generic",
    )
    decision = decide_safety_verdict(
        model_witnesses=model,
        environment_witnesses=environment,
        l2_calibrated_model_scope=True,
    )
    assert decision.verdict == ENVIRONMENT_VIOLATION
    assert decision.witness == environment[0]


def test_high_calibration_cannot_hide_model_witness_and_absence_is_inconclusive():
    witness = collect_invariant_safety_witnesses(
        [[{"x": -1.0}]], _strict_height_formula(), source="model", spec_id="generic",
    )
    assert decide_safety_verdict(
        model_witnesses=witness, l2_calibrated_model_scope=True,
    ).verdict == MODEL_VIOLATION
    assert decide_safety_verdict().verdict == INCONCLUSIVE
    assert decide_safety_verdict(l2_calibrated_model_scope=True).verdict == L2_CALIBRATED_MODEL_SCOPE


def test_only_unconditional_always_clauses_become_safety_witnesses():
    # (G x) OR (G y) does not permit treating a failed x branch as a violation
    # while y remains true.  This guards against spec-specific shortcut logic.
    formula = {
        "type": "or",
        "left": _strict_height_formula(),
        "right": {
            "type": "always", "a": 0, "b": 100,
            "child": {"type": "atom", "dim": "y", "op": ">", "threshold": 0.0},
        },
    }
    assert not collect_invariant_safety_witnesses(
        [[{"x": -1.0, "y": 1.0}]], formula, source="model", spec_id="generic",
    )


def test_main_returns_model_violation_at_strict_equality_before_lppm_calibration():
    from main import VerifyConfig, verify
    from specs.ltl_specs import get_ltl_spec_by_id

    spec = get_ltl_spec_by_id("ltl_height_safety")
    result = verify(
        [[{"height": 0.6}, {"height": 0.8}]],
        spec,
        VerifyConfig(verbose=False),
    )
    assert result.safety_verdict.verdict == MODEL_VIOLATION
    assert result.safety_verdict.witness is not None
    assert result.safety_verdict.witness.time_index == 0


def test_main_never_labels_an_untrained_lppm_fallback_as_l2_calibrated_safe():
    from main import VerifyConfig, verify
    from specs.ltl_specs import get_ltl_spec_by_id

    spec = get_ltl_spec_by_id("ltl_height_safety")
    result = verify(
        [[{"height": 1.0}, {"height": 1.1}] for _ in range(3)],
        spec,
        VerifyConfig(verbose=False),
    )
    assert result.safety_verdict.verdict == INCONCLUSIVE
    assert not result.is_safe()
