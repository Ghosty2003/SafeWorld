"""
Evaluation protocol helpers.

Metrics stay unchanged; this layer normalizes method outputs into
PredictionResult objects and applies OursMethod's 100-trial stochastic
trajectory protocol.
"""

from __future__ import annotations

from eval.method_wrappers import OursMethod
from eval.metrics import PredictionResult
from eval.semantics import OURS_EVAL_TRIALS


def predict_for_metrics(
    method,
    spec_id: str,
    spec: dict,
    trace_steps: list[dict],
    true_safe: bool,
) -> PredictionResult:
    """Return one metric-ready prediction for a single oracle trajectory."""
    if isinstance(method, OursMethod):
        verdict, confidence = method.predict_trials(
            spec_id,
            spec,
            trace_steps,
            trials=OURS_EVAL_TRIALS,
        )
    else:
        verdict, confidence = method.predict(spec_id, spec, trace_steps)

    return PredictionResult(
        verdict=verdict,
        confidence=confidence,
        true_safe=true_safe,
    )
