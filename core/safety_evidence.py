"""Generic safety evidence and verdict precedence.

This module intentionally knows nothing about a particular AP, specification
identifier, or world-model implementation.  It is the common boundary between
three independently evolving layers:

* a wrapper/evaluator supplies model and (optionally) environment AP traces;
* this module turns violated ``always`` clauses into replayable evidence;
* a caller combines that evidence with an L2 result without allowing a sampled
  calibration statistic to hide a concrete counterexample.

``L2_CALIBRATED_MODEL_SCOPE`` is deliberately not an infinite-horizon
environment guarantee.  ``L2_DEDUCTIVE_SAFE`` is reserved for a future
support-wide verifier that actually proves the premises of Theorem 5.2.  The
current sampled LPPM path must never emit that verdict.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Sequence


MODEL_VIOLATION = "MODEL_VIOLATION"
ENVIRONMENT_VIOLATION = "ENVIRONMENT_VIOLATION"
L2_DEDUCTIVE_SAFE = "L2_DEDUCTIVE_SAFE"
L2_CALIBRATED_MODEL_SCOPE = "L2_CALIBRATED_MODEL_SCOPE"
INCONCLUSIVE = "INCONCLUSIVE"

WitnessSource = Literal["model", "environment"]


def _json_default(value: Any) -> Any:
    """Make numpy scalars/arrays and paths safe to place in audit metadata."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SafetyWitness:
    """A replayable finite-trace counterexample to an invariant clause.

    ``trajectory_hash`` identifies the AP trace that was checked.  A wrapper
    can additionally provide an action hash, an anchor/state identifier, and
    any checkpoint-specific replay instructions through ``provenance``.  The
    core does not prescribe their representation, which keeps this usable for
    TD-MPC2, Dreamer, CarDreamer, and future wrappers alike.
    """

    source: WitnessSource
    spec_id: str
    rollout_index: int
    time_index: int
    formula_fragment: dict[str, Any]
    ap_key: str | None = None
    operator: str | None = None
    threshold: float | None = None
    observed_value: float | None = None
    trajectory_hash: str = ""
    action_hash: str | None = None
    checkpoint_id: str | None = None
    anchor_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    witness_id: str = ""

    @classmethod
    def create(
        cls,
        *,
        source: WitnessSource,
        spec_id: str,
        rollout_index: int,
        time_index: int,
        formula_fragment: dict[str, Any],
        trajectory: Sequence[dict[str, Any]],
        ap_key: str | None = None,
        operator: str | None = None,
        threshold: float | None = None,
        observed_value: float | None = None,
        checkpoint_id: str | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> "SafetyWitness":
        provenance = dict(provenance or {})
        actions = provenance.get("actions")
        action_hash = _stable_hash(actions) if actions is not None else None
        trajectory_hash = _stable_hash(list(trajectory))
        payload = {
            "source": source,
            "spec_id": spec_id,
            "rollout_index": rollout_index,
            "time_index": time_index,
            "formula_fragment": formula_fragment,
            "ap_key": ap_key,
            "operator": operator,
            "threshold": threshold,
            "observed_value": observed_value,
            "trajectory_hash": trajectory_hash,
            "action_hash": action_hash,
            "checkpoint_id": checkpoint_id,
            "anchor_id": provenance.get("anchor_id"),
        }
        return cls(
            **payload,
            provenance=provenance,
            witness_id=_stable_hash(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation for an experiment artifact or ledger."""
        return asdict(self)


@dataclass(frozen=True)
class SafetyVerdict:
    """The unambiguous safety conclusion, separate from legacy UI labels."""

    verdict: str
    reason: str
    witness: SafetyWitness | None = None

    @property
    def is_deductively_safe(self) -> bool:
        return self.verdict == L2_DEDUCTIVE_SAFE


def decide_safety_verdict(
    *,
    model_witnesses: Sequence[SafetyWitness] = (),
    environment_witnesses: Sequence[SafetyWitness] = (),
    l2_deductive_verified: bool = False,
    l2_calibrated_model_scope: bool = False,
) -> SafetyVerdict:
    """Apply the non-negotiable evidence order used by every caller.

    Environment evidence wins because it is an observed violation of the
    deployed system.  A model witness is next: it falsifies the model-side
    safety claim on its declared rollout scope.  Only in the absence of both
    can a certificate result be reported.  This is deliberately a pure
    function so wrapper-specific paired replay code cannot accidentally
    implement a different precedence policy.
    """
    if environment_witnesses:
        return SafetyVerdict(
            ENVIRONMENT_VIOLATION,
            "A paired environment rollout violated an invariant safety clause; no sampled "
            "L2 statistic may override this concrete counterexample.",
            environment_witnesses[0],
        )
    if model_witnesses:
        return SafetyVerdict(
            MODEL_VIOLATION,
            "A model rollout violated an invariant safety clause; this scope is unsafe "
            "regardless of p_hat, loss values, or certificate-training outcome.",
            model_witnesses[0],
        )
    if l2_deductive_verified:
        return SafetyVerdict(
            L2_DEDUCTIVE_SAFE,
            "A support-wide L2 verifier established the certificate premises."
        )
    if l2_calibrated_model_scope:
        return SafetyVerdict(
            L2_CALIBRATED_MODEL_SCOPE,
            "Held-out sampled L2 calibration met its configured criterion on the declared "
            "model rollout scope; this is not a real-environment infinite-horizon proof."
        )
    return SafetyVerdict(
        INCONCLUSIVE,
        "No concrete invariant counterexample was observed, but no support-wide L2 proof exists."
    )


def _atom_holds(atom: dict[str, Any], step: dict[str, Any]) -> bool:
    value = float(step.get(atom["dim"], 0.0))
    threshold = float(atom["threshold"])
    if atom["op"] == ">":
        return value > threshold
    if atom["op"] == "<":
        return value < threshold
    raise ValueError(f"Unsupported atomic comparison {atom['op']!r}")


def _formula_holds(formula: dict[str, Any], trajectory: Sequence[dict[str, Any]], t: int) -> bool:
    """Finite-trace Boolean semantics used only to localise an invariant breach.

    Unlike quantitative robustness, strict inequalities deliberately reject
    equality.  That is required for APs such as ``height > h_min`` and is
    independent of any model/spec identifier.
    """
    if t < 0 or t >= len(trajectory):
        return False
    kind = formula["type"]
    if kind == "atom":
        return _atom_holds(formula, trajectory[t])
    if kind == "not":
        return not _formula_holds(formula["child"], trajectory, t)
    if kind == "and":
        return _formula_holds(formula["left"], trajectory, t) and _formula_holds(formula["right"], trajectory, t)
    if kind == "or":
        return _formula_holds(formula["left"], trajectory, t) or _formula_holds(formula["right"], trajectory, t)
    if kind == "implies":
        return (not _formula_holds(formula["left"], trajectory, t)) or _formula_holds(formula["right"], trajectory, t)
    if kind == "next":
        return _formula_holds(formula["child"], trajectory, t + 1)
    if kind == "always":
        hi = min(t + int(formula["b"]), len(trajectory) - 1)
        return all(_formula_holds(formula["child"], trajectory, tp) for tp in range(t + int(formula["a"]), hi + 1))
    if kind == "eventually":
        hi = min(t + int(formula["b"]), len(trajectory) - 1)
        return any(_formula_holds(formula["child"], trajectory, tp) for tp in range(t + int(formula["a"]), hi + 1))
    if kind == "until":
        hi = min(t + int(formula["b"]), len(trajectory) - 1)
        for tp in range(t + int(formula["a"]), hi + 1):
            if _formula_holds(formula["right"], trajectory, tp) and all(
                _formula_holds(formula["left"], trajectory, tpp) for tpp in range(t, tp)
            ):
                return True
        return False
    raise ValueError(f"Unsupported formula node {kind!r}")


def _first_failed_atom(formula: dict[str, Any], trajectory: Sequence[dict[str, Any]], t: int) -> tuple[int, dict[str, Any]] | None:
    """Find one concrete failed atom in a false invariant body when possible."""
    kind = formula["type"]
    if kind == "atom":
        return (t, formula) if not _atom_holds(formula, trajectory[t]) else None
    if kind == "not":
        # A negated atom is represented as its full fragment; its raw AP is
        # still retained in the witness, but no misleading reversed operator
        # is fabricated here.
        if not _formula_holds(formula, trajectory, t):
            return (t, formula)
        return None
    if kind in ("and", "or", "implies"):
        for child_name in ("left", "right"):
            child = formula[child_name]
            if not _formula_holds(child, trajectory, t):
                candidate = _first_failed_atom(child, trajectory, t)
                return candidate or (t, child)
        return None
    if kind == "next":
        return _first_failed_atom(formula["child"], trajectory, t + 1)
    return (t, formula)


def _always_clauses(formula: dict[str, Any]) -> list[dict[str, Any]]:
    """Return unconditional G-clauses, including an Obligation's G half.

    Only conjunction preserves the obligation to satisfy a child independently.
    A ``G`` below ``or``, ``eventually``, ``until``, negation, or implication is
    not an unconditional invariant of the top-level formula, so turning its
    local failure into a safety counterexample would be unsound.
    """
    kind = formula["type"]
    if kind == "always":
        return [formula]
    if kind == "and":
        out = _always_clauses(formula["left"])
        out.extend(_always_clauses(formula["right"]))
        return out
    return []


def collect_invariant_safety_witnesses(
    trajectories: Sequence[Sequence[dict[str, Any]]],
    formula: dict[str, Any],
    *,
    source: WitnessSource,
    spec_id: str,
    checkpoint_id: str | None = None,
    rollout_provenance: Sequence[dict[str, Any]] | None = None,
) -> list[SafetyWitness]:
    """Collect one counterexample per trace for every violated ``G`` clause.

    This intentionally does *not* call an unfulfilled ``F``/``U`` clause a
    safety violation: a finite prefix cannot refute such liveness claims.
    It does handle safety clauses inside larger formulas, e.g. the ``G p``
    half of an Obligation specification.
    """
    clauses = _always_clauses(formula)
    if rollout_provenance is not None and len(rollout_provenance) != len(trajectories):
        raise ValueError(
            "rollout_provenance must have exactly one entry per trajectory; refusing to "
            "attach actions/anchors to the wrong safety witness."
        )
    witnesses: list[SafetyWitness] = []
    for rollout_index, trajectory in enumerate(trajectories):
        provenance = dict(rollout_provenance[rollout_index]) if rollout_provenance else {}
        for clause in clauses:
            lo = int(clause["a"])
            hi = min(int(clause["b"]), len(trajectory) - 1)
            for time_index in range(lo, hi + 1):
                body = clause["child"]
                if _formula_holds(body, trajectory, time_index):
                    continue
                failed = _first_failed_atom(body, trajectory, time_index)
                failed_t, fragment = failed or (time_index, body)
                if fragment.get("type") == "atom":
                    ap_key = fragment["dim"]
                    observed = float(trajectory[failed_t].get(ap_key, 0.0))
                    operator = fragment["op"]
                    threshold = float(fragment["threshold"])
                else:
                    ap_key = operator = threshold = observed = None
                witnesses.append(SafetyWitness.create(
                    source=source,
                    spec_id=spec_id,
                    rollout_index=rollout_index,
                    time_index=failed_t,
                    formula_fragment=fragment,
                    trajectory=trajectory,
                    ap_key=ap_key,
                    operator=operator,
                    threshold=threshold,
                    observed_value=observed,
                    checkpoint_id=checkpoint_id,
                    provenance=provenance,
                ))
                break
    return witnesses
