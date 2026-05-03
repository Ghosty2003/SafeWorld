"""
core/soft_buchi.py

Differentiable Büchi Monitoring (Section 4.5).

Replaces hard Boolean predicate evaluation with a sigmoid relaxation so that
the acceptance score is differentiable with respect to the trajectory — enabling
gradient-based falsification (searching for violating trajectories).

Equations (paper §4.5)
----------------------
Soft AP:
  p̃_i(z) = σ( (z_{d_i} - τ_i) / T )          [Eq. 3]

Soft guard (transition guarded by P⁺, P⁻):
  g̃(z, P⁺, P⁻) = Π_{p∈P⁺} p̃(z)  ×  Π_{p∈P⁻} (1 - p̃(z))   [Eq. 4]

Soft state occupancy (forward propagation):
  w₀(q_init) = 1,  w₀(q) = 0 otherwise
  w_{t+1}(q') = Σ_q  w_t(q) · g̃(z_t, P⁺_{q→q'}, P⁻_{q→q'})

Acceptance score:
  a* = max_{t, q∈F}  w_t(q)

A score near 1 indicates a likely violation; near 0 indicates no violation
pattern detected.

Status note
-----------
The world model transition function is a non-differentiable numpy call in this
implementation (Section 4.2 limitation), so optimisation is limited to searching
over initial conditions.  Gradient flow operates only over the soft AP
evaluation and guard satisfaction stages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.cegar.buchi import BuchiAutomaton, build_negation_buchi
from core.lppm.automaton import collect_atom_map


# ─── soft AP evaluation ───────────────────────────────────────────────────────

def soft_ap(
    val:         float,
    threshold:   float,
    temperature: float = 0.1,
) -> float:
    """
    Sigmoid relaxation of the threshold predicate z_d > τ.

    σ((val - threshold) / temperature)

    As temperature → 0 this recovers the hard indicator ⟦val > threshold⟧.
    """
    x = (val - threshold) / max(temperature, 1e-8)
    # Numerically stable sigmoid
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# ─── soft guard evaluation ────────────────────────────────────────────────────

def _guard_to_ap_sets(
    guard_expr: str,
    ap_order:   list[str],
    atom_map:   dict[str, dict[str, Any]],
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """
    Parse a HOA guard expression into (positive_aps, negative_aps) where each
    item is (dim, threshold).  Only works for simple conjunctive guards over
    indexed or named APs; complex disjunctive guards default to ([], []).
    """
    pos: list[tuple[str, float]] = []
    neg: list[tuple[str, float]] = []

    # Simple name-based guards (template automata)
    if guard_expr == "t" or guard_expr == "":
        return pos, neg
    if guard_expr in atom_map:
        atom = atom_map[guard_expr]
        pos.append((atom["dim"], float(atom.get("threshold", 0.0))))
        return pos, neg
    if guard_expr.startswith("!") and guard_expr[1:] in atom_map:
        atom = atom_map[guard_expr[1:]]
        neg.append((atom["dim"], float(atom.get("threshold", 0.0))))
        return pos, neg

    # HOA numeric guards: try to parse simple clauses like "0 & !1" → pos=[ap0], neg=[ap1]
    import re
    tokens = re.findall(r"!?\d+", guard_expr)
    for tok in tokens:
        if tok.startswith("!"):
            idx = int(tok[1:])
            if idx < len(ap_order):
                name = ap_order[idx]
                atom = atom_map.get(name, {})
                neg.append((atom.get("dim", name), float(atom.get("threshold", 0.0))))
        else:
            idx = int(tok)
            if idx < len(ap_order):
                name = ap_order[idx]
                atom = atom_map.get(name, {})
                pos.append((atom.get("dim", name), float(atom.get("threshold", 0.0))))

    return pos, neg


def soft_guard_eval(
    z:           dict[str, float],
    pos_aps:     list[tuple[str, float]],
    neg_aps:     list[tuple[str, float]],
    temperature: float,
) -> float:
    """
    Soft guard satisfaction (Eq. 4):
      g̃ = Π_{(d,τ)∈pos} σ((z_d-τ)/T) × Π_{(d,τ)∈neg} (1-σ((z_d-τ)/T))
    """
    score = 1.0
    for dim, thr in pos_aps:
        score *= soft_ap(float(z.get(dim, 0.0)), thr, temperature)
    for dim, thr in neg_aps:
        score *= 1.0 - soft_ap(float(z.get(dim, 0.0)), thr, temperature)
    return score


# ─── soft Büchi propagation ───────────────────────────────────────────────────

def _propagate_weights(
    w:           dict[str, float],
    z:           dict[str, float],
    buchi:       BuchiAutomaton,
    atom_map:    dict[str, dict[str, Any]],
    temperature: float,
) -> dict[str, float]:
    """
    One-step soft occupancy propagation: w_{t+1}(q') = Σ_q w_t(q)·g̃(z,P⁺,P⁻).

    Each per-state weight is clamped to [0, 1] so the acceptance score stays
    in [0, 1] even for nondeterministic template automata (where multiple
    unconstrained "t" transitions would otherwise make weights grow without bound).
    Deterministic Spot automata never exceed 1 naturally.
    """
    w_next: dict[str, float] = {s: 0.0 for s in buchi.states}
    for src, trans in buchi.transitions.items():
        w_src = w.get(src, 0.0)
        if w_src < 1e-12:
            continue
        for guard_expr, dst in trans:
            pos, neg = _guard_to_ap_sets(guard_expr, buchi.ap_order, atom_map)
            g = soft_guard_eval(z, pos, neg, temperature)
            w_next[dst] = min(1.0, w_next.get(dst, 0.0) + w_src * g)
    return w_next


# ─── main function ────────────────────────────────────────────────────────────

@dataclass
class SoftBuchiResult:
    """
    Output of soft Büchi monitoring.

    acceptance_score : max_{t,q∈F} w_t(q)   ∈ [0, 1]
        Near 1 → likely violation pattern.
        Near 0 → no violation pattern detected.
    verdict          : "VIOLATION" if score ≥ epsilon, else "SAFE"
    witness_traj_idx : index of the rollout with highest acceptance score.
    """

    acceptance_score: float
    verdict:          str
    epsilon:          float
    witness_traj_idx: int = 0
    per_traj_scores:  list[float] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.per_traj_scores is None:
            self.per_traj_scores = []

    def summary(self) -> str:
        return (
            f"[SoftBüchi] {self.verdict}  |  "
            f"a*={self.acceptance_score:.4f}  ε={self.epsilon:.4f}  "
            f"witness_traj={self.witness_traj_idx}"
        )


def run_soft_buchi(
    trajectories: list[list[dict[str, float]]],
    spec:         dict[str, Any],
    temperature:  float = 0.1,
    epsilon:      float = 0.5,
) -> SoftBuchiResult:
    """
    Compute the soft Büchi acceptance score over all trajectories (Section 4.5).

    Parameters
    ----------
    trajectories : N latent rollouts (AP-dicts).
    spec         : specification dict with formula.
    temperature  : sigmoid sharpness T.  Smaller → closer to hard threshold.
    epsilon      : acceptance threshold; score ≥ ε → VIOLATION.

    Returns
    -------
    SoftBuchiResult with max acceptance score and verdict.
    """
    buchi    = build_negation_buchi(spec)
    atom_map = collect_atom_map(spec.get("formula", {}))

    per_traj: list[float] = []
    global_max = 0.0
    witness    = 0

    for i, traj in enumerate(trajectories):
        # Initialise occupancy at Büchi initial state
        w: dict[str, float] = {s: (1.0 if s == buchi.initial else 0.0)
                                for s in buchi.states}
        traj_max = max(w.get(q, 0.0) for q in buchi.accepting) if buchi.accepting else 0.0

        for z in traj:
            w = _propagate_weights(w, z, buchi, atom_map, temperature)
            step_max = max(w.get(q, 0.0) for q in buchi.accepting) if buchi.accepting else 0.0
            traj_max = max(traj_max, step_max)

        per_traj.append(traj_max)
        if traj_max > global_max:
            global_max = traj_max
            witness    = i

    verdict = "VIOLATION" if global_max >= epsilon else "SAFE"
    return SoftBuchiResult(
        acceptance_score=global_max,
        verdict=verdict,
        epsilon=epsilon,
        witness_traj_idx=witness,
        per_traj_scores=per_traj,
    )
