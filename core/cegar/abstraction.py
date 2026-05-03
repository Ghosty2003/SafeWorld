"""
core/cegar/abstraction.py

CEGAR abstraction layer: partition the latent space using Boolean predicates.

Given N trajectories of AP-dicts and a list of Predicate objects, builds the
abstract system T = (S, E) whose states are predicate bit-vectors and whose
transitions are the pairs (s_t, s_{t+1}) actually observed in the data.

Public API
----------
Predicate           — threshold predicate p(z) = [z_dim op τ]
AbstractSystem      — data-driven abstract graph
build_abstract_system(trajectories, predicates) -> AbstractSystem
derive_predicates_from_spec(spec)               -> list[Predicate]
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any


# ─── predicate ────────────────────────────────────────────────────────────────

@dataclass
class Predicate:
    """Boolean threshold predicate p(z) = [z_dim op threshold]."""

    dim:       str
    threshold: float
    op:        str = ">"   # ">" or "<"
    name:      str = ""

    def evaluate(self, state: dict[str, float]) -> bool:
        val = float(state.get(self.dim, 0.0))
        return (val > self.threshold) if self.op == ">" else (val < self.threshold)

    def __hash__(self):
        return hash((self.dim, round(self.threshold, 9), self.op))

    def __eq__(self, other):
        return (isinstance(other, Predicate)
                and self.dim == other.dim
                and abs(self.threshold - other.threshold) < 1e-9
                and self.op == other.op)

    def __repr__(self) -> str:
        return self.name or f"{self.dim}{self.op}{self.threshold:.4f}"


# ─── abstract state key ───────────────────────────────────────────────────────

AbstractStateKey = tuple[bool, ...]   # one bit per predicate


# ─── abstract system ──────────────────────────────────────────────────────────

@dataclass
class AbstractSystem:
    """Data-driven abstract graph derived from concrete trajectory data."""

    predicates:      list[Predicate]
    state_keys:      list[AbstractStateKey]               # indexed list of unique abstract states
    state_index:     dict[AbstractStateKey, int]          # key → index
    transitions:     set[tuple[int, int]]                 # (from_idx, to_idx)
    initial_indices: set[int]                             # indices from trajectory starts
    concrete_map:    dict[int, list[dict[str, float]]]    # abstract_idx → concrete states


# ─── building the abstract system ─────────────────────────────────────────────

def _eval_key(state: dict[str, float], preds: list[Predicate]) -> AbstractStateKey:
    return tuple(p.evaluate(state) for p in preds)


def build_abstract_system(
    trajectories: list[list[dict[str, float]]],
    predicates:   list[Predicate],
) -> AbstractSystem:
    """
    Build abstract system T from sampled concrete trajectories + predicates.

    Each concrete state z is mapped to an abstract state s = (p1(z), p2(z), ...).
    Observed transitions (s_t, s_{t+1}) form the edge set.
    """
    state_index:     dict[AbstractStateKey, int]       = {}
    state_keys:      list[AbstractStateKey]            = []
    transitions:     set[tuple[int, int]]              = set()
    initial_indices: set[int]                          = set()
    concrete_map:    dict[int, list[dict[str, float]]] = {}

    def get_idx(key: AbstractStateKey) -> int:
        if key not in state_index:
            idx = len(state_keys)
            state_index[key] = idx
            state_keys.append(key)
            concrete_map[idx] = []
        return state_index[key]

    for traj in trajectories:
        if not traj:
            continue
        prev_idx: int | None = None
        for t, z in enumerate(traj):
            key = _eval_key(z, predicates)
            idx = get_idx(key)
            concrete_map[idx].append(z)
            if t == 0:
                initial_indices.add(idx)
            if prev_idx is not None:
                transitions.add((prev_idx, idx))
            prev_idx = idx

    return AbstractSystem(
        predicates=predicates,
        state_keys=state_keys,
        state_index=state_index,
        transitions=transitions,
        initial_indices=initial_indices,
        concrete_map=concrete_map,
    )


# ─── spec-derived predicates (for ONE-SHOT mode) ──────────────────────────────

def derive_predicates_from_spec(spec: dict[str, Any]) -> list[Predicate]:
    """
    Extract atomic propositions from the formula as Predicates.

    For ONE-SHOT (Section 4.6): "for each AP appearing in φ that corresponds to a
    threshold condition z_d > τ, directly create the predicate p(z) = [z_d > τ]."
    """
    from core.lppm.automaton import collect_atom_map
    atom_map = collect_atom_map(spec.get("formula", {}))
    seen: set[tuple] = set()
    predicates: list[Predicate] = []
    for label, atom in atom_map.items():
        key = (atom["dim"], round(float(atom.get("threshold", 0.0)), 9), atom.get("op", ">"))
        if key not in seen:
            seen.add(key)
            predicates.append(Predicate(
                dim       = atom["dim"],
                threshold = float(atom.get("threshold", 0.0)),
                op        = atom.get("op", ">"),
                name      = label,
            ))
    return predicates


# ─── helpers for product automaton construction ───────────────────────────────

def active_aps_for_abstract_state(
    abstract_idx: int,
    abstract_sys: AbstractSystem,
    spec:         dict[str, Any],
) -> frozenset[str]:
    """
    Return the active AP frozenset for an abstract state by evaluating a
    representative concrete state from the cell.

    For ONESHOT predicates are exactly the formula atoms, so every state in a
    cell has the same formula-AP truth values; any representative works.
    """
    from core.lppm.automaton import extract_active_aps
    concretes = abstract_sys.concrete_map.get(abstract_idx, [])
    if not concretes:
        # No concrete data: synthesize a representative state from predicates
        syn = _synthesize_representative(abstract_sys.state_keys[abstract_idx],
                                         abstract_sys.predicates)
        return extract_active_aps(syn, spec)
    return extract_active_aps(concretes[0], spec)


def _synthesize_representative(
    abstract_key: AbstractStateKey,
    predicates:   list[Predicate],
) -> dict[str, float]:
    """Construct a synthetic concrete state consistent with the abstract key."""
    state: dict[str, float] = {}
    for pred, val in zip(predicates, abstract_key):
        if pred.op == ">":
            state[pred.dim] = pred.threshold + (0.1 if val else -0.1)
        else:
            state[pred.dim] = pred.threshold - (0.1 if val else -0.1)
    return state


# ─── variance-guided predicate splitting ──────────────────────────────────────

def variance_split_predicate(
    suspect_indices: set[int],
    abstract_sys:    AbstractSystem,
) -> Predicate | None:
    """
    Find the abstract state in `suspect_indices` with maximum per-dimension
    variance among its concrete states, then return a new predicate that splits
    it along d* = argmax_d Var_d(z for z in cell) at the median threshold.

    Returns None if no valid split is found.
    """
    best_pred: Predicate | None = None
    best_var = -1.0

    for idx in suspect_indices:
        concretes = abstract_sys.concrete_map.get(idx, [])
        if len(concretes) < 2:
            continue
        all_keys = list(concretes[0].keys())
        for dim in all_keys:
            vals = [float(z.get(dim, 0.0)) for z in concretes]
            var = statistics.variance(vals)
            if var > best_var:
                best_var = var
                median_val = statistics.median(vals)
                best_pred = Predicate(
                    dim       = dim,
                    threshold = median_val,
                    op        = ">",
                    name      = f"{dim}>{median_val:.4f}",
                )

    return best_pred
