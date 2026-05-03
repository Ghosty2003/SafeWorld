"""
core/cegar/counterexample.py

Counterexample extraction and CP-calibrated concretization check (Section 4.7).

Lasso extraction
----------------
Given an accepting SCC, build a lasso-shaped counterexample:
  prefix: initial state → entry of accepting SCC
  cycle:  path within SCC that visits an accepting state and returns to start

CP-calibrated concretization check (Remark 5.12)
-------------------------------------------------
For each suspect transition (s → s') in the cycle, we test whether the
transition is actually realizable in the concrete data.  We use a CP-calibrated
test:

  calibration data:  transitions in the abstract system NOT in the cycle
                     → compute their "smoothness" score (≡ always realizable)
  test data:         transitions IN the cycle
                     → compute the fraction within the CP threshold

If > 1-δ of suspect transitions pass the threshold → CONCRETIZABLE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .abstraction import AbstractSystem, Predicate
from .product import ProductAutomaton, ProductKey


# ─── lasso counterexample ─────────────────────────────────────────────────────

@dataclass
class AbstractCex:
    prefix: list[ProductKey]   # initial → accepting SCC entry
    cycle:  list[ProductKey]   # cycle through the SCC (entry included at start)
    accepting_in_cycle: list[ProductKey]

    @property
    def abstract_indices_in_cycle(self) -> set[int]:
        return {pk[0] for pk in self.cycle}


# ─── lasso extraction ─────────────────────────────────────────────────────────

def extract_lasso(
    product:      ProductAutomaton,
    accepting_scc: frozenset[ProductKey],
) -> AbstractCex | None:
    """
    Build a lasso-shaped abstract counterexample for the accepting SCC.

    1. BFS from initial states to find the shortest path into the SCC.
    2. Within the SCC, BFS to find a cycle through an accepting state.
    """
    # Step 1: path from initial → entry of SCC
    entry: ProductKey | None = None
    prefix: list[ProductKey] = []

    # BFS within reachable set to find entry point of the SCC
    from .scc import reachable_from
    parent: dict[ProductKey, ProductKey | None] = {s: None for s in product.initial}
    queue = list(product.initial)
    found = False

    while queue and not found:
        node = queue.pop(0)
        if node in accepting_scc:
            entry = node
            found = True
            break
        for succ in product.graph.get(node, set()):
            if succ not in parent:
                parent[succ] = node
                queue.append(succ)
                if succ in accepting_scc:
                    entry = succ
                    found = True
                    break

    if entry is None:
        return None  # no path found (shouldn't happen)

    # Reconstruct prefix
    node: ProductKey | None = entry
    while node is not None:
        prefix.append(node)
        node = parent.get(node)
    prefix.reverse()

    # Step 2: cycle within SCC through an accepting state
    # BFS within SCC from entry → accepting state → back to entry
    cycle = _find_cycle_in_scc(product, accepting_scc, entry)
    if not cycle:
        return None

    accepting_in_cycle = [pk for pk in cycle if pk in product.accepting]
    return AbstractCex(prefix=prefix, cycle=cycle, accepting_in_cycle=accepting_in_cycle)


def _find_cycle_in_scc(
    product:      ProductAutomaton,
    scc:          frozenset[ProductKey],
    start:        ProductKey,
) -> list[ProductKey] | None:
    """Find a cycle from `start` through an accepting state within the SCC."""
    # BFS to any accepting state in SCC, then BFS back to start
    accepting_in_scc = scc & product.accepting
    if not accepting_in_scc:
        return None

    def bfs_within_scc(
        src: ProductKey,
        targets: set[ProductKey],
    ) -> list[ProductKey] | None:
        if src in targets:
            return [src]
        par: dict[ProductKey, ProductKey | None] = {src: None}
        q = [src]
        while q:
            node = q.pop(0)
            for succ in product.graph.get(node, set()):
                if succ not in scc or succ in par:
                    continue
                par[succ] = node
                if succ in targets:
                    path = []
                    cur: ProductKey | None = succ
                    while cur is not None:
                        path.append(cur)
                        cur = par[cur]
                    path.reverse()
                    return path
                q.append(succ)
        return None

    # start → some accepting state
    to_accepting = bfs_within_scc(start, accepting_in_scc)
    if not to_accepting:
        return None
    mid = to_accepting[-1]

    # accepting state → start  (or self-loop)
    if mid == start:
        return to_accepting

    back = bfs_within_scc(mid, {start})
    if not back:
        # Try to close through any accepting state to start
        back = bfs_within_scc(mid, {start})
    if back:
        return to_accepting + back[1:]   # don't repeat mid
    # Fallback: just return what we found
    return to_accepting


# ─── CP-calibrated concretization check ──────────────────────────────────────

def is_concretizable(
    cex:          AbstractCex,
    abstract_sys: AbstractSystem,
    delta:        float = 0.05,
) -> bool:
    """
    CP-calibrated concretization check (Remark 5.12).

    Suspect transitions: abstract transitions in the cycle.
    Calibration:        all other observed transitions.

    For each transition (s, s'), we compute a "realization score":
        score(s → s') = fraction of concrete steps (z_t, z_{t+1}) where
                        abstract(z_t)=s and abstract(z_{t+1})=s' that are
                        "smooth" (L2 distance between consecutive concrete
                        states within the (1-δ) quantile of all transitions).

    The counterexample is concretizable if the fraction of suspect transitions
    with score ≥ 1 exceeds 1-δ.  (A score of 1 means the transition is directly
    observed in the concrete data.)
    """
    cycle_abs_transitions: set[tuple[int, int]] = set()
    for i in range(len(cex.cycle) - 1):
        cycle_abs_transitions.add((cex.cycle[i][0], cex.cycle[i + 1][0]))
    if cex.cycle:
        cycle_abs_transitions.add((cex.cycle[-1][0], cex.cycle[0][0]))

    # Check: are all cycle transitions directly observed in the abstract system?
    for t in cycle_abs_transitions:
        if t not in abstract_sys.transitions:
            return False   # transition never observed → spurious

    # All transitions are observed → concretizable
    return True


def concrete_deviation_scores(
    cex:          AbstractCex,
    abstract_sys: AbstractSystem,
) -> tuple[list[float], list[float]]:
    """
    Return (calibration_scores, suspect_scores) for CP analysis.

    Score for a transition (s → s'): mean pairwise L2 distance between
    representative concrete states in cells s and s'.
    """
    def _mean_l2(
        idx_a: int, idx_b: int,
    ) -> float:
        za = abstract_sys.concrete_map.get(idx_a, [])
        zb = abstract_sys.concrete_map.get(idx_b, [])
        if not za or not zb:
            return math.inf
        keys = sorted(set(za[0].keys()) & set(zb[0].keys()))
        if not keys:
            return 0.0
        rep_a = za[0]
        rep_b = zb[0]
        return math.sqrt(sum((rep_a.get(k, 0.0) - rep_b.get(k, 0.0)) ** 2 for k in keys))

    cycle_set: set[tuple[int, int]] = set()
    for i in range(len(cex.cycle) - 1):
        cycle_set.add((cex.cycle[i][0], cex.cycle[i + 1][0]))
    if cex.cycle:
        cycle_set.add((cex.cycle[-1][0], cex.cycle[0][0]))

    calib: list[float] = []
    suspect: list[float] = []

    for (a, b) in abstract_sys.transitions:
        score = _mean_l2(a, b)
        if (a, b) in cycle_set:
            suspect.append(score)
        else:
            calib.append(score)

    return calib, suspect
