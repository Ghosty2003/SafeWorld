"""
core/cegar/product.py

Build the product automaton  P = T × B¬φ  used by CEGAR.

States  : (abstract_idx, buchi_state)
Edges   : for each observed abstract transition (s_i → s_j) and each Büchi
          transition (b_k, guard, b_l) where guard is satisfied by the APs of
          s_i, add product edge (s_i, b_k) → (s_j, b_l).
Accepting: product states whose Büchi component is in the Büchi accepting set.
Initial  : {(idx, buchi.initial) for idx in abstract_sys.initial_indices}

A reachable non-trivial accepting SCC in P witnesses a potential VIOLATION of φ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .abstraction import AbstractSystem, active_aps_for_abstract_state
from .buchi import BuchiAutomaton


# ─── types ────────────────────────────────────────────────────────────────────

ProductKey = tuple[int, str]   # (abstract_idx, buchi_state)


@dataclass
class ProductAutomaton:
    graph:    dict[ProductKey, set[ProductKey]]   # adjacency list
    accepting: set[ProductKey]                    # Büchi-accepting product states
    initial:   set[ProductKey]                    # initial product states


# ─── builder ─────────────────────────────────────────────────────────────────

def build_product_automaton(
    abstract_sys: AbstractSystem,
    buchi:        BuchiAutomaton,
    spec:         dict[str, Any],
) -> ProductAutomaton:
    """
    Construct T × B¬φ.

    We enumerate all observed abstract transitions (s_i → s_j) and all Büchi
    transitions (b_k, guard, b_l).  For each pair where the guard is satisfied
    by the APs of s_i, we add the product edge (s_i, b_k) → (s_j, b_l).
    """
    graph:     dict[ProductKey, set[ProductKey]] = {}
    accepting: set[ProductKey]                   = set()
    reachable: set[ProductKey]                   = set()

    # Pre-compute AP frozensets per abstract state
    ap_cache: dict[int, frozenset[str]] = {
        idx: active_aps_for_abstract_state(idx, abstract_sys, spec)
        for idx in range(len(abstract_sys.state_keys))
    }

    # Initial product states
    initial: set[ProductKey] = {
        (idx, buchi.initial) for idx in abstract_sys.initial_indices
    }
    for pk in initial:
        graph.setdefault(pk, set())
        if pk[1] in buchi.accepting:
            accepting.add(pk)

    # Build all reachable product states via BFS over observed abstract transitions
    frontier = list(initial)
    visited: set[ProductKey] = set(initial)

    while frontier:
        pk = frontier.pop()
        abs_idx, b_state = pk
        active = ap_cache[abs_idx]

        for b_guard, b_dst in buchi.transitions.get(b_state, []):
            from .buchi import _eval_guard
            if not _eval_guard(b_guard, buchi.ap_order, active):
                continue
            # Advance the abstract system to all observed successors
            for a_src, a_dst in abstract_sys.transitions:
                if a_src != abs_idx:
                    continue
                succ: ProductKey = (a_dst, b_dst)
                graph[pk].add(succ)
                graph.setdefault(succ, set())
                if b_dst in buchi.accepting:
                    accepting.add(succ)
                if succ not in visited:
                    visited.add(succ)
                    frontier.append(succ)

    return ProductAutomaton(graph=graph, accepting=accepting, initial=initial)
