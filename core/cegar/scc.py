"""
core/cegar/scc.py

SCC-based accepting cycle detection for the product automaton (Section 4.6/4.7).

We use an iterative implementation of Tarjan's algorithm to avoid Python's
recursion depth limit on large product graphs.

An accepting cycle exists iff there is an SCC that:
  (1) is reachable from at least one initial product state,
  (2) contains at least one accepting product state, and
  (3) is non-trivial (|SCC| > 1  OR  has a self-loop).
"""

from __future__ import annotations

from typing import Any, TypeVar

from .product import ProductAutomaton, ProductKey


# ─── Tarjan's SCC (iterative) ─────────────────────────────────────────────────

def tarjan_sccs(
    graph: dict[ProductKey, set[ProductKey]],
) -> list[frozenset[ProductKey]]:
    """
    Compute all SCCs of `graph` via iterative Tarjan's algorithm.
    Returns SCCs in reverse topological order.
    """
    index:    dict[ProductKey, int]  = {}
    lowlink:  dict[ProductKey, int]  = {}
    on_stack: dict[ProductKey, bool] = {}
    stack:    list[ProductKey]       = []
    sccs:     list[frozenset[ProductKey]] = []
    counter                          = [0]

    # Work-stack frames: (node, iterator_over_successors, is_first_visit)
    call_stack: list[tuple[ProductKey, Any, bool]] = []

    def visit(v: ProductKey) -> None:
        index[v] = lowlink[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack[v] = True
        call_stack.append((v, iter(graph.get(v, set())), True))

    for start in graph:
        if start in index:
            continue
        visit(start)
        while call_stack:
            v, it, _ = call_stack[-1]
            try:
                w = next(it)
                if w not in index:
                    visit(w)
                elif on_stack.get(w):
                    lowlink[v] = min(lowlink[v], index[w])
            except StopIteration:
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[v])
                # Check if v is a root of an SCC
                if lowlink[v] == index[v]:
                    scc: set[ProductKey] = set()
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        scc.add(w)
                        if w == v:
                            break
                    sccs.append(frozenset(scc))

    return sccs


# ─── reachability from initial states ─────────────────────────────────────────

def reachable_from(
    graph:   dict[ProductKey, set[ProductKey]],
    initial: set[ProductKey],
) -> set[ProductKey]:
    visited: set[ProductKey] = set()
    frontier = list(initial)
    while frontier:
        node = frontier.pop()
        if node in visited:
            continue
        visited.add(node)
        for succ in graph.get(node, set()):
            if succ not in visited:
                frontier.append(succ)
    return visited


# ─── accepting cycle detection ────────────────────────────────────────────────

def find_accepting_scc(
    product: ProductAutomaton,
) -> frozenset[ProductKey] | None:
    """
    Return the first non-trivial accepting SCC reachable from an initial state,
    or None if no such SCC exists.

    A "non-trivial" SCC has at least two states, or is a singleton with a
    self-loop.
    """
    if not product.accepting:
        return None

    reachable = reachable_from(product.graph, product.initial)
    sccs = tarjan_sccs(product.graph)

    for scc in sccs:
        # Must intersect reachable states
        if not scc & reachable:
            continue
        # Must contain at least one accepting state
        if not scc & product.accepting:
            continue
        # Must be non-trivial: |SCC| > 1  OR  self-loop
        if len(scc) == 1:
            node = next(iter(scc))
            if node not in product.graph.get(node, set()):
                continue   # singleton without self-loop — trivial
        return scc

    return None
