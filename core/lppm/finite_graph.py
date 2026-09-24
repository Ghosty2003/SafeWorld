"""Exact binary L2 verification for explicitly finite transition support."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping


@dataclass(frozen=True)
class FiniteL2Result:
    verdict: Literal["SAFE", "VIOLATION"]
    reachable_states: tuple[str, ...]
    bad_states: tuple[str, ...]
    eta: float
    values: dict[str, float] | None
    max_bad_visits: int | None
    lasso_prefix: tuple[str, ...] | None
    lasso_cycle: tuple[str, ...] | None
    p1_verified: bool
    p2_verified: bool


def _reachable(graph: Mapping[str, tuple[str, ...]], initials: tuple[str, ...]) -> set[str]:
    reached = set(initials)
    queue = deque(initials)
    while queue:
        state = queue.popleft()
        for successor in graph[state]:
            if successor not in reached:
                reached.add(successor)
                queue.append(successor)
    return reached


def _tarjan(graph: Mapping[str, tuple[str, ...]], reachable: set[str]):
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    components: list[tuple[str, ...]] = []

    def visit(state: str) -> None:
        nonlocal index
        indices[state] = lowlink[state] = index
        index += 1
        stack.append(state)
        on_stack.add(state)
        for successor in graph[state]:
            if successor not in reachable:
                continue
            if successor not in indices:
                visit(successor)
                lowlink[state] = min(lowlink[state], lowlink[successor])
            elif successor in on_stack:
                lowlink[state] = min(lowlink[state], indices[successor])
        if lowlink[state] == indices[state]:
            component = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == state:
                    break
            components.append(tuple(sorted(component)))

    for state in sorted(reachable):
        if state not in indices:
            visit(state)
    return components


def _bfs_path(
    graph: Mapping[str, tuple[str, ...]],
    starts: Iterable[str],
    goal: str,
    allowed: set[str] | None = None,
) -> tuple[str, ...]:
    parents: dict[str, str | None] = {}
    queue = deque()
    for state in starts:
        if allowed is None or state in allowed:
            parents[state] = None
            queue.append(state)
    while queue:
        state = queue.popleft()
        if state == goal:
            path = []
            cursor: str | None = state
            while cursor is not None:
                path.append(cursor)
                cursor = parents[cursor]
            return tuple(reversed(path))
        for successor in graph[state]:
            if (allowed is None or successor in allowed) and successor not in parents:
                parents[successor] = state
                queue.append(successor)
    raise AssertionError(f"no path to reachable goal {goal!r}")


def verify_finite_cobuchi(
    transitions: Mapping[str, Iterable[str]],
    initial_states: Iterable[str],
    bad_states: Iterable[str],
    *,
    eta: float = 1.0,
) -> FiniteL2Result:
    """Return an exact SAFE certificate or a reachable bad-cycle lasso.

    The semantics are universal over all infinite paths of the explicitly
    supplied finite graph. A reachable SCC containing a bad state and a cycle
    is a co-Buchi violation. Otherwise SCC condensation yields the maximum
    remaining bad-visit count, whose scaled value is an exact LPPM.
    """
    if eta <= 0:
        raise ValueError("eta must be positive")
    graph = {str(state): tuple(map(str, successors)) for state, successors in transitions.items()}
    initials = tuple(map(str, initial_states))
    bad = set(map(str, bad_states))
    if not initials:
        raise ValueError("at least one initial state is required")
    all_states = set(graph)
    all_states.update(successor for successors in graph.values() for successor in successors)
    for state in all_states:
        graph.setdefault(state, ())
    unknown_initials = set(initials) - all_states
    if unknown_initials:
        raise ValueError(f"unknown initial states: {sorted(unknown_initials)}")
    reachable = _reachable(graph, initials)
    dead_ends = sorted(state for state in reachable if not graph[state])
    if dead_ends:
        raise ValueError(f"infinite-run graph has dead ends: {dead_ends}")

    components = _tarjan(graph, reachable)
    component_of = {
        state: index for index, component in enumerate(components) for state in component
    }
    violating_component: tuple[str, ...] | None = None
    for component in components:
        cyclic = len(component) > 1 or component[0] in graph[component[0]]
        if cyclic and bad.intersection(component):
            violating_component = component
            break

    if violating_component is not None:
        component_set = set(violating_component)
        bad_member = sorted(bad & component_set)[0]
        prefix = _bfs_path(graph, initials, bad_member)
        if bad_member in graph[bad_member]:
            cycle = (bad_member, bad_member)
        else:
            first = next(
                successor for successor in graph[bad_member]
                if successor in component_set
            )
            return_path = _bfs_path(
                graph, (first,), bad_member, allowed=component_set
            )
            cycle = (bad_member, *return_path)
        return FiniteL2Result(
            verdict="VIOLATION",
            reachable_states=tuple(sorted(reachable)),
            bad_states=tuple(sorted(bad & reachable)),
            eta=eta,
            values=None,
            max_bad_visits=None,
            lasso_prefix=prefix,
            lasso_cycle=cycle,
            p1_verified=False,
            p2_verified=False,
        )

    successors_by_component: dict[int, set[int]] = {
        index: set() for index in range(len(components))
    }
    for state in reachable:
        source = component_of[state]
        for successor in graph[state]:
            target = component_of[successor]
            if source != target:
                successors_by_component[source].add(target)

    memo: dict[int, int] = {}

    def remaining_bad_visits(component_index: int) -> int:
        if component_index in memo:
            return memo[component_index]
        component = components[component_index]
        own = int(bool(bad.intersection(component)))
        downstream = max(
            (remaining_bad_visits(item) for item in successors_by_component[component_index]),
            default=0,
        )
        memo[component_index] = own + downstream
        return memo[component_index]

    ranks = {
        state: remaining_bad_visits(component_of[state]) for state in reachable
    }
    values = {state: eta * rank for state, rank in ranks.items()}
    p1 = all(
        values[successor] <= values[state]
        for state in reachable for successor in graph[state]
    )
    p2 = all(
        state not in bad or values[successor] <= values[state] - eta
        for state in reachable for successor in graph[state]
    )
    if not (p1 and p2):
        raise AssertionError("constructed finite-graph LPPM failed P1/P2")
    return FiniteL2Result(
        verdict="SAFE",
        reachable_states=tuple(sorted(reachable)),
        bad_states=tuple(sorted(bad & reachable)),
        eta=eta,
        values=values,
        max_bad_visits=max(ranks[state] for state in initials),
        lasso_prefix=None,
        lasso_cycle=None,
        p1_verified=True,
        p2_verified=True,
    )
