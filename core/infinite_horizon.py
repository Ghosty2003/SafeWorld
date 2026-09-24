"""Fail-closed helpers for infinite-horizon executable-model checking.

Finite rollouts can propose recurrent behavior, but only an exact repeated
complete state (for a deterministic executable transition) or a separately
sound region proof can close an infinite argument. Sampled/quantized SCCs are
therefore diagnostics and can never directly produce SAFE or VIOLATE here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import networkx as nx
import numpy as np


SAFE = "SAFE"
VIOLATE = "VIOLATE"
ABSTAIN = "ABSTAIN"


def linear_probe_simnorm_bounds(
    coefficients: np.ndarray, intercept: float, group_dim: int
) -> tuple[float, float]:
    """Exact range of a linear probe over a product of probability simplices.

    SimNorm partitions the latent into groups whose entries are nonnegative
    and sum to one. A linear functional reaches its extrema at a simplex
    vertex, independently in every group.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if group_dim < 1 or len(coefficients) % group_dim:
        raise ValueError("group_dim must positively divide the coefficient count")
    groups = coefficients.reshape(-1, group_dim)
    lower = float(intercept + groups.min(axis=1).sum())
    upper = float(intercept + groups.max(axis=1).sum())
    return lower, upper


@dataclass(frozen=True)
class ExactLasso:
    prefix_length: int
    cycle_length: int
    repeated_fingerprint: str
    cycle_has_bad: bool
    bad_cycle_offsets: tuple[int, ...]

    @property
    def verdict(self) -> str:
        return VIOLATE if self.cycle_has_bad else SAFE

    def as_dict(self) -> dict[str, Any]:
        return {
            "prefix_length": self.prefix_length,
            "cycle_length": self.cycle_length,
            "repeated_fingerprint": self.repeated_fingerprint,
            "cycle_has_bad": self.cycle_has_bad,
            "bad_cycle_offsets": list(self.bad_cycle_offsets),
            "verdict": self.verdict,
            "scope": "one exact deterministic executable closed-loop run",
        }


def detect_exact_lasso(
    fingerprints: Sequence[str], bad: Sequence[bool]
) -> ExactLasso | None:
    """Find the first repeated complete state and classify its exact cycle."""
    if len(fingerprints) != len(bad):
        raise ValueError("fingerprints and bad flags must have equal length")
    first_seen: dict[str, int] = {}
    for end, fingerprint in enumerate(fingerprints):
        start = first_seen.get(fingerprint)
        if start is not None:
            offsets = tuple(
                index - start for index in range(start, end) if bool(bad[index])
            )
            return ExactLasso(
                prefix_length=start,
                cycle_length=end - start,
                repeated_fingerprint=fingerprint,
                cycle_has_bad=bool(offsets),
                bad_cycle_offsets=offsets,
            )
        first_seen[fingerprint] = end
    return None


def replay_validate_exact_lasso(
    states: Sequence[Any],
    lasso: ExactLasso,
    *,
    fingerprint: Callable[[Any], str],
    step: Callable[[Any], Any],
) -> dict[str, Any]:
    """Re-execute every lasso edge and require bit-exact successor equality."""
    start = lasso.prefix_length
    end = start + lasso.cycle_length
    if end >= len(states):
        raise ValueError("states must include the repeated state closing the cycle")
    failures = []
    for index in range(start, end):
        observed = step(states[index])
        expected_fingerprint = fingerprint(states[index + 1])
        observed_fingerprint = fingerprint(observed)
        if observed_fingerprint != expected_fingerprint:
            failures.append(
                {
                    "edge_offset": index - start,
                    "expected": expected_fingerprint,
                    "observed": observed_fingerprint,
                }
            )
    closure_equal = fingerprint(states[start]) == fingerprint(states[end])
    verified = closure_equal and not failures
    return {
        "verified": verified,
        "closure_equal": closure_equal,
        "edges_checked": lasso.cycle_length,
        "edge_failures": failures,
        "verdict": lasso.verdict if verified else ABSTAIN,
        "scope": "executable floating-point transition semantics",
    }


def _cell_key(row: np.ndarray, widths: np.ndarray) -> tuple[int, ...]:
    return tuple(np.floor(row / widths).astype(np.int64).tolist())


def build_sampled_partition_graph(
    features: np.ndarray,
    bad: np.ndarray,
    *,
    cell_widths: np.ndarray,
) -> tuple[nx.DiGraph, list[tuple[int, ...]]]:
    """Build a sampled abstraction used only to propose recurrent candidates.

    ``features`` is ``[trajectory,time,dimension]``. The returned graph is not
    a transition over-approximation: unobserved successors are absent. Every
    consumer must preserve its ``sound_for_verdict=False`` marker.
    """
    features = np.asarray(features, dtype=np.float64)
    bad = np.asarray(bad, dtype=bool)
    widths = np.asarray(cell_widths, dtype=np.float64)
    if features.ndim != 3 or bad.shape != features.shape[:2]:
        raise ValueError("features must be [N,T,D] and bad must be [N,T]")
    if widths.shape != (features.shape[-1],) or np.any(widths <= 0):
        raise ValueError("cell_widths must be one positive value per dimension")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(widths)):
        raise ValueError("partition inputs must be finite")

    graph = nx.DiGraph(
        construction="sampled_quantized",
        sound_for_verdict=False,
        n_trajectories=int(features.shape[0]),
    )
    initial_nodes = []
    for trajectory in range(features.shape[0]):
        keys = [_cell_key(row, widths) for row in features[trajectory]]
        initial_nodes.append(keys[0])
        for time_index, key in enumerate(keys):
            attrs = graph.nodes[key] if graph.has_node(key) else {}
            graph.add_node(
                key,
                visits=int(attrs.get("visits", 0)) + 1,
                bad_visits=int(attrs.get("bad_visits", 0))
                + int(bad[trajectory, time_index]),
            )
        for source, target in zip(keys[:-1], keys[1:]):
            count = int(graph.edges[source, target].get("count", 0)) \
                if graph.has_edge(source, target) else 0
            graph.add_edge(source, target, count=count + 1)
    return graph, initial_nodes


def sampled_bad_scc_candidates(
    graph: nx.DiGraph, initial_nodes: Sequence[tuple[int, ...]]
) -> list[dict[str, Any]]:
    """Return reachable sampled cyclic SCCs containing an observed bad state."""
    reachable: set[tuple[int, ...]] = set()
    for initial in initial_nodes:
        if initial in graph:
            reachable.add(initial)
            reachable.update(nx.descendants(graph, initial))
    candidates = []
    for component in nx.strongly_connected_components(graph.subgraph(reachable)):
        cyclic = len(component) > 1 or any(graph.has_edge(node, node) for node in component)
        bad_nodes = [
            node for node in component if int(graph.nodes[node].get("bad_visits", 0)) > 0
        ]
        if not cyclic or not bad_nodes:
            continue
        candidates.append(
            {
                "nodes": [list(node) for node in sorted(component)],
                "n_nodes": len(component),
                "bad_nodes": [list(node) for node in sorted(bad_nodes)],
                "observed_internal_edges": int(
                    sum(1 for u, v in graph.edges if u in component and v in component)
                ),
                "verdict": ABSTAIN,
                "reason": (
                    "sampled quantized SCC is only a candidate; cell closure and "
                    "a concrete infinite execution are not proved"
                ),
            }
        )
    return sorted(candidates, key=lambda item: (-item["n_nodes"], item["nodes"]))


def aggregate_exact_run_verdict(lassos: Sequence[ExactLasso | None]) -> str:
    """Aggregate exact seeded executions without overstating their scope."""
    observed = [item for item in lassos if item is not None]
    if any(item.verdict == VIOLATE for item in observed):
        return VIOLATE
    if len(observed) == len(lassos) and observed and all(
        item.verdict == SAFE for item in observed
    ):
        return SAFE
    return ABSTAIN
