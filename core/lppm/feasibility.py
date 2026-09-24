"""Structural feasibility checks for strict-descent LPPM objectives."""

from __future__ import annotations

from dataclasses import dataclass

from .automaton import ParityAutomaton


@dataclass(frozen=True)
class LPPMFeasibility:
    eligible: bool
    absorbing_odd_states: tuple[str, ...]
    reason: str


def _outgoing_states(dpa: ParityAutomaton, state: str) -> set[str]:
    if dpa.edge_guards:
        return {dst for _guard, dst, _priority in dpa.edge_guards.get(state, [])}
    return {dst for (src, _guard), dst in dpa.transition.items() if src == state}


def absorbing_odd_states(dpa: ParityAutomaton) -> tuple[str, ...]:
    """Return odd states whose every represented outgoing edge is a self-loop."""
    result = []
    for state in dpa.states:
        if dpa.priority.get(state, 0) % 2 != 1:
            continue
        outgoing = _outgoing_states(dpa, state)
        if outgoing and outgoing == {state}:
            result.append(state)
    return tuple(sorted(result))


def analyze_lppm_feasibility(
    dpa: ParityAutomaton,
    *,
    mp_class: str,
    eta: float,
) -> LPPMFeasibility:
    """Reject pure Safety automata with a permanent odd trap.

    P2 requires ``V(next) <= V(curr) - eta`` whenever the current priority is
    odd. For ``eta > 0``, that condition cannot hold indefinitely in an
    absorbing odd state for a nonnegative V. In the current pure-Safety
    template the only bad state is exactly such a permanent trap, so training
    on a violation creates an irreducible strict-descent contradiction while
    training only on safe traces leaves P1's constant optimum.

    Other Manna--Pnueli classes may contain rejecting sink states too, but can
    also have meaningful non-sink odd progress states. This narrow guard routes
    only pure Safety here; broader support-reachability analysis is separate.
    """
    absorbing = absorbing_odd_states(dpa)
    if mp_class == "Safety" and eta > 0.0 and absorbing:
        return LPPMFeasibility(
            eligible=False,
            absorbing_odd_states=absorbing,
            reason=(
                "Pure Safety automaton contains permanent odd state(s) "
                f"{list(absorbing)!r}. P2 demands a positive strict decrease "
                f"eta={eta:g} on every odd-state step, which a nonnegative V "
                "cannot sustain in an absorbing trap. Route G(safe) to direct "
                "invariant/barrier evidence, or use a semantically distinct "
                "recoverable Persistence property F G(safe)."
            ),
        )
    return LPPMFeasibility(
        eligible=True,
        absorbing_odd_states=absorbing,
        reason="No pure-Safety absorbing-odd strict-descent contradiction detected.",
    )
