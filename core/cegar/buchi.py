"""
core/cegar/buchi.py

Büchi automaton for ¬φ — used by CEGAR and Soft Büchi monitoring.

The product automaton T × B¬φ captures potential VIOLATIONS of φ: an
accepting run in the product witnesses a trajectory that satisfies ¬φ, i.e.,
violates φ.

Strategy
--------
1. Try Spot: negate the formula string, translate("Buchi","Deterministic","Complete"),
   parse the HOA output.
2. Fall back to template automata covering Safety / Guarantee / Obligation /
   Recurrence patterns (the types used in the benchmark).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

try:
    import spot  # type: ignore
except ImportError:
    spot = None


# ─── data structure ──────────────────────────────────────────────────────────

@dataclass
class BuchiAutomaton:
    states:      list[str]
    initial:     str
    transitions: dict[str, list[tuple[str, str]]]   # state → [(guard_expr, dst)]
    accepting:   set[str]
    ap_order:    list[str]
    backend:     str = "spot"

    def successors(self, state: str, active_aps: frozenset[str]) -> list[str]:
        result = []
        for guard_expr, dst in self.transitions.get(state, []):
            if _eval_guard(guard_expr, self.ap_order, active_aps):
                result.append(dst)
        return result


# ─── public builder ───────────────────────────────────────────────────────────

def build_negation_buchi(spec: dict[str, Any]) -> BuchiAutomaton:
    """
    Build Büchi automaton for ¬φ.  Tries Spot first; falls back to templates.
    """
    if spot is not None:
        result = _build_spot_negation_buchi(spec)
        if result is not None:
            return result
    return _build_template_negation_buchi(spec)


# ─── Spot path ────────────────────────────────────────────────────────────────

def _build_spot_negation_buchi(spec: dict[str, Any]) -> BuchiAutomaton | None:
    try:
        from core.lppm.automaton import _formula_to_spot_ltl
        ltl_str, ap_order = _formula_to_spot_ltl(spec["formula"])
        neg = spot.formula(f"!({ltl_str})")
        aut = neg.translate("Buchi", "Deterministic", "Complete")
        hoa = aut.to_str("hoa")
        return _parse_buchi_hoa(hoa, ap_order)
    except Exception:
        return None


def _parse_buchi_hoa(hoa: str, ap_order: list[str]) -> BuchiAutomaton:
    states:      list[str]                        = []
    accepting:   set[str]                         = set()
    transitions: dict[str, list[tuple[str, str]]] = {}
    initial = "0"
    current: str | None = None

    for line in hoa.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Start:"):
            initial = line.split(":", 1)[1].strip()
            continue
        if line.startswith("State:"):
            parts = line.split(":", 1)[1].strip().split()
            current = parts[0]
            if current not in states:
                states.append(current)
            transitions.setdefault(current, [])
            # State-based acceptance: {0} in the State line
            acc_in_state = re.findall(r"\{([^}]*)\}", line)
            if acc_in_state and "0" in acc_in_state[0].split():
                accepting.add(current)
            continue
        if not line.startswith("[") or current is None:
            continue
        m = re.match(r"\[(.*?)\]\s+([^\s]+)(?:\s+\{([^}]*)\})?", line)
        if not m:
            continue
        guard_expr = m.group(1).strip()
        dst        = m.group(2).strip()
        edge_acc   = (m.group(3) or "").strip()
        # Edge-based Büchi: edge in acceptance set 0
        if edge_acc and "0" in edge_acc.split():
            accepting.add(current)
        transitions[current].append((guard_expr, dst))
        if dst not in states:
            states.append(dst)
            transitions.setdefault(dst, [])

    return BuchiAutomaton(
        states=states, initial=initial,
        transitions=transitions, accepting=accepting,
        ap_order=ap_order, backend="spot",
    )


# ─── template path ────────────────────────────────────────────────────────────

def _build_template_negation_buchi(spec: dict[str, Any]) -> BuchiAutomaton:
    from utils.spec_analysis import analyze_spec_structure
    from core.lppm.automaton import collect_atom_map, atom_label as _al

    analysis   = spec.get("analysis") or analyze_spec_structure(spec)
    spec["analysis"] = analysis
    mp_class   = analysis["mp_class"]
    objectives = analysis["objectives"]
    atom_map   = collect_atom_map(spec.get("formula", {}))
    ap_order   = list(atom_map.keys())

    # ¬□p  =  ♢¬p  — Büchi accepting = any state where some safety AP violated
    if mp_class == "Safety":
        safe_aps = objectives["safety"] or ap_order
        trans = {
            "ok":   [("t", "ok")] + [(f"!{ap}", "trap") for ap in safe_aps],
            "trap": [("t", "trap")],
        }
        return BuchiAutomaton("ok", "ok", trans, {"trap"}, ap_order, "template")  # type: ignore[arg-type]

    # ¬♢p  =  □¬p  — accepts iff never reach any goal AP
    if mp_class == "Guarantee":
        goal_aps = objectives["guarantee"] or ap_order
        trans = {
            "seek": [("t", "seek")] + [(ap, "trap") for ap in goal_aps],
            "trap": [("t", "trap")],
        }
        return BuchiAutomaton(["seek", "trap"], "seek", trans, {"seek"}, ap_order, "template")

    # ¬(□p ∧ ♢q)  ≈  template covering common Obligation patterns
    if mp_class == "Obligation":
        safe_aps = objectives["safety"]
        goal_aps = objectives["guarantee"]
        # Accept if safety violated OR goal never reached
        trans = {
            "wait":     [("t", "wait")]
                        + [(f"!{ap}", "safe_viol") for ap in safe_aps]
                        + [(ap, "trap") for ap in goal_aps],
            "safe_viol":[("t", "safe_viol")],
            "trap":     [("t", "trap")],
        }
        return BuchiAutomaton(
            ["wait", "safe_viol", "trap"], "wait",
            trans, {"safe_viol", "wait"}, ap_order, "template",
        )

    # ¬□♢p  =  ♢□¬p  — eventually permanently avoid p
    if mp_class == "Recurrence":
        recur_aps = objectives["recurrence"] or ap_order
        trans = {
            "pre":    [("t", "pre"), ("t", "stable")],
            "stable": [("t", "stable")] + [(ap, "reset") for ap in recur_aps],
            "reset":  [("t", "stable")],
        }
        return BuchiAutomaton(
            ["pre", "stable", "reset"], "pre",
            trans, {"stable"}, ap_order, "template",
        )

    # Fallback: single self-looping accepting state (accepts everything)
    return BuchiAutomaton(["q0"], "q0", {"q0": [("t", "q0")]}, {"q0"}, ap_order, "template_fallback")


# ─── guard evaluation ─────────────────────────────────────────────────────────

def _eval_guard(guard_expr: str, ap_order: list[str], active_aps: frozenset[str]) -> bool:
    """Evaluate a HOA/template guard expression against the active AP set."""
    if guard_expr == "t" or guard_expr == "":
        return True
    if guard_expr == "f":
        return False
    # Simple named-AP guards (template automata)
    if guard_expr.startswith("!"):
        ap = guard_expr[1:]
        return ap not in active_aps
    if guard_expr in active_aps:
        return True
    if guard_expr in ap_order:
        return guard_expr in active_aps
    # HOA numeric guard expression
    from core.lppm.automaton import _evaluate_hoa_label
    return _evaluate_hoa_label(guard_expr, ap_order, active_aps)
