from __future__ import annotations

from typing import Any


UNBOUNDED_SENTINEL = 10_000


def analyze_spec_structure(spec: dict[str, Any]) -> dict[str, Any]:
    formula = spec["formula"]
    bounded = is_bounded_formula(formula)
    objectives = extract_objectives(formula)
    level = infer_level(formula, bounded, objectives)
    verification_mode = "finite_stl" if bounded else "infinite_parity"
    if bounded:
        support = "sound"
        note = "Bounded STL robustness plus transfer calibration is directly supported."
    else:
        support = "approximate"
        note = (
            "Infinite-horizon parity/LPPM verification uses template automata and heuristic "
            "progress measures, so results are approximate rather than fully general."
        )
    mp_class = infer_mp_class(objectives, bounded)
    return {
        "bounded": bounded,
        "verification_mode": verification_mode,
        "support_level": support,
        "support_note": note,
        "task_level": level,
        "objectives": objectives,
        "mp_class": mp_class,
    }


def is_bounded_formula(formula: dict[str, Any]) -> bool:
    ftype = formula["type"]
    if ftype == "atom":
        return True
    if ftype in {"not", "next"}:
        return is_bounded_formula(formula["child"])
    if ftype in {"and", "or", "implies"}:
        return is_bounded_formula(formula["left"]) and is_bounded_formula(formula["right"])
    if ftype in {"always", "eventually", "until"}:
        if int(formula["b"]) >= UNBOUNDED_SENTINEL:
            return False
        children = [formula.get("child"), formula.get("left"), formula.get("right")]
        return all(is_bounded_formula(child) for child in children if child is not None)
    return False


def extract_objectives(formula: dict[str, Any]) -> dict[str, Any]:
    objectives = {
        "safety": [],
        "guarantee": [],
        "recurrence": [],
        "persistence": [],
        "responses": [],
        "other": [],
    }
    for clause in _flatten_conjunction(formula):
        if _is_invariant(clause):
            objectives["safety"].append(_atom_name(clause["child"]))
        elif _is_reachability(clause):
            objectives["guarantee"].extend(_extract_atom_names(clause["child"]))
        elif _is_recurrence(clause):
            objectives["recurrence"].append(_atom_name(clause["child"]["child"]))
        elif _is_persistence(clause):
            objectives["persistence"].append(_atom_name(clause["child"]["child"]))
        else:
            response = _extract_response(clause)
            if response is not None:
                objectives["responses"].append(response)
            else:
                objectives["other"].append(clause)
    return objectives


def infer_level(formula: dict[str, Any], bounded: bool, objectives: dict[str, Any]) -> str:
    if bounded:
        if objectives["safety"] and not objectives["guarantee"] and not objectives["responses"]:
            return "L1" if _is_simple_atom_formula(formula) else "L3"
        if objectives["guarantee"] and not objectives["safety"] and not objectives["responses"]:
            return "L2" if _is_simple_atom_formula(formula) else "L3"
        return "L4"
    if objectives["responses"] and (
        len(objectives["responses"]) > 1
        or objectives["recurrence"]
        or objectives["guarantee"]
        or objectives["safety"]
        or objectives["other"]
    ):
        return "L8"
    if objectives["responses"]:
        return "L7"
    if objectives["persistence"]:
        return "L6"
    if objectives["recurrence"]:
        return "L5"
    if objectives["safety"] and objectives["guarantee"]:
        return "L4"
    if objectives["guarantee"]:
        return "L2"
    return "L1"


def infer_mp_class(objectives: dict[str, Any], bounded: bool) -> str:
    if bounded:
        if objectives["safety"] and objectives["guarantee"]:
            return "Obligation"
        if objectives["guarantee"]:
            return "Guarantee"
        return "Safety"
    if objectives["responses"]:
        return "Streett" if len(objectives["responses"]) > 1 or objectives["recurrence"] else "Reactivity"
    if objectives["persistence"]:
        return "Persistence"
    if objectives["recurrence"]:
        return "Recurrence"
    if objectives["safety"] and objectives["guarantee"]:
        return "Obligation"
    if objectives["guarantee"]:
        return "Guarantee"
    return "Safety"


def _flatten_conjunction(formula: dict[str, Any]) -> list[dict[str, Any]]:
    if formula["type"] == "and":
        return _flatten_conjunction(formula["left"]) + _flatten_conjunction(formula["right"])
    return [formula]


def _is_invariant(node: dict[str, Any]) -> bool:
    return node["type"] == "always" and node["child"]["type"] == "atom"


def _is_reachability(node: dict[str, Any]) -> bool:
    # F(atom) or F(compound) — but not F(G(…)) which is Persistence
    return node["type"] == "eventually" and not _is_persistence(node)


def _is_recurrence(node: dict[str, Any]) -> bool:
    return (
        node["type"] == "always"
        and node["child"]["type"] == "eventually"
        and node["child"]["child"]["type"] == "atom"
        and int(node["b"]) >= UNBOUNDED_SENTINEL
        and int(node["child"]["b"]) >= UNBOUNDED_SENTINEL
    )


def _is_persistence(node: dict[str, Any]) -> bool:
    return (
        node["type"] == "eventually"
        and node["child"]["type"] == "always"
        and node["child"]["child"]["type"] == "atom"
        and int(node["b"]) >= UNBOUNDED_SENTINEL
        and int(node["child"]["b"]) >= UNBOUNDED_SENTINEL
    )


def _extract_response(node: dict[str, Any]) -> dict[str, str] | None:
    if node["type"] != "always":
        return None
    inner = node["child"]
    if inner["type"] == "implies":
        left = inner["left"]
        right = inner["right"]
    elif inner["type"] == "or" and inner["left"]["type"] == "not":
        left = inner["left"]["child"]
        right = inner["right"]
    else:
        return None
    if left["type"] != "atom":
        return None
    if right["type"] == "eventually" and right["child"]["type"] == "atom":
        return {"trigger": _atom_name(left), "response": _atom_name(right["child"])}
    return None


def _atom_name(node: dict[str, Any]) -> str:
    if node["type"] != "atom":
        raise ValueError(f"Expected atom node, got {node['type']}")
    return str(node["dim"])


def _extract_atom_names(node: dict[str, Any]) -> list[str]:
    """Recursively collect every atom dim referenced in a formula subtree."""
    if node["type"] == "atom":
        return [str(node["dim"])]
    names: list[str] = []
    for key in ("child", "left", "right"):
        child = node.get(key)
        if child is not None:
            names.extend(_extract_atom_names(child))
    return names


def _is_simple_atom_formula(formula: dict[str, Any]) -> bool:
    if formula["type"] in {"always", "eventually"}:
        return formula["child"]["type"] == "atom"
    return False
