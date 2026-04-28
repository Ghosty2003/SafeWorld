from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .spec_analysis import analyze_spec_structure, UNBOUNDED_SENTINEL


CONFIDENCE_PROFILES = {
    "quick": {"n_rollouts": 20},
    "moderate": {"n_rollouts": 100},
    "high-confidence": {"n_rollouts": 1000},
}


def load_task_spec(path_or_obj: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_obj, dict):
        raw = dict(path_or_obj)
    else:
        path = Path(path_or_obj)
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        raw["_source_path"] = str(path)

    predicates = raw.get("predicates", [])
    spec_block = raw.get("specification", {})
    formula = spec_block.get("formula")
    if isinstance(formula, str):
        raw["formula"] = parse_formula_string(formula, predicates)
    elif isinstance(formula, dict):
        raw["formula"] = formula
    elif "formula" not in raw:
        raise ValueError("Task spec must provide specification.formula")

    raw["aps"] = [pred["name"] for pred in predicates]
    raw["name"] = raw.get("name", raw.get("task_name", raw.get("id", "task_spec")))
    raw["id"] = raw.get("id", raw.get("task_name", raw["name"]))
    raw["type"] = spec_block.get("type", raw.get("type", "STL"))
    raw["mp_class"] = raw.get("mp_class", "TaskDefined")
    raw["rollout"] = raw.get("rollout", {})
    raw["calibration"] = raw.get("calibration", {})
    raw["predicate_map"] = {pred["name"]: pred for pred in predicates}
    raw["horizon"] = int(raw["rollout"].get("horizon", raw.get("horizon", 50)))
    analysis = analyze_spec_structure(raw)
    raw["analysis"] = analysis
    raw.update(analysis)
    return raw


def apply_confidence_profile(
    rollout: dict[str, Any] | None,
    profile: str | None,
    explicit_n: int | None = None,
) -> dict[str, Any]:
    merged = dict(rollout or {})
    if profile:
        if profile not in CONFIDENCE_PROFILES:
            raise ValueError(
                f"Unknown confidence profile '{profile}'. "
                f"Choose from {sorted(CONFIDENCE_PROFILES)}."
            )
        merged.update(CONFIDENCE_PROFILES[profile])
        merged["confidence_profile"] = profile
    if explicit_n is not None:
        merged["num_samples"] = explicit_n
        merged["n_rollouts"] = explicit_n
    elif "num_samples" in merged and "n_rollouts" not in merged:
        merged["n_rollouts"] = int(merged["num_samples"])
    elif "n_rollouts" in merged and "num_samples" not in merged:
        merged["num_samples"] = int(merged["n_rollouts"])
    return merged


def evaluate_predicate(state: dict[str, float], predicate: dict[str, Any]) -> float:
    ptype = predicate.get("type", "scalar")
    operator = predicate.get("operator", predicate.get("op", ">"))
    source = predicate.get("source", predicate.get("dim", predicate.get("object", predicate["name"])))
    threshold = float(predicate.get("threshold", 0.0))
    scale = float(predicate.get("scale", 1.0))
    offset = float(predicate.get("offset", 0.0))

    value = float(state.get(source, 0.0)) * scale + offset
    if ptype in {"scalar", "distance", "region", "decoded_scalar"}:
        if operator in (">", ">="):
            return value - threshold
        if operator in ("<", "<="):
            return threshold - value
    raise ValueError(f"Unsupported predicate type/operator combination: {ptype}/{operator}")


def evaluate_predicates(
    trajectory: list[dict[str, float]],
    predicates: list[dict[str, Any]],
    include_raw_state: bool = False,
) -> list[dict[str, float]]:
    compiled: list[dict[str, float]] = []
    for state in trajectory:
        row = dict(state) if include_raw_state else {}
        for pred in predicates:
            row[pred["name"]] = evaluate_predicate(state, pred)
        compiled.append(row)
    return compiled


TOKEN_RE = re.compile(
    r"\s*("
    r"G\[\d+,\d+\]|F\[\d+,\d+\]|U\[\d+,\d+\]|"
    r"G|F|X|!|&|\||->|\(|\)|"
    r"[A-Za-z_][A-Za-z0-9_]*"
    r")"
)


def parse_formula_string(formula: str, predicates: list[dict[str, Any]]) -> dict[str, Any]:
    predicate_names = {pred["name"] for pred in predicates}
    tokens = [tok for tok in TOKEN_RE.findall(formula) if tok.strip()]
    pos = 0

    def parse_expr() -> dict[str, Any]:
        return parse_implies()

    def parse_implies() -> dict[str, Any]:
        nonlocal pos
        node = parse_or()
        if pos < len(tokens) and tokens[pos] == "->":
            pos += 1
            node = {"type": "implies", "left": node, "right": parse_implies()}
        return node

    def parse_or() -> dict[str, Any]:
        nonlocal pos
        node = parse_and()
        while pos < len(tokens) and tokens[pos] == "|":
            pos += 1
            node = {"type": "or", "left": node, "right": parse_and()}
        return node

    def parse_and() -> dict[str, Any]:
        nonlocal pos
        node = parse_until()
        while pos < len(tokens) and tokens[pos] == "&":
            pos += 1
            node = {"type": "and", "left": node, "right": parse_until()}
        return node

    def parse_until() -> dict[str, Any]:
        nonlocal pos
        node = parse_unary()
        while pos < len(tokens) and tokens[pos].startswith("U"):
            op = tokens[pos]
            pos += 1
            a, b = _parse_bounds(op)
            node = {"type": "until", "a": a, "b": b, "left": node, "right": parse_unary()}
        return node

    def parse_unary() -> dict[str, Any]:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError(f"Unexpected end of formula while parsing '{formula}'")
        tok = tokens[pos]
        if tok == "!":
            pos += 1
            return {"type": "not", "child": parse_unary()}
        if tok == "X":
            pos += 1
            return {"type": "next", "child": parse_unary()}
        if tok == "G" or tok.startswith("G["):
            pos += 1
            a, b = _parse_bounds(tok, default=(0, UNBOUNDED_SENTINEL))
            return {"type": "always", "a": a, "b": b, "child": parse_unary()}
        if tok == "F" or tok.startswith("F["):
            pos += 1
            a, b = _parse_bounds(tok, default=(0, UNBOUNDED_SENTINEL))
            return {"type": "eventually", "a": a, "b": b, "child": parse_unary()}
        if tok == "(":
            pos += 1
            node = parse_expr()
            if pos >= len(tokens) or tokens[pos] != ")":
                raise ValueError(f"Missing ')' in formula '{formula}'")
            pos += 1
            return node
        if tok in predicate_names:
            pos += 1
            return {"type": "atom", "dim": tok, "threshold": 0.0, "op": ">"}
        raise ValueError(f"Unknown token '{tok}' in formula '{formula}'")

    node = parse_expr()
    if pos != len(tokens):
        raise ValueError(f"Unparsed tokens remaining in formula '{formula}': {tokens[pos:]}")
    return node


def _parse_bounds(token: str, default: tuple[int, int] | None = None) -> tuple[int, int]:
    match = re.search(r"\[(\d+),(\d+)\]", token)
    if match:
        return int(match.group(1)), int(match.group(2))
    if default is not None:
        return default
    raise ValueError(f"Operator '{token}' is missing bounds")
