from __future__ import annotations

from typing import Any


UNBOUNDED_SENTINEL = 10_000

# support_level enum. "sound" is reserved exclusively for a deductive,
# support-wide NN-verification branch (zero miscoverage) -- no code path in
# this project implements that branch, so no code path may ever produce
# SUPPORT_DEDUCTIVE. Bounded STL's guarantee is Theorem 5.1's calibrated
# (1-delta_cp-delta_err) confidence, not a deterministic proof, hence
# SUPPORT_CALIBRATED rather than "sound".
SUPPORT_DEDUCTIVE = "deductive"
SUPPORT_CALIBRATED = "calibrated"
SUPPORT_APPROXIMATE = "approximate"

# Batch-2 L2 audit fix (Option B, approved): infer_mp_class() must never
# silently default an unrecognized clause into Safety/Obligation/etc. When
# extract_objectives() cannot place a top-level (post-AND-flattening) clause
# into any known Manna-Pnueli bucket, the WHOLE spec's classification is
# untrustworthy -- some other clause could shift the true class arbitrarily
# (e.g. an unrecognized clause could itself be Recurrence-class, which must
# never be routed into the co-Büchi L2 pipeline, Lemma E.6). mp_class is set
# to this sentinel instead of guessing, and analyze_spec_structure() also
# sets classification_uncertain=True so callers don't have to string-compare.
UNCLASSIFIED_MP_CLASS = "Unclassified"


def analyze_spec_structure(spec: dict[str, Any]) -> dict[str, Any]:
    formula = spec["formula"]
    bounded = is_bounded_formula(formula)
    objectives = extract_objectives(formula)
    level = infer_level(formula, bounded, objectives)
    # Prefer the spec's hardcoded level field when present — structural inference
    # cannot distinguish L4/L5/L6/L8 within the recurrence class without additional
    # heuristics, and the spec author's intent is authoritative.
    if "level" in spec:
        level = f"L{spec['level']}"
    verification_mode = "finite_stl" if bounded else "infinite_parity"
    if bounded:
        support = SUPPORT_CALIBRATED
        note = (
            "Bounded STL robustness plus transfer calibration (Theorem 5.1): "
            "confidence 1-delta_cp-delta_err, not a deterministic guarantee."
        )
    else:
        support = SUPPORT_APPROXIMATE
        note = (
            "Infinite-horizon parity/LPPM verification uses template automata and heuristic "
            "progress measures, so results are approximate rather than fully general."
        )
    mp_class = infer_mp_class(objectives, bounded)
    classification_uncertain = mp_class == UNCLASSIFIED_MP_CLASS
    if classification_uncertain:
        support = SUPPORT_APPROXIMATE
        note = (
            f"Classification uncertain: {len(objectives['other'])} top-level clause(s) "
            "did not match any recognized Safety/Guarantee/Recurrence/Persistence/Response "
            "AST shape (utils/spec_analysis.py::extract_objectives()'s 'other' bucket is "
            "non-empty). Refusing to guess an mp_class -- see EXPERIMENT_CONFIG.md."
        )
    return {
        "bounded": bounded,
        "verification_mode": verification_mode,
        "support_level": support,
        "support_note": note,
        "task_level": level,
        "objectives": objectives,
        "mp_class": mp_class,
        "classification_uncertain": classification_uncertain,
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
        # Batch 3a: G(a1 v a2 v ... v an) -- a PURE disjunction of atoms
        # under an invariant. Kept as a SEPARATE bucket from "safety" (not
        # merged into a flat atom list) because the two have different
        # automaton semantics: "safety" entries are each independently
        # required (AND across atoms: trap if ANY one is ever violated);
        # each entry here is a single clause's disjuncts, which must ALL be
        # violated SIMULTANEOUSLY to trap (see core/lppm/automaton.py's
        # Safety branch). Flattening these into "safety" would silently
        # reproduce the "G(a v b) treated as G(a) and G(b)" bug this batch
        # exists to fix.
        "safety_disjunctions": [],
        "guarantee": [],
        # Batch 3b: F(A and F(B and F(C ...))) -- nested sequential
        # reachability. Kept SEPARATE from "guarantee" (an unordered flat
        # atom list) for the same reason safety_disjunctions is separate
        # from "safety": order is load-bearing here (see
        # core/lppm/automaton.py's Guarantee branch, which builds a linear
        # chain for entries here instead of a powerset-of-unordered-remaining
        # automaton). A single-atom F(atom) still goes into "guarantee" —
        # this bucket is only for genuine multi-goal chains (length >= 2).
        "guarantee_sequences": [],
        "recurrence": [],
        "persistence": [],
        "responses": [],
        "other": [],
    }
    for clause in _flatten_conjunction(formula):
        if _is_invariant(clause):
            objectives["safety"].append(_atom_name(clause["child"]))
        elif _is_invariant_disjunction(clause):
            disjuncts = _flatten_or_of_atoms(clause["child"])
            objectives["safety_disjunctions"].append([_atom_name(a) for a in disjuncts])
        elif _is_reachability(clause):
            sequence = _extract_ordered_sequence(clause)
            if sequence is not None and len(sequence) >= 2:
                objectives["guarantee_sequences"].append(sequence)
            elif sequence is not None:
                objectives["guarantee"].extend(sequence)
            else:
                # _is_reachability matched (top-level eventually, not
                # persistence) but the body isn't the plain-atom or clean
                # nested-and-eventually shape _extract_ordered_sequence
                # understands (e.g. F(A or B)). Conservative: unrecognized,
                # not guessed via the old unordered _extract_atom_names path.
                objectives["other"].append(clause)
        elif _is_recurrence(clause):
            # child may be F(atom), F(compound), or U(p,q) — collect all referenced APs
            objectives["recurrence"].extend(_extract_atom_names(clause["child"]))
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
    has_safety = bool(objectives["safety"]) or bool(objectives.get("safety_disjunctions"))
    has_guarantee = bool(objectives["guarantee"]) or bool(objectives.get("guarantee_sequences"))
    if bounded:
        if has_safety and not has_guarantee and not objectives["responses"]:
            return "L1" if _is_simple_atom_formula(formula) else "L3"
        if has_guarantee and not has_safety and not objectives["responses"]:
            return "L2" if _is_simple_atom_formula(formula) else "L3"
        return "L4"
    if objectives["responses"] and (
        len(objectives["responses"]) > 1
        or objectives["recurrence"]
        or has_guarantee
        or has_safety
        or objectives["other"]
    ):
        return "L8"
    if objectives["responses"]:
        return "L7"
    if objectives["persistence"]:
        return "L6"
    if objectives["recurrence"]:
        return "L5"
    if has_safety and has_guarantee:
        return "L4"
    if has_guarantee:
        return "L2"
    return "L1"


def infer_mp_class(objectives: dict[str, Any], bounded: bool) -> str:
    if objectives["other"]:
        # At least one top-level clause matched none of the known AST shapes.
        # It could be ANY Manna-Pnueli class -- in particular Recurrence,
        # which must never be silently folded into Safety/Obligation and
        # routed into the co-Büchi L2 pipeline (Lemma E.6). Refuse to guess.
        return UNCLASSIFIED_MP_CLASS
    has_safety = bool(objectives["safety"]) or bool(objectives["safety_disjunctions"])
    has_guarantee = bool(objectives["guarantee"]) or bool(objectives["guarantee_sequences"])
    if bounded:
        # Recurrence/Response dominate: G(F·), G(·U·), G(p→F(q)) are all Recurrence class
        # regardless of whether bounds are finite. Safety/Guarantee are pointwise and
        # co-safety objectives; Recurrence requires repetition and is semantically stronger.
        if objectives["recurrence"] or objectives["responses"]:
            return "Recurrence"
        if has_safety and has_guarantee:
            return "Obligation"
        if has_guarantee:
            return "Guarantee"
        return "Safety"
    if objectives["responses"]:
        return "Streett" if len(objectives["responses"]) > 1 or objectives["recurrence"] else "Reactivity"
    if objectives["persistence"]:
        return "Persistence"
    if objectives["recurrence"]:
        return "Recurrence"
    if has_safety and has_guarantee:
        return "Obligation"
    if has_guarantee:
        return "Guarantee"
    return "Safety"


def _flatten_conjunction(formula: dict[str, Any]) -> list[dict[str, Any]]:
    if formula["type"] == "and":
        return _flatten_conjunction(formula["left"]) + _flatten_conjunction(formula["right"])
    return [formula]


def _is_invariant(node: dict[str, Any]) -> bool:
    return node["type"] == "always" and node["child"]["type"] == "atom"


def _flatten_or_of_atoms(node: dict[str, Any]) -> list[dict[str, Any]] | None:
    """
    Returns the flat list of atom nodes if `node` is a PURE disjunction of
    atoms (possibly nested, e.g. or(or(a,b),c)), else None. Deliberately
    conservative: any leaf that isn't a plain atom (e.g. a temporal operator,
    a negation, a further compound) makes the whole subtree unrecognized
    (None) rather than guessed at -- see _is_invariant_disjunction()'s
    docstring for why this must not be stretched to cover mixed shapes like
    or(atom, F(atom)) (that is a different, unrelated gap, not this one).
    """
    if node["type"] == "atom":
        return [node]
    if node["type"] == "or":
        left = _flatten_or_of_atoms(node["left"])
        right = _flatten_or_of_atoms(node["right"])
        if left is None or right is None:
            return None
        return left + right
    return None


def _is_invariant_disjunction(node: dict[str, Any]) -> bool:
    """
    G(a1 v a2 v ... v an), n>=2 -- a PURE disjunction of atoms under an
    invariant (specs/ltl_specs.py::ltl_conditional_speed's actual shape:
    G(lor(atom, atom))). Deliberately narrow: does NOT match mixed
    disjunctions like or(atom, F(atom)) (specs/ltl_specs.py::
    ltl_hazard_response's shape -- a response pattern with negation encoded
    via atom polarity instead of an explicit not(...), a different AST-
    recognition gap with a different fix, not covered here). Widening this
    to guess at every equivalent way of writing an implication was
    considered and rejected in favor of staying conservative (Batch-2's
    Option-B principle: unrecognized stays Unclassified, never guessed).
    """
    if node["type"] != "always":
        return False
    disjuncts = _flatten_or_of_atoms(node["child"])
    return disjuncts is not None and len(disjuncts) >= 2


def _is_reachability(node: dict[str, Any]) -> bool:
    # F(atom) or F(compound) — but not F(G(…)) which is Persistence
    return node["type"] == "eventually" and not _is_persistence(node)


def _is_recurrence(node: dict[str, Any]) -> bool:
    # Manna-Pnueli Recurrence: G(F(·)) or G(· U ·) — bounded or unbounded.
    # Routing (finite_stl vs infinite_parity) is determined separately by is_bounded_formula();
    # mp_class is determined by temporal structure, independent of bound magnitude.
    if node["type"] != "always":
        return False
    return node["child"]["type"] in ("eventually", "until")


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


def _extract_ordered_sequence(node: dict[str, Any]) -> list[str] | None:
    """
    Batch 3b. Parses F(A), or the nested chain F(A and F(B and F(C ...))),
    into an ORDERED list of atom names [A, B, C, ...] -- specs/ltl_specs.py::
    ltl_sequential_goals / ltl_three_stage's exact shape. Returns None for
    anything else (e.g. F(A or B), F(compound-non-atom)): deliberately
    narrow, matching Batch-2's Option-B conservatism -- callers must route a
    None result to objectives["other"] (Unclassified), not fall back to the
    old unordered _extract_atom_names() extraction, which is exactly the bug
    this batch fixes (order lost -> B-before-A wrongly accepted).

    A single-atom result (length 1) is the ordinary F(atom) case -- callers
    should treat that the same as before (the "guarantee" bucket), reserving
    the new "guarantee_sequences" bucket for genuine multi-goal chains.
    """
    if node["type"] != "eventually":
        return None
    child = node["child"]
    if child["type"] == "atom":
        return [_atom_name(child)]
    if child["type"] == "and":
        left, right = child["left"], child["right"]
        if left["type"] != "atom":
            return None
        rest = _extract_ordered_sequence(right)
        if rest is None:
            return None
        return [_atom_name(left)] + rest
    return None


def _is_simple_atom_formula(formula: dict[str, Any]) -> bool:
    if formula["type"] in {"always", "eventually"}:
        return formula["child"]["type"] == "atom"
    return False
