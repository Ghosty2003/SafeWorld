"""
core/cegar/loop.py

Main CEGAR loop and one-shot verification (Algorithm 1, lines 14-41 / Section 4.6).

CEGAR loop
----------
for k = 1..K:
  T   = build_abstract_system(trajectories, predicates)
  P   = build_product_automaton(T, B¬φ)
  scc = find_accepting_scc(P)
  if scc is None:            return SAFE
  cex = extract_lasso(P, scc)
  if is_concretizable(cex):  return VIOLATION(cex)
  p_new = variance_split(cex, T)
  predicates += [p_new]
return INCONCLUSIVE

One-shot (Section 4.6)
----------------------
Same as CEGAR with K=1 and P0 derived directly from the spec's atoms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .abstraction import (
    AbstractSystem,
    Predicate,
    build_abstract_system,
    derive_predicates_from_spec,
    variance_split_predicate,
)
from .buchi import BuchiAutomaton, build_negation_buchi
from .product import build_product_automaton
from .scc import find_accepting_scc
from .counterexample import AbstractCex, extract_lasso, is_concretizable


# ─── result ───────────────────────────────────────────────────────────────────

CEGAR_SAFE        = "SAFE"
CEGAR_VIOLATION   = "VIOLATION"
CEGAR_INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class CegarResult:
    """
    Result of the CEGAR or ONE-SHOT verification procedure.

    Verdicts
    --------
    SAFE         — no accepting cycle found; φ holds over all observed abstract
                   transitions.  Sound when the sample covers all reachable cells.
    VIOLATION    — an accepting cycle was found AND is concretizable; a concrete
                   trajectory segment witnesses a violation.
    INCONCLUSIVE — accepting cycles found but all are spurious; iteration budget
                   exhausted without refinement resolving the ambiguity.
    """

    verdict:     str                        # SAFE | VIOLATION | INCONCLUSIVE
    iterations:  int = 0
    predicates:  list[Predicate] = field(default_factory=list)
    counterexample: AbstractCex | None = None
    abstract_sys:   AbstractSystem | None = None
    detail:      str = ""


# ─── CEGAR configuration ──────────────────────────────────────────────────────

@dataclass
class CegarConfig:
    max_iterations: int   = 5
    delta_cex:      float = 0.05    # CP failure prob for concretization check
    initial_predicates: list[Predicate] = field(default_factory=list)
    verbose:        bool  = False


# ─── main loops ───────────────────────────────────────────────────────────────

def run_cegar(
    trajectories: list[list[dict[str, float]]],
    spec:         dict[str, Any],
    config:       CegarConfig | None = None,
) -> CegarResult:
    """
    Run the CEGAR verification loop (Algorithm 1, lines 14-41).

    Parameters
    ----------
    trajectories : N latent rollouts (list of AP-dicts).
    spec         : specification dict.
    config       : CegarConfig with initial predicates and iteration budget.
    """
    cfg = config or CegarConfig()

    # Build ¬φ Büchi automaton once (reused across iterations)
    buchi: BuchiAutomaton = build_negation_buchi(spec)

    # Decompose conjunctions for compositional verification (Section 4.9)
    conjuncts = _decompose_conjuncts(spec)

    predicates: list[Predicate] = list(cfg.initial_predicates)

    for k in range(1, cfg.max_iterations + 1):
        if cfg.verbose:
            print(f"    [CEGAR] iter {k}: {len(predicates)} predicates", flush=True)

        abstract_sys = build_abstract_system(trajectories, predicates)

        # Find accepting cycle across conjuncts (compositional check)
        found_scc = None
        for conj in conjuncts:
            buchi_i = build_negation_buchi(conj) if len(conjuncts) > 1 else buchi
            product = build_product_automaton(abstract_sys, buchi_i, conj)
            scc = find_accepting_scc(product)
            if scc is not None:
                found_scc  = scc
                found_prod = product
                found_conj = conj
                break

        if found_scc is None:
            return CegarResult(
                verdict=CEGAR_SAFE,
                iterations=k,
                predicates=predicates,
                abstract_sys=abstract_sys,
                detail=f"No accepting cycle found after {k} iteration(s).",
            )

        # Extract lasso counterexample
        cex = extract_lasso(found_prod, found_scc)
        if cex is None:
            # Shouldn't happen; treat as spurious
            if cfg.verbose:
                print("    [CEGAR] lasso extraction failed — spurious?")
        elif is_concretizable(cex, abstract_sys, cfg.delta_cex):
            return CegarResult(
                verdict=CEGAR_VIOLATION,
                iterations=k,
                predicates=predicates,
                counterexample=cex,
                abstract_sys=abstract_sys,
                detail=f"Concretizable accepting cycle found at iteration {k}.",
            )

        # Spurious cycle — refine predicates via variance split
        suspect_indices = cex.abstract_indices_in_cycle if cex else {
            pk[0] for pk in found_scc
        }
        new_pred = variance_split_predicate(suspect_indices, abstract_sys)
        if new_pred is None or new_pred in predicates:
            if cfg.verbose:
                print("    [CEGAR] no useful split found — stopping early")
            return CegarResult(
                verdict=CEGAR_INCONCLUSIVE,
                iterations=k,
                predicates=predicates,
                counterexample=cex,
                abstract_sys=abstract_sys,
                detail="Variance split found no new predicate; iteration limit would not help.",
            )
        predicates.append(new_pred)
        if cfg.verbose:
            print(f"    [CEGAR] added predicate: {new_pred}")

    return CegarResult(
        verdict=CEGAR_INCONCLUSIVE,
        iterations=cfg.max_iterations,
        predicates=predicates,
        abstract_sys=abstract_sys if 'abstract_sys' in dir() else None,
        detail=f"Iteration budget ({cfg.max_iterations}) exhausted without resolving spurious cycles.",
    )


def run_oneshot(
    trajectories: list[list[dict[str, float]]],
    spec:         dict[str, Any],
    verbose:      bool = False,
    delta_cex:    float = 0.05,
) -> CegarResult:
    """
    One-shot verification (Section 4.6): K=1 CEGAR with spec-derived predicates.

    Cost: O(N·n + 2n·|Q|) — a single CEGAR iteration.
    Returns SAFE, VIOLATION, or INCONCLUSIVE (if spurious cycle found).
    """
    predicates = derive_predicates_from_spec(spec)
    cfg = CegarConfig(
        max_iterations=1,
        delta_cex=delta_cex,
        initial_predicates=predicates,
        verbose=verbose,
    )
    result = run_cegar(trajectories, spec, cfg)
    return result


# ─── conjunct decomposition ───────────────────────────────────────────────────

def _decompose_conjuncts(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Split a top-level conjunction φ = φ1 ∧ ... ∧ φk into sub-specs.

    If the formula is not a conjunction (or the spec has no structured
    conjuncts field), returns [spec] as a singleton list.
    """
    formula = spec.get("formula", {})
    if formula.get("type") != "and":
        return [spec]

    conjuncts: list[dict[str, Any]] = []
    _collect_conjuncts(formula, conjuncts, spec)
    return conjuncts if conjuncts else [spec]


def _collect_conjuncts(
    node:      dict[str, Any],
    out:       list[dict[str, Any]],
    base_spec: dict[str, Any],
) -> None:
    if node.get("type") == "and":
        _collect_conjuncts(node["left"],  out, base_spec)
        _collect_conjuncts(node["right"], out, base_spec)
    else:
        sub_spec = dict(base_spec)
        sub_spec["formula"] = node
        sub_spec.pop("analysis", None)   # force re-analysis
        out.append(sub_spec)
