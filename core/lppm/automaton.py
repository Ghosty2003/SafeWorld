from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from utils.spec_analysis import UNCLASSIFIED_MP_CLASS, analyze_spec_structure

try:
    import spot  # type: ignore
except ImportError:
    spot = None


@dataclass
class ParityAutomaton:
    states: list[str]
    initial: str
    priority: dict[str, int]
    transition: dict[tuple[str, frozenset[str]], str]
    state_meta: dict[str, dict] = field(default_factory=dict)
    backend: str = "template"
    exact: bool = False
    ap_order: list[str] = field(default_factory=list)
    edge_guards: dict[str, list[tuple[str, str, int]]] = field(default_factory=dict)

    def step(self, q: str, active_aps: frozenset[str]) -> str:
        q_next, _ = self.step_with_priority(q, active_aps)
        return q_next

    def step_with_priority(self, q: str, active_aps: frozenset[str]) -> tuple[str, int]:
        if self.edge_guards:
            for guard_expr, dst, priority in self.edge_guards.get(q, []):
                if _evaluate_hoa_label(guard_expr, self.ap_order, active_aps):
                    return dst, priority
            return q, self.priority.get(q, 0)

        key = (q, active_aps)
        if key in self.transition:
            return self.transition[key], self.priority.get(q, 0)

        candidates = sorted(
            self.transition.items(),
            key=lambda item: len(item[0][1]),
            reverse=True,
        )
        for (src, guard), dst in candidates:
            if src == q and guard <= active_aps:
                return dst, self.priority.get(q, 0)
        return q, self.priority.get(q, 0)

    @property
    def odd_priorities(self) -> list[int]:
        if self.edge_guards:
            edge_priorities = {
                priority
                for edges in self.edge_guards.values()
                for _, _, priority in edges
            }
            return sorted({p for p in edge_priorities if p % 2 == 1})
        return sorted({p for p in self.priority.values() if p % 2 == 1})


@dataclass
class ProductState:
    t: int
    z: dict[str, float]
    q: str
    q_next: str
    priority: int


def build_parity_automaton(spec: dict[str, Any]) -> ParityAutomaton:
    analysis = spec.get("analysis") or analyze_spec_structure(spec)
    spec["analysis"] = analysis
    exact_dpa = _build_spot_parity_automaton(spec)
    if exact_dpa is not None:
        return exact_dpa

    mp = analysis["mp_class"]
    objectives = analysis["objectives"]

    if mp == "Safety":
        # Batch 3a: safety_disjunctions holds G(a1 v a2 v ... v an) clauses
        # (see utils/spec_analysis.py) SEPARATELY from plain single-atom
        # safety objectives -- they need a different trap condition (ALL
        # disjuncts simultaneously violated), not "any one atom violated".
        safety_disjunctions = objectives.get("safety_disjunctions", [])
        # Only fall back to the declared aps list when NEITHER bucket
        # produced anything -- previously this fell back whenever
        # objectives["safety"] alone was empty, which would have silently
        # mistreated a purely-disjunctive spec's atoms as independent
        # must-never-violate atoms (reintroducing the G(a v b) -> G(a) & G(b)
        # bug this batch exists to fix).
        if objectives["safety"] or safety_disjunctions:
            safe_aps = objectives["safety"]
        else:
            safe_aps = spec.get("aps", [])
        transitions = {("trap", frozenset()): "trap"}
        for ap in safe_aps:
            transitions[("ok", frozenset({f"not_{ap}"}))] = "trap"
            transitions[("trap", frozenset({f"not_{ap}"}))] = "trap"
        for clause in safety_disjunctions:
            # Trap iff EVERY disjunct's own atom condition is simultaneously
            # false (guard <= active_aps subset-matching in
            # step_with_priority() already gives exactly this semantics: the
            # guard only matches when ALL its "not_x" labels are present).
            guard = frozenset(f"not_{ap}" for ap in clause)
            transitions[("ok", guard)] = "trap"
            transitions[("trap", guard)] = "trap"
        transitions[("ok", frozenset())] = "ok"
        return ParityAutomaton(
            states=["ok", "trap"],
            initial="ok",
            priority={"ok": 0, "trap": 1},
            transition=transitions,
            state_meta={"ok": {"kind": "safe"}, "trap": {"kind": "trap"}},
            backend="template",
        )

    if mp == "Guarantee":
        sequences = objectives.get("guarantee_sequences", [])
        if not sequences:
            # Unchanged path (no sequences present): identical to the
            # pre-Batch-3b code, zero behavior change for plain F(atom) /
            # unordered-multi-goal specs.
            goals = objectives["guarantee"] or spec.get("aps", [])
            states, priority, transitions, state_meta = [], {}, {}, {}
            for remaining in _powerset_strings(goals):
                name = _remaining_state_name("wait", remaining)
                states.append(name)
                priority[name] = 1 if remaining else 0
                state_meta[name] = {"kind": "waiting", "remaining_goals": list(remaining)}
            initial = _remaining_state_name("wait", tuple(goals))
            for remaining in _powerset_strings(goals):
                src = _remaining_state_name("wait", remaining)
                transitions[(src, frozenset())] = src
                for active in _all_label_sets(goals):
                    reduced = tuple(goal for goal in remaining if goal not in active)
                    transitions[(src, active)] = _remaining_state_name("wait", reduced)
            return ParityAutomaton(states, initial, priority, transitions, state_meta, backend="template")

        # Batch 3b: at least one F(A and F(B and F(C ...))) ordered chain is
        # present. State = (remaining UNORDERED atoms, tuple of per-sequence
        # progress indices). Each sequence's progress can only advance by
        # ONE step per tick, and ONLY via its own next-required atom -- seeing
        # a later atom in the chain out of order does not advance it, fixing
        # the "B before A wrongly accepted" bug (see utils/spec_analysis.py's
        # guarantee_sequences bucket and its Batch-3b comment for the extraction
        # side of this fix).
        goals = objectives["guarantee"]  # any remaining plain (unordered) F(atom) goals
        all_labels = list(goals) + [ap for seq in sequences for ap in seq]

        def seq_states():
            ranges = [range(len(seq) + 1) for seq in sequences]
            result = [()]
            for r in ranges:
                result = [t + (i,) for t in result for i in r]
            return result

        def state_name(remaining: tuple[str, ...], progress: tuple[int, ...]) -> str:
            if not remaining and all(p == len(seq) for p, seq in zip(progress, sequences)):
                return "wait_done"
            prog_str = ",".join(str(p) for p in progress)
            return "wait:" + ",".join(remaining) + "|" + prog_str

        def advance_progress(progress: tuple[int, ...], active: frozenset[str]) -> tuple[int, ...]:
            # External-review fix: F(A and F(B and F(C ...))) is satisfied
            # the instant its remaining required atoms are ALL true
            # simultaneously (F(B) only needs B true at some time >= now,
            # including right now) -- a single `if` here only ever consumed
            # ONE atom per observation, even when several consecutive
            # required atoms were active in the very same observation,
            # wrongly leaving the automaton "still waiting" for an atom that
            # had, in fact, already been satisfied at this same instant.
            # `while` keeps advancing through as many consecutive required
            # atoms as are simultaneously active.
            new_progress = []
            for p, seq in zip(progress, sequences):
                while p < len(seq) and seq[p] in active:
                    p += 1
                new_progress.append(p)
            return tuple(new_progress)

        states, priority, transitions, state_meta = [], {}, {}, {}
        for remaining in _powerset_strings(goals):
            for progress in seq_states():
                name = state_name(remaining, progress)
                if name in priority:
                    continue
                states.append(name)
                done = not remaining and all(p == len(seq) for p, seq in zip(progress, sequences))
                priority[name] = 0 if done else 1
                state_meta[name] = {
                    "kind": "waiting", "remaining_goals": list(remaining),
                    "sequence_progress": list(progress),
                }
        initial_progress = tuple(0 for _ in sequences)
        initial = state_name(tuple(goals), initial_progress)
        for remaining in _powerset_strings(goals):
            for progress in seq_states():
                src = state_name(remaining, progress)
                transitions[(src, frozenset())] = src
                for active in _all_label_sets(all_labels):
                    reduced = tuple(goal for goal in remaining if goal not in active)
                    new_progress = advance_progress(progress, active)
                    transitions[(src, active)] = state_name(reduced, new_progress)
        return ParityAutomaton(states, initial, priority, transitions, state_meta, backend="template")

    if mp == "Obligation":
        safe_aps = objectives["safety"]
        goals = objectives["guarantee"]
        states = ["trap"]
        priority = {"trap": 1}
        transitions = {("trap", frozenset()): "trap"}
        state_meta = {"trap": {"kind": "trap"}}
        for remaining in _powerset_strings(goals):
            name = _remaining_state_name("wait_ok", remaining)
            states.append(name)
            priority[name] = 1 if remaining else 0
            state_meta[name] = {"kind": "waiting", "remaining_goals": list(remaining)}
        initial = _remaining_state_name("wait_ok", tuple(goals))
        for remaining in _powerset_strings(goals):
            src = _remaining_state_name("wait_ok", remaining)
            transitions[(src, frozenset())] = src
            for active in _all_label_sets(goals + safe_aps):
                reduced = tuple(goal for goal in remaining if goal not in active)
                transitions[(src, active)] = _remaining_state_name("wait_ok", reduced)
            for ap in safe_aps:
                transitions[(src, frozenset({f"not_{ap}"}))] = "trap"
        for ap in safe_aps:
            transitions[("trap", frozenset({f"not_{ap}"}))] = "trap"
        return ParityAutomaton(states, initial, priority, transitions, state_meta, backend="template")

    if mp == "Recurrence":
        recur_aps = objectives["recurrence"] or spec.get("aps", [])
        states, priority, transitions, state_meta = [], {}, {}, {}
        for remaining in _powerset_strings(recur_aps):
            name = _remaining_state_name("seek", remaining)
            states.append(name)
            priority[name] = 2 if not remaining else 1
            state_meta[name] = {"kind": "recurrence", "remaining_recur": list(remaining)}
        initial = _remaining_state_name("seek", tuple(recur_aps))
        for remaining in _powerset_strings(recur_aps):
            src = _remaining_state_name("seek", remaining)
            for active in _all_label_sets(recur_aps):
                base_remaining = tuple(recur_aps) if not remaining else tuple(remaining)
                reduced = tuple(ap for ap in base_remaining if ap not in active)
                transitions[(src, active)] = _remaining_state_name("seek", reduced)
            transitions[(src, frozenset())] = _remaining_state_name(
                "seek", tuple(recur_aps) if not remaining else remaining
            )
        return ParityAutomaton(states, initial, priority, transitions, state_meta, backend="template")

    if mp == "Persistence":
        stable_aps = objectives["persistence"] or spec.get("aps", [])
        transitions = {
            ("pre", frozenset()): "pre",
            ("absorbed", frozenset()): "pre",
        }
        for active in _all_label_sets(stable_aps):
            if all(ap in active for ap in stable_aps):
                transitions[("pre", active)] = "absorbed"
                transitions[("absorbed", active)] = "absorbed"
            else:
                transitions[("pre", active)] = "pre"
                transitions[("absorbed", active)] = "pre"
        return ParityAutomaton(
            states=["pre", "absorbed"],
            initial="pre",
            priority={"pre": 1, "absorbed": 0},
            transition=transitions,
            state_meta={"pre": {"kind": "pre"}, "absorbed": {"kind": "absorbed"}},
            backend="template",
        )

    if mp in {"Reactivity", "Streett"}:
        safety_aps = objectives["safety"]
        responses = objectives["responses"]
        triggers = [item["trigger"] for item in responses]
        response_labels = [item["response"] for item in responses]
        pending_items = tuple(f"{t}->{r}" for t, r in zip(triggers, response_labels))
        states = ["trap"]
        priority = {"trap": 1}
        transitions = {("trap", frozenset()): "trap"}
        state_meta = {"trap": {"kind": "trap"}}
        for pending in _powerset_strings(list(pending_items)):
            name = _remaining_state_name("pending", pending)
            states.append(name)
            priority[name] = 1 if pending else 0
            state_meta[name] = {"kind": "pending", "pending": list(pending)}
        initial = _remaining_state_name("pending", ())
        labels = safety_aps + triggers + response_labels
        for pending in _powerset_strings(list(pending_items)):
            src = _remaining_state_name("pending", pending)
            for active in _all_label_sets(labels):
                next_pending = set(pending)
                for trig, resp in zip(triggers, response_labels):
                    key = f"{trig}->{resp}"
                    if trig in active and resp not in active:
                        next_pending.add(key)
                    if resp in active and key in next_pending:
                        next_pending.remove(key)
                transitions[(src, active)] = _remaining_state_name("pending", tuple(sorted(next_pending)))
            transitions[(src, frozenset())] = src
            for ap in safety_aps:
                transitions[(src, frozenset({f"not_{ap}"}))] = "trap"
        return ParityAutomaton(states, initial, priority, transitions, state_meta, backend="template")

    # Batch-2 L2 audit fix (Option B): every legitimate mp_class value
    # infer_mp_class() can produce has an explicit branch above. Reaching
    # here means either UNCLASSIFIED_MP_CLASS (extract_objectives() found an
    # unrecognized clause -- see utils/spec_analysis.py) or some other
    # unrecognized value, possibly injected directly via spec["analysis"]
    # rather than computed by analyze_spec_structure(). Previously this
    # silently returned a trivial single-state "accept everything" automaton
    # (states=["q0"], priority 0 -- always in the accepting region regardless
    # of the trajectory), which is at least as dangerous as the classifier
    # defaulting to Safety/Obligation: it fabricates a co-Büchi warrant for a
    # specification this code never actually understood. Fail loud instead.
    raise ValueError(
        f"build_parity_automaton(): no automaton construction branch for "
        f"mp_class={mp!r} (spec id={spec.get('id', '<unknown>')!r}). Refusing "
        "to fall back to a trivial always-accepting automaton -- this would "
        "silently issue an unsupported co-Büchi warrant. If mp_class == "
        f"{UNCLASSIFIED_MP_CLASS!r}, the caller should have already routed "
        "this spec to INCONCLUSIVE before reaching build_parity_automaton() "
        "(see main.py::verify()'s Step-3 classification_uncertain check)."
    )


def extract_active_aps(state: dict[str, float], spec: dict[str, Any]) -> frozenset[str]:
    active = set()
    atom_map = spec.get("atom_map") or collect_atom_map(spec["formula"])
    spec["atom_map"] = atom_map
    for label, atom in atom_map.items():
        value = float(state.get(atom["dim"], 0.0))
        threshold = float(atom.get("threshold", 0.0))
        op = atom.get("op", ">")
        if (op == ">" and value > threshold) or (op == "<" and value < threshold):
            active.add(label)
        else:
            active.add(f"not_{label}")
    return frozenset(active)


def collect_atom_map(formula: dict[str, Any], atoms: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    if atoms is None:
        atoms = {}
    ftype = formula["type"]
    if ftype == "atom":
        atoms[atom_label(formula)] = formula
        return atoms
    if ftype in {"not", "next"}:
        return collect_atom_map(formula["child"], atoms)
    if ftype in {"and", "or", "implies", "until"}:
        collect_atom_map(formula["left"], atoms)
        collect_atom_map(formula["right"], atoms)
        return atoms
    if ftype in {"always", "eventually"}:
        return collect_atom_map(formula["child"], atoms)
    return atoms


def atom_label(node: dict[str, Any]) -> str:
    return str(node["dim"])


def _remaining_state_name(prefix: str, remaining: tuple[str, ...]) -> str:
    if not remaining:
        return f"{prefix}_done"
    return f"{prefix}:" + ",".join(remaining)


def _powerset_strings(items: list[str]) -> list[tuple[str, ...]]:
    ordered = sorted(dict.fromkeys(items))
    result = [()]
    for item in ordered:
        result += [tuple(sorted(existing + (item,))) for existing in result]
    return sorted(set(result), key=lambda tup: (len(tup), tup))


def _all_label_sets(items: list[str]) -> list[frozenset[str]]:
    ordered = sorted(dict.fromkeys(items))
    sets = [frozenset()]
    for item in ordered:
        sets += [frozenset(set(existing) | {item}) for existing in sets]
    return sorted(set(sets), key=lambda s: (len(s), tuple(sorted(s))))


def _build_spot_parity_automaton(spec: dict[str, Any]) -> ParityAutomaton | None:
    if spot is None:
        return None
    analysis = spec.get("analysis") or analyze_spec_structure(spec)
    if analysis["verification_mode"] != "infinite_parity":
        return None
    try:
        ltl_str, ap_order = _formula_to_spot_ltl(spec["formula"])
        automaton = spot.formula(ltl_str).translate(
            "parity",
            "Deterministic",
            "Colored",
            "Complete",
        )
        hoa = automaton.to_str("hoa")
        dpa = _parse_spot_hoa(hoa, ap_order)
        dpa.backend = "spot"
        dpa.exact = True
        return dpa
    except Exception:
        return None


def _formula_to_spot_ltl(formula: dict[str, Any]) -> tuple[str, list[str]]:
    atom_names: list[str] = []

    def walk(node: dict[str, Any]) -> str:
        ftype = node["type"]
        if ftype == "atom":
            label = atom_label(node)
            if label not in atom_names:
                atom_names.append(label)
            return label
        if ftype == "not":
            return f"!({walk(node['child'])})"
        if ftype == "and":
            return f"({walk(node['left'])} & {walk(node['right'])})"
        if ftype == "or":
            return f"({walk(node['left'])} | {walk(node['right'])})"
        if ftype == "implies":
            return f"({walk(node['left'])} -> {walk(node['right'])})"
        if ftype == "next":
            return f"X({walk(node['child'])})"
        if ftype == "always":
            if int(node["b"]) < 10_000:
                raise ValueError("Spot backend only supports unbounded temporal operators.")
            return f"G({walk(node['child'])})"
        if ftype == "eventually":
            if int(node["b"]) < 10_000:
                raise ValueError("Spot backend only supports unbounded temporal operators.")
            return f"F({walk(node['child'])})"
        if ftype == "until":
            if int(node["b"]) < 10_000:
                raise ValueError("Spot backend only supports unbounded temporal operators.")
            return f"({walk(node['left'])} U {walk(node['right'])})"
        raise ValueError(f"Unsupported formula node for Spot serialization: {ftype}")

    return walk(formula), atom_names


def _parse_spot_hoa(hoa: str, ap_order: list[str]) -> ParityAutomaton:
    states: list[str] = []
    priority: dict[str, int] = {}
    edge_guards: dict[str, list[tuple[str, str, int]]] = {}
    initial = "0"
    current_state: str | None = None

    for line in hoa.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Start:"):
            initial = line.split(":", 1)[1].strip()
            continue
        if line.startswith("State:"):
            current_state = line.split(":", 1)[1].strip().split()[0]
            if current_state not in states:
                states.append(current_state)
            priority.setdefault(current_state, 0)
            edge_guards.setdefault(current_state, [])
            continue
        if not line.startswith("[") or current_state is None:
            continue
        match = re.match(r"\[(.*?)\]\s+([^\s]+)(?:\s+\{([^}]*)\})?", line)
        if not match:
            continue
        guard_expr = match.group(1).strip()
        dst = match.group(2).strip()
        acc_sets = (match.group(3) or "").strip()
        color = 0
        if acc_sets:
            nums = [int(x) for x in acc_sets.split() if x.strip()]
            if nums:
                color = nums[0]
        edge_guards.setdefault(current_state, []).append((guard_expr, dst, color))
        if dst not in states:
            states.append(dst)
            priority.setdefault(dst, 0)
            edge_guards.setdefault(dst, [])

    return ParityAutomaton(
        states=states,
        initial=initial,
        priority=priority,
        transition={},
        state_meta={state: {} for state in states},
        backend="spot",
        exact=True,
        ap_order=ap_order,
        edge_guards=edge_guards,
    )


def _evaluate_hoa_label(expr: str, ap_order: list[str], active_aps: frozenset[str]) -> bool:
    tokens = re.findall(r"!|\(|\)|\&|\||\d+|t|f", expr.replace(" ", ""))
    pos = 0

    def parse_or() -> bool:
        nonlocal pos
        value = parse_and()
        while pos < len(tokens) and tokens[pos] == "|":
            pos += 1
            value = value or parse_and()
        return value

    def parse_and() -> bool:
        nonlocal pos
        value = parse_unary()
        while pos < len(tokens) and tokens[pos] == "&":
            pos += 1
            value = value and parse_unary()
        return value

    def parse_unary() -> bool:
        nonlocal pos
        tok = tokens[pos]
        if tok == "!":
            pos += 1
            return not parse_unary()
        if tok == "(":
            pos += 1
            value = parse_or()
            pos += 1
            return value
        pos += 1
        if tok == "t":
            return True
        if tok == "f":
            return False
        idx = int(tok)
        return idx < len(ap_order) and ap_order[idx] in active_aps

    return parse_or() if tokens else False
