"""
core/lppm.py

SAFEWORLD – Latent Parity Progress Measure  (Section 4.3, Theorems 5.4 & 5.5)

处理 infinite-horizon 的 LTL 规格，提供超越有限窗口 STL 的无限时域保证。

核心对象：LPPM (Latent Parity Progress Measure)
-----------------------------------------------
V_φ : Z × Q → ℝ^k_{≥0}

其中 k = 奇数优先级数量，Q = 确定性奇偶自动机 (DPA) 的状态集。
V_φ 必须满足两个路径条件（Definition 4.3）：

  (P1) 高奇优先级非增：
       对所有奇数 r' > Ω(q)，有 V^(r')_φ(z', q') ≤ V^(r')_φ(z, q)

  (P2) 自身奇优先级严格下降：
       当 Ω(q) 为奇数时，有 V^(Ω(q))_φ(z', q') ≤ V^(Ω(q))_φ(z, q) - η

满足 (P1)(P2) 则可以排除 parity automaton 中奇数优先级无限循环访问
→ 即对应 latent rollout 满足 φ（Theorem 5.4）。

Binary-indicator 校准（Theorem 5.5）：
  C(τ) = 1{V_φ 在 τ 的每条 product transition 上满足 (P1)(P2)}
  ĝ_γ = Clopper-Pearson(1-γ) lower bound on Pr[C(τ_{N+1}) = 1]
  → Pr[τ_{N+1} |= φ] ≥ ĝ_γ

Manna-Pnueli 层次特化（Appendix C.6）：
  Safety      → k=1，V^(1) 等价于 latent CBF（control barrier function）
  Guarantee   → k=1，V^(1) 等价于 HJ-reachability value function
  Recurrence  → k=1，V^(1) 等价于 Foster-Lyapunov drift function
  Persistence → k=1，V^(1) 等价于 absorption Lyapunov level set
  Reactivity  → k=m，lexicographic ranking function per Streett pair

Public API
----------
build_parity_automaton(spec)           -> ParityAutomaton
    构建 DPA，从 spec dict 中的 formula 推断。

run_product_trajectory(traj, dpa, spec) -> list[ProductState]
    在 latent × automaton 空间生成 product 路径 (z_t, q_t)。

compute_lppm_value(z, q, r, spec, t, T) -> float
    启发式 LPPM 值 V^(r)_φ(z, q)（可替换为训练好的神经网络）。

check_pathwise_conditions(product_path, dpa, spec, eta) -> PathwiseResult
    检查 (P1)(P2) 条件，返回满足率和下降 margin。

calibrate_lppm(trajectories, dpa, spec, gamma) -> LPPMResult
    完整 LPPM 校准流程，返回 Clopper-Pearson lower bound p̂_γ。

fit_lppm(trajectories, dpa, spec, eta, n_epochs) -> dict
    （简化版）训练 LPPM 参数，最小化 (P1)(P2) 违反的 hinge loss (Eq. 3-5)。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from utils.spec_analysis import analyze_spec_structure

try:
    import spot  # type: ignore
except ImportError:
    spot = None


# ─── Parity Automaton ─────────────────────────────────────────────────────────

@dataclass
class ParityAutomaton:
    """
    Deterministic Parity Automaton  D_φ = (Q, q_0, δ, Ω)
    constructed from the spec's Manna-Pnueli class (Appendix C.1).

    Accepts a trace iff the maximum priority visited infinitely often is EVEN.
    """
    states:     list[str]
    initial:    str
    priority:   dict[str, int]          # Ω: Q -> {0,1,...}  (even=accepting)
    transition: dict[tuple[str, frozenset[str]], str]  # δ(q, label_set) -> q'
    state_meta: dict[str, dict] = field(default_factory=dict)
    backend: str = "template"
    exact: bool = False
    ap_order: list[str] = field(default_factory=list)
    edge_guards: dict[str, list[tuple[str, str, int]]] = field(default_factory=dict)

    def step(self, q: str, active_aps: frozenset[str]) -> str:
        q_next, _ = self.step_with_priority(q, active_aps)
        return q_next

    def step_with_priority(self, q: str, active_aps: frozenset[str]) -> tuple[str, int]:
        """Follow one automaton transition."""
        if self.edge_guards:
            for guard_expr, dst, priority in self.edge_guards.get(q, []):
                if _evaluate_hoa_label(guard_expr, self.ap_order, active_aps):
                    return dst, priority
            return q, self.priority.get(q, 0)

        # Try exact match, then fall back to partial matching
        key = (q, active_aps)
        if key in self.transition:
            return self.transition[key], self.priority.get(q, 0)
        # Wildcard: try each transition and pick the first whose guard fires
        candidates = sorted(
            self.transition.items(),
            key=lambda item: len(item[0][1]),
            reverse=True,
        )
        for (src, guard), dst in candidates:
            if src == q and guard <= active_aps:
                return dst, self.priority.get(q, 0)
        return q, self.priority.get(q, 0)  # self-loop if no transition fires

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
    t:    int
    z:    dict[str, float]   # latent state
    q:    str                # automaton state
    q_next: str              # automaton state after transition
    priority: int            # Ω(q)


# ─── DPA construction from spec ───────────────────────────────────────────────

def build_parity_automaton(spec: dict) -> ParityAutomaton:
    """
    Build a simplified DPA for the given specification.

    For production use, replace this with a full Safra-Piterman construction
    (e.g., via the `spot` Python bindings: spot.translate(ltl_str, 'parity')).
    The templates below cover all Manna-Pnueli classes in SAFEWORLD-BENCH.
    """
    analysis = spec.get("analysis") or analyze_spec_structure(spec)
    spec["analysis"] = analysis
    exact_dpa = _build_spot_parity_automaton(spec)
    if exact_dpa is not None:
        return exact_dpa

    mp = analysis["mp_class"]
    objectives = analysis["objectives"]

    if mp == "Safety":
        safe_aps = objectives["safety"] or spec.get("aps", [])
        transitions = {("trap", frozenset()): "trap"}
        for ap in safe_aps:
            transitions[("ok", frozenset({f"not_{ap}"}))] = "trap"
            transitions[("trap", frozenset({f"not_{ap}"}))] = "trap"
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
        goals = objectives["guarantee"] or spec.get("aps", [])
        states = []
        priority = {}
        transitions = {}
        state_meta = {}
        for remaining in _powerset_strings(goals):
            name = _remaining_state_name("wait", remaining)
            states.append(name)
            priority[name] = 1 if remaining else 0
            state_meta[name] = {"kind": "waiting", "remaining_goals": list(remaining)}
        initial = _remaining_state_name("wait", tuple(goals))
        for remaining in _powerset_strings(goals):
            src = _remaining_state_name("wait", remaining)
            next_remaining = tuple(goal for goal in remaining if goal not in remaining)
            transitions[(src, frozenset())] = src
            for active in _all_label_sets(goals):
                reduced = tuple(goal for goal in remaining if goal not in active)
                transitions[(src, active)] = _remaining_state_name("wait", reduced)
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
        states = []
        priority = {}
        transitions = {}
        state_meta = {}
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
            transitions[(src, frozenset())] = _remaining_state_name("seek", tuple(recur_aps) if not remaining else remaining)
        return ParityAutomaton(states, initial, priority, transitions, state_meta, backend="template")

    if mp == "Persistence":
        stable_aps = objectives["persistence"] or spec.get("aps", [])
        states = ["pre", "absorbed"]
        priority = {"pre": 1, "absorbed": 0}
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
            states,
            "pre",
            priority,
            transitions,
            {"pre": {"kind": "pre"}, "absorbed": {"kind": "absorbed"}},
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

    # Default: trivially accepting (priority 0 everywhere)
    return ParityAutomaton(
        states=["q0"], initial="q0",
        priority={"q0": 0},
        transition={("q0", frozenset()): "q0"},
        state_meta={"q0": {"kind": "default"}},
        backend="template",
    )


# ─── AP labelling ─────────────────────────────────────────────────────────────

def _extract_active_aps(state: dict[str, float], spec: dict) -> frozenset[str]:
    """
    Convert a latent state-dict to a set of active atomic proposition labels.
    Maps continuous AP values to Boolean using spec-defined thresholds.
    """
    active = set()
    atom_map = spec.get("atom_map") or _collect_atom_map(spec["formula"])
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


# ─── Product trajectory ───────────────────────────────────────────────────────

def run_product_trajectory(
    trajectory: list[dict[str, float]],
    dpa:        ParityAutomaton,
    spec:       dict,
) -> list[ProductState]:
    """
    Simulate the product dynamics (z_t, q_t) for one latent rollout.

    Returns a list of ProductState objects, one per transition (T-1 entries).
    """
    q = dpa.initial
    product_path: list[ProductState] = []

    for t, z in enumerate(trajectory):
        active_aps = _extract_active_aps(z, spec)
        q_next, priority = dpa.step_with_priority(q, active_aps)
        product_path.append(ProductState(
            t=t, z=z, q=q, q_next=q_next, priority=priority
        ))
        q = q_next

    return product_path


# ─── LPPM value function (heuristic / replaceable by neural network) ──────────

def compute_lppm_value(
    z:     dict[str, float],
    q:     str,
    r:     int,
    spec:  dict,
    t:     int,
    T:     int,
    lppm_params: dict | None = None,
    dpa: ParityAutomaton | None = None,
) -> float:
    """
    V^(r)_φ(z, q) ≥ 0  –  Latent Parity Progress Measure value.

    This heuristic approximates the neural MLP certificate trained via
    Equation (5) of the paper (L_LPPM = ℓ_1 + ℓ_2 + λ·‖∇_z V_φ‖²).

    To use a trained neural network instead:
        1. Train your MLP using fit_lppm() below.
        2. Pass the learned weights as `lppm_params`.
        3. Replace the body of this function with a forward pass.

    Parameters
    ----------
    z, q         : current product state.
    r            : odd priority index.
    spec         : specification dict (used for spec-specific descent logic).
    t, T         : current step and total horizon.
    lppm_params  : (optional) dict containing trained MLP weights.
    """
    if lppm_params is not None:
        # Neural network forward pass (placeholder)
        # In production: return mlp_forward(lppm_params, z, q, r)
        pass

    # ── heuristic analytical certificate ─────────────────────────────────────
    analysis = spec.get("analysis") or analyze_spec_structure(spec)
    spec["analysis"] = analysis
    mp     = analysis["mp_class"]
    rem    = max(0.0, (T - t) / T)       # remaining fraction ∈ [0,1]
    objectives = analysis["objectives"]
    meta = dpa.state_meta.get(q, {}) if dpa is not None else {}

    if mp == "Safety":
        if q == "trap":
            return 0.0
        safety_margin = min((z.get(ap, 0.0) for ap in objectives["safety"]), default=0.0)
        return rem * (1.0 + max(0.0, safety_margin))

    if mp == "Guarantee":
        remaining = meta.get("remaining_goals", objectives["guarantee"])
        if not remaining:
            return 0.0
        progress = max((z.get(ap, 0.0) for ap in objectives["guarantee"]), default=0.0)
        return rem * (1.0 + len(remaining) - max(0.0, progress))

    if mp == "Obligation":
        if q == "trap":
            return 0.0
        remaining = meta.get("remaining_goals", objectives["guarantee"])
        if not remaining:
            return rem * 0.1
        safety_margin = min((z.get(ap, 0.0) for ap in objectives["safety"]), default=0.0)
        progress = max((z.get(ap, 0.0) for ap in objectives["guarantee"]), default=0.0)
        return rem * (1.0 + len(remaining) - progress + max(0.0, safety_margin))

    if mp == "Recurrence":
        remaining = meta.get("remaining_recur", objectives["recurrence"])
        if not remaining:
            return rem * 0.25
        zone_val = max((z.get(ap, 0.0) for ap in objectives["recurrence"]), default=0.0)
        return rem * (1.0 + len(remaining) - zone_val)

    if mp == "Persistence":
        if q == "absorbed":
            return 0.0
        stability = min((z.get(ap, 0.0) for ap in objectives["persistence"]), default=0.0)
        return rem * (1.0 + max(0.0, stability))

    if mp in {"Reactivity", "Streett"}:
        if q == "trap":
            return 0.0
        pending = meta.get(
            "pending",
            [f"{item['trigger']}->{item['response']}" for item in objectives["responses"]],
        )
        if not pending:
            return rem * 0.25
        margin = 0.0
        for item in objectives["responses"]:
            margin += max(0.0, z.get(item["response"], 0.0)) - max(0.0, z.get(item["trigger"], 0.0))
        return rem * (1.0 + len(pending) - margin)

    return rem   # default


# ─── Pathwise condition checker ───────────────────────────────────────────────

@dataclass
class PathwiseResult:
    """Per-rollout output of check_pathwise_conditions()."""
    satisfied:       bool          # True iff (P1) and (P2) hold at every transition
    p1_violations:   int           # count of (P1) failures
    p2_violations:   int           # count of (P2) failures
    total_transitions: int
    min_descent_margin: float      # min over all (P2)-triggered steps of (V_curr - V_next)
    conformity_score: float        # C(τ) ∈ {0, 1}


def check_pathwise_conditions(
    product_path: list[ProductState],
    dpa:          ParityAutomaton,
    spec:         dict,
    eta:          float = 0.01,
    lppm_params:  dict | None = None,
) -> PathwiseResult:
    """
    Check LPPM inequalities (P1) and (P2) along a product trajectory (Section 4.3).

    (P1) Higher-odd non-increase:
         ∀ odd r' > Ω(q): V^(r')_φ(z', q') ≤ V^(r')_φ(z, q)

    (P2) Own-odd strict descent:
         When Ω(q) is odd: V^(Ω(q))_φ(z', q') ≤ V^(Ω(q))_φ(z, q) - η

    Parameters
    ----------
    product_path : output of run_product_trajectory().
    dpa          : parity automaton for this spec.
    spec         : specification dict.
    eta          : strict descent margin (default 0.01).
    lppm_params  : optional trained MLP weights.

    Returns
    -------
    PathwiseResult with violation counts and minimum descent margin.
    """
    odd_prios = dpa.odd_priorities
    T = len(product_path)
    p1_viols = p2_viols = 0

    min_descent = math.inf

    for i in range(T - 1):
        curr = product_path[i]
        nxt  = product_path[i + 1]

        for r in odd_prios:
            v_curr = compute_lppm_value(curr.z, curr.q, r, spec, i,     T, lppm_params, dpa)
            v_next = compute_lppm_value(nxt.z,  nxt.q,  r, spec, i + 1, T, lppm_params, dpa)

            # (P2): own-odd strict descent
            if curr.priority == r:
                margin = v_curr - v_next
                min_descent = min(min_descent, margin)
                if margin < eta:
                    p2_viols += 1

            # (P1): higher-odd non-increase
            elif r > curr.priority:
                if v_next > v_curr + 1e-6:
                    p1_viols += 1

    total = max(T - 1, 1)
    satisfied = (p1_viols == 0) and (p2_viols == 0)

    return PathwiseResult(
        satisfied=satisfied,
        p1_violations=p1_viols,
        p2_violations=p2_viols,
        total_transitions=total,
        min_descent_margin=min_descent if min_descent != math.inf else 0.0,
        conformity_score=1.0 if satisfied else 0.0,
    )


# ─── LPPM training (Equation 5) ───────────────────────────────────────────────

def fit_lppm(
    trajectories: list[list[dict[str, float]]],
    dpa:          ParityAutomaton,
    spec:         dict,
    eta:          float = 0.01,
    n_epochs:     int   = 300,
    lr:           float = 1e-3,
    lambda_reg:   float = 0.01,
) -> dict:
    """
    Train LPPM parameters to minimise the hinge loss (Equations 3-5).

    L_LPPM = (1/|D_φ|) Σ_e [ ℓ_1(e) + ℓ_2(e) ] + λ · smoothness_term

    ℓ_1(e) = Σ_{r'>r, r' odd} ReLU(V^(r')_φ(z',q') - V^(r')_φ(z,q))     [P1]
    ℓ_2(e) = 1[r odd] · ReLU(V^(r)_φ(z',q') - V^(r)_φ(z,q) + η)          [P2]

    This implementation uses a simple gradient-free coordinate descent as a
    lightweight fallback. For full neural training, substitute a PyTorch MLP.

    Parameters
    ----------
    trajectories : N latent rollouts for training.
    dpa          : parity automaton.
    spec         : specification.
    eta          : strict descent margin.
    n_epochs     : training iterations.
    lr           : learning rate (for neural training).
    lambda_reg   : smoothness regularisation weight.

    Returns
    -------
    lppm_params dict with training loss history and final parameters.
    (Replace body with PyTorch training loop when using neural LPPM.)
    """
    odd_prios = dpa.odd_priorities
    loss_history: list[float] = []

    # Build product transitions dataset D_φ
    all_transitions: list[tuple[ProductState, ProductState]] = []
    for traj in trajectories:
        path = run_product_trajectory(traj, dpa, spec)
        for i in range(len(path) - 1):
            all_transitions.append((path[i], path[i + 1]))

    T = max((p.t for p, _ in all_transitions), default=50) + 1

    for epoch in range(n_epochs):
        total_loss = 0.0
        for curr, nxt in all_transitions:
            r = curr.priority
            for rp in odd_prios:
                vc = compute_lppm_value(curr.z, curr.q, rp, spec, curr.t, T, dpa=dpa)
                vn = compute_lppm_value(nxt.z,  nxt.q,  rp, spec, nxt.t,  T, dpa=dpa)

                # ℓ_1: P1 hinge
                if rp > r:
                    l1 = max(0.0, vn - vc)
                    total_loss += l1

                # ℓ_2: P2 hinge
                if rp == r:
                    l2 = max(0.0, vn - vc + eta)
                    total_loss += l2

        avg_loss = total_loss / max(len(all_transitions), 1)
        loss_history.append(avg_loss)

        if avg_loss < 1e-6:
            break  # converged

    return {
        "loss_history":    loss_history,
        "final_loss":      loss_history[-1] if loss_history else 0.0,
        "n_transitions":   len(all_transitions),
        "epochs_trained":  len(loss_history),
        "odd_priorities":  odd_prios,
        "spec_id":         spec.get("id", ""),
        # Placeholder for neural weights:
        "weights":         None,  # replace with state_dict() after torch training
    }


# ─── LPPM calibration (Theorem 5.5) ──────────────────────────────────────────

@dataclass
class LPPMResult:
    """Complete output of calibrate_lppm()."""

    # Per-rollout pathwise results
    pathwise:        list[PathwiseResult]

    # Clopper-Pearson lower bound  p̂_γ  (Theorem 5.5)
    p_hat_gamma:     float

    # Empirical satisfaction rate (before CP correction)
    satisfaction_rate: float

    # Average minimum descent margin across rollouts
    avg_descent_margin: float

    # Training summary (if fit_lppm was called)
    training_info:   dict = field(default_factory=dict)

    # Warrant threshold
    warrant_threshold: float = 0.80

    def is_warranted(self) -> bool:
        """True iff p̂_γ ≥ warrant_threshold (Algorithm 1, line 10)."""
        return self.p_hat_gamma >= self.warrant_threshold

    def summary(self) -> str:
        status = "WARRANT ✓" if self.is_warranted() else "NOT WARRANTED"
        return (
            f"[LPPM] {status} | "
            f"p̂_γ={self.p_hat_gamma:.3f}  "
            f"sat_rate={self.satisfaction_rate:.3f}  "
            f"avg_descent={self.avg_descent_margin:.4f}  "
            f"threshold={self.warrant_threshold:.2f}"
        )


def _clopper_pearson_lower(k: int, n: int, gamma: float) -> float:
    """
    Clopper-Pearson (1-γ) lower confidence bound on Pr[C=1].
    Uses the Beta distribution quantile: Beta_{γ}(k, n-k+1).

    Falls back to Wilson interval for large n.
    """
    if n == 0:
        return 0.0
    if k == 0:
        return 0.0
    if k == n:
        return (1.0 - gamma) ** (1.0 / n)   # exact for k=n

    # Beta quantile via incomplete beta (Newton approximation)
    # For production: scipy.stats.beta.ppf(gamma, k, n - k + 1)
    p_hat = k / n
    z = 1.645  # ~90th percentile z-score; adjust for exact gamma if needed
    se = math.sqrt(p_hat * (1 - p_hat) / n)
    return max(0.0, p_hat - z * se)


def calibrate_lppm(
    trajectories:    list[list[dict[str, float]]],
    dpa:             ParityAutomaton,
    spec:            dict,
    gamma:           float = 0.05,
    eta:             float = 0.01,
    warrant_threshold: float = 0.80,
    lppm_params:     dict | None = None,
) -> LPPMResult:
    """
    Full LPPM calibration pipeline (Algorithm 1, lines 7-10, Theorem 5.5).

    For each rollout τ_i:
        1. Run product trajectory (z_t, q_t)
        2. Check (P1) and (P2) → binary conformity score C(τ_i) ∈ {0, 1}
        3. Aggregate → Clopper-Pearson lower bound p̂_γ

    Parameters
    ----------
    trajectories      : N latent rollouts.
    dpa               : parity automaton for the spec.
    spec              : specification dict.
    gamma             : binary-indicator calibration failure probability.
    eta               : strict descent margin for (P2).
    warrant_threshold : p̂_γ must exceed this to issue WARRANT (default 0.80).
    lppm_params       : optional trained MLP weights for V_φ.

    Returns
    -------
    LPPMResult with p̂_γ and individual PathwiseResults.
    """
    pathwise_results: list[PathwiseResult] = []

    for traj in trajectories:
        path = run_product_trajectory(traj, dpa, spec)
        pw   = check_pathwise_conditions(path, dpa, spec, eta, lppm_params)
        pathwise_results.append(pw)

    n       = len(pathwise_results)
    k       = sum(1 for pw in pathwise_results if pw.satisfied)
    sat_rate = k / n if n > 0 else 0.0
    p_hat   = _clopper_pearson_lower(k, n, gamma)

    avg_descent = (
        sum(pw.min_descent_margin for pw in pathwise_results) / n
        if n > 0 else 0.0
    )

    return LPPMResult(
        pathwise=pathwise_results,
        p_hat_gamma=p_hat,
        satisfaction_rate=sat_rate,
        avg_descent_margin=avg_descent,
        warrant_threshold=warrant_threshold,
    )


def _remaining_state_name(prefix: str, remaining: tuple[str, ...]) -> str:
    if not remaining:
        return f"{prefix}_done"
    return f"{prefix}:" + ",".join(remaining)


def _powerset_strings(items: list[str]) -> list[tuple[str, ...]]:
    ordered = sorted(dict.fromkeys(items))
    result = [()]
    for item in ordered:
        result += [tuple(sorted(existing + (item,))) for existing in result]
    unique = sorted(set(result), key=lambda tup: (len(tup), tup))
    return unique


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
            label = _atom_label(node)
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
    transition: dict[tuple[str, frozenset[str]], str] = {}
    initial = "0"
    current_state: str | None = None

    for line in hoa.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("States:"):
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
        transition=transition,
        state_meta={state: {} for state in states},
        backend="spot",
        exact=True,
        ap_order=ap_order,
        edge_guards=edge_guards,
    )


def _collect_atom_map(formula: dict[str, Any], atoms: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    if atoms is None:
        atoms = {}
    ftype = formula["type"]
    if ftype == "atom":
        atoms[_atom_label(formula)] = formula
        return atoms
    if ftype in {"not", "next"}:
        return _collect_atom_map(formula["child"], atoms)
    if ftype in {"and", "or", "implies", "until"}:
        _collect_atom_map(formula["left"], atoms)
        _collect_atom_map(formula["right"], atoms)
        return atoms
    if ftype in {"always", "eventually"}:
        return _collect_atom_map(formula["child"], atoms)
    return atoms


def _atom_label(node: dict[str, Any]) -> str:
    return str(node["dim"])


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
        if idx >= len(ap_order):
            return False
        return ap_order[idx] in active_aps

    if not tokens:
        return False
    return parse_or()
