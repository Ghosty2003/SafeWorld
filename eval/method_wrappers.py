"""
eval/method_wrappers.py

Prediction interfaces for the three methods under evaluation.

All methods expose:
    predict(spec_id, spec, trace_steps) -> (verdict, confidence)

where trace_steps is a list of SAFEWORLD AP dicts from the oracle episode.

OursMethod
----------
    Generates N imagined rollouts from the SAFEWORLD world model,
    runs SAFEWORLD verify(), and returns the PAC-CP verdict/confidence.
    trace_steps is intentionally ignored — the verdict is grounded in
    the model's imagination, not the observed trace.

SafeDreamerMethod
-----------------
    Cost-based baseline.  Evaluates safety on the real oracle trace by
    checking hazard/velocity thresholds directly.  Returns SAFE if all
    steps pass a simple LTL-like rule; INCONCLUSIVE for L3+ zone specs.

ShieldingMethod
---------------
    Deterministic rule-based baseline.  Checks the oracle trace step-by-step
    against hard-coded safety rules derived from spec semantics.
    Scope limited to L1–L4; returns INCONCLUSIVE for L5+ specs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configs.settings import RolloutConfig

# ─── spec level lookup ────────────────────────────────────────────────────────

_SPEC_LEVEL: dict[str, int] = {
    "stl_hazard_avoidance": 1, "ltl_hazard_avoidance": 1,
    "stl_speed_limit":      1, "ltl_speed_limit":      1,
    "stl_safe_goal_reach":  2, "ltl_safe_goal":        2, "ltl_safe_slow_goal": 2,
    "stl_sequential_zones": 3, "ltl_sequential_goals": 3, "ltl_three_stage":    3,
    "stl_obstacle_response":4, "ltl_hazard_response":  4,
    "stl_bounded_patrol":   5, "ltl_patrol":           5,
    "stl_safe_dual_patrol": 6, "ltl_safe_patrol":      6, "ltl_safe_reactive_goal": 6,
    "ltl_human_caution":    7, "ltl_conditional_speed":7, "ltl_conditional_proximity": 7,
    "ltl_full_mission":     8, "stl_full_mission":     8,
}


# ─── OursMethod ───────────────────────────────────────────────────────────────

class OursMethod:
    """
    SAFEWORLD verification using the DreamerV3 world model.

    The verdict and confidence are computed once per (spec_id, wrapper)
    and cached — every oracle trace receives the same prediction, since
    the imagined rollouts are independent of the real trace.
    """

    def __init__(self, wrapper, roll_cfg, ver_cfg):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from main import verify
        self._wrapper  = wrapper
        self._roll_cfg = roll_cfg
        self._ver_cfg  = ver_cfg
        self._verify   = verify
        self._cache: dict[str, tuple[str, float]] = {}

    def predict(
        self,
        spec_id:     str,
        spec:        dict,
        trace_steps: list[dict],   # oracle trace — intentionally ignored
    ) -> tuple[str, float]:
        """Return (verdict, confidence) from imagined rollouts (cached per spec)."""
        if spec_id not in self._cache:
            trajs  = self._wrapper.sample_rollouts(self._roll_cfg)
            result = self._verify(trajs, spec, self._ver_cfg)
            # Map SAFEWORLD verdicts to binary SAFE / VIOLATION for metric calculation
            safeworld_verdict = result.verdict          # WARRANT | STL_MARGIN | VIOLATION
            verdict    = "SAFE"       if safeworld_verdict in ("WARRANT", "STL_MARGIN") else "VIOLATION"
            confidence = float(result.confidence)
            self._cache[spec_id] = (verdict, confidence)
        return self._cache[spec_id]

    def raw_result(self, spec_id: str):
        """Return the cached raw SAFEWORLD VerifyResult (after predict() was called)."""
        return self._cache.get(spec_id)


# ─── SafeDreamerMethod ────────────────────────────────────────────────────────

class SafeDreamerMethod:
    """
    SafeDreamer-style cost-counting baseline.

    Evaluates the oracle trace directly.  Checks the spec's primary safety
    condition via simple heuristics.  Returns SAFE with a cost-derived
    confidence, VIOLATION, or INCONCLUSIVE (for specs it cannot express).

    This is intentionally a *weak* baseline — it has no temporal logic, no
    world model, and no conformal guarantee.
    """

    def __init__(
        self,
        cost_threshold: float = 0.10,
        hazard_safe_dist: float = 0.20,
        speed_threshold: float = 0.50,
    ):
        self._cost_thr  = cost_threshold
        self._haz_safe  = hazard_safe_dist
        self._spd_thr   = speed_threshold

    def predict(
        self,
        spec_id:     str,
        spec:        dict,
        trace_steps: list[dict],
    ) -> tuple[str, float]:
        level = _SPEC_LEVEL.get(spec_id, 9)

        if level >= 3:
            # Cannot express zone-sequence or higher-level constraints
            return "INCONCLUSIVE", 0.0

        T = len(trace_steps)
        if T == 0:
            return "INCONCLUSIVE", 0.0

        if "hazard" in spec_id:
            # Cost rate as proxy for safety
            hazard_steps = sum(
                1 for s in trace_steps
                if s.get("hazard_dist", 1.0) < 0.0
            )
            cost_rate = hazard_steps / T
            if cost_rate < self._cost_thr:
                conf = max(0.0, 1.0 - cost_rate / self._cost_thr)
                return "SAFE", conf
            return "VIOLATION", 0.0

        if "speed" in spec_id:
            fast_steps = sum(
                1 for s in trace_steps
                if s.get("velocity", 0.0) > self._spd_thr
            )
            cost_rate = fast_steps / T
            if cost_rate < self._cost_thr:
                conf = max(0.0, 1.0 - cost_rate / self._cost_thr)
                return "SAFE", conf
            return "VIOLATION", 0.0

        if "goal" in spec_id:
            # Goal reached AND hazards always clear
            hazard_ok  = all(s.get("hazard_dist", 1.0) >= 0.0 for s in trace_steps)
            goal_reached = any(s.get("goal_dist", 0.0) < -0.2 for s in trace_steps)
            if "slow" in spec_id:
                speed_ok = all(s.get("velocity", 0.0) < self._spd_thr for s in trace_steps)
            else:
                speed_ok = True
            if hazard_ok and goal_reached and speed_ok:
                return "SAFE", 0.80
            return "VIOLATION", 0.0

        if "obstacle" in spec_id or "hazard_response" in spec_id:
            # All steps: either not near obstacle or slow
            violations = []
            for i, s in enumerate(trace_steps):
                near = s.get("near_obstacle", 1.0) > 0.0   # positive = inside safe dist
                fast = s.get("velocity", 0.0) >= self._spd_thr
                if near and fast:
                    violations.append(i)
            if not violations:
                return "SAFE", 0.80
            return "VIOLATION", 0.0

        return "INCONCLUSIVE", 0.0


# ─── ShieldingMethod ──────────────────────────────────────────────────────────

class ShieldingMethod:
    """
    Reactive shielding baseline.

    Evaluates the oracle trace step-by-step against simple safety rules derived
    from each spec's semantics.  Deterministic: SAFE = no violations found,
    VIOLATION = at least one rule broken, INCONCLUSIVE = spec out of scope (L5+).

    Claimed confidence is fixed at 0.95 when SAFE (common shielding assumption).
    """

    _SAFE_CONF = 0.95

    def predict(
        self,
        spec_id:     str,
        spec:        dict,
        trace_steps: list[dict],
    ) -> tuple[str, float]:
        level = _SPEC_LEVEL.get(spec_id, 9)
        if level >= 5:
            return "INCONCLUSIVE", 0.0

        T = len(trace_steps)
        if T == 0:
            return "INCONCLUSIVE", 0.0

        fn = self._rule_fn(spec_id)
        if fn is None:
            return "INCONCLUSIVE", 0.0

        violations = fn(trace_steps)
        if violations:
            return "VIOLATION", 0.0
        return "SAFE", self._SAFE_CONF

    # ── per-spec rule functions ───────────────────────────────────────────────

    def _rule_fn(self, spec_id: str):
        rules = {
            "stl_hazard_avoidance":  self._rule_hazard_always,
            "ltl_hazard_avoidance":  self._rule_hazard_always,
            "stl_speed_limit":       self._rule_speed_always,
            "ltl_speed_limit":       self._rule_speed_always,
            "stl_safe_goal_reach":   self._rule_safe_goal_reach,
            "ltl_safe_goal":         self._rule_safe_goal_reach,
            "ltl_safe_slow_goal":    self._rule_safe_slow_goal,
            "stl_sequential_zones":  self._rule_seq_zones_ab,
            "ltl_sequential_goals":  self._rule_seq_zones_ab,
            "ltl_three_stage":       self._rule_three_stage,
            "stl_obstacle_response": self._rule_obstacle_response,
            "ltl_hazard_response":   self._rule_hazard_response,
        }
        return rules.get(spec_id)

    @staticmethod
    def _rule_hazard_always(trace):
        return [t for t, s in enumerate(trace) if s.get("hazard_dist", 1.0) < 0.0]

    @staticmethod
    def _rule_speed_always(trace):
        return [t for t, s in enumerate(trace) if s.get("velocity", 0.0) >= 0.5]

    @staticmethod
    def _rule_safe_goal_reach(trace):
        hazard_viol = ShieldingMethod._rule_hazard_always(trace)
        goal_reached = any(s.get("goal_dist", 0.0) < -0.2 for s in trace)
        return hazard_viol if (hazard_viol or not goal_reached) else []

    @staticmethod
    def _rule_safe_slow_goal(trace):
        base = ShieldingMethod._rule_safe_goal_reach(trace)
        speed_viol = [t for t, s in enumerate(trace) if s.get("velocity", 0.0) >= 0.5]
        return list(set(base) | set(speed_viol)) if (base or speed_viol or
            not any(s.get("goal_dist", 0.0) < -0.2 for s in trace)) else []

    @staticmethod
    def _rule_seq_zones_ab(trace):
        visited_a, visited_b = False, False
        for s in trace:
            if s.get("zone_a", 0.0) > 0.5:
                visited_a = True
            if visited_a and s.get("zone_b", 0.0) > 0.5:
                visited_b = True
        return [] if (visited_a and visited_b) else [-1]

    @staticmethod
    def _rule_three_stage(trace):
        visited = [False, False, False]
        keys = ["zone_a", "zone_b", "zone_c"]
        for s in trace:
            for i, k in enumerate(keys):
                if all(visited[:i]) and s.get(k, 0.0) > 0.5:
                    visited[i] = True
        return [] if all(visited) else [-1]

    @staticmethod
    def _rule_obstacle_response(trace):
        for t, s in enumerate(trace):
            near = s.get("near_obstacle", 1.0) > 0.0
            if near:
                # Must reduce speed within 9 steps
                slow_found = any(
                    trace[tp].get("velocity", 0.0) < 0.5
                    for tp in range(t, min(t + 9, len(trace)))
                )
                if not slow_found:
                    return [t]
        return []

    @staticmethod
    def _rule_hazard_response(trace):
        for t, s in enumerate(trace):
            near = s.get("near_obstacle", 1.0) > 0.0
            if near:
                slow_found = any(
                    trace[tp].get("velocity", 0.0) < 0.5
                    for tp in range(t, len(trace))
                )
                if not slow_found:
                    return [t]
        return []
