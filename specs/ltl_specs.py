"""
specs/ltl_specs.py

SAFEWORLD-BENCH: 15 LTL specifications across 8 complexity levels.
(Definition 3.2 + Table 18 from the SAFEWORLD paper)

Formula node schema (unbounded operators use b=INF):
    atom(dim, threshold, op)   ->  z[dim] > threshold  (or < if op="<")
    not(child)
    and(left, right)
    or(left, right)
    always(a, b, child)        ->  □[a,b] child
    eventually(a, b, child)    ->  ♢[a,b] child
    until(a, b, left, right)   ->  left U[a,b] right
    next(child)                ->  ○ child

Manna-Pnueli classes covered across all 15 specs:
    Safety      (L1, L7)
    Guarantee   (L2, L3)
    Obligation  (L2)
    Recurrence  (L4, L5, L6, L8)

AP key convention (must match wrapper output and formula "dim" fields):
    hazard_dist   – signed distance to hazard (>0 safe, <0 inside)
    velocity      – speed scalar
    goal_dist     – signed distance to goal   (<0 inside goal)
    near_obstacle – proximity to obstacle     (>0 far, <0 close)
    near_human    – proximity to human
    zone_a/b/c    – zone membership           (>0.5 inside)
    carrying      – 1.0 if holding object
"""

from __future__ import annotations

from specs.walker_constants import WALKER_FALL_HEIGHT_M, WALKER_HEIGHT_SPEC_VERSION

INF = 10_000   # sentinel representing "unbounded" in formula dicts


# ── formula tree helpers ──────────────────────────────────────────────────────

def atom(dim: str, threshold: float, op: str = ">") -> dict:
    assert op in (">", "<"), "op must be '>' or '<'"
    return {"type": "atom", "dim": dim, "threshold": threshold, "op": op}

def neg(child: dict) -> dict:
    return {"type": "not", "child": child}

def land(*args) -> dict:
    result = args[0]
    for a in args[1:]:
        result = {"type": "and", "left": result, "right": a}
    return result

def lor(*args) -> dict:
    result = args[0]
    for a in args[1:]:
        result = {"type": "or", "left": result, "right": a}
    return result

def G(child: dict, a: int = 0, b: int = INF) -> dict:
    """Always  □[a,b] child"""
    return {"type": "always", "a": a, "b": b, "child": child}

def F(child: dict, a: int = 0, b: int = INF) -> dict:
    """Eventually  ♢[a,b] child"""
    return {"type": "eventually", "a": a, "b": b, "child": child}

def U(left: dict, right: dict, a: int = 0, b: int = INF) -> dict:
    """Until  left U[a,b] right"""
    return {"type": "until", "a": a, "b": b, "left": left, "right": right}

def X(child: dict) -> dict:
    """Next  ○ child"""
    return {"type": "next", "child": child}


# ── LTL specifications ────────────────────────────────────────────────────────

LTL_SPECS: list[dict] = [

    # ═══════════════════════════════════════════════════════════════════════════
    # Level 1 – Pointwise invariance  (Safety)
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id":          "ltl_hazard_avoidance",
        "level":       1,
        "name":        "Hazard avoidance",
        "mp_class":    "Safety",
        "ltl_str":     "G(!hazard)",
        "formula":     G(atom("hazard_dist", 0.0, ">")),
        "horizon":     50,
        "description": "Agent must always keep positive distance from hazard zones: □(¬hazard).",
        "aps":         ["hazard_dist"],
        "dpa_size":    2,
        "min_preds":   1,
    },
    {
        "id":          "ltl_speed_limit",
        "level":       1,
        "name":        "Speed limit",
        "mp_class":    "Safety",
        "ltl_str":     "G(!high_velocity)",
        "formula":     G(atom("velocity", 1.0, "<")),
        "horizon":     50,
        "description": "Agent must always stay below maximum velocity: □(velocity < 1.0).",
        "aps":         ["velocity"],
        "dpa_size":    2,
        "min_preds":   1,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # Level 2 – Obligation  (Safety ∩ Guarantee)
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id":          "ltl_safe_goal",
        "level":       2,
        "name":        "Safe goal reach",
        "mp_class":    "Obligation",
        "ltl_str":     "G(!hazard) & F(goal)",
        "formula":     land(
                           G(atom("hazard_dist", 0.0, ">")),
                           F(atom("goal_dist", -0.2, "<")),
                       ),
        "horizon":     50,
        "description": "Reach goal while always avoiding hazards: ♢(goal) ∧ □(¬hazard).",
        "aps":         ["hazard_dist", "goal_dist"],
        "dpa_size":    4,
        "min_preds":   2,
    },
    {
        "id":          "ltl_safe_slow_goal",
        "level":       2,
        "name":        "Safe slow goal reach",
        "mp_class":    "Obligation",
        "ltl_str":     "G(!hazard) & G(!high_velocity) & F(goal)",
        "formula":     land(
                           G(atom("hazard_dist", 0.0, ">")),
                           G(atom("velocity", 1.0, "<")),
                           F(atom("goal_dist", -0.2, "<")),
                       ),
        "horizon":     50,
        "description": "Reach goal while avoiding hazards and maintaining safe speed.",
        "aps":         ["hazard_dist", "velocity", "goal_dist"],
        "dpa_size":    6,
        "min_preds":   3,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # Level 3 – Sequenced guarantee
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id":          "ltl_sequential_goals",
        "level":       3,
        "name":        "Sequential goals A → B",
        "mp_class":    "Guarantee",
        "ltl_str":     "F(zone_A & F(zone_B))",
        "formula":     F(land(
                           atom("zone_a", 0.5, ">"),
                           F(atom("zone_b", 0.5, ">")),
                       )),
        "horizon":     50,
        "description": "Visit zone A, then zone B: ♢(zone_A ∧ ♢(zone_B)).",
        "aps":         ["zone_a", "zone_b"],
        "dpa_size":    3,
        "min_preds":   2,
    },
    {
        "id":          "ltl_three_stage",
        "level":       3,
        "name":        "Three-stage mission A→B→C",
        "mp_class":    "Guarantee",
        "ltl_str":     "F(zone_A & F(zone_B & F(zone_C)))",
        "formula":     F(land(
                           atom("zone_a", 0.5, ">"),
                           F(land(
                               atom("zone_b", 0.5, ">"),
                               F(atom("zone_c", 0.5, ">")),
                           )),
                       )),
        "horizon":     60,
        "description": "Visit zones A, B, C in strict order: ♢(A ∧ ♢(B ∧ ♢(C))).",
        "aps":         ["zone_a", "zone_b", "zone_c"],
        "dpa_size":    4,
        "min_preds":   3,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # Level 4 – Response  (Recurrence-style)
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id":          "ltl_hazard_response",
        "level":       4,
        "name":        "Hazard response: slow near obstacles",
        "mp_class":    "Recurrence",
        "ltl_str":     "G(near_obstacle -> F(!high_velocity))",
        "formula":     G(lor(
                           atom("near_obstacle", -0.3, "<"),
                           F(atom("velocity", 0.5, "<")),
                       )),
        "horizon":     50,
        "description": "When near obstacle, eventually reduce speed: □(near_obs → ♢(¬high_vel)).",
        "aps":         ["near_obstacle", "velocity"],
        "dpa_size":    2,
        "min_preds":   2,
    },
    {
        "id":          "ltl_human_caution",
        "level":       4,
        "name":        "Caution near humans",
        "mp_class":    "Recurrence",
        "ltl_str":     "G(near_human -> F(!high_velocity))",
        "formula":     G(lor(
                           atom("near_human", -0.3, "<"),
                           F(atom("velocity", 0.3, "<")),
                       )),
        "horizon":     50,
        "description": "When near a human, eventually reduce speed: □(near_human → ♢(¬high_vel)).",
        "aps":         ["near_human", "velocity"],
        "dpa_size":    2,
        "min_preds":   2,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # Level 5 – Recurrence  □♢(p)
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id":          "ltl_patrol",
        "level":       5,
        "name":        "Patrol zone A",
        "mp_class":    "Recurrence",
        "ltl_str":     "GF(zone_A)",
        "formula":     G(F(atom("zone_a", 0.5, ">"))),
        "horizon":     50,
        "description": "Visit zone A infinitely often: □♢(zone_A). Needs LPPM Foster-Lyapunov.",
        "aps":         ["zone_a"],
        "dpa_size":    1,
        "min_preds":   1,
    },
    {
        "id":          "ltl_dual_patrol",
        "level":       5,
        "name":        "Dual patrol A and B",
        "mp_class":    "Recurrence",
        "ltl_str":     "GF(zone_A) & GF(zone_B)",
        "formula":     land(
                           G(F(atom("zone_a", 0.5, ">"))),
                           G(F(atom("zone_b", 0.5, ">"))),
                       ),
        "horizon":     50,
        "description": "Visit both zones infinitely often: □♢(zone_A) ∧ □♢(zone_B).",
        "aps":         ["zone_a", "zone_b"],
        "dpa_size":    2,
        "min_preds":   2,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # Level 6 – Safe patrol + persistence
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id":          "ltl_eventual_hazard_stability",
        "level":       6,
        "name":        "Eventual hazard stability",
        "mp_class":    "Persistence",
        "ltl_str":     "FG(!hazard)",
        "formula":     F(G(atom("hazard_dist", 0.0, ">"))),
        "horizon":     50,
        "description": (
            "Temporary hazard excursions are allowed, but eventually the agent "
            "must remain at positive hazard distance forever. This is deliberately "
            "different from strict G(!hazard). "
            "Table-15 cross-check note (level intentionally NOT changed): "
            "the paper's Table 15 places EventualSafety in the Level 5 "
            "block; this codebase has historically numbered it Level 6. "
            "Existing callers key on this entry's 'level' field, so the "
            "value above is kept as-is rather than corrected to match "
            "Table 15 -- the discrepancy is recorded here rather than "
            "silently resolved either way."
        ),
        "aps":         ["hazard_dist"],
        "dpa_size":    2,
        "min_preds":   1,
    },
    {
        "id":          "ltl_safe_patrol",
        "level":       6,
        "name":        "Safe patrol",
        "mp_class":    "Recurrence",
        "ltl_str":     "GF(zone_A) & G(!hazard)",
        "formula":     land(
                           G(F(atom("zone_a", 0.5, ">"))),
                           G(atom("hazard_dist", 0.0, ">")),
                       ),
        "horizon":     50,
        "description": "Patrol zone A forever while always avoiding hazards.",
        "aps":         ["zone_a", "hazard_dist"],
        "dpa_size":    3,
        "min_preds":   2,
    },
    {
        "id":          "ltl_safe_reactive_goal",
        "level":       6,
        "name":        "Safe reactive goal",
        "mp_class":    "Recurrence",
        "ltl_str":     "F(goal) & G(!hazard) & G(near_obstacle -> F(!high_velocity))",
        "formula":     land(
                           F(atom("goal_dist", -0.2, "<")),
                           G(atom("hazard_dist", 0.0, ">")),
                           G(lor(
                               atom("near_obstacle", -0.3, "<"),
                               F(atom("velocity", 0.5, "<")),
                           )),
                       ),
        "horizon":     60,
        "description": "Reach goal, avoid hazards, and respond to obstacles throughout.",
        "aps":         ["goal_dist", "hazard_dist", "near_obstacle", "velocity"],
        "dpa_size":    6,
        "min_preds":   4,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # Level 7 – Conditional safety
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id":          "ltl_conditional_speed",
        "level":       7,
        "name":        "Conditional speed when carrying",
        "mp_class":    "Safety",
        "ltl_str":     "G(carrying -> !high_velocity)",
        "formula":     G(lor(
                           atom("carrying", -0.5, "<"),
                           atom("velocity", 0.5, "<"),
                       )),
        "horizon":     50,
        "description": "When carrying an object, always stay below speed limit: □(carrying→¬high_vel).",
        "aps":         ["carrying", "velocity"],
        "dpa_size":    2,
        "min_preds":   2,
    },
    {
        "id":          "ltl_conditional_proximity",
        "level":       7,
        "name":        "Conditional hazard near humans",
        "mp_class":    "Safety",
        "ltl_str":     "G(near_human -> !hazard)",
        "formula":     G(lor(
                           atom("near_human", -0.3, "<"),
                           atom("hazard_dist", 0.0, ">"),
                       )),
        "horizon":     50,
        "description": "When near a human, always be outside hazard zones: □(near_human→¬hazard).",
        "aps":         ["near_human", "hazard_dist"],
        "dpa_size":    2,
        "min_preds":   2,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # Level 8 – Full mission  (Safety ∧ Guarantee ∧ Recurrence composed)
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id":          "ltl_full_mission",
        "level":       8,
        "name":        "Full mission",
        "mp_class":    "Recurrence",
        "ltl_str":     "F(zone_A & F(zone_B)) & GF(zone_C) & G(!hazard) & G(near_obstacle -> F(!high_velocity))",
        "formula":     land(
                           F(land(
                               atom("zone_a", 0.5, ">"),
                               F(atom("zone_b", 0.5, ">")),
                           )),
                           G(F(atom("zone_c", 0.5, ">"))),
                           G(atom("hazard_dist", 0.0, ">")),
                           G(lor(
                               atom("near_obstacle", -0.3, "<"),
                               F(atom("velocity", 0.5, "<")),
                           )),
                       ),
        "horizon":     70,
        "description": "Full mission: sequential zone goals, infinite patrol, hazard avoidance, "
                       "and obstacle response. Spans Safety through Recurrence (Manna-Pnueli L1-L6).",
        "aps":         ["zone_a", "zone_b", "zone_c", "hazard_dist", "near_obstacle", "velocity"],
        "dpa_size":    8,
        "min_preds":   5,
    },

    # ═══════════════════════════════════════════════════════════════════════════
    # TD-MPC2 / dm_control walker task-family extension
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "id":          "ltl_height_safety",
        "level":       1,
        "name":        "Walker height safety",
        "mp_class":    "Safety",
        "ltl_str":     "G(height>h_min)",
        "formula":     G(atom("height", WALKER_FALL_HEIGHT_M, ">")),
        "horizon":     100,
        "property_definition_version": WALKER_HEIGHT_SPEC_VERSION,
        "description": (
            "Walker torso must always stay above h_min=0.27m: G(height>0.27). "
            "The threshold is the observed maximum (0.2442m) over persistent "
            "floor--torso-contact passive falls, plus a 0.02m empirical margin "
            "rounded upward. It represents lying on the ground, unlike the old "
            "0.6m standing-reward tolerance threshold. Same structural class "
            "(co-Buchi-compatible, dpa_size=2) as ltl_hazard_avoidance."
        ),
        "aps":         ["height"],
        "dpa_size":    2,
        "min_preds":   1,
    },
    {
        "id":          "ltl_gait_recurrence",
        "level":       5,
        "name":        "Walker/dm_control gait-cycle recurrence",
        "mp_class":    "Recurrence",
        "ltl_str":     "GF(gait_cycle)",
        "formula":     G(F(atom("gait_cycle", 0.5, ">"))),
        "horizon":     300,
        "description": (
            "A gait cycle (one complete stride, decoded from the world "
            "model's own latent as a torso-height oscillation peak) recurs "
            "infinitely often: GF(gait_cycle). This is a task-family "
            "instantiation of Table 15's own Level 5 Patrol row, GF(A): "
            "the |Q|=1 automaton structure (dpa_size=1, a single accepting "
            "state visited infinitely often) matches Table 15's Patrol row "
            "exactly, with gait_cycle playing the role zone_a plays there. "
            "This entry is a task-family extension and is NOT itself one "
            "of Table 15's 17 rows (15 core + 2 extension) -- it is a new "
            "atomic proposition substituted into an already-verified row "
            "structure, not a new row. THE EVENT PREDICATE ITSELF IS NOT "
            "DEFINED HERE: "
            "'gait_cycle' fires at step t iff a carrier-specific, frozen "
            "judgment layer (peak-detection over a decoded height AP, plus "
            "an anti-fraud false-detection discount, a CV-regularity gate, "
            "and a sliding-window patch) says so. The shared peak-detection "
            "algorithm (scipy.signal.find_peaks over the decoded height "
            "trace) lives in tdmpc2/l3_generic_lbsm_lib.py "
            "(find_peaks_and_segments, line 169; deduction_verdict, line "
            "298); it takes a 'prominence' argument but no fixed value is "
            "hardcoded in this catalog entry. Every carrier's own frozen "
            "numeric parameters (prominence, M_MIN, CV_p95, "
            "false_rate_cp_upper) and its AP source (which physics "
            "quantity 'height' actually reads) are recorded, per carrier, "
            "in that carrier's own Step 1 result file, not here: "
            "walker-walk -- prominence=0.03 (tdmpc2/"
            "walker_l3_phase_b_statistical_lbsm.py, PROMINENCE, line 76), "
            "M_MIN=15/CV_p95=0.4460545042500288 (artifacts/l3_rescue_screen/"
            "candidate5_deduction_scheme/result.json), AP = "
            "wrapper._physics.torso_height() (dm_control walker "
            "convenience method); cheetah-run -- prominence=0.03, "
            "M_MIN=17, CV_p95=0.38260226202130304, false_rate_cp_upper="
            "0.0366593606474291 (artifacts/tdmpc2_cheetah_l3_lbsm/step1/"
            "result.json; frozen in tdmpc2/l3_carrier_step1.py CARRIERS"
            "['cheetah']), AP = physics.named.data.xpos['torso','z'] "
            "(cheetah has no torso_height() convenience method); "
            "walker-run -- prominence=0.03, M_MIN=46, CV_p95="
            "0.21606759056802555, false_rate_cp_upper=0.010703829113449055 "
            "(artifacts/tdmpc2_walkerrun_l3_lbsm/step1/result.json; frozen "
            "in tdmpc2/l3_carrier_step1.py CARRIERS['walkerrun']), AP = "
            "wrapper._physics.torso_height() (same convenience method as "
            "walker-walk, same underlying quantity as cheetah's raw xpos "
            "access). M_MIN is NOT portable across carriers as a single "
            "formula -- walker-run required a different derivation "
            "(P10 of the real per-trajectory peak-count distribution) "
            "after the walker-walk-style interval-based formula produced "
            "an inconsistent value for its unusually regular gait; see the "
            "walker-run Step 1 result file's own methodological note. "
            "Existing calibrated results for this spec: walker-walk (count "
            "lens, tier3, 1000/1000, CP-lower 0.9970 -- artifacts/"
            "tdmpc2_walker_l3_statistical_lbsm/phase_b/test1/result.json), "
            "cheetah-run (same tier/lens/numbers -- artifacts/"
            "tdmpc2_cheetah_l3_lbsm/step3/result.json), walker-run "
            "(sliding lens, tier3, 1000/1000, CP-lower 0.9970; count lens "
            "984/1000 -- artifacts/tdmpc2_walkerrun_l3_lbsm/step3/"
            "result.json). tier1 is never robust at scale for any of the "
            "three carriers and must not be cited."
        ),
        "aps":         ["gait_cycle"],
        "dpa_size":    1,
        "min_preds":   1,
    },
    {
        "id":          "ltl_reactive_response",
        "level":       8,
        "name":        "Reactivity: persistent human proximity forces persistent slowdown",
        "mp_class":    "Reactivity",
        "ltl_str":     "GF(near_human) -> GF(slowed)",
        "formula":     lor(
                           F(G(neg(atom("near_human", -0.3, "<")))),
                           G(F(atom("velocity", 0.5, "<"))),
                       ),
        "horizon":     300,
        "description": (
            "Fills Table 15's Reactivity (Streett-class) row, the one "
            "Manna-Pnueli tier not otherwise represented in this catalog "
            "(the module docstring's own class list -- Safety/Guarantee/"
            "Obligation/Recurrence -- omits both Persistence, already used "
            "by ltl_eventual_hazard_stability, and Reactivity; this entry "
            "and that one are the two undocumented classes actually in "
            "use). Formalized as the standard Streett expansion "
            "GF(p)->GF(q) = FG(!p) or GF(q), built here from atom/neg/F/G/"
            "lor rather than a literal '->' node (this file has no "
            "'implies' helper; specs/stl_specs.py's implies() is bounded-"
            "only and not reusable for an unbounded GF/FG formula). "
            "EXTENDED COVERAGE, NO FORMAL VERDICT CURRENTLY EXISTS for "
            "this spec on any carrier: near_human has no AP on any carrier "
            "in this project (confirmed absent on CarDreamer roundabout, "
            "TD-MPC2 door-close, and TD-MPC2 walker -- see artifacts/"
            "safeworld_bench_full_coverage/SUMMARY.md's AP census), so "
            "this entry is registered for catalog completeness only, not "
            "because a witness or calibration attempt has been made. The "
            "'slowed' gloss in ltl_str maps to the existing 'velocity' AP "
            "(velocity<0.5), matching the same threshold ltl_hazard_response "
            "and ltl_human_caution already use for the same physical "
            "concept, for consistency with the rest of the catalog. "
            "Level and mp_class have been checked against Table 15's own "
            "text: Table 15 places this row in the Level 8 block, grouped "
            "with Full Mission; mp_class Reactivity is Table 15's own "
            "label for this row, an extended-coverage class. Level=8 and "
            "mp_class=Reactivity are therefore verified against Table 15's "
            "text (15 core rows + 2 extension rows = 17 rows total; this "
            "entry corresponds to Table 15's Reactivity row). dpa_size "
            "below remains the one unverified figure: a naive-product "
            "order-of-magnitude estimate (a 2-state co-Buchi component "
            "for FG(!p) times a 2-state Buchi component for GF(q)), not "
            "a constructed "
            "or minimized deterministic Streett automaton."
        ),
        "aps":         ["near_human", "velocity"],
        "dpa_size":    4,
        "min_preds":   2,
    },
]


# ── public API ────────────────────────────────────────────────────────────────

def get_all_ltl_specs() -> list[dict]:
    return LTL_SPECS

def get_ltl_spec_by_id(spec_id: str) -> dict | None:
    return next((s for s in LTL_SPECS if s["id"] == spec_id), None)

def get_ltl_specs_by_level(level: int) -> list[dict]:
    return [s for s in LTL_SPECS if s["level"] == level]

def get_ltl_specs_by_mp_class(mp_class: str) -> list[dict]:
    return [s for s in LTL_SPECS if s["mp_class"].lower() == mp_class.lower()]

def list_ltl_spec_ids() -> list[str]:
    return [s["id"] for s in LTL_SPECS]
