"""Walker-native task specifications.

These are environment extensions, not additions to the fixed cross-domain
SAFEWORLD-BENCH catalog. ``progress`` is trajectory-relative MuJoCo root-x
displacement in metres, so it requires an explicit initial-state accumulator.
"""

from __future__ import annotations

from specs.ltl_specs import F, G, atom, land
from specs.walker_constants import (
    WALKER_FALL_HEIGHT_M,
    WALKER_HEIGHT_SPEC_VERSION,
)


WALKER_PROGRESS_GOAL_M = 20.0
WALKER_CURRENT_SAFETY_SPEC_ID = 'walker_progress_and_eventual_torso_contact_avoidance'
WALKER_CONTACT_AP = 'torso_contact_during_last_agent_step'
# Backward-compatible symbol; its semantics are now explicitly "fallen onto
# the ground", not dm_control's older standing-reward tolerance floor.
WALKER_HEIGHT_MIN_M = WALKER_FALL_HEIGHT_M


WALKER_SPECS = [
    {
        'id': WALKER_CURRENT_SAFETY_SPEC_ID,
        'name': 'Walker goal and eventual torso-contact avoidance',
        'mp_class': 'GuaranteeAndPersistence',
        'ltl_str': 'F(progress > 20m) & F G(torso_contact_during_last_agent_step < 0.5)',
        'formula': land(F(atom('progress', WALKER_PROGRESS_GOAL_M, '>')),
                        F(G(atom(WALKER_CONTACT_AP, 0.5, '<')))),
        'aps': ['progress', WALKER_CONTACT_AP],
        'required_binary_aps': [WALKER_CONTACT_AP],
        'property_definition_version': 'walker-torso-contact-any-substep-v1',
        'native_horizon_agent_steps': 500,
        'description': 'Bad iff torso touches floor at any physics substep of the preceding agent step; reset uses endpoint contact.',
        'model_ap_requirement': 'Explicit predicted contact-event monitor required; no height or stale real-latch fallback.',
    },
    {
        "id": "walker_progress_goal",
        "name": "Walker forward-progress goal",
        "mp_class": "Guarantee",
        "ltl_str": "F(progress > 20m)",
        "formula": F(atom("progress", WALKER_PROGRESS_GOAL_M, ">")),
        "aps": ["progress"],
        "native_horizon_agent_steps": 500,
        "description": (
            "Eventually displace the root more than 20 metres forward from "
            "the episode's initial root-x position."
        ),
    },
    {
        "id": "walker_progress_and_eventual_height_stability",
        "legacy_height_proxy": True,
        "superseded_by": WALKER_CURRENT_SAFETY_SPEC_ID,
        "name": "Walker goal achievement with eventual height stability",
        "mp_class": "GuaranteeAndPersistence",
        "ltl_str": "F(progress > 20m) & F G(height > 0.27m)",
        "formula": land(
            F(atom("progress", WALKER_PROGRESS_GOAL_M, ">")),
            F(G(atom("height", WALKER_HEIGHT_MIN_M, ">"))),
        ),
        "aps": ["progress", "height"],
        "native_horizon_agent_steps": 500,
        "property_definition_version": WALKER_HEIGHT_SPEC_VERSION,
        "finite_evidence_semantics": (
            "goal must be reached within 500 steps; after the last bad height "
            "state, at least 100 consecutive safe steps must be observed"
        ),
        "description": (
            "Reach the forward displacement goal and eventually recover into "
            "persistent ground-fall avoidance. Finite experiments use the declared "
            "100-step recovery-dwell proxy and cannot prove the unbounded F G clause."
        ),
    },
]


def get_walker_spec_by_id(spec_id: str) -> dict | None:
    return next((spec for spec in WALKER_SPECS if spec["id"] == spec_id), None)
