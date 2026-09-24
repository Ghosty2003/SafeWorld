from __future__ import annotations

from .ltl_specs import LTL_SPECS, get_ltl_spec_by_id
from .stl_specs import STL_SPECS, get_stl_spec_by_id

ALL_SPECS = [*LTL_SPECS, *STL_SPECS]


def get_spec_by_id(spec_id: str) -> dict | None:
    from .walker_specs import get_walker_spec_by_id
    return get_ltl_spec_by_id(spec_id) or get_stl_spec_by_id(spec_id) or get_walker_spec_by_id(spec_id)
