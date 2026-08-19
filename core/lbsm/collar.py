"""
core/lbsm/collar.py

Section 4.4 / Appendix C.3's three collar-discharge routes for alpha_collar.
This file implements only route (a), elimination:

Appendix C.3, "(a) Elimination": "When white-box bounds on the label function
... certify that the closure of the collar does not meet the region on which
the run must be retained ... collar entry is impossible before retention is
decided and alpha_collar = 0 is certified; the L3 instantiation of Section
6.3 takes this route."

Appendix E.2, concrete instance: "alpha_collar = 0 certified here via the
elimination route of Section 4.4: the label sets F, I, and C are closed-form
norm balls, so ball-boundary intersection is decided by interval arithmetic
on the anchor norm rather than estimated from samples."
"""

from __future__ import annotations


def ball_crosses_boundary(anchor_norm: float, boundary_radius: float, mesh_delta: float) -> bool:
    """
    Interval-arithmetic check: does a ball of radius `mesh_delta` centered at
    a point of norm `anchor_norm` cross the sphere ||z|| = boundary_radius?

    The ball's norm-interval is [anchor_norm - mesh_delta, anchor_norm +
    mesh_delta]; it crosses the boundary iff boundary_radius falls strictly
    inside that interval. (This 1-D radial check assumes anchor_norm >=
    mesh_delta, i.e. the ball does not contain the origin; not an issue for
    this module's callers, which only ever use mesh_delta == 0, see below.)
    """
    return (anchor_norm - mesh_delta) < boundary_radius < (anchor_norm + mesh_delta)


def eliminate_collar_stage1_analytic(r_F: float) -> float:
    """
    Route (a) elimination for the Stage-1 ANALYTIC construction (Appendix
    E.2). Returns alpha_collar, always 0.0 for this construction, but
    computed via ball_crosses_boundary() rather than hardcoded as a bare
    literal, so Stage 2/3 (which DO build a positive-radius covering net) can
    reuse the same primitive with mesh_delta > 0 and get a genuine nonzero
    answer when warranted.

    Why Delta = 0 here: core/lbsm/drift.py::exact_drift_e2() evaluates the
    drift in EXACT closed form at every point of the certified annuli
    (I \\ F and C \\ I), not via a finite covering net of anchor balls -- so
    there is no discretization mesh radius at all (Delta = 0, no covering net
    built). A zero-radius ball can only "cross" a boundary it sits exactly
    on, which the checked regions (defined by strict inequality relative to
    r_F) never do by construction.
    """
    crosses = ball_crosses_boundary(anchor_norm=r_F, boundary_radius=r_F, mesh_delta=0.0)
    if crosses:
        # Unreachable for mesh_delta=0 (a degenerate point can't straddle its
        # own boundary under a strict-inequality check) -- fail loud if this
        # ever changes rather than silently returning a wrong alpha_collar.
        raise AssertionError(
            "eliminate_collar_stage1_analytic(): ball_crosses_boundary() unexpectedly "
            "reported a crossing at Delta=0 -- the Stage-1 analytic elimination-route "
            "argument no longer holds; do not fall back to alpha_collar=0 silently."
        )
    return 0.0


def assert_anchor_ball_does_not_cross_f_boundary(
    anchor_norm: float, r_F: float, mesh_delta: float,
) -> None:
    """
    Route (a) elimination, Stage-2 COVERING-NET version (mesh_delta > 0, as
    opposed to Stage 1's mesh_delta == 0 degenerate case above).

    Stage 2 genuinely builds a positive-radius covering net over I \\ F
    (core/lbsm/verifier.py::verify_trained_recurrence_warrant()), so unlike
    Stage 1 there IS a mesh radius that could straddle the F-boundary
    ||z|| = r_F. The caller is responsible for excluding any anchor within
    mesh_delta of r_F from the certified net BEFORE calling this (i.e.
    starting I \\ F's inner radius at r_F + mesh_delta, not r_F) -- this
    function is the explicit, checked assertion that the exclusion was done
    correctly for every RETAINED anchor, not a substitute for doing it. Any
    anchor this raises on indicates a construction bug (an anchor that should
    have been excluded from the certified region, i.e. genuine uncertified
    collar, was certified anyway).
    """
    if ball_crosses_boundary(anchor_norm, r_F, mesh_delta):
        raise AssertionError(
            f"anchor at radius {anchor_norm} has a mesh_delta={mesh_delta} ball crossing "
            f"the F-boundary r_F={r_F} -- this anchor should have been excluded from the "
            "covering net as uncertified collar, not certified via route (a) elimination."
        )
