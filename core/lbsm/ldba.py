"""
core/lbsm/ldba.py

SAFEWORLD L3 -- the 2-state limit-deterministic Buchi automaton (LDBA) for
phi = Box Diamond p (paper Section 4.4 "Product-state dynamics"; Appendix
E.2's concrete instance).

Paper correspondence
---------------------
Section 4.4: "We first compile the specification phi into a limit-
deterministic Buchi automaton A = (Q, q0, Sigma, delta_A, F)... The accepting
product states are F_X = {(z,q) in X : q in F}."

Section 6.3 / Appendix E.2, "Specification and product acceptance set
(state-based)": for phi = Box Diamond p, the natural automaton is "the
memoryless single-state machine with a self-loop accepting on reading a
p-labelled step -- a transition-based acceptance condition." Following the
"acceptance-bit augmentation" of Section 4.4, this is put into STATE-BASED
Moore normal form with two states Q = {q_pbar, q_p}, accepting set F = {q_p}:
"any label satisfying p leads to q_p and any label violating p leads to
q_pbar, from EITHER state" (memoryless -- q_t depends only on L(z_t), never
on q_{t-1}; this is specific to this construction, not a general LDBA
property).

Timing convention (load-bearing, see Stage-0 audit note below)
----------------------------------------------------------------
"the step into x_t consumes L(z_t), so the augmented bit at x_t equals
[L(z_t) = p]" -- i.e. the automaton state x_t = (z_t, q_t) used for FX
membership at time t is the DESTINATION state entered by consuming the
CURRENT step's label, not the state the automaton was in before this step.

core/lppm/automaton.py::ParityAutomaton.step_with_priority() (the L2 co-Buchi
automaton builder) returns the priority of the SOURCE state q, which is
correct for L2's (P1)-(P2) conditions (defined at the source: "if q in B
then..."), but is the WRONG convention for L3's F_X membership. This module
therefore does NOT reuse core/lppm/automaton.py or core/lppm/verifier.py's
run_product_trajectory() -- it is a new, independent implementation, per the
Stage-0 investigation.

To keep this timing decision auditable at a glance, `step()` below returns
ONLY the destination state (no bundled acceptance/priority info -- there is
deliberately no `step_with_priority`-style combined call here), and acceptance
is always judged by a SEPARATE call to `accepting()` on that destination
state. See tests/test_lbsm_analytic.py for a test that would fail if a future
edit accidentally judged acceptance on the source state instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Q_PBAR = "q_pbar"
Q_P = "q_p"


@dataclass(frozen=True)
class LabelBuchiAutomaton:
    """
    2-state Moore-form Buchi automaton for phi = Box Diamond p.

    label_predicate(z) == p(z), the atomic proposition tested at each step.
    For Appendix E.2: p(z) := ||z|| <= r_F.
    """

    label_predicate: Callable[[np.ndarray], bool]

    def step(self, q_prev: str | None, z: np.ndarray) -> str:
        """
        Pure transition: returns ONLY the destination state q_t, consuming
        the CURRENT label L(z). q_prev is accepted (and ignored) to match the
        general product-state calling convention -- this automaton happens to
        be memoryless, but a future LDBA reusing this loop shape need not be.
        """
        return Q_P if self.label_predicate(z) else Q_PBAR

    def accepting(self, q: str) -> bool:
        """F = {q_p}."""
        return q == Q_P


def build_e2_recurrence_automaton(r_F: float) -> LabelBuchiAutomaton:
    """Appendix E.2's p(z) := ||z|| <= r_F, i.e. ||z||^2 <= r_F^2."""
    r_F_sq = r_F ** 2
    return LabelBuchiAutomaton(label_predicate=lambda z: float(np.dot(z, z)) <= r_F_sq)


def run_trajectory(automaton: LabelBuchiAutomaton, zs: list[np.ndarray]) -> list[str]:
    """
    Returns [q_0, ..., q_{T-1}], the DESTINATION state entered after
    consuming each z_t -- the paper's "entering x_t consumes L(z_t)"
    convention. q_{-1} (the automaton's notional pre-trajectory state) is
    never exposed, since this automaton's step() ignores q_prev entirely.
    """
    q_states: list[str] = []
    q_prev: str | None = None
    for z in zs:
        q_prev = automaton.step(q_prev, z)
        q_states.append(q_prev)
    return q_states


def accepting_mask(automaton: LabelBuchiAutomaton, q_states: list[str]) -> list[bool]:
    """
    F_X membership at each t, using the DESTINATION states from
    run_trajectory() -- NOT source states. This is the function whose output
    both core/lbsm/verifier.py and the timing-misdirection test in
    tests/test_lbsm_analytic.py depend on.
    """
    return [automaton.accepting(q) for q in q_states]
