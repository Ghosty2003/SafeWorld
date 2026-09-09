"""
core/lbsm/model.py

SAFEWORLD L3 -- LBSM return certificate W and companion certificate U
(paper Section 4.4; Appendix E.2's analytic instantiation).

Paper correspondence
---------------------
- W : X -> [0, B]    the LBSM return certificate (Eq. 6, Theorems 5.5/5.6).
- U : X -> [0, B_U]  the companion retention certificate (Section 4.4,
                     "Retention certificate"; Theorem 5.6 hypotheses (H-a)/(H-b)).

Both are functions of the LATENT z only in this module (they do not depend on
the automaton state q). This matches Appendix E.2's construction exactly: "the
certificates W and U depend only on z (constant in q)", which is also why the
paper's product (B1) drift domain reduces from X\\F_X to the latent-only region
{z not in F}.

Stage 1 (this file): W and U are HAND-WRITTEN CLOSED FORMS, not neural
networks. Appendix E.2, verbatim: "Take U(z) = W(z) := ||z||^2". No training
occurs in this stage -- see core/lbsm/trainer.py (Stage 2, not yet built) for
the trained-network version with the sigmoid/softplus-bounded architecture.

Boundedness note: Theorem 5.6 requires W, U bounded on their full domain X.
||z||^2 is NOT bounded on all of R^d. The paper resolves this by restricting
evaluation to the certified operative region C = {||z||^2 <= ell}: "the
certificate is never evaluated outside C (where ||z||^2 is of course
unbounded)" (Appendix E.2, "Companion and drift"). This module does not
architecturally clip the function -- clipping would change the exact drift
equality this construction relies on (see core/lbsm/drift.py::exact_drift_e2).
Instead, AnalyticCertificate carries `ell` (== B == B_U here, "the covering
cap is B_bar = ell") as an explicit cap, and core/lbsm/verifier.py is
responsible for never evaluating W/U outside C.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnalyticCertificate:
    """
    W(z) = U(z) = ||z||^2, restricted (by the caller, not by this class) to
    the certified operative region C = {||z||^2 <= ell}.

    Paper: Appendix E.2, "Companion and drift" -- "Take U(z) = W(z) := ||z||^2
    and C = {||z||^2 <= ell} with ell > r_inv^2. On the operative region C
    this is a bounded certificate, U = ||z||^2 <= ell, so the covering cap is
    B_bar = ell".
    """

    ell: float  # == B == B_U (the paper's "covering cap B_bar = ell")

    def W(self, z: np.ndarray) -> float:
        return float(np.dot(z, z))

    def U(self, z: np.ndarray) -> float:
        # Identical function to W, per the paper's own construction -- not a
        # coincidence of this codebase, this IS the paper's stated choice.
        return self.W(z)

    @property
    def B(self) -> float:
        return self.ell

    @property
    def B_U(self) -> float:
        return self.ell
