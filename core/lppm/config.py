"""
core/lppm/config.py

Batch-4 audit fix: single source of truth for LPPM hyperparameters shared
across fit_lppm()/calibrate_lppm()/check_pathwise_conditions() and any
script that trains/calibrates/verifies an LPPM certificate.

Before this fix, eta=0.01 was hardcoded independently in FIVE places (3 core
functions -- core/lppm/trainer.py::fit_lppm, core/lppm/calibrator.py::
calibrate_lppm, core/lppm/verifier.py::check_pathwise_conditions -- plus 2
hand-written eval scripts, cardreamer/eval_ltl_hazard_real_lppm.py and
tdmpc2/eval_ltl_height_safety_lppm.py), with nothing enforcing they stay in
sync. They all happened to agree (0.01) at the time of the audit, but a
future edit to any one of them would silently desynchronize the descent
margin used during training from the one used to score calibration/
verification -- exactly the kind of drift that produces a p_hat_gamma that
doesn't mean what it claims to.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LPPMConfig:
    eta: float = 0.01


DEFAULT_LPPM_CONFIG = LPPMConfig()
