"""
wrappers/tdmpc2_probes.py

Pluggable AP-extraction components for TDMPC2Wrapper latent trajectories.

Per project convention (see cardreamer/probe_common.py): probes are fit and
used OUTSIDE the world-model wrapper. TDMPC2Wrapper.sample_rollouts() only
accepts an externally-supplied `ap_extractor` callable — it contains no
probe-specific logic itself.

Height probe validation (walker-walk, checkpoint seed=3, 15-episode pilot
+ 200-anchor MPC-imagined decomposition, GPU):
    C0 (episode-grouped CV regression accuracy): R²=0.9986, MAE=0.0132m
    C1 (MPC-real vs MPC-imagined, matched action mechanism AND matched
        target distribution -- the only methodologically clean measurement):
        depth 100: p50=0.0218m p90=0.0637m max=0.1123m, consistent across
        all 8 contributing episodes (no tail-statistic domination by a
        single trajectory).
Both results hold only for walker-walk / seed=3. Do not assume they transfer
to walker-run/stand/backwards or other seeds without re-running C0/C1 there.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def fit_height_probe(npz_path: str, alpha: float = 10.0):
    """
    Ridge probe z -> real torso height, fit on all posterior (z, height) pairs
    in npz_path (expects arrays "z" and "h", as saved by the validated pilot
    collection scripts).

    npz_path is a required argument (not defaulted to a /tmp scratch path):
    the caller must point this at whatever posterior dataset is the current
    production probe-training set (pilot-scale or scaled-up), since that
    decision is independent of the wrapper/probe plumbing itself.
    """
    from sklearn.linear_model import Ridge

    data = np.load(npz_path)
    z, h = data["z"], data["h"]
    return Ridge(alpha=alpha).fit(z, h)


def make_height_ap_extractor(
    probe,
    key: str = "height",
) -> Callable[[np.ndarray], dict[str, float]]:
    """
    Wrap a fitted probe (anything with .predict(z_batch) -> (N,) or (N,1))
    into the ap_extractor callable signature TDMPC2Wrapper.sample_rollouts()
    / TDMPC2Wrapper.set_ap_extractor() expect: z (D,) -> {key: float}.
    """

    def extractor(z: np.ndarray) -> dict[str, float]:
        pred = probe.predict(z.reshape(1, -1))
        return {key: float(np.asarray(pred).reshape(-1)[0])}

    return extractor
