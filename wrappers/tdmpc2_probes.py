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

``forward_speed`` has a deliberately narrower meaning than the generic
``velocity`` key used by some benchmark environments: it is dm_control
walker's signed centre-of-mass horizontal velocity in metres/second, exactly
the quantity returned by ``Physics.horizontal_velocity()`` and used by the
walker-walk reward.  It is *not* ``obs[15]`` (the root generalized velocity),
and it is not the absolute speed.  Validation and artifact creation live in
``tdmpc2/validate_forward_speed_probe.py``.

Forward-speed validation (walker-walk, checkpoint seed=3, 2026-08-30):
    C0 nested episode-grouped CV on 15 independent real episodes (8 MPC,
    7 random; 7,500 states): R²=0.9867, MAE=0.0617m/s, overall threshold
    agreement=99.43%; on the MPC subset, p90 absolute error=0.0647m/s and
    threshold agreement=99.65%.
    C1 on 200 exact anchors from 8 additional MPC episodes, with model MPC
    actions replayed from the identical physics state: depth-1 p90=0.0779m/s,
    depth-5 p90=0.1275m/s, depth-10 p90=0.2288m/s, but depth-100
    p90=2.3679m/s.  The fixed C1 criteria hold only through depth 15.  Thus the
    probe itself is validated on real posterior latents, while 100-step
    world-model transfer is INCONCLUSIVE.  Do not use it for a 100-step L2
    claim without pricing that model error.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def _load_scalar_probe_data(npz_path: str, target_key: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)
    if "z" not in data or target_key not in data:
        raise ValueError(f"{npz_path!r} must contain arrays 'z' and {target_key!r}")
    z = np.asarray(data["z"], dtype=np.float32)
    target = np.asarray(data[target_key], dtype=np.float32).reshape(-1)
    if z.ndim != 2 or len(z) != len(target) or len(z) == 0:
        raise ValueError("probe data must have z.shape=(n,d), target.shape=(n,), n>0")
    if not np.isfinite(z).all() or not np.isfinite(target).all():
        raise ValueError("probe data must be finite")
    return z, target


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

    z, h = _load_scalar_probe_data(npz_path, "h")
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


def fit_forward_speed_probe(
    npz_path: str,
    alpha: float = 10.0,
    *,
    standardize: bool = True,
):
    """Fit ``latent z -> signed walker forward_speed (m/s)``.

    The input dataset must contain ``z`` and ``forward_speed``.  A scaler is
    fit inside the returned sklearn pipeline by default; callers performing
    cross-validation must fit the whole pipeline inside every fold to avoid
    leakage.  The validation runner does exactly that and stores its selected
    alpha in the result artifact.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    z, speed = _load_scalar_probe_data(npz_path, "forward_speed")
    model = (
        make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
        if standardize
        else Ridge(alpha=alpha, solver="lsqr")
    )
    return model.fit(z, speed)


def make_forward_speed_ap_extractor(
    probe,
    key: str = "forward_speed",
) -> Callable[[np.ndarray], dict[str, float]]:
    """Wrap a fitted forward-speed probe for ``TDMPC2Wrapper``."""

    def extractor(z: np.ndarray) -> dict[str, float]:
        pred = probe.predict(np.asarray(z).reshape(1, -1))
        return {key: float(np.asarray(pred).reshape(-1)[0])}

    return extractor


def walker_forward_speed_from_physics(physics) -> float:
    """Return the task-semantic signed COM horizontal velocity in m/s."""
    value = float(physics.horizontal_velocity())
    if not np.isfinite(value):
        raise ValueError("walker physics returned a non-finite horizontal velocity")
    return value
