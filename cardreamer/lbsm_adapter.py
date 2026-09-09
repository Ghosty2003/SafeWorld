"""CarDreamer RSSM adapter for the model-agnostic distributional L3 core.

The product state is the full Markov RSSM state used by imagination
(``deter`` + flattened categorical ``stoch``), followed by the current
acceptance bit and a normalized 2-D position probe output.  The position is
included so region predicates remain auditable; it is not used as a lossy
replacement for the RSSM state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from core.lbsm.operator_model import AnchorSuccessorBatch
from cardreamer.probe_common import RING_C


@dataclass(frozen=True)
class CarDreamerLatentAnchor:
    deter: np.ndarray
    stoch: np.ndarray
    logit: np.ndarray
    sample_id: str
    episode_id: int
    real_xy: np.ndarray


@dataclass(frozen=True)
class CarDreamerProductLayout:
    deter_dim: int
    stoch_size: int
    position_scale: float = 64.0

    @property
    def q_index(self) -> int:
        return self.deter_dim + self.stoch_size

    @property
    def xy_slice(self) -> slice:
        return slice(self.q_index + 1, self.q_index + 3)

    @property
    def state_dim(self) -> int:
        return self.q_index + 3

    def position(self, product_state: np.ndarray) -> np.ndarray:
        return np.asarray(product_state[self.xy_slice]) * self.position_scale


def encode_product_states(
    deter: np.ndarray,
    stoch: np.ndarray,
    positions: np.ndarray,
    accepting: np.ndarray,
    layout: CarDreamerProductLayout,
) -> np.ndarray:
    deter = np.asarray(deter, dtype=np.float32)
    stoch = np.asarray(stoch, dtype=np.float32).reshape(deter.shape[0], -1)
    positions = np.asarray(positions, dtype=np.float32)
    accepting = np.asarray(accepting, dtype=np.float32).reshape(-1, 1)
    if deter.ndim != 2 or deter.shape[1] != layout.deter_dim:
        raise ValueError("deter shape does not match product layout")
    if stoch.shape[1] != layout.stoch_size:
        raise ValueError("stoch shape does not match product layout")
    if positions.shape != (deter.shape[0], 2):
        raise ValueError("positions must have shape (n,2)")
    return np.concatenate(
        [deter, stoch, accepting, positions / layout.position_scale], axis=1
    ).astype(np.float32)


def make_roundabout_region_predicates(
    layout: CarDreamerProductLayout,
    *,
    accepting_radius: float = 24.9,
    retention_radius: float = 30.0,
):
    """Return F and I predicates for GF(zone_a), with F strictly inside I."""
    if not accepting_radius < retention_radius:
        raise ValueError("accepting_radius must be strictly smaller than retention_radius")

    def is_accepting(x: np.ndarray) -> bool:
        # The bit is the automaton destination after consuming the current AP.
        return bool(np.asarray(x)[layout.q_index] > 0.5)

    def in_retention_set(x: np.ndarray) -> bool:
        return float(np.linalg.norm(layout.position(x) - RING_C)) < retention_radius

    return is_accepting, in_retention_set


def build_one_step_rssm_sampler(jax_agent: Any, kappa: int):
    """Build a JIT sampler returning kappa policy/RSSM successors per anchor."""
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    from dreamerv3 import ninjax as nj
    import jax.numpy as jnp
    from wrappers.cardreamer_wrapper import _get_actor

    wm = jax_agent.agent.wm
    actor = _get_actor(jax_agent)
    device = jax_agent.policy_devices[0]

    def _sample(deter, stoch, logit):
        # This checkpoint runs its RSSM state in float16.  JAX lax.scan
        # requires the carry input/output dtypes to match exactly; live
        # posterior anchors may otherwise arrive as float32 via NumPy.
        deter = deter.astype(jnp.float16)
        stoch = stoch.astype(jnp.float16)
        logit = logit.astype(jnp.float16)
        repeat = lambda value: jnp.repeat(value, kappa, axis=0)
        start = {
            "deter": repeat(deter),
            "stoch": repeat(stoch),
            "logit": repeat(logit),
            "is_terminal": jnp.zeros((deter.shape[0] * kappa,), jnp.float32),
        }

        def policy(state):
            return actor(state).sample(seed=nj.rng())

        trajectory = wm.imagine(policy, start, 1)
        return trajectory["deter"][1], trajectory["stoch"][1]

    return nj.jit(nj.pure(_sample), device=device)


def sample_product_successor_batches(
    *,
    jax_agent: Any,
    anchors: Sequence[CarDreamerLatentAnchor],
    position_probe: Any,
    kappa: int,
    sampling_scope: str,
    accepting_radius: float = 24.9,
    chunk_size: int = 4,
) -> tuple[list[AnchorSuccessorBatch], CarDreamerProductLayout]:
    """Convert live posterior anchors into distributional-L3 query batches."""
    if not anchors:
        raise ValueError("anchors must not be empty")
    if not sampling_scope:
        raise ValueError("sampling_scope must be non-empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    import jax

    first = anchors[0]
    layout = CarDreamerProductLayout(
        deter_dim=int(np.asarray(first.deter).size),
        stoch_size=int(np.asarray(first.stoch).size),
    )
    sampler = build_one_step_rssm_sampler(jax_agent, kappa)
    device = jax_agent.policy_devices[0]
    result: list[AnchorSuccessorBatch] = []

    for start_index in range(0, len(anchors), chunk_size):
        real_chunk = list(anchors[start_index : start_index + chunk_size])
        padded = real_chunk + [real_chunk[-1]] * (chunk_size - len(real_chunk))
        deter = np.stack([np.asarray(item.deter) for item in padded])
        stoch = np.stack([np.asarray(item.stoch) for item in padded])
        logit = np.stack([np.asarray(item.logit) for item in padded])
        rng = jax_agent._next_rngs(jax_agent.policy_devices)
        successor_deter, successor_stoch = sampler(
            jax_agent.varibs,
            rng,
            jax.device_put(deter, device),
            jax.device_put(stoch, device),
            jax.device_put(logit, device),
        )[0]
        successor_deter = np.asarray(jax.device_get(successor_deter)).reshape(
            chunk_size, kappa, -1
        )
        successor_stoch = np.asarray(jax.device_get(successor_stoch)).reshape(
            chunk_size, kappa, *np.asarray(first.stoch).shape
        )

        current_xy = position_probe.predict(deter[: len(real_chunk)])
        for local_index, anchor in enumerate(real_chunk):
            succ_deter = successor_deter[local_index]
            succ_stoch = successor_stoch[local_index]
            successor_xy = position_probe.predict(succ_deter)
            current_accepting = (
                np.linalg.norm(current_xy[local_index] - RING_C) < accepting_radius
            )
            successor_accepting = (
                np.linalg.norm(successor_xy - RING_C, axis=1) < accepting_radius
            )
            current_state = encode_product_states(
                deter[local_index : local_index + 1],
                stoch[local_index : local_index + 1],
                current_xy[local_index : local_index + 1],
                np.asarray([current_accepting]),
                layout,
            )[0]
            successor_states = encode_product_states(
                succ_deter,
                succ_stoch,
                successor_xy,
                successor_accepting,
                layout,
            )
            result.append(AnchorSuccessorBatch(
                anchor=current_state,
                successors=successor_states,
                sample_id=anchor.sample_id,
                sampling_scope=sampling_scope,
            ))
    return result, layout


def select_core_points(
    batches: Sequence[AnchorSuccessorBatch],
    layout: CarDreamerProductLayout,
    *,
    retention_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose non-degenerate U-shaping examples from the certificate split."""
    states = np.stack([batch.anchor for batch in batches])
    radii = np.linalg.norm(
        np.stack([layout.position(state) for state in states]) - RING_C,
        axis=1,
    )
    inside = states[radii < retention_radius]
    outside = states[radii >= retention_radius]
    if len(inside) == 0 or len(outside) == 0:
        raise ValueError(
            "certificate split must contain states both inside and outside retention set I"
        )
    return inside, outside
