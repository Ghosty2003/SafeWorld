"""Collect diverse, training-only L2 data for CarDreamer hazard avoidance.

The deployment-policy calibration split must remain independent.  This runner
therefore writes only a labelled LPPM *training candidate* pool and marks every
artifact as calibration-excluded.  Collection starts from replay-buffer
posterior anchors and follows the checkpoint actor in imagination.

Raw batches are ``*.npz`` files (ignored by this repository).  A small JSON
manifest records hashes, counts, stopping criteria, and the checkpoint/config
identity without copying either file.

Example:

    conda run --no-capture-output -n cardreamer python -u \
      cardreamer/collect_l2_hazard_training_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time
from typing import Any

# Must be set before importing the wrapper (and therefore JAX).  The same flags
# made the existing four-lane L2 experiment reproducible across process runs.
os.environ.setdefault(
    "XLA_FLAGS", "--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"
)

import numpy as np

from configs.settings import RolloutConfig
from wrappers.cardreamer_wrapper import CarDreamerWrapper


AP_KEYS = ("hazard_dist", "near_obstacle", "goal_dist", "velocity")
SCHEMA_VERSION = 1


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory_fingerprint(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _pack_trajectories(
    trajectories: list[list[dict[str, float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not trajectories:
        raise ValueError("cannot pack an empty trajectory batch")
    horizon = len(trajectories[0])
    if horizon == 0 or any(len(traj) != horizon for traj in trajectories):
        raise ValueError("all trajectories must have the same non-zero horizon")

    values = np.asarray(
        [[[float(step[key]) for key in AP_KEYS] for step in traj] for traj in trajectories],
        dtype=np.float32,
    )
    trap = np.any(values[:, :, AP_KEYS.index("hazard_dist")] <= 0.0, axis=1)
    fingerprints = np.asarray(
        [_trajectory_fingerprint(values[i]) for i in range(len(values))], dtype="U64"
    )
    return values, trap, fingerprints


def _read_batches(output_dir: pathlib.Path) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("batch_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if int(data["schema_version"]) != SCHEMA_VERSION:
                raise ValueError(f"unsupported schema in {path}")
            batches.append(
                {
                    "path": path,
                    "values": data["values"].astype(np.float32, copy=False),
                    "trap": data["trap"].astype(bool, copy=False),
                    "fingerprints": data["fingerprints"].astype("U64", copy=False),
                    "seed": int(data["seed"]),
                }
            )
    return batches


def _summarize(batches: list[dict[str, Any]]) -> dict[str, Any]:
    if not batches:
        return {
            "n_rollouts": 0,
            "n_trap_rollouts": 0,
            "trap_rate": 0.0,
            "n_unique_trajectories": 0,
            "n_unique_trap_trajectories": 0,
            "hazard_min": {},
        }

    values = np.concatenate([batch["values"] for batch in batches], axis=0)
    trap = np.concatenate([batch["trap"] for batch in batches], axis=0)
    fingerprints = np.concatenate([batch["fingerprints"] for batch in batches])
    hazard = values[:, :, AP_KEYS.index("hazard_dist")]
    minima = np.min(hazard, axis=1)
    trap_fingerprints = fingerprints[trap]
    return {
        "n_rollouts": int(len(values)),
        "n_trap_rollouts": int(np.sum(trap)),
        "trap_rate": float(np.mean(trap)),
        "n_unique_trajectories": int(len(set(fingerprints.tolist()))),
        "n_unique_trap_trajectories": int(len(set(trap_fingerprints.tolist()))),
        "n_trap_transitions": int(np.sum(hazard <= 0.0)),
        "hazard_min": {
            "min": float(np.min(minima)),
            "p10": float(np.percentile(minima, 10)),
            "p50": float(np.percentile(minima, 50)),
            "p90": float(np.percentile(minima, 90)),
            "max": float(np.max(minima)),
        },
    }


def _write_manifest(
    output_dir: pathlib.Path,
    args: argparse.Namespace,
    checkpoint: pathlib.Path,
    config: pathlib.Path,
    batches: list[dict[str, Any]],
    started_at: float,
) -> dict[str, Any]:
    summary = _summarize(batches)
    files = [
        {
            "name": batch["path"].name,
            "sha256": _sha256_file(batch["path"]),
            "n_rollouts": int(len(batch["values"])),
            "n_trap_rollouts": int(np.sum(batch["trap"])),
            "seed": batch["seed"],
        }
        for batch in batches
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "cardreamer_four_lane_l2_hazard_training_collection",
        "data_role": "lppm_training_candidate_only",
        "calibration_eligible": False,
        "calibration_note": (
            "Never use these stop-rule-selected batches as the held-out calibration split. "
            "Collect a fresh, independent actor-policy calibration sample."
        ),
        "task": "carla_four_lane",
        "spec": "ltl_hazard_avoidance",
        "failure_definition": "trajectory contains at least one hazard_dist <= 0.0 step",
        "action_source": args.action_source,
        "epsilon_random": (
            args.epsilon_random if args.action_source == "actor_epsilon" else 0.0
        ),
        "anchor_source": "checkpoint replay buffer with posterior burn-in",
        "replay_anchor_strategy": args.anchor_strategy,
        "replay_low_hazard_fraction": args.low_hazard_fraction,
        "horizon": args.horizon,
        "base_seed": args.seed,
        "batch_size": args.batch_size,
        "replay_burn_in_steps": args.burn_in,
        "replay_pool_rollouts": args.replay_pool_rollouts,
        "replay_pool_multiplier": args.replay_pool_multiplier,
        "target_trap_rollouts": args.target_trap_rollouts,
        "max_rollouts": args.max_rollouts,
        "stopping_rule_met": summary["n_trap_rollouts"] >= args.target_trap_rollouts,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "size": checkpoint.stat().st_size,
            "sha256": _sha256_file(checkpoint),
        },
        "config": {
            "path": str(config.resolve()),
            "size": config.stat().st_size,
            "sha256": _sha256_file(config),
        },
        "ap_keys": list(AP_KEYS),
        "summary": summary,
        "batches": files,
        "elapsed_seconds": time.time() - started_at,
    }
    target = output_dir / "manifest.json"
    temporary = output_dir / "manifest.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, target)
    return manifest


def _resolve_config(checkpoint: pathlib.Path, explicit: str | None) -> pathlib.Path:
    if explicit:
        result = pathlib.Path(explicit)
    else:
        candidates = sorted(checkpoint.parent.glob("config_*.yaml"))
        if not candidates:
            raise FileNotFoundError(
                f"no config_*.yaml beside checkpoint {checkpoint}; pass --config"
            )
        result = candidates[-1]
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def collect(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.time()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = pathlib.Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = _resolve_config(checkpoint, args.config)

    batches = _read_batches(output_dir) if args.resume else []
    if not args.resume and any(output_dir.glob("batch_*.npz")):
        raise FileExistsError(
            f"{output_dir} already contains batches; pass --resume or choose a new directory"
        )
    summary = _summarize(batches)
    print(
        f"Existing: {summary['n_rollouts']} rollouts, "
        f"{summary['n_trap_rollouts']} trap rollouts",
        flush=True,
    )
    if (
        summary["n_trap_rollouts"] >= args.target_trap_rollouts
        or summary["n_rollouts"] >= args.max_rollouts
    ):
        return _write_manifest(
            output_dir, args, checkpoint, config, batches, started_at
        )

    # This only controls replay-pool preloading.  Individual imagination calls
    # still use --batch-size, keeping decoder memory bounded by wrapper chunks.
    preload_rollouts = min(args.max_rollouts, args.replay_pool_rollouts)
    load_cfg = RolloutConfig(
        n_rollouts=preload_rollouts,
        horizon=args.horizon,
        seed=args.seed,
        action_source=args.action_source,
    )
    wrapper = CarDreamerWrapper(
        load_cfg,
        use_replay_start=True,
        replay_burn_in_steps=args.burn_in,
        replay_pool_multiplier=args.replay_pool_multiplier,
        epsilon_random=args.epsilon_random,
        replay_anchor_strategy=args.anchor_strategy,
        replay_low_hazard_fraction=args.low_hazard_fraction,
    )
    print(f"Loading checkpoint: {checkpoint}", flush=True)
    wrapper.load(checkpoint_path=str(checkpoint), config_path=str(config))
    if wrapper._replay_obs_pool is None:
        raise RuntimeError("replay anchors are required; cold-start fallback is refused")
    anchor_hazard = wrapper._replay_anchor_hazard
    if anchor_hazard is not None:
        print(
            "Replay anchors after selection: "
            f"n={len(anchor_hazard)} min={np.min(anchor_hazard):.3f} "
            f"p50={np.median(anchor_hazard):.3f} max={np.max(anchor_hazard):.3f}",
            flush=True,
        )

    next_batch = len(batches)
    try:
        while True:
            summary = _summarize(batches)
            remaining = args.max_rollouts - summary["n_rollouts"]
            if remaining <= 0 or summary["n_trap_rollouts"] >= args.target_trap_rollouts:
                break
            n = min(args.batch_size, remaining)
            batch_seed = args.seed + next_batch
            cfg = RolloutConfig(
                n_rollouts=n,
                horizon=args.horizon,
                seed=batch_seed,
                action_source=args.action_source,
            )
            print(
                f"Batch {next_batch:04d}: collecting n={n}, seed={batch_seed}", flush=True
            )
            trajectories = wrapper.sample_rollouts(cfg)
            values, trap, fingerprints = _pack_trajectories(trajectories)
            path = output_dir / f"batch_{next_batch:04d}.npz"
            np.savez_compressed(
                path,
                schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
                ap_keys=np.asarray(AP_KEYS, dtype="U32"),
                values=values,
                trap=trap,
                fingerprints=fingerprints,
                seed=np.asarray(batch_seed, dtype=np.int64),
            )
            batches = _read_batches(output_dir)
            manifest = _write_manifest(
                output_dir, args, checkpoint, config, batches, started_at
            )
            aggregate = manifest["summary"]
            print(
                f"  aggregate={aggregate['n_rollouts']}  "
                f"trap={aggregate['n_trap_rollouts']} "
                f"({aggregate['trap_rate']:.2%})  "
                f"unique_trap={aggregate['n_unique_trap_trajectories']}",
                flush=True,
            )
            next_batch += 1
    finally:
        wrapper.close()

    return _write_manifest(output_dir, args, checkpoint, config, batches, started_at)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="/home/bot/CarDreamer/logdir/carla_four_lane/checkpoint.ckpt",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--output-dir",
        default="artifacts/l2_hazard_training_data",
    )
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument(
        "--action-source", choices=("actor", "actor_epsilon"), default="actor"
    )
    parser.add_argument("--epsilon-random", type=float, default=0.20)
    parser.add_argument(
        "--anchor-strategy", choices=("uniform", "low_hazard"), default="uniform"
    )
    parser.add_argument("--low-hazard-fraction", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-rollouts", type=int, default=1000)
    parser.add_argument("--target-trap-rollouts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=91000)
    parser.add_argument("--burn-in", type=int, default=3)
    parser.add_argument("--replay-pool-rollouts", type=int, default=200)
    parser.add_argument("--replay-pool-multiplier", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in (
        "horizon",
        "batch_size",
        "max_rollouts",
        "target_trap_rollouts",
        "replay_pool_rollouts",
        "replay_pool_multiplier",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 <= args.epsilon_random <= 1.0:
        raise ValueError("--epsilon-random must lie in [0, 1]")
    if not 0.0 < args.low_hazard_fraction <= 1.0:
        raise ValueError("--low-hazard-fraction must lie in (0, 1]")
    manifest = collect(args)
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    if not manifest["stopping_rule_met"]:
        print(
            "WARNING: maximum rollout budget reached before the trap target; "
            "use --resume with a larger --max-rollouts.",
            flush=True,
        )


if __name__ == "__main__":
    main()
