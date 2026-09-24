"""Collect full RSSM-state rollouts for ``F G(hazard_dist > 0)``.

The default mode deliberately produces biased training-only data using
low-margin replay anchors and epsilon exploration. ``--calibration`` instead
produces a separately labelled candidate split with uniform anchors and the
deployment actor. Raw NPZ batches and later PT models remain Git-ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import time

os.environ.setdefault(
    "XLA_FLAGS", "--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"
)

import numpy as np

from configs.settings import RolloutConfig
from wrappers.cardreamer_wrapper import CarDreamerWrapper


SCHEMA_VERSION = 1


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack_latent_rollouts(rollouts):
    hazard = np.asarray(
        [[float(step["hazard_dist"]) for step in item["aps"]] for item in rollouts],
        dtype=np.float32,
    )
    deter = np.stack([item["deter"] for item in rollouts]).astype(np.float16)
    stoch = np.stack([item["stoch"] for item in rollouts]).astype(np.float16)
    if deter.shape[:2] != hazard.shape or stoch.shape[:2] != hazard.shape:
        raise ValueError("AP/deter/stoch time axes are not aligned")
    trap = np.any(hazard <= 0.0, axis=1)
    fingerprints = []
    for i in range(len(hazard)):
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(hazard[i]).tobytes())
        digest.update(np.ascontiguousarray(deter[i]).tobytes())
        digest.update(np.ascontiguousarray(stoch[i]).tobytes())
        fingerprints.append(digest.hexdigest())
    return hazard, deter, stoch, trap, np.asarray(fingerprints, dtype="U64")


def recovered_after_failure(hazard: np.ndarray, min_recovery_suffix: int = 5) -> np.ndarray:
    """Flag trajectories whose final failure is followed by a safe suffix."""
    hazard = np.asarray(hazard)
    if hazard.ndim != 2:
        raise ValueError("hazard must have shape [trajectory, time]")
    recovered = np.zeros(len(hazard), dtype=bool)
    for i, row in enumerate(hazard):
        bad = np.flatnonzero(row <= 0.0)
        if len(bad) and len(row) - 1 - int(bad[-1]) >= min_recovery_suffix:
            recovered[i] = True
    return recovered


def read_batch_headers(output_dir: pathlib.Path, min_recovery_suffix: int = 5):
    result = []
    for path in sorted(output_dir.glob("batch_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            hazard = data["hazard_dist"].astype(np.float32, copy=False)
            recovered = recovered_after_failure(hazard, min_recovery_suffix)
            result.append(
                {
                    "path": path,
                    "n": int(len(data["hazard_dist"])),
                    "trap": int(np.sum(data["trap"])),
                    "fingerprints": data["fingerprints"].astype("U64").tolist(),
                    "trap_flags": data["trap"].astype(bool).tolist(),
                    "recovered_flags": recovered.tolist(),
                    "seed": int(data["seed"]),
                    "deter_dim": int(data["deter"].shape[-1]),
                    "stoch_size": int(np.prod(data["stoch"].shape[2:])),
                }
            )
    return result


def summarize(headers):
    fingerprints = [fp for batch in headers for fp in batch["fingerprints"]]
    trap_fingerprints = [
        fp
        for batch in headers
        for fp, is_trap in zip(batch["fingerprints"], batch["trap_flags"])
        if is_trap
    ]
    recovered_fingerprints = [
        fp
        for batch in headers
        for fp, is_recovered in zip(batch["fingerprints"], batch["recovered_flags"])
        if is_recovered
    ]
    n = sum(batch["n"] for batch in headers)
    n_trap = sum(batch["trap"] for batch in headers)
    return {
        "n_rollouts": n,
        "n_trap_rollouts": n_trap,
        "trap_rate": n_trap / n if n else 0.0,
        "n_unique_rollouts": len(set(fingerprints)),
        "n_unique_trap_rollouts": len(set(trap_fingerprints)),
        "n_recovered_rollouts": len(recovered_fingerprints),
        "n_unique_recovered_rollouts": len(set(recovered_fingerprints)),
    }


def stopping_rule_met(summary, args) -> bool:
    if args.calibration:
        return summary["n_rollouts"] >= args.max_rollouts
    if args.target_recovered_rollouts > 0:
        return summary["n_unique_recovered_rollouts"] >= args.target_recovered_rollouts
    return summary["n_unique_trap_rollouts"] >= args.target_trap_rollouts


def write_manifest(output_dir, args, checkpoint, config, headers, started):
    summary = summarize(headers)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "cardreamer_eventual_hazard_stability_full_rssm_training_data",
        "spec": "ltl_eventual_hazard_stability",
        "formula": "F G(hazard_dist > 0)",
        "data_role": (
            "heldout_lppm_calibration" if args.calibration
            else "lppm_training_candidate_only"
        ),
        "calibration_eligible": bool(args.calibration),
        "action_source": "actor" if args.calibration else "actor_epsilon",
        "epsilon_random": 0.0 if args.calibration else args.epsilon_random,
        "anchor_strategy": "uniform" if args.calibration else "low_hazard",
        "low_hazard_fraction": None if args.calibration else args.low_hazard_fraction,
        "horizon": args.horizon,
        "burn_in": args.burn_in,
        "base_seed": args.seed,
        "target_trap_rollouts": args.target_trap_rollouts,
        "target_recovered_rollouts": args.target_recovered_rollouts,
        "min_recovery_suffix": args.min_recovery_suffix,
        "max_rollouts": args.max_rollouts,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "size": checkpoint.stat().st_size,
        },
        "config": {
            "path": str(config.resolve()),
            "sha256": sha256_file(config),
            "size": config.stat().st_size,
        },
        "layout": {
            "deter_dim": headers[0]["deter_dim"] if headers else None,
            "stoch_size": headers[0]["stoch_size"] if headers else None,
            "product_state_dim_without_automaton": (
                headers[0]["deter_dim"] + headers[0]["stoch_size"]
                if headers else None
            ),
        },
        "summary": summary,
        "stopping_rule_met": stopping_rule_met(summary, args),
        "batches": [
            {
                "name": batch["path"].name,
                "sha256": sha256_file(batch["path"]),
                "n_rollouts": batch["n"],
                "n_trap_rollouts": batch["trap"],
                "seed": batch["seed"],
            }
            for batch in headers
        ],
        "elapsed_seconds": time.time() - started,
        "scope_note": (
            "Held-out uniform-anchor deployment-actor calibration candidate. "
            "Eligibility still requires fingerprint disjointness from every model-selection split."
            if args.calibration else
            "Biased training-only collection. A fresh uniform-anchor actor-policy "
            "split is mandatory for any formal calibration."
        ),
    }
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_dir / "manifest.json")
    return manifest


def resolve_config(checkpoint: pathlib.Path, explicit: str | None):
    if explicit:
        path = pathlib.Path(explicit)
    else:
        candidates = sorted(checkpoint.parent.glob("config_*.yaml"))
        if not candidates:
            raise FileNotFoundError("no config beside checkpoint; pass --config")
        path = candidates[-1]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="/home/bot/CarDreamer/logdir/carla_four_lane/checkpoint.ckpt",
    )
    parser.add_argument("--config")
    parser.add_argument("--output-dir", default="artifacts/l2_persistence_latents")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-rollouts", type=int, default=1000)
    parser.add_argument("--target-trap-rollouts", type=int, default=30)
    parser.add_argument(
        "--target-recovered-rollouts", type=int, default=0,
        help="if positive, stop on this many unique failure-then-recovery trajectories",
    )
    parser.add_argument("--min-recovery-suffix", type=int, default=5)
    parser.add_argument("--seed", type=int, default=92000)
    parser.add_argument("--epsilon-random", type=float, default=0.20)
    parser.add_argument("--low-hazard-fraction", type=float, default=0.25)
    parser.add_argument("--burn-in", type=int, default=3)
    parser.add_argument("--replay-pool-rollouts", type=int, default=40)
    parser.add_argument("--replay-pool-multiplier", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--calibration", action="store_true",
        help="collect held-out uniform-anchor deployment-actor rollouts",
    )
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = pathlib.Path(args.checkpoint)
    config = resolve_config(checkpoint, args.config)
    if args.min_recovery_suffix < 1:
        raise ValueError("--min-recovery-suffix must be positive")
    headers = read_batch_headers(output_dir, args.min_recovery_suffix) if args.resume else []
    if args.resume and (output_dir / "manifest.json").exists():
        existing_manifest = json.loads((output_dir / "manifest.json").read_text())
        if bool(existing_manifest.get("calibration_eligible")) != args.calibration:
            raise ValueError("--calibration role does not match resumed manifest")
    if not args.resume and any(output_dir.glob("batch_*.npz")):
        raise FileExistsError("output contains batches; pass --resume")
    started = time.time()
    summary = summarize(headers)
    print(f"Existing: {summary}", flush=True)
    if summary["n_rollouts"] >= args.max_rollouts or stopping_rule_met(summary, args):
        report = write_manifest(output_dir, args, checkpoint, config, headers, started)
        print(json.dumps(report["summary"], indent=2))
        return

    action_source = "actor" if args.calibration else "actor_epsilon"
    load_cfg = RolloutConfig(
        n_rollouts=args.replay_pool_rollouts,
        horizon=args.horizon,
        seed=args.seed,
        action_source=action_source,
    )
    wrapper = CarDreamerWrapper(
        load_cfg,
        use_replay_start=True,
        replay_burn_in_steps=args.burn_in,
        replay_pool_multiplier=args.replay_pool_multiplier,
        epsilon_random=0.0 if args.calibration else args.epsilon_random,
        replay_anchor_strategy="uniform" if args.calibration else "low_hazard",
        replay_low_hazard_fraction=args.low_hazard_fraction,
    )
    wrapper.load(checkpoint_path=str(checkpoint), config_path=str(config))
    if wrapper._replay_obs_pool is None:
        raise RuntimeError("replay anchors required")
    anchor_hazard = wrapper._replay_anchor_hazard
    print(
        f"Selected anchors n={len(anchor_hazard)} min={np.min(anchor_hazard):.3f} "
        f"p50={np.median(anchor_hazard):.3f} max={np.max(anchor_hazard):.3f}",
        flush=True,
    )

    batch_index = len(headers)
    try:
        while True:
            summary = summarize(headers)
            if summary["n_rollouts"] >= args.max_rollouts or stopping_rule_met(summary, args):
                break
            n = min(args.batch_size, args.max_rollouts - summary["n_rollouts"])
            batch_seed = args.seed + batch_index
            print(f"Batch {batch_index:04d}: n={n} seed={batch_seed}", flush=True)
            rollouts = wrapper.sample_latent_rollouts(
                RolloutConfig(
                    n_rollouts=n,
                    horizon=args.horizon,
                    seed=batch_seed,
                    action_source=action_source,
                )
            )
            hazard, deter, stoch, trap, fingerprints = pack_latent_rollouts(rollouts)
            recovered = recovered_after_failure(hazard, args.min_recovery_suffix)
            path = output_dir / f"batch_{batch_index:04d}.npz"
            np.savez_compressed(
                path,
                schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
                hazard_dist=hazard,
                deter=deter,
                stoch=stoch,
                trap=trap,
                recovered=recovered,
                fingerprints=fingerprints,
                seed=np.asarray(batch_seed, dtype=np.int64),
            )
            headers = read_batch_headers(output_dir, args.min_recovery_suffix)
            report = write_manifest(output_dir, args, checkpoint, config, headers, started)
            print(f"  aggregate: {report['summary']}", flush=True)
            batch_index += 1
    finally:
        wrapper.close()
    report = write_manifest(output_dir, args, checkpoint, config, headers, started)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
