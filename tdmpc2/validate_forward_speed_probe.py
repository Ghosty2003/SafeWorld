"""Collect and validate a TD-MPC2 walker-walk forward-speed probe.

The target is dm_control's signed centre-of-mass horizontal velocity in m/s,
``Physics.horizontal_velocity()``, which is the signal used by walker-walk's
reward.  Validation has two disjoint stages:

C0: nested episode-grouped CV on real posterior latents.  Eight MPC and seven
    random-policy episodes are the defaults; adjacent frames never cross a
    train/test fold.
C1: exact physics-state anchors from eight additional MPC episodes.  The model
    imagines with MPC, then the exact recorded actions are replayed in dm_control
    and probe predictions are compared with the real physical signal.

The C1 error is a combined probe + world-model error.  It is reported separately
from C0 and must not be described as pure probe error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
from typing import Any

import numpy as np

from wrappers.tdmpc2_probes import (
    fit_forward_speed_probe,
    make_forward_speed_ap_extractor,
    walker_forward_speed_from_physics,
)
from wrappers.tdmpc2_wrapper import TDMPC2Wrapper


ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
WALK_TARGET_MPS = 1.0
# Fixed before observing this run.  C1's 0.5 m/s limit is walker-walk's own
# reward tolerance margin (_WALK_SPEED / 2), not a post-hoc fitted threshold.
C0_MIN_R2 = 0.95
C0_MAX_MAE_MPS = 0.10
C0_MAX_MPC_P90_MPS = 0.20
C1_MAX_DEPTH100_P90_MPS = 0.50
C1_MIN_DEPTH100_THRESHOLD_AGREEMENT = 0.80


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_commit(tdmpc2_src: pathlib.Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(tdmpc2_src), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _load_wrapper(args: argparse.Namespace, seed: int) -> TDMPC2Wrapper:
    wrapper = TDMPC2Wrapper(tdmpc2_src=args.tdmpc2_src)
    wrapper.load(checkpoint=args.checkpoint, task="walker-walk", seed=seed)
    return wrapper


def _collect_real_episode(
    wrapper: TDMPC2Wrapper,
    *,
    policy: str,
    episode_id: int,
    steps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs = wrapper._env.reset()
    z_rows, speed_rows, root_velocity_rows, height_rows = [], [], [], []
    for step in range(steps):
        z = wrapper.encode(obs)
        z_rows.append(z[0].detach().cpu().numpy().copy())
        speed_rows.append(walker_forward_speed_from_physics(wrapper._physics))
        root_velocity_rows.append(float(wrapper._physics.velocity()[0]))
        height_rows.append(float(wrapper._physics.torso_height()))
        if policy == "mpc_plan":
            action = wrapper.plan_action(z, t0=(step == 0), eval_mode=True)[0].cpu()
        elif policy == "random":
            action = torch.as_tensor(
                rng.uniform(-1.0, 1.0, size=wrapper._cfg.action_dim),
                dtype=torch.float32,
            )
        else:
            raise ValueError(f"unknown collection policy {policy!r}")
        obs, _, done, _ = wrapper._env.step(action)
        if done and step + 1 < steps:
            raise RuntimeError(f"episode {episode_id} terminated early at step {step}")
    n = len(z_rows)
    return {
        "z": np.stack(z_rows).astype(np.float32),
        "forward_speed": np.asarray(speed_rows, dtype=np.float32),
        "root_velocity": np.asarray(root_velocity_rows, dtype=np.float32),
        "height": np.asarray(height_rows, dtype=np.float32),
        "episode_id": np.full(n, episode_id, dtype=np.int32),
        "policy": np.full(n, policy),
    }


def collect_c0(args: argparse.Namespace, path: pathlib.Path) -> None:
    wrapper = _load_wrapper(args, args.seed)
    episodes = []
    try:
        schedule = ["mpc_plan"] * args.c0_mpc_episodes + ["random"] * args.c0_random_episodes
        for episode_id, policy in enumerate(schedule):
            episode = _collect_real_episode(
                wrapper,
                policy=policy,
                episode_id=episode_id,
                steps=args.episode_steps,
                seed=args.seed * 1000 + episode_id,
            )
            episodes.append(episode)
            speed = episode["forward_speed"]
            print(
                f"  C0 episode {episode_id + 1}/{len(schedule)} policy={policy} "
                f"speed=[{speed.min():+.3f},{speed.max():+.3f}] mean={speed.mean():+.3f}",
                flush=True,
            )
    finally:
        wrapper.close()
    merged = {key: np.concatenate([episode[key] for episode in episodes]) for key in episodes[0]}
    np.savez_compressed(path, **merged)


def _new_ridge(alpha: float):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))


def _choose_alpha(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GroupKFold

    n_splits = min(4, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("alpha selection requires at least two episode groups")
    splitter = GroupKFold(n_splits=n_splits)
    scores = []
    for alpha in ALPHA_GRID:
        fold_mae = []
        for train, test in splitter.split(x, y, groups):
            model = _new_ridge(alpha).fit(x[train], y[train])
            fold_mae.append(mean_absolute_error(y[test], model.predict(x[test])))
        scores.append((float(np.mean(fold_mae)), alpha))
    return min(scores)[1]


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import r2_score

    error = np.asarray(prediction) - np.asarray(y)
    absolute = np.abs(error)
    true_fast = np.asarray(y) >= WALK_TARGET_MPS
    predicted_fast = np.asarray(prediction) >= WALK_TARGET_MPS
    return {
        "n": int(len(y)),
        "r2": float(r2_score(y, prediction)),
        "mae_mps": float(np.mean(absolute)),
        "rmse_mps": float(np.sqrt(np.mean(error ** 2))),
        "p90_abs_mps": float(np.percentile(absolute, 90)),
        "p99_abs_mps": float(np.percentile(absolute, 99)),
        "max_abs_mps": float(np.max(absolute)),
        "walk_threshold_agreement": float(np.mean(true_fast == predicted_fast)),
        "true_walk_rate": float(np.mean(true_fast)),
        "predicted_walk_rate": float(np.mean(predicted_fast)),
    }


def validate_c0(dataset_path: pathlib.Path, model_path: pathlib.Path) -> tuple[Any, dict]:
    import joblib
    from sklearn.model_selection import GroupKFold

    data = np.load(dataset_path, allow_pickle=False)
    x = np.asarray(data["z"], dtype=np.float32)
    y = np.asarray(data["forward_speed"], dtype=np.float32)
    groups = np.asarray(data["episode_id"])
    policies = np.asarray(data["policy"])
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    predictions = np.empty_like(y)
    outer_alphas = []
    for train, test in splitter.split(x, y, groups):
        alpha = _choose_alpha(x[train], y[train], groups[train])
        outer_alphas.append(alpha)
        predictions[test] = _new_ridge(alpha).fit(x[train], y[train]).predict(x[test])

    final_alpha = _choose_alpha(x, y, groups)
    probe = fit_forward_speed_probe(str(dataset_path), alpha=final_alpha, standardize=True)
    joblib.dump(probe, model_path)
    result = {
        "protocol": "nested_episode_grouped_cv",
        "alpha_grid": list(ALPHA_GRID),
        "outer_selected_alphas": outer_alphas,
        "final_alpha": final_alpha,
        "overall": _metrics(y, predictions),
        "mpc_plan": _metrics(y[policies == "mpc_plan"], predictions[policies == "mpc_plan"]),
        "random": _metrics(y[policies == "random"], predictions[policies == "random"]),
        "per_episode": {
            str(int(group)): _metrics(y[groups == group], predictions[groups == group])
            for group in np.unique(groups)
        },
    }
    result["passed"] = bool(
        result["overall"]["r2"] >= C0_MIN_R2
        and result["overall"]["mae_mps"] <= C0_MAX_MAE_MPS
        and result["mpc_plan"]["p90_abs_mps"] <= C0_MAX_MPC_P90_MPS
    )
    return probe, result


def collect_c1_anchors(args: argparse.Namespace, path: pathlib.Path) -> None:
    import torch

    wrapper = _load_wrapper(args, args.seed + 1000)
    physics_states, episode_ids, steps, speeds = [], [], [], []
    try:
        for episode in range(args.c1_episodes):
            torch.manual_seed(args.seed * 10000 + episode)
            obs = wrapper._env.reset()
            for step in range(args.episode_steps):
                z = wrapper.encode(obs)
                if step % args.anchor_every == 0:
                    physics_states.append(wrapper._physics.get_state().copy())
                    episode_ids.append(episode)
                    steps.append(step)
                    speeds.append(walker_forward_speed_from_physics(wrapper._physics))
                action = wrapper.plan_action(z, t0=(step == 0), eval_mode=True)[0].cpu()
                obs, _, done, _ = wrapper._env.step(action)
                if done and step + 1 < args.episode_steps:
                    raise RuntimeError(f"C1 source episode {episode} terminated early")
            print(
                f"  C1 anchor episode {episode + 1}/{args.c1_episodes}: "
                f"anchors={sum(value == episode for value in episode_ids)}",
                flush=True,
            )
    finally:
        wrapper.close()
    np.savez_compressed(
        path,
        physics_state=np.stack(physics_states),
        episode_id=np.asarray(episode_ids, dtype=np.int32),
        step=np.asarray(steps, dtype=np.int32),
        anchor_forward_speed=np.asarray(speeds, dtype=np.float32),
    )


def collect_c1_predictions(
    args: argparse.Namespace,
    probe: Any,
    probe_sha256: str,
    anchors_path: pathlib.Path,
    predictions_path: pathlib.Path,
) -> None:
    import torch

    anchors = np.load(anchors_path, allow_pickle=False)
    states = np.asarray(anchors["physics_state"])
    model_values, real_values = [], []
    wrapper = _load_wrapper(args, args.seed + 2000)
    model_extractor = make_forward_speed_ap_extractor(probe)

    def real_extractor(_obs):
        return {"forward_speed": walker_forward_speed_from_physics(wrapper._physics)}

    try:
        for start in range(0, len(states), args.c1_chunk):
            stop = min(start + args.c1_chunk, len(states))
            torch.manual_seed(args.seed * 100000 + start)
            pairs, _ = wrapper.sample_paired_rollouts_from_states(
                states[start:stop],
                action_source="mpc_plan",
                horizon=args.c1_horizon,
                model_ap_extractor=model_extractor,
                environment_ap_extractor=real_extractor,
            )
            for model_traj, real_traj in pairs:
                model_values.append([step["forward_speed"] for step in model_traj])
                real_values.append([step["forward_speed"] for step in real_traj])
            print(f"  C1 paired anchors {stop}/{len(states)}", flush=True)
    finally:
        wrapper.close()
    np.savez_compressed(
        predictions_path,
        model_forward_speed=np.asarray(model_values, dtype=np.float32),
        real_forward_speed=np.asarray(real_values, dtype=np.float32),
        episode_id=anchors["episode_id"],
        anchor_step=anchors["step"],
        probe_sha256=np.asarray(probe_sha256),
    )


def _c1_cache_matches_probe(path: pathlib.Path, probe_sha256: str) -> bool:
    if not path.exists():
        return False
    data = np.load(path, allow_pickle=False)
    return "probe_sha256" in data and str(data["probe_sha256"]) == probe_sha256


def validate_c1(path: pathlib.Path) -> dict:
    data = np.load(path, allow_pickle=False)
    model = np.asarray(data["model_forward_speed"])
    real = np.asarray(data["real_forward_speed"])
    if model.shape != real.shape or model.ndim != 2:
        raise ValueError("C1 model/real predictions must have identical (n,horizon) shapes")
    depths = sorted(depth for depth in {1, 5, 10, 25, 50, model.shape[1]} if depth <= model.shape[1])
    all_depth_metrics = [
        _metrics(real[:, depth - 1], model[:, depth - 1])
        for depth in range(1, model.shape[1] + 1)
    ]
    by_depth = {str(depth): all_depth_metrics[depth - 1] for depth in depths}
    max_prefix = 0
    for depth, metrics in enumerate(all_depth_metrics, start=1):
        if (
            metrics["p90_abs_mps"] <= C1_MAX_DEPTH100_P90_MPS
            and metrics["walk_threshold_agreement"] >= C1_MIN_DEPTH100_THRESHOLD_AGREEMENT
        ):
            max_prefix = depth
        else:
            break
    result = {
        "protocol": "exact_anchor_model_mpc_actions_replayed_open_loop_in_environment",
        "n_anchors": int(model.shape[0]),
        "horizon": int(model.shape[1]),
        "combined_probe_and_world_model_error": True,
        "max_prefix_depth_meeting_fixed_c1_criteria": max_prefix,
        "all_steps": _metrics(real.reshape(-1), model.reshape(-1)),
        "by_depth": by_depth,
    }
    final = by_depth[str(model.shape[1])]
    result["passed"] = bool(
        final["p90_abs_mps"] <= C1_MAX_DEPTH100_P90_MPS
        and final["walk_threshold_agreement"] >= C1_MIN_DEPTH100_THRESHOLD_AGREEMENT
    )
    return result


def main(args: argparse.Namespace) -> None:
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    c0_path = output_dir / "c0_real_posterior.npz"
    model_path = output_dir / "forward_speed_probe.joblib"
    anchors_path = output_dir / "c1_exact_mpc_anchors.npz"
    predictions_path = output_dir / "c1_paired_predictions.npz"
    result_path = output_dir / "validation.json"

    print("[1/4] C0 real-posterior collection", flush=True)
    if not (args.resume and c0_path.exists()):
        collect_c0(args, c0_path)
    else:
        print(f"  resume {c0_path}", flush=True)

    print("[2/4] Nested episode-grouped C0 validation", flush=True)
    probe, c0 = validate_c0(c0_path, model_path)
    print(
        f"  C0 r2={c0['overall']['r2']:.4f} mae={c0['overall']['mae_mps']:.4f} "
        f"mpc_p90={c0['mpc_plan']['p90_abs_mps']:.4f} passed={c0['passed']}",
        flush=True,
    )

    print("[3/4] Independent exact MPC anchors", flush=True)
    if not (args.resume and anchors_path.exists()):
        collect_c1_anchors(args, anchors_path)
    else:
        print(f"  resume {anchors_path}", flush=True)

    print("[4/4] Matched-action C1 validation", flush=True)
    probe_sha256 = _sha256(model_path)
    if not (args.resume and _c1_cache_matches_probe(predictions_path, probe_sha256)):
        collect_c1_predictions(
            args, probe, probe_sha256, anchors_path, predictions_path,
        )
    else:
        print(f"  resume {predictions_path}", flush=True)
    c1 = validate_c1(predictions_path)
    final_depth = c1["by_depth"][str(c1["horizon"])]
    print(
        f"  C1 depth={c1['horizon']} p90={final_depth['p90_abs_mps']:.4f} "
        f"threshold_agreement={final_depth['walk_threshold_agreement']:.4f} "
        f"passed={c1['passed']}",
        flush=True,
    )

    if c0["passed"] and c1["passed"]:
        verdict = "VALIDATED_WALKER_WALK_SEED3_MPC"
    elif c0["passed"]:
        verdict = "C0_VALID_C1_MODEL_TRANSFER_INCONCLUSIVE"
    else:
        verdict = "INCONCLUSIVE_PROBE"
    result = {
        "verdict": verdict,
        "target": {
            "name": "forward_speed",
            "definition": "dm_control walker Physics.horizontal_velocity()",
            "signed": True,
            "unit": "m/s",
            "walk_target_mps": WALK_TARGET_MPS,
            "not_observation_index_15": True,
        },
        "scope": {
            "task": "walker-walk",
            "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
            "checkpoint_sha256": _sha256(pathlib.Path(args.checkpoint)),
            "checkpoint_seed": args.seed,
            "tdmpc2_source": str(pathlib.Path(args.tdmpc2_src).resolve()),
            "tdmpc2_commit": _source_commit(pathlib.Path(args.tdmpc2_src)),
            "action_source": "mpc_plan",
        },
        "fixed_acceptance_criteria": {
            "c0_min_r2": C0_MIN_R2,
            "c0_max_mae_mps": C0_MAX_MAE_MPS,
            "c0_max_mpc_p90_mps": C0_MAX_MPC_P90_MPS,
            "c1_max_depth100_p90_mps": C1_MAX_DEPTH100_P90_MPS,
            "c1_min_depth100_threshold_agreement": C1_MIN_DEPTH100_THRESHOLD_AGREEMENT,
        },
        "c0": c0,
        "c1": c1,
        "artifacts": {
            "c0_dataset": str(c0_path.resolve()),
            "probe_model": str(model_path.resolve()),
            "c1_anchors": str(anchors_path.resolve()),
            "c1_predictions": str(predictions_path.resolve()),
        },
        "args": vars(args),
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"  verdict={verdict}", flush=True)
    print(f"  result={result_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/bot/SafeWorld/models/walker-walk-3.pt")
    parser.add_argument("--tdmpc2-src", default="/tmp/tdmpc2_src/tdmpc2")
    parser.add_argument(
        "--output-dir",
        default="/home/bot/SafeWorld/artifacts/tdmpc2_forward_speed",
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--c0-mpc-episodes", type=int, default=8)
    parser.add_argument("--c0-random-episodes", type=int, default=7)
    parser.add_argument("--c1-episodes", type=int, default=8)
    parser.add_argument("--anchor-every", type=int, default=20)
    parser.add_argument("--c1-horizon", type=int, default=100)
    parser.add_argument("--c1-chunk", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
