"""First real CarDreamer distribution-scoped L3 experiment.

Specification: ``ltl_patrol = GF(zone_a)`` on carla_roundabout.
This is genuine unbounded recurrence syntax, unlike the historical bounded
``stl_sequential_zones`` probe.  The result remains distribution-scoped: it
does not invoke the paper's global Lipschitz covering or Ville warrant.

The statistical splits use independent CARLA episodes.  Certificate/operator
training may use every periodic anchor in their episodes.  Residual calibration
and warrant validation use exactly one seeded random anchor per episode, so CP
and split-conformal samples are not silently treated as independent when they
come from the same physical episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from dataclasses import asdict

import numpy as np

sys.path.insert(0, "/home/bot/CarDreamer")
sys.path.insert(0, "/home/bot/SafeWorld")

from cardreamer.lbsm_adapter import (
    CarDreamerLatentAnchor,
    make_roundabout_region_predicates,
    sample_product_successor_batches,
    select_core_points,
)
from cardreamer.probe_common import (
    RING_C,
    ROUNDABOUT_CKPT_FROZEN,
    ROUNDABOUT_CONFIG_FROZEN,
    wrap_obs,
)


def _reset_with_retry(env, retries: int = 3):
    for attempt in range(retries):
        try:
            return env.reset()
        except RuntimeError:
            if attempt == retries - 1:
                raise
            time.sleep(5.0)


def _drive_episode(jax_agent, env, episode_id: int, anchor_every: int):
    raw = _reset_with_retry(env)
    policy_state = None
    done = False
    step = 0
    posterior_deter, posterior_xy = [], []
    anchors: list[CarDreamerLatentAnchor] = []
    while not done:
        outputs, policy_state = jax_agent.policy(
            wrap_obs(raw, step == 0), policy_state, mode="eval"
        )
        latent = policy_state[0][0]
        action = np.asarray(outputs["action"])[0].astype(np.float32)
        raw, _reward, done, info = env.step(int(np.argmax(action)))
        xy = np.asarray(
            [float(info.get("ego_x", np.nan)), float(info.get("ego_y", np.nan))],
            dtype=np.float32,
        )
        if np.isfinite(xy).all():
            deter = np.asarray(latent["deter"])[0]
            posterior_deter.append(deter)
            posterior_xy.append(xy)
            if step % anchor_every == 0:
                anchors.append(CarDreamerLatentAnchor(
                    deter=deter.copy(),
                    stoch=np.asarray(latent["stoch"])[0].copy(),
                    logit=np.asarray(latent["logit"])[0].copy(),
                    sample_id=f"episode-{episode_id}-step-{step}",
                    episode_id=episode_id,
                    real_xy=xy.copy(),
                ))
        step += 1
    return posterior_deter, posterior_xy, anchors, step


def _checkpoint_scope(checkpoint: str, args) -> str:
    stat = os.stat(checkpoint)
    payload = (
        f"{pathlib.Path(checkpoint).resolve()}:{stat.st_size}:{stat.st_mtime_ns}:"
        f"carla_roundabout:ltl_patrol:anchor_every={args.anchor_every}:seed={args.seed}"
    )
    return "cardreamer-roundabout-patrol-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _one_per_episode(episodes, rng):
    selected = []
    for candidates in episodes:
        if not candidates:
            raise RuntimeError("an episode produced no eligible L3 anchor")
        selected.append(candidates[int(rng.integers(len(candidates)))])
    return selected


def _save_anchor_cache(path, *, probe_h, probe_xy, heldout_deter, heldout_xy,
                       all_anchors, split_sizes):
    np.savez_compressed(
        path,
        probe_h=np.stack(probe_h),
        probe_xy=np.stack(probe_xy),
        heldout_deter=np.stack(heldout_deter),
        heldout_xy=np.stack(heldout_xy),
        deter=np.stack([anchor.deter for anchor in all_anchors]),
        stoch=np.stack([anchor.stoch for anchor in all_anchors]),
        logit=np.stack([anchor.logit for anchor in all_anchors]),
        sample_id=np.asarray([anchor.sample_id for anchor in all_anchors]),
        episode_id=np.asarray([anchor.episode_id for anchor in all_anchors]),
        real_xy=np.stack([anchor.real_xy for anchor in all_anchors]),
        split_sizes=np.asarray(split_sizes),
    )


def _load_anchor_cache(path):
    data = np.load(path, allow_pickle=False)
    anchors = [CarDreamerLatentAnchor(
        deter=data["deter"][i],
        stoch=data["stoch"][i],
        logit=data["logit"][i],
        sample_id=str(data["sample_id"][i]),
        episode_id=int(data["episode_id"][i]),
        real_xy=data["real_xy"][i],
    ) for i in range(len(data["deter"]))]
    return (
        data["probe_h"], data["probe_xy"], data["heldout_deter"],
        data["heldout_xy"], anchors, tuple(map(int, data["split_sizes"])),
    )


def main(args):
    # XLA autotuning is a known reproducibility source in this repository.
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true")
    import jax
    import car_dreamer
    from sklearn.linear_model import Ridge

    from configs.settings import RolloutConfig
    from core.lbsm import run_distributional_l3_pipeline
    from wrappers.cardreamer_wrapper import CarDreamerWrapper

    total_l3_episodes = (
        args.certificate_episodes + args.operator_episodes
        + args.residual_episodes + args.warrant_episodes
    )
    print("=" * 72, flush=True)
    print("CarDreamer distributional L3: ltl_patrol = GF(zone_a)", flush=True)
    print("NOT Theorem-5.6 global L3; no RSSM Lipschitz constant is used", flush=True)
    print("=" * 72, flush=True)
    print(f"[1/5] Loading checkpoint {args.checkpoint}", flush=True)
    config = RolloutConfig(n_rollouts=4, horizon=1, seed=args.seed, action_source="actor")
    wrapper = CarDreamerWrapper(config, use_replay_start=False)
    wrapper.load(checkpoint_path=args.checkpoint, config_path=args.config)
    jax_agent = wrapper._jax_agent

    if args.resume_cache and os.path.exists(args.anchor_cache):
        print(f"[2/5] Loading independent-episode anchor cache {args.anchor_cache}", flush=True)
        (probe_h, probe_xy, heldout_deter, heldout_xy,
         all_anchors, split_sizes) = _load_anchor_cache(args.anchor_cache)
    else:
        print(f"[2/5] Connecting CARLA :{args.port}; collecting "
              f"{args.probe_episodes} probe + {total_l3_episodes} L3 episodes", flush=True)
        env, _ = car_dreamer.create_task(
            "carla_roundabout", ["--env.world.carla_port", str(args.port)]
        )
        probe_h, probe_xy = [], []
        for episode in range(args.probe_episodes):
            hs, xys, _anchors, length = _drive_episode(
                jax_agent, env, episode, args.anchor_every
            )
            probe_h.extend(hs)
            probe_xy.extend(xys)
            print(f"  probe episode {episode + 1}/{args.probe_episodes}: len={length}", flush=True)
        if len(probe_h) < 20:
            raise RuntimeError("too few finite-position posterior samples to fit the position probe")

        episode_anchors = []
        heldout_deter, heldout_xy = [], []
        for offset in range(total_l3_episodes):
            episode_id = args.probe_episodes + offset
            _hs, _xys, anchors, length = _drive_episode(
                jax_agent, env, episode_id, args.anchor_every
            )
            episode_anchors.append(anchors)
            heldout_deter.extend([anchor.deter for anchor in anchors])
            heldout_xy.extend([anchor.real_xy for anchor in anchors])
            print(f"  L3 episode {offset + 1}/{total_l3_episodes}: "
                  f"len={length} anchors={len(anchors)}", flush=True)
        env.close()

        cursor = 0
        cert_eps = episode_anchors[cursor : cursor + args.certificate_episodes]
        cursor += args.certificate_episodes
        operator_eps = episode_anchors[cursor : cursor + args.operator_episodes]
        cursor += args.operator_episodes
        residual_eps = episode_anchors[cursor : cursor + args.residual_episodes]
        cursor += args.residual_episodes
        warrant_eps = episode_anchors[cursor : cursor + args.warrant_episodes]
        rng = np.random.default_rng(args.seed)
        certificate_anchors = [anchor for episode in cert_eps for anchor in episode]
        operator_anchors = [anchor for episode in operator_eps for anchor in episode]
        residual_anchors = _one_per_episode(residual_eps, rng)
        warrant_anchors = _one_per_episode(warrant_eps, rng)
        all_anchors = certificate_anchors + operator_anchors + residual_anchors + warrant_anchors
        split_sizes = tuple(map(len, (
            certificate_anchors, operator_anchors, residual_anchors, warrant_anchors,
        )))
        _save_anchor_cache(
            args.anchor_cache,
            probe_h=probe_h,
            probe_xy=probe_xy,
            heldout_deter=heldout_deter,
            heldout_xy=heldout_xy,
            all_anchors=all_anchors,
            split_sizes=split_sizes,
        )
        print(f"  anchor cache saved: {args.anchor_cache}", flush=True)

    position_probe = Ridge(alpha=10.0).fit(np.stack(probe_h), np.stack(probe_xy))

    heldout_pred = position_probe.predict(np.stack(heldout_deter))
    heldout_xy_arr = np.stack(heldout_xy)
    probe_errors = np.linalg.norm(heldout_pred - heldout_xy_arr, axis=1)
    true_zone = np.linalg.norm(heldout_xy_arr - RING_C, axis=1) < args.accepting_radius
    predicted_zone = np.linalg.norm(heldout_pred - RING_C, axis=1) < args.accepting_radius
    zone_accuracy = float(np.mean(true_zone == predicted_zone))
    print(f"  held-out posterior probe: RMSE={np.sqrt(np.mean(probe_errors**2)):.3f}m "
          f"p90={np.percentile(probe_errors, 90):.3f}m zone_accuracy={zone_accuracy:.4f}",
          flush=True)

    print(f"[3/5] Sampling {args.kappa} one-step successors per anchor; "
          f"splits cert/op/residual/warrant={split_sizes}", flush=True)
    scope = _checkpoint_scope(args.checkpoint, args)
    all_batches, layout = sample_product_successor_batches(
        jax_agent=jax_agent,
        anchors=all_anchors,
        position_probe=position_probe,
        kappa=args.kappa,
        sampling_scope=scope,
        accepting_radius=args.accepting_radius,
        chunk_size=args.successor_chunk,
    )
    n_cert, n_op, n_resid, n_warrant = split_sizes
    certificate_batches = all_batches[:n_cert]
    operator_batches = all_batches[n_cert : n_cert + n_op]
    residual_batches = all_batches[n_cert + n_op : n_cert + n_op + n_resid]
    warrant_batches = all_batches[-n_warrant:]
    is_accepting, in_retention = make_roundabout_region_predicates(
        layout,
        accepting_radius=args.accepting_radius,
        retention_radius=args.retention_radius,
    )
    core_inside, core_boundary = select_core_points(
        certificate_batches, layout, retention_radius=args.retention_radius,
    )

    print(f"[4/5] Training W/U and H_W/H_U on CPU; product_dim={layout.state_dim}", flush=True)
    result = run_distributional_l3_pipeline(
        certificate_train_batches=certificate_batches,
        operator_train_batches=operator_batches,
        residual_calibration_batches=residual_batches,
        warrant_batches=warrant_batches,
        is_accepting=is_accepting,
        in_retention_set=in_retention,
        core_inside=core_inside,
        core_boundary=core_boundary,
        core_inside_upper=args.B_U * 0.25,
        core_boundary_lower=args.B_U * 0.75,
        seed=args.seed,
        B_W=args.B_W,
        B_U=args.B_U,
        eps_W=args.eps_W,
        eps_U=args.eps_U,
        certificate_hidden_dim=args.hidden_dim,
        certificate_epochs=args.certificate_epochs,
        certificate_successors_per_anchor=args.certificate_kappa,
        operator_hidden_dim=args.hidden_dim,
        operator_epochs=args.operator_epochs,
        alpha_W=args.alpha,
        alpha_U=args.alpha,
        delta_mc_W=args.delta_mc,
        delta_mc_U=args.delta_mc,
        gamma=args.gamma,
        warrant_threshold=args.warrant_threshold,
    )

    verification = result.verification
    # AP-probe gating is separate from certificate statistics and can only
    # downgrade the report.
    report_verdict = verification.verdict
    if zone_accuracy < args.min_zone_accuracy:
        report_verdict = "INCONCLUSIVE_AP_PROBE"
    print("[5/5] Result", flush=True)
    print(verification.summary(), flush=True)
    print(f"  report_verdict={report_verdict}", flush=True)
    print(f"  q_W={result.calibration.q_W:.6f} eta_W={result.calibration.eta_W:.6f} "
          f"q_U={result.calibration.q_U:.6f} eta_U={result.calibration.eta_U:.6f}",
          flush=True)
    print(f"  conditions={verification.n_surrogate_conditions_met}/"
          f"{verification.n_warrant}; failed={verification.failed_anchor_ids}", flush=True)

    artifact = {
        "method": "L3_DISTRIBUTIONAL_POST_EXPECTATION",
        "global_theorem_5_6": False,
        "uses_dynamics_lipschitz": False,
        "spec": "ltl_patrol",
        "formula": "GF(zone_a)",
        "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
        "sampling_scope": scope,
        "split_sizes": split_sizes,
        "product_state_dim": layout.state_dim,
        "heldout_probe_rmse": float(np.sqrt(np.mean(probe_errors**2))),
        "heldout_probe_p90": float(np.percentile(probe_errors, 90)),
        "heldout_zone_accuracy": zone_accuracy,
        "report_verdict": report_verdict,
        "verification": asdict(verification),
        "calibration": {
            key: value for key, value in asdict(result.calibration).items()
            if "fingerprint" not in key
        },
        "certificate_final_loss": result.certificates.loss_history[-1],
        "operator_final_loss": result.operators.loss_history[-1],
        "args": vars(args),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    print(f"  artifact={args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=2100)
    parser.add_argument("--checkpoint", default=ROUNDABOUT_CKPT_FROZEN)
    parser.add_argument("--config", default=ROUNDABOUT_CONFIG_FROZEN)
    parser.add_argument("--probe-episodes", type=int, default=5)
    parser.add_argument("--certificate-episodes", type=int, default=5)
    parser.add_argument("--operator-episodes", type=int, default=5)
    parser.add_argument("--residual-episodes", type=int, default=20)
    parser.add_argument("--warrant-episodes", type=int, default=40)
    parser.add_argument("--anchor-every", type=int, default=10)
    parser.add_argument("--kappa", type=int, default=128)
    parser.add_argument("--certificate-kappa", type=int, default=16)
    parser.add_argument("--successor-chunk", type=int, default=4)
    parser.add_argument("--accepting-radius", type=float, default=24.9)
    parser.add_argument("--retention-radius", type=float, default=30.0)
    parser.add_argument("--B-W", dest="B_W", type=float, default=0.5)
    parser.add_argument("--B-U", dest="B_U", type=float, default=0.5)
    parser.add_argument("--eps-W", dest="eps_W", type=float, default=0.005)
    parser.add_argument("--eps-U", dest="eps_U", type=float, default=0.005)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--certificate-epochs", type=int, default=100)
    parser.add_argument("--operator-epochs", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta-mc", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--warrant-threshold", type=float, default=0.80)
    parser.add_argument("--min-zone-accuracy", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--anchor-cache", default="/tmp/cardreamer_l3_patrol_anchors.npz")
    parser.add_argument("--resume-cache", action="store_true")
    parser.add_argument("--output", default="/tmp/cardreamer_l3_distributional_patrol.json")
    main(parser.parse_args())
