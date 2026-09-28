"""Bounded STL under SafeDreamer's CCEPlanner in learned-model imagination.

This is model-scope statistical verification, NOT the paired model/environment
transfer theorem. No random-action error budget is reused. Formula semantics
and thresholds are preserved; insufficient trajectory lengths fail explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.settings import RolloutConfig
from core.stl_monitor import monitor_rollouts
from specs.stl_specs import get_stl_spec_by_id

GROUNDED_APS = {"hazard_dist", "goal_dist", "velocity", "near_obstacle"}
ALL_GROUNDED = ["stl_hazard_avoidance", "stl_speed_limit", "stl_safe_goal_reach",
                "stl_obstacle_response", "stl_gap_recovery", "stl_safe_flow_patrol",
                "stl_gap_response"]


def formula_requirements(f):
    """Return largest required state index and actual predicate dimensions."""
    kind = f["type"]
    if kind == "atom":
        return 0, {f["dim"]}
    if kind in ("not", "next", "always", "eventually"):
        horizon, dims = formula_requirements(f["child"])
    elif kind in ("and", "or", "implies", "until"):
        lh, ld = formula_requirements(f["left"])
        rh, rd = formula_requirements(f["right"])
        horizon, dims = max(lh, rh), ld | rd
    else:
        raise ValueError(f"Unsupported STL operator: {kind}")
    if kind in ("always", "eventually", "until"):
        a, b = f["a"], f["b"]
        if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a <= b:
            raise ValueError("Only finite integer-step STL intervals are supported")
        horizon += b
    elif kind == "next":
        horizon += 1
    return horizon, dims


def select_specs(ids):
    specs = []
    for sid in dict.fromkeys(ids):
        spec = get_stl_spec_by_id(sid)
        if spec is None:
            raise ValueError(f"Unknown bounded STL specification: {sid}")
        _, dims = formula_requirements(spec["formula"])
        if not dims <= GROUNDED_APS:
            raise ValueError(f"{sid}: APs not grounded in SafeDreamer: {dims - GROUNDED_APS}")
        specs.append(spec)
    if not specs:
        raise ValueError("Select at least one specification")
    return specs


def evaluate(spec, paths, gamma):
    from scipy.stats import beta
    required, dims = formula_requirements(spec["formula"])
    if not paths or not 0 < gamma < 1:
        raise ValueError("Nonempty data and gamma in (0,1) required")
    for path in paths:
        if len(path) <= required:
            raise ValueError(f"{spec['id']} needs states 0..{required}; refusing truncated evaluation")
        for state in path[:required + 1]:
            if any(d not in state or not math.isfinite(state[d]) for d in dims):
                raise ValueError("Missing/nonfinite grounded AP")
    mon = monitor_rollouts(spec["formula"], paths)
    k, n = mon.n_satisfied, len(paths)
    lower = float(beta.ppf(gamma, k, n-k+1)) if k else 0.0
    return dict(n=n, successes=k, empirical_rate=k/n, cp_lower=lower,
                confidence=1-gamma, rho_star=mon.rho_star,
                margins=mon.margins, observed_nonpositive_margins=n-k)


def collect(wrapper, horizon, count, seed, extra):
    cfg = RolloutConfig(horizon=horizon, n_rollouts=count, seed=seed,
                        action_source="policy", extra=extra)
    data = wrapper.sample_latent_rollouts(cfg)
    if data["action_source"] != "policy":
        raise RuntimeError("Refusing non-policy trajectories")
    if len(data["aps"]) != count or any(len(p) != horizon+1 for p in data["aps"]):
        raise RuntimeError("Expected H+1 states for exactly H imagination transitions")
    return dict(aps=data["aps"], actions=data["actions"].tolist(), seed=seed,
                action_source="policy", horizon=horizon)


def write_new(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def main(argv=None, *, default_specs=None, default_n=100):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", nargs="+", default=default_specs or ["stl_hazard_avoidance"])
    parser.add_argument("--horizon", type=int, help="Model transitions; auto-computed if omitted")
    parser.add_argument("--n-cal", type=int, default=default_n)
    parser.add_argument("--n-test", type=int, default=default_n)
    parser.add_argument("--gamma", type=float, default=.05)
    parser.add_argument("--threshold", type=float, default=.95)
    parser.add_argument("--seed", type=int, default=46501000)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--repo-root", default=os.environ.get("SAFEDREAMER_REPO_ROOT", str(ROOT.parent / "SafeDreamer")))
    parser.add_argument("--checkpoint", default=os.environ.get("SAFEDREAMER_CHECKPOINT_PATH", str(
        ROOT.parent / "SafeDreamer/checkpoint/20240307-010600_osrp_vector_safetygymcoor_SafetyPointGoal1-v0_0.ckpt")))
    parser.add_argument("--output", type=Path, required=True, help="New output directory (never overwritten)")
    args = parser.parse_args(argv)
    specs = select_specs(args.specs)
    required = max(formula_requirements(s["formula"])[0] for s in specs)
    horizon = max(1, required) if args.horizon is None else args.horizon
    if horizon < required or horizon < 1:
        parser.error(f"--horizon must be >= {max(1, required)}; formulas are not shortened")
    if min(args.n_cal, args.n_test) < 1 or not 0 < args.gamma < 1 or not 0 < args.threshold <= 1:
        parser.error("Positive split sizes, gamma in (0,1), threshold in (0,1] required")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    extra = dict(repo_root=args.repo_root, checkpoint_path=str(checkpoint), method="osrp_vector",
                 task="safetygymcoor_SafetyPointGoal1-v0", action_source="policy",
                 config_overrides={"jax": {"platform": args.device}})
    if args.device == "gpu":
        extra["config_overrides"]["jax"]["logical_gpus"] = 0
    args.output.mkdir(parents=True, exist_ok=False)
    # Bonferroni handles sharing a calibration batch across preselected specs.
    per_spec_gamma = args.gamma / len(specs)
    h = hashlib.sha256()
    with checkpoint.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    plan = dict(scope="MODEL_ONLY_BOUNDED_STL", action_source="policy", controller="CCEPlanner",
                world_model="SafeDreamer RSSM", checkpoint_sha256=h.hexdigest(), extra=extra,
                specs=specs, horizon=horizon, states_per_rollout=horizon+1,
                n_cal=args.n_cal, n_test=args.n_test, gamma=args.gamma,
                per_spec_gamma=per_spec_gamma, threshold=args.threshold,
                seeds=dict(calibration=args.seed, test=args.seed+1),
                assumptions="Independent same-distribution trajectories; policy and specs fixed before calibration. "
                            "Simulator resets supply initial observations; reset RNG not fully controlled by imagination seeds.",
                interpretation="CP lower bound of positive STL robustness, not environment transfer or infinite-horizon proof.")
    plan["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (Path(__file__).resolve(), ROOT/"wrappers/safedreamer_wrapper.py",
                                       ROOT/"core/stl_monitor.py", ROOT/"specs/stl_specs.py")}
    write_new(args.output / "plan.json", plan)
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    logging.basicConfig(level=logging.INFO)
    cfg = RolloutConfig(horizon=horizon, n_rollouts=1, seed=args.seed,
                        action_source="policy", extra=extra)
    print(f"MODEL_ONLY_BOUNDED_STL: CCEPlanner + RSSM, H={horizon}; no real-world warrant", flush=True)
    with SafeDreamerWrapper(cfg) as wrapper:
        wrapper.load()
        import jax
        if jax.default_backend() != args.device:
            raise RuntimeError(f"Requested {args.device}, got {jax.default_backend()}; no silent fallback")
        if wrapper._config.expl_behavior != "CCEPlanner":
            raise RuntimeError("Expected native osrp_vector CCEPlanner")
        write_new(args.output / "policy_config.json", dict(wrapper._config))
        cal = collect(wrapper, horizon, args.n_cal, args.seed, extra)
        write_new(args.output / "calibration.json", cal)
        prediction = {}
        for spec in specs:
            row = evaluate(spec, cal["aps"], per_spec_gamma)
            row["decision"] = "MODEL_STL_WARRANT" if row["cp_lower"] >= args.threshold else "NO_WARRANT"
            prediction[spec["id"]] = row
        # Persist predictions before inspecting independent test outcomes.
        write_new(args.output / "prediction.json", prediction)
        test = collect(wrapper, horizon, args.n_test, args.seed+1, extra)
        write_new(args.output / "test.json", test)
    report = {}
    for spec in specs:
        sid = spec["id"]
        row = evaluate(spec, test["aps"], per_spec_gamma)
        report[sid] = dict(calibration=prediction[sid], test=row,
                          empirical_rate_gap=row["empirical_rate"]-prediction[sid]["empirical_rate"],
                          note="Test proportions are observations, not guaranteed future batch counts; no reselection.")
        print(f"{sid}: calibration {prediction[sid]['successes']}/{args.n_cal}, "
              f"lower={prediction[sid]['cp_lower']:.4f}, threshold={args.threshold:.2f}, "
              f"{prediction[sid]['decision']}; test {row['successes']}/{args.n_test}")
    write_new(args.output / "report.json", dict(scope=plan["scope"], results=report))


if __name__ == "__main__":
    main()
