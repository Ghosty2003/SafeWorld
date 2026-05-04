"""
run_pointbutton_benchmark.py

Run SAFEWORLD verification against the three PointButton1 DreamerV3 checkpoints
and print a structured comparison table.

Usage
─────
  cd SafeWorld_V2
  python run_pointbutton_benchmark.py                          # all three models
  python run_pointbutton_benchmark.py --model largedim         # single model
  python run_pointbutton_benchmark.py --n 50 --horizon 50
  python run_pointbutton_benchmark.py --method cegar
  python run_pointbutton_benchmark.py --output results_pb.json

Available models
────────────────
  largedim           largedim_usingdata.json   (deter=1024, offline+online)
  smalldim           smalldim_usingdata.json   (deter=512,  offline+online)
  smalldim_rl_only   smalldim_only_RL_data.json (deter=512,  RL only)

AP coverage for all three models
─────────────────────────────────
  SUPPORTED (latent stats)   hazard_dist, goal_dist
  N/A                        velocity, near_obstacle, near_human, zone_a/b/c, carrying

Supported specs (non-N/A)
──────────────────────────
  L1 — stl_hazard_avoidance, ltl_hazard_avoidance
  L2 — stl_safe_goal_reach,  ltl_safe_goal
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from configs.settings import RolloutConfig
from specs import ALL_SPECS, get_spec_by_id
from specs.spec_calibrator import load_env_config
from main import verify, VerifyConfig
from wrappers import PointButtonWrapper

# ─── model registry ───────────────────────────────────────────────────────────

MODELS: dict[str, str] = {
    "largedim":         "configs/environments/largedim_usingdata.json",
    "smalldim":         "configs/environments/smalldim_usingdata.json",
    "smalldim_rl_only": "configs/environments/smalldim_only_RL_data.json",
}

# ─── applicable specs (only APs we can actually provide) ─────────────────────

_REAL_APS  = {"hazard_dist", "goal_dist"}
_ZERO_APS  = {
    "velocity", "near_obstacle", "near_human",
    "zone_a", "zone_b", "zone_c", "carrying", "model_cost",
    # PointButton-specific names that may appear in specs
    "near_hazard", "near_gremlin", "at_button",
}


def _collect_dims(node: dict) -> set[str]:
    if not isinstance(node, dict):
        return set()
    if node.get("type") == "atom":
        return {node["dim"]}
    dims: set[str] = set()
    for v in node.values():
        if isinstance(v, dict):
            dims |= _collect_dims(v)
    return dims


def _applicability(spec: dict) -> tuple[str, list[str]]:
    dims     = _collect_dims(spec.get("formula", {}))
    zero_hit = sorted(dims & _ZERO_APS)
    if zero_hit:
        return "N/A", [f"requires {d} (always 0.0)" for d in zero_hit]
    return "SUPPORTED", []


def _emoji(verdict: str) -> str:
    return {"WARRANT": "✓", "STL_MARGIN": "~", "VIOLATION": "✗"}.get(verdict, "?")


# ─── per-model benchmark ──────────────────────────────────────────────────────

def run_one_model(
    model_key:   str,
    config_path: str,
    n_rollouts:  int,
    horizon:     int,
    seed:        int,
    device:      str,
    delta_cp:    float,
    delta_err:   float,
    c_hat:       float,
    spec_filter: str | None,
    verbose:     bool,
    method:      str,
    cegar_iterations:       int,
    soft_buchi_temperature: float,
    soft_buchi_epsilon:     float,
) -> list[dict[str, Any]]:

    env_config = load_env_config(config_path)

    roll_cfg = RolloutConfig(
        horizon=horizon, n_rollouts=n_rollouts, seed=seed,
        extra={"device": device},
    )

    wrapper = PointButtonWrapper(roll_cfg)
    if verbose:
        print(f"  Loading {model_key} from {env_config.get('checkpoint_path', '?')} …",
              flush=True)
    wrapper.load(env_config=env_config, device=device)
    if verbose:
        print(f"  Model loaded (deter_dim={wrapper._deter_dim}).")

    ver_cfg = VerifyConfig(
        delta_cp=delta_cp,
        delta_err=delta_err,
        model_error_budget=c_hat,
        verbose=False,
        method=method,
        cegar_iterations=cegar_iterations,
        soft_buchi_temperature=soft_buchi_temperature,
        soft_buchi_epsilon=soft_buchi_epsilon,
    )

    results: list[dict[str, Any]] = []

    spec_ids: list[str] = [
        "stl_hazard_avoidance", "ltl_hazard_avoidance",
        "stl_safe_goal_reach",  "ltl_safe_goal",
        "ltl_safe_slow_goal",
        "stl_speed_limit", "ltl_speed_limit",
    ]

    for spec_id in spec_ids:
        if spec_filter and spec_id != spec_filter:
            continue

        spec = get_spec_by_id(spec_id)
        if spec is None:
            continue

        status, reasons = _applicability(spec)

        row: dict[str, Any] = {
            "model":      model_key,
            "spec_id":    spec_id,
            "status":     status,
            "reasons":    reasons,
            "verdict":    None,
            "rho_star":   None,
            "rho_net":    None,
            "guarantee":  None,
            "confidence": None,
            "runtime_s":  None,
            "error":      None,
        }

        if status == "N/A":
            row["verdict"] = "N/A"
            results.append(row)
            continue

        try:
            t0     = time.perf_counter()
            trajs  = wrapper.sample_rollouts(roll_cfg)
            result = verify(trajs, spec, ver_cfg)
            elapsed = time.perf_counter() - t0

            row.update({
                "verdict":    result.verdict,
                "rho_star":   round(result.monitor.rho_star, 4),
                "rho_net":    round(result.transfer.rho_net, 4),
                "guarantee":  result.guarantee_type,
                "confidence": round(result.confidence, 3),
                "runtime_s":  round(elapsed, 1),
            })

            if verbose:
                print(
                    f"    {_emoji(result.verdict)} {result.verdict:12s}"
                    f"  {spec_id:42s}"
                    f"  ρ*={result.monitor.rho_star:+.3f}"
                    f"  ρ_net={result.transfer.rho_net:+.3f}"
                    f"  [{result.guarantee_type}, {result.confidence:.0%}]"
                )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            row["error"]      = str(exc)
            row["verdict"]    = "ERROR"
            row["runtime_s"]  = round(elapsed, 1)
            if verbose:
                print(f"    ERROR  {spec_id}  →  {exc}")

        results.append(row)

    wrapper.close()
    return results


# ─── summary table ────────────────────────────────────────────────────────────

def print_summary(all_results: list[dict[str, Any]]) -> None:
    print("\n" + "═" * 90)
    print("  POINTBUTTON  ·  SAFEWORLD BENCHMARK SUMMARY")
    print("═" * 90)
    print(f"  {'Model':20s}  {'Spec':36s}  {'Verdict':12s}  {'ρ*':>7}  {'ρ_net':>7}  {'Conf':>5}")
    print(f"  {'─'*20}  {'─'*36}  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*5}")

    for r in all_results:
        verdict = r["verdict"] or "—"
        rho_s   = f"{r['rho_star']:+.3f}" if r["rho_star"] is not None else "  —  "
        rho_n   = f"{r['rho_net']:+.3f}"  if r["rho_net"]  is not None else "  —  "
        conf    = f"{r['confidence']:.0%}" if r["confidence"] is not None else "  —"
        print(
            f"  {r['model']:20s}  {r['spec_id']:36s}  {verdict:12s}"
            f"  {rho_s:>7}  {rho_n:>7}  {conf:>5}"
        )

    print("═" * 90)
    total   = len(all_results)
    n_na    = sum(1 for r in all_results if r["verdict"] == "N/A")
    n_err   = sum(1 for r in all_results if r["verdict"] == "ERROR")
    n_ran   = total - n_na - n_err
    n_w     = sum(1 for r in all_results if r["verdict"] == "WARRANT")
    n_stl   = sum(1 for r in all_results if r["verdict"] == "STL_MARGIN")
    n_viol  = sum(1 for r in all_results if r["verdict"] == "VIOLATION")
    print(
        f"  Ran {n_ran}  |  WARRANT {n_w}  STL_MARGIN {n_stl}  VIOLATION {n_viol}"
        f"  |  N/A {n_na}  ERROR {n_err}"
    )
    print()
    print("  N/A: spec requires velocity / zone / carrying — unsupported by latent-stats AP extraction")
    print("  hazard_dist & goal_dist extracted from top-variance h dims (data-driven, approximate)")
    print("═" * 90)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PointButton1 DreamerV3 — SAFEWORLD benchmark"
    )
    parser.add_argument(
        "--model", default=None,
        choices=list(MODELS.keys()) + ["all"],
        help="Which model to test (default: all three)",
    )
    parser.add_argument("--n",        type=int,   default=30,    help="rollouts per spec")
    parser.add_argument("--horizon",  type=int,   default=50,    help="steps per rollout")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--c-hat",    type=float, default=0.08,  help="model error budget")
    parser.add_argument("--spec",     default=None,              help="run a single spec only")
    parser.add_argument("--output",   default=None,              help="save JSON to file")
    parser.add_argument("--method",   default="stl",
                        choices=["stl", "cegar", "oneshot", "soft_buchi"])
    parser.add_argument("--cegar-iters",       type=int,   default=5)
    parser.add_argument("--soft-buchi-temp",   type=float, default=0.1)
    parser.add_argument("--soft-buchi-eps",    type=float, default=0.5)
    args = parser.parse_args()

    models_to_run = (
        {args.model: MODELS[args.model]}
        if args.model and args.model != "all"
        else MODELS
    )

    all_results: list[dict[str, Any]] = []

    for model_key, config_path in models_to_run.items():
        print(f"\n{'─'*70}")
        print(f"  Model: {model_key}  ({config_path})")
        print(f"{'─'*70}")
        results = run_one_model(
            model_key   = model_key,
            config_path = config_path,
            n_rollouts  = args.n,
            horizon     = args.horizon,
            seed        = args.seed,
            device      = args.device,
            delta_cp    = 0.05,
            delta_err   = 0.05,
            c_hat       = args.c_hat,
            spec_filter = args.spec,
            verbose     = True,
            method      = args.method,
            cegar_iterations        = args.cegar_iters,
            soft_buchi_temperature  = args.soft_buchi_temp,
            soft_buchi_epsilon      = args.soft_buchi_eps,
        )
        all_results.extend(results)

    print_summary(all_results)

    if args.output:
        Path(args.output).write_text(json.dumps(all_results, indent=2))
        print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
