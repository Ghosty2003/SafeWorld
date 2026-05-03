"""
main.py

SAFEWORLD – Main verification entry point.

This file exposes the primary public function:

    verify(trajectories, spec, config) -> VerificationResult

which takes already-obtained latent trajectory data (from any wrapper or your
own source) and runs the full SAFEWORLD Algorithm 1 pipeline:

    1. Latent Monitor     – STL robustness ρ(φ, τ_i, 0) for each rollout
    2. Transfer Calibrator – ρ_net = ρ* - ĉ_err (model→environment transfer)
    3. LPPM Certificate   – p̂_γ for infinite-horizon LTL warrant

Also provides:

    verify_from_wrapper(wrapper, spec, rollout_config, verify_config)
        Convenience wrapper: sample trajectories then verify in one call.

    run_benchmark(wrapper, spec_ids, rollout_config, verify_config)
        Run verification across multiple specifications.
"""

from __future__ import annotations

import logging
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from specs import get_spec_by_id, ALL_SPECS
from specs.spec_calibrator import apply_env_config_to_spec, load_env_config 
from configs.settings import build_rollout_config, load_settings_config, settings_path_for_model
from core.stl_monitor import monitor_rollouts, MonitorResult, net_robustness
from core.transfer_calibrator import (
    fit_conformal_error_budget,
    calibrate_robustness_quantile,
    transfer_verdict,
    TransferResult,
)
from core.lppm import (
    build_parity_automaton,
    calibrate_lppm,
    fit_lppm,
    LPPMResult,
)
from utils.spec_analysis import analyze_spec_structure
from utils.task_parser import apply_confidence_profile, evaluate_predicates, load_task_spec

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class VerifyConfig:
    """
    Hyperparameters for the SAFEWORLD verification pipeline.
    Corresponds to Algorithm 1 inputs: δ_cp, δ_err, γ, η.
    """

    # Transfer Calibrator
    delta_cp:       float = 0.05
    """Conformal coverage failure prob for model-side robustness bound (Theorem D.4)."""

    delta_err:      float = 0.05
    """Conformal coverage failure prob for model-error budget (Corollary 5.2)."""

    model_error_budget: float = 0.08
    """
    ĉ_err: pre-set model/environment mismatch budget.
    Used when no paired rollouts are available for conformal calibration.
    Values from paper Table 22: ~0.05-0.12 depending on AP.
    """

    paired_rollouts: list[tuple[list[dict], list[dict]]] | None = None
    """
    Paired (model_traj, env_traj) rollout pairs for conformal ĉ_err estimation.
    If None, model_error_budget is used directly.
    """

    # LPPM Certificate
    gamma:          float = 0.05
    """Binary-indicator calibration failure probability for LPPM (Theorem 5.5)."""

    eta:            float = 0.01
    """Strict descent margin η for LPPM (P2) condition."""

    warrant_threshold: float = 0.80
    """p̂_γ must exceed this to issue a WARRANT verdict (Algorithm 1, line 10)."""

    fit_lppm_params: bool = False
    """
    If True, run fit_lppm() to train LPPM parameters before calibration.
    Adds training time but improves p̂_γ for Recurrence/Persistence specs.
    """

    lppm_epochs:    int = 300
    """Number of training epochs for fit_lppm (if fit_lppm_params=True)."""

    # General
    verbose:        bool = True
    """Print progress and summary to stdout."""

    auto_collect_paired_rollouts: bool = False
    """If True, ask the wrapper for paired model/environment rollouts when available."""


# ─── Verdict enum ─────────────────────────────────────────────────────────────

WARRANT    = "WARRANT"
STL_MARGIN = "STL_MARGIN"
VIOLATION  = "VIOLATION"


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """
    Complete output of the SAFEWORLD verification pipeline (Algorithm 1).

    Verdict logic:
        WARRANT    – transfer guarantee holds (strict or conformal) AND p̂_γ ≥ threshold
        STL_MARGIN – ρ* > 0 but no transfer guarantee (ρ_net ≤ 0 and ρ_net_cp ≤ 0)
        VIOLATION  – ρ* < 0  (witnessing rollout available)

    Guarantee types (inspect via `guarantee_type`):
        "strict"    – ρ_net > 0: all N rollouts satisfied spec with margin > ĉ_err.
                      Confidence ≥ 1 - δ_err.
        "conformal" – ρ_net_cp > 0: (1-δ_cp) fraction satisfied with margin > ĉ_err.
                      Confidence ≥ 1 - δ_cp - δ_err  (Theorem 5.5).
        "none"      – no transfer guarantee achieved.
    """

    verdict:    str    # WARRANT | STL_MARGIN | VIOLATION

    # ── Layer 1: Latent Monitor ───────────────────────────────────────────────
    monitor:    MonitorResult

    # ── Layer 2: Transfer Calibrator ─────────────────────────────────────────
    transfer:   TransferResult

    # ── Layer 3: LPPM Certificate ─────────────────────────────────────────────
    lppm:       LPPMResult | None   # None for bounded-only specs

    # ── Meta ──────────────────────────────────────────────────────────────────
    spec_id:    str   = ""
    spec_name:  str   = ""
    mp_class:   str   = ""
    level:      int   = 0
    task_level: str   = ""
    verification_mode: str = ""
    support_level: str = ""
    support_note: str = ""
    wall_time:  float = 0.0   # seconds

    # Which guarantee path produced the WARRANT/STL_MARGIN verdict
    guarantee_type: str = "none"   # "strict" | "conformal" | "none"

    # Effective statistical confidence of the verdict (0.0 if VIOLATION or no transfer)
    confidence: float = 0.0

    # LPPM training info (if fit_lppm_params=True)
    lppm_training: dict = field(default_factory=dict)

    # Witness trajectory (only when VIOLATION)
    witness_trajectory: list[dict] | None = None

    # ── convenience properties ────────────────────────────────────────────────

    @property
    def rho_star(self) -> float:
        return self.monitor.rho_star

    @property
    def rho_net(self) -> float:
        return self.transfer.rho_net

    @property
    def rho_net_cp(self) -> float:
        return self.transfer.rho_net_cp

    @property
    def q_hat(self) -> float:
        return self.transfer.q_hat

    @property
    def p_hat(self) -> float:
        return self.lppm.p_hat_gamma if self.lppm else 0.0

    @property
    def c_hat_err(self) -> float:
        return self.transfer.c_hat_err

    def is_safe(self) -> bool:
        return self.verdict in (WARRANT, STL_MARGIN)

    def summary(self) -> str:
        lines = [
            f"{'─'*60}",
            f" SAFEWORLD Verification: {self.spec_name} (Level {self.level}, {self.mp_class})",
            f"{'─'*60}",
            f" Verdict:        {self.verdict}",
            f" Guarantee:      {self.guarantee_type}",
            f" Confidence:     {self.confidence:.3f}",
            f" Task level:     {self.task_level or 'n/a'}",
            f" Mode:           {self.verification_mode or 'n/a'}",
            f" Support:        {self.support_level or 'n/a'}",
            f" ρ* (worst-case margin):  {self.rho_star:+.4f}",
            f" q̂_δ (CP quantile):       {self.q_hat:+.4f}",
            f" ĉ_err (model error):      {self.c_hat_err:.4f}",
            f" ρ_net = ρ* - ĉ_err:       {self.rho_net:+.4f}  [strict]",
            f" ρ_net_cp = q̂_δ - ĉ_err:  {self.rho_net_cp:+.4f}  [conformal, Theorem 5.5]",
        ]
        if self.lppm:
            lines += [
                f" p̂_γ (LPPM warrant):      {self.p_hat:.3f}",
                f" avg descent margin η:    {self.lppm.avg_descent_margin:.4f}",
            ]
        lines += [
            f" Note:                    {self.support_note}",
            f" Wall time:               {self.wall_time:.2f}s",
            f"{'─'*60}",
        ]
        if self.verdict == VIOLATION:
            lines.append(f" ⚠ Witness rollout index: {self.monitor.witness_idx}")
        return "\n".join(lines)


# ─── Core verify() function ───────────────────────────────────────────────────

def verify(
    trajectories: list[list[dict[str, float]]],
    spec:         dict,
    config:       VerifyConfig | None = None,
) -> VerificationResult:
    """
    Run the full SAFEWORLD verification pipeline on pre-collected latent trajectories.

    This is the main function to call when you already have latent trajectory data
    (e.g. from a wrapper, from your own model, or from a file).

    Parameters
    ----------
    trajectories : N latent rollouts, each a list of T state-dicts.
                   Keys must match the AP "dim" fields in spec["formula"].
    spec         : specification dict from specs/ (use get_spec_by_id() to look up).
    config       : VerifyConfig with calibration hyperparameters.

    Returns
    -------
    VerificationResult with verdict, robustness margins, transfer result, and LPPM.

    Example
    -------
    >>> from main import verify, VerifyConfig
    >>> from specs import get_spec_by_id
    >>> from wrappers import DreamerV3Wrapper, RolloutConfig
    >>>
    >>> spec = get_spec_by_id("ltl_patrol")
    >>> cfg  = RolloutConfig(horizon=50, n_rollouts=20,
    ...                      extra={"env_name": "SafetyPointGoal1-v0"})
    >>> with DreamerV3Wrapper(cfg) as w:
    ...     w.load(checkpoint_path="my_checkpoint.pkl")
    ...     trajs = w.sample_rollouts()
    >>>
    >>> result = verify(trajs, spec)
    >>> print(result.summary())
    """
    cfg = config or VerifyConfig()
    t0  = time.perf_counter()

    if not trajectories:
        raise ValueError("trajectories must be non-empty")
    if not spec:
        raise ValueError("spec must be a valid specification dict")

    predicate_defs = spec.get("predicates", [])
    if predicate_defs:
        trajectories = [
            evaluate_predicates(traj, predicate_defs, include_raw_state=True)
            for traj in trajectories
        ]
        if cfg.paired_rollouts:
            cfg.paired_rollouts = [
                (
                    evaluate_predicates(model_traj, predicate_defs, include_raw_state=True),
                    evaluate_predicates(env_traj, predicate_defs, include_raw_state=True),
                )
                for model_traj, env_traj in cfg.paired_rollouts
            ]

    analysis = spec.get("analysis") or analyze_spec_structure(spec)
    spec["analysis"] = analysis
    formula  = spec["formula"]
    spec_id  = spec.get("id", "")
    mp_class = analysis["mp_class"]

    if cfg.verbose:
        print(f"\n[SAFEWORLD] Verifying '{spec.get('name', spec_id)}' "
              f"(Level {spec.get('level',0)}, {mp_class})")
        print(f"            N={len(trajectories)} rollouts, T={len(trajectories[0])}")

    # ─── Step 1: Latent Monitor (Section 4.2) ────────────────────────────────
    if cfg.verbose:
        print("  [1/3] Latent Monitor: computing STL robustness...")

    monitor_res = monitor_rollouts(formula, trajectories)

    if cfg.verbose:
        print(f"        ρ*={monitor_res.rho_star:+.4f}  "
              f"sat={monitor_res.n_satisfied}/{monitor_res.n_rollouts}")

    # ─── Step 2: Transfer Calibrator (Section 4.2, Corollary 5.2) ────────────
    if cfg.verbose:
        print("  [2/3] Transfer Calibrator: conformal error budget...")

    if cfg.paired_rollouts:
        c_hat_err = fit_conformal_error_budget(
            paired_rollouts=cfg.paired_rollouts,
            formula_aps=spec.get("aps", []),
            delta_err=cfg.delta_err,
            predicate_map=spec.get("predicate_map"),
        )
    else:
        c_hat_err = cfg.model_error_budget

    transfer_res = transfer_verdict(
        rho_star=monitor_res.rho_star,
        c_hat_err=c_hat_err,
        delta_cp=cfg.delta_cp,
        delta_err=cfg.delta_err,
        margins=monitor_res.margins,
    )

    if cfg.verbose:
        print(f"        ĉ_err={c_hat_err:.4f}  ρ_net={transfer_res.rho_net:+.4f}  "
              f"transfers={transfer_res.transfers()}")

    # ─── Early exit: VIOLATION (Algorithm 1, line 6) ─────────────────────────
    if monitor_res.rho_star < 0:
        if cfg.verbose:
            print(f"  → VIOLATION (ρ*<0, witness rollout #{monitor_res.witness_idx})")
        return VerificationResult(
            verdict=VIOLATION,
            monitor=monitor_res,
            transfer=transfer_res,
            lppm=None,
            spec_id=spec_id,
            spec_name=spec.get("name", ""),
            mp_class=mp_class,
            level=spec.get("level", 0),
            task_level=analysis["task_level"],
            verification_mode=analysis["verification_mode"],
            support_level=analysis["support_level"],
            support_note=analysis["support_note"],
            wall_time=time.perf_counter() - t0,
            guarantee_type="none",
            confidence=0.0,
            witness_trajectory=trajectories[monitor_res.witness_idx],
        )

    if analysis["verification_mode"] == "finite_stl":
        # Strict path: all N rollouts satisfied with margin > ĉ_err
        if transfer_res.transfers():
            verdict        = WARRANT
            guarantee_type = "strict"
            confidence     = max(0.0, 1.0 - cfg.delta_err)
        # Conformal path (Theorem 5.5): (1-δ_cp) fraction satisfied with margin > ĉ_err
        elif transfer_res.transfers_cp():
            verdict        = WARRANT
            guarantee_type = "conformal"
            confidence     = transfer_res.effective_confidence()
        else:
            verdict        = STL_MARGIN
            guarantee_type = "none"
            confidence     = 0.0
        if cfg.verbose:
            print("  [3/3] LPPM: skipped for bounded STL task.")
            print(f"  → {verdict} ({guarantee_type}, confidence={confidence:.3f})")
        return VerificationResult(
            verdict=verdict,
            monitor=monitor_res,
            transfer=transfer_res,
            lppm=None,
            spec_id=spec_id,
            spec_name=spec.get("name", ""),
            mp_class=mp_class,
            level=spec.get("level", 0),
            task_level=analysis["task_level"],
            verification_mode=analysis["verification_mode"],
            support_level=analysis["support_level"],
            support_note=analysis["support_note"],
            wall_time=time.perf_counter() - t0,
            guarantee_type=guarantee_type,
            confidence=confidence,
        )

    # ─── Step 3: LPPM Certificate (Section 4.3, Theorem 5.5) ─────────────────
    if cfg.verbose:
        print("  [3/3] LPPM: building parity automaton and calibrating certificate...")

    dpa          = build_parity_automaton(spec)
    support_level = analysis["support_level"]
    support_note = analysis["support_note"]
    if getattr(dpa, "exact", False):
        support_level = "sound"
        support_note = (
            "Infinite-horizon LTL was translated to an exact deterministic parity automaton "
            "via Spot; the remaining verification path uses the parity/LPPM certificate."
        )
        if cfg.verbose:
            print("        backend=spot exact_parity=True")
    else:
        if cfg.verbose:
            print(f"        backend={getattr(dpa, 'backend', 'template')} exact_parity=False")
    lppm_params  = None

    if cfg.fit_lppm_params:
        if cfg.verbose:
            print(f"        Training LPPM ({cfg.lppm_epochs} epochs)...")
        training_info = fit_lppm(
            trajectories=trajectories,
            dpa=dpa,
            spec=spec,
            eta=cfg.eta,
            n_epochs=cfg.lppm_epochs,
        )
        lppm_params = training_info
        if cfg.verbose:
            print(f"        Training loss: {training_info['final_loss']:.5f}")
    else:
        training_info = {}

    lppm_res = calibrate_lppm(
        trajectories=trajectories,
        dpa=dpa,
        spec=spec,
        gamma=cfg.gamma,
        eta=cfg.eta,
        warrant_threshold=cfg.warrant_threshold,
        lppm_params=lppm_params,
    )

    if cfg.verbose:
        print(f"        p̂_γ={lppm_res.p_hat_gamma:.3f}  "
              f"sat={lppm_res.satisfaction_rate:.3f}  "
              f"warranted={lppm_res.is_warranted()}")

    # ─── Verdict (Algorithm 1, lines 10-11) ──────────────────────────────────
    # Determine transfer guarantee type before checking LPPM
    if transfer_res.transfers():
        guarantee_type = "strict"
        confidence     = max(0.0, 1.0 - cfg.delta_err)
        transfer_ok    = True
    elif transfer_res.transfers_cp():
        guarantee_type = "conformal"
        confidence     = transfer_res.effective_confidence()
        transfer_ok    = True
    else:
        guarantee_type = "none"
        confidence     = 0.0
        transfer_ok    = False

    if transfer_ok and lppm_res.is_warranted():
        verdict = WARRANT
    elif transfer_ok:
        verdict = STL_MARGIN   # transfer holds but LPPM p̂_γ < threshold
    else:
        verdict = STL_MARGIN   # no transfer guarantee, no explicit violation

    if cfg.verbose:
        print(f"  → {verdict} ({guarantee_type}, confidence={confidence:.3f})")

    return VerificationResult(
        verdict=verdict,
        monitor=monitor_res,
        transfer=transfer_res,
        lppm=lppm_res,
        spec_id=spec_id,
        spec_name=spec.get("name", ""),
        mp_class=mp_class,
        level=spec.get("level", 0),
        task_level=analysis["task_level"],
        verification_mode=analysis["verification_mode"],
        support_level=support_level,
        support_note=support_note,
        wall_time=time.perf_counter() - t0,
        guarantee_type=guarantee_type,
        confidence=confidence,
        lppm_training=training_info,
    )


# ─── Convenience: verify_from_wrapper() ─────────────────────────────────────

def verify_from_wrapper(
    wrapper,
    spec:          dict,
    rollout_config=None,
    verify_config: VerifyConfig | None = None,
) -> VerificationResult:
    """
    Sample rollouts from a wrapper, then run verify().

    Parameters
    ----------
    wrapper       : any WorldModelWrapper subclass (already loaded).
    spec          : specification dict.
    rollout_config: RolloutConfig for sampling.
    verify_config : VerifyConfig for calibration.

    Returns
    -------
    VerificationResult.
    """
    vcfg = verify_config or VerifyConfig()
    if vcfg.auto_collect_paired_rollouts:
        paired = wrapper.sample_paired_rollouts(rollout_config)
        trajectories = [tau_model for tau_model, _ in paired]
        vcfg.paired_rollouts = paired
    else:
        trajectories = wrapper.sample_rollouts(rollout_config)

    # Warn about missing AP keys
    if spec.get("predicates"):
        required_keys = [
            pred.get("source", pred.get("dim", pred.get("object", pred["name"])))
            for pred in spec["predicates"]
        ]
    else:
        required_keys = spec.get("aps", [])
    warnings_list = wrapper.validate_trajectories(trajectories, required_keys)
    for w in warnings_list:
        logger.warning(w)

    return verify(trajectories, spec, vcfg)


# ─── run_benchmark() ─────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    results:    dict[str, VerificationResult]   # spec_id -> result
    wall_time:  float

    def summary_table(self) -> str:
        rows = [
            f"{'Spec ID':<35} {'Level':>5} {'MP':>12} {'Verdict':>12} "
            f"{'ρ*':>8} {'ρ_net':>8} {'p̂_γ':>6} {'Time':>6}"
        ]
        rows.append("─" * 100)
        for sid, r in self.results.items():
            p_hat_str = f"{r.p_hat:.3f}" if r.lppm else "  n/a"
            rows.append(
                f"{sid:<35} {r.level:>5} {r.mp_class:>12} {r.verdict:>12} "
                f"{r.rho_star:>+8.3f} {r.rho_net:>+8.3f} {p_hat_str:>6} "
                f"{r.wall_time:>5.1f}s"
            )
        rows.append("─" * 100)
        rows.append(f"Total wall time: {self.wall_time:.1f}s")
        return "\n".join(rows)


def run_benchmark(
    wrapper,
    spec_ids:      list[str] | None = None,
    rollout_config=None,
    verify_config: VerifyConfig | None = None,
) -> BenchmarkResult:
    """
    Run SAFEWORLD verification across multiple specifications.

    Parameters
    ----------
    wrapper       : WorldModelWrapper (already loaded).
    spec_ids      : list of spec IDs to test; if None, uses all 23 specs.
    rollout_config: RolloutConfig for sampling.
    verify_config : VerifyConfig for calibration.

    Returns
    -------
    BenchmarkResult with one VerificationResult per spec.
    """
    specs_to_run = (
        [get_spec_by_id(sid) for sid in spec_ids if get_spec_by_id(sid) is not None]
        if spec_ids
        else ALL_SPECS
    )

    vcfg = verify_config or VerifyConfig(verbose=False)
    results: dict[str, VerificationResult] = {}
    t0 = time.perf_counter()

    for spec in specs_to_run:
        sid = spec["id"]
        print(f"  Verifying {sid}...", end=" ", flush=True)
        try:
            res = verify_from_wrapper(wrapper, spec, rollout_config, vcfg)
            results[sid] = res
            print(res.verdict)
        except Exception as exc:
            print(f"ERROR: {exc}")
            logger.exception(f"Failed to verify {sid}")

    return BenchmarkResult(results=results, wall_time=time.perf_counter() - t0)


def _json_object_arg(value: str | None, flag_name: str) -> dict:
    if value is None:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{flag_name} must be a valid JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{flag_name} must be a JSON object.")
    return parsed


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAFEWORLD verification pipeline")
    # ② CLI 参数：加 --env-config 参数
    parser.add_argument("--env-config", default=None,
                        help="Path to environment config JSON for AP threshold overrides")
    parser.add_argument("--spec",    default="ltl_hazard_avoidance",
                        help="Specification ID (see specs/__init__.py for full list)")
    parser.add_argument("--task-config", default=None,
                        help="Path to a JSON task spec with predicates + formula")
    parser.add_argument("--settings-config", default=None,
                        help="Path to a JSON runtime settings file")
    parser.add_argument("--model",   default="random",
                        choices=["random", "dreamerv3", "safety_point_goal", "simple_pointgoal2", "goal2_dreamer"],
                        help="World model to use")
    parser.add_argument("--env-name", default=None,
                        help="Gymnasium environment name for env-backed wrappers")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--n",       type=int, default=None, help="Number of rollouts")
    parser.add_argument("--confidence-profile", default=None,
                        choices=["quick", "moderate", "high-confidence"],
                        help="Preset rollout counts: quick=20, moderate=100, high-confidence=1000")
    parser.add_argument("--seed",    type=int, default=None)
    parser.add_argument("--action-source", default=None,
                        choices=["random", "env", "zeros", "adversarial"],
                        help="Action sampling mode for rollout wrappers")
    parser.add_argument("--fidelity",type=float, default=0.75,
                        help="Model fidelity for random wrapper (0=unsafe, 1=safe)")
    parser.add_argument("--c-hat",   type=float, default=None,
                        help="Model error budget ĉ_err")
    parser.add_argument("--auto-paired", action="store_true",
                        help="Use wrapper-provided paired model/environment rollouts when available")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run all specs (SAFEWORLD-BENCH)")
    parser.add_argument("--checkpoint", default=None,
                        help="DreamerV3 checkpoint path")
    parser.add_argument("--env-kwargs", default=None,
                        help="JSON object passed to environment construction")
    parser.add_argument("--reset-kwargs", default=None,
                        help="JSON object passed to env.reset() for paired rollouts")
    parser.add_argument("--stop-on-done", action="store_true",
                        help="Stop model-only rollouts when done_probability exceeds threshold")
    parser.add_argument("--done-threshold", type=float, default=None,
                        help="done_probability threshold used with --stop-on-done")
    args = parser.parse_args()

    # Import wrappers here to avoid circular imports at module level
    from wrappers import (
        DreamerV3Wrapper,
        Goal2WorldModelWrapper,
        RandomWorldModelWrapper,
        SafetyPointGoalWrapper,
        SimplePointGoal2WorldModelWrapper,
    )

    task_spec = load_task_spec(args.task_config) if args.task_config else None
    settings_path = Path(args.settings_config) if args.settings_config else settings_path_for_model(args.model)
    settings = load_settings_config(settings_path if settings_path.exists() else None)
    env_config_path = (
        args.env_config                                          # CLI 优先
        or settings.get("environment", {}).get("config")        # settings JSON 次之
    )
    env_config = load_env_config(env_config_path) if env_config_path else {}
    roll_cfg = build_rollout_config(settings)
    rollout_meta = apply_confidence_profile({}, args.confidence_profile, explicit_n=args.n)
    if args.horizon is not None:
        rollout_meta["horizon"] = args.horizon
    env_kwargs = _json_object_arg(args.env_kwargs, "--env-kwargs")
    reset_kwargs = _json_object_arg(args.reset_kwargs, "--reset-kwargs")
    if "horizon" in rollout_meta:
        roll_cfg.horizon = int(rollout_meta["horizon"])
    if "n_rollouts" in rollout_meta:
        roll_cfg.n_rollouts = int(rollout_meta["n_rollouts"])
    if args.seed is not None:
        roll_cfg.seed = args.seed
    if args.action_source is not None:
        roll_cfg.action_source = args.action_source
        roll_cfg.extra["action_source"] = args.action_source
    roll_cfg.extra.setdefault("fidelity", args.fidelity)
    roll_cfg.extra["spec_type"] = args.spec.replace("ltl_", "").replace("stl_", "")
    if args.env_name is not None:
        roll_cfg.extra["env_name"] = args.env_name
    if env_kwargs:
        roll_cfg.extra["env_kwargs"] = env_kwargs
    if reset_kwargs:
        roll_cfg.extra["reset_kwargs"] = reset_kwargs
    if args.stop_on_done:
        roll_cfg.extra["stop_on_done"] = True
    if args.done_threshold is not None:
        roll_cfg.extra["done_threshold"] = args.done_threshold
    verification_settings = dict(settings.get("verification", {}))
    ver_cfg = VerifyConfig(
        model_error_budget=float(args.c_hat if args.c_hat is not None else verification_settings.get("model_error_budget", 0.08)),
        verbose=not args.benchmark,
        auto_collect_paired_rollouts=bool(args.auto_paired or verification_settings.get("auto_collect_paired_rollouts", False)),
    )

    if args.model == "dreamerv3":
        w = DreamerV3Wrapper(roll_cfg)
        w.load(checkpoint_path=args.checkpoint)
    elif args.model == "safety_point_goal":
        w = SafetyPointGoalWrapper(roll_cfg)
        w.load(env_name=roll_cfg.extra.get("env_name", "SafetyPointGoal1-v0"))
    elif args.model == "simple_pointgoal2":
        roll_cfg.extra.setdefault("env_name", "SafetyPointGoal2Gymnasium-v0")
        w = SimplePointGoal2WorldModelWrapper(roll_cfg)
        w.load(checkpoint_path=args.checkpoint or None, env_name=roll_cfg.extra["env_name"])
    elif args.model == "goal2_dreamer":
        w = Goal2WorldModelWrapper(roll_cfg)
        ckpt = args.checkpoint or settings.get("model", {}).get("checkpoint_path")
        mdir = settings.get("model", {}).get("model_dir")
        w.load(checkpoint_path=ckpt, model_dir=mdir, env_config=env_config)
    else:
        w = RandomWorldModelWrapper(roll_cfg)
        w.load()

    if args.benchmark:
        print("\nRunning SAFEWORLD-BENCH across all 23 specifications...\n")
        bench = run_benchmark(w, rollout_config=roll_cfg, verify_config=ver_cfg)
        print("\n" + bench.summary_table())
    else:
        spec = task_spec or get_spec_by_id(args.spec)
        if spec is None:
            print(f"Unknown spec ID: {args.spec}")
            print("Available IDs:")
            for s in ALL_SPECS:
                print(f"  {s['id']}")
        else:
            if env_config:
                spec = apply_env_config_to_spec(spec, env_config)
            result = verify_from_wrapper(w, spec, roll_cfg, ver_cfg)
            print(result.summary())
