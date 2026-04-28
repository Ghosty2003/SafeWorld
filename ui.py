"""
ui.py

SAFEWORLD – Interactive terminal UI.

Lets the user configure and run verification without writing code:
    • Select world model and load checkpoint
    • Browse and select specifications by level / MP class
    • Adjust all verification hyperparameters interactively
    • Run single spec verification or full benchmark
    • View results in formatted tables with colour highlighting

Usage:
    python ui.py                     # interactive TUI
    python ui.py --no-color          # disable ANSI colour

Requires only the Python standard library (no curses / rich needed).
"""

from __future__ import annotations

import os
import sys
import time
import textwrap
from dataclasses import dataclass, field
from typing import Any

# ── ANSI colour helpers ───────────────────────────────────────────────────────

USE_COLOR = sys.stdout.isatty() and "--no-color" not in sys.argv

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def green(t):   return _c("32", t)
def red(t):     return _c("31", t)
def yellow(t):  return _c("33", t)
def cyan(t):    return _c("36", t)
def bold(t):    return _c("1",  t)
def dim(t):     return _c("2",  t)


def _verdict_colored(v: str) -> str:
    if v == "WARRANT":    return green(bold(v))
    if v == "VIOLATION":  return red(bold(v))
    if v == "STL_MARGIN": return yellow(bold(v))
    return v


# ── UI state ──────────────────────────────────────────────────────────────────

@dataclass
class UIState:
    """All user-adjustable settings."""

    # Model
    model_type:      str   = "random"          # "random" | "dreamerv3" | "tdmpc2"
    checkpoint_path: str   = ""
    env_name:        str   = "SafetyPointGoal1-v0"
    latent_dim:      int   = 32
    use_decoder:     bool  = True
    use_stoch:       bool  = False
    action_source:   str   = "random"          # "random" | "policy" | "zeros"
    fidelity:        float = 0.75              # random wrapper only

    # Rollout
    horizon:         int   = 50
    n_rollouts:      int   = 20
    seed:            int   = 0

    # Verification
    model_error_budget: float = 0.08
    delta_cp:           float = 0.05
    delta_err:          float = 0.05
    gamma:              float = 0.05
    eta:                float = 0.01
    warrant_threshold:  float = 0.80
    fit_lppm_params:    bool  = False
    lppm_epochs:        int   = 300

    # Spec filter
    filter_level:    int | None = None
    filter_mp:       str | None = None


# ── helpers ───────────────────────────────────────────────────────────────────

def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _input(prompt: str, default: Any = None) -> str:
    dflt_str = f" [{default}]" if default is not None else ""
    try:
        val = input(f"{prompt}{dflt_str}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return str(default) if default is not None else ""
    return val if val else (str(default) if default is not None else "")


def _input_float(prompt: str, default: float, lo: float = 0.0, hi: float = 1e9) -> float:
    while True:
        raw = _input(prompt, default)
        try:
            val = float(raw)
            if lo <= val <= hi:
                return val
            print(f"  Must be in [{lo}, {hi}]")
        except ValueError:
            print("  Invalid number.")


def _input_int(prompt: str, default: int, lo: int = 1, hi: int = 10_000) -> int:
    while True:
        raw = _input(prompt, default)
        try:
            val = int(raw)
            if lo <= val <= hi:
                return val
            print(f"  Must be in [{lo}, {hi}]")
        except ValueError:
            print("  Invalid integer.")


def _input_bool(prompt: str, default: bool) -> bool:
    raw = _input(f"{prompt} (y/n)", "y" if default else "n")
    return raw.lower() in ("y", "yes", "1", "true")


def _choose(prompt: str, options: list[str]) -> int:
    """
    Display a numbered menu and return the 0-based index of the chosen option.
    """
    for i, opt in enumerate(options, 1):
        print(f"  {dim(str(i)+'.')} {opt}")
    while True:
        raw = _input(prompt, 1)
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(options)}.")


def _header(title: str):
    width = 62
    print()
    print(bold(cyan("╔" + "═" * width + "╗")))
    pad = (width - len(title)) // 2
    print(bold(cyan("║")) + " " * pad + bold(title) + " " * (width - pad - len(title)) + bold(cyan("║")))
    print(bold(cyan("╚" + "═" * width + "╝")))


def _section(title: str):
    print(f"\n{bold(cyan('── ' + title + ' ──'))}")


# ── Menu screens ──────────────────────────────────────────────────────────────

def screen_main(state: UIState) -> str:
    """Main menu. Returns the chosen action string."""
    _header("SAFEWORLD Verifier")
    print(textwrap.dedent(f"""
    Current configuration:
      Model:     {bold(state.model_type)}  ·  env={state.env_name}  ·  d={state.latent_dim}
      Rollouts:  N={state.n_rollouts}  T={state.horizon}  seed={state.seed}
      ĉ_err:     {state.model_error_budget}   δ_cp={state.delta_cp}  δ_err={state.delta_err}
      LPPM:      γ={state.gamma}  η={state.eta}  threshold={state.warrant_threshold}
                 fit_params={state.fit_lppm_params}
    """))

    options = [
        "Configure world model",
        "Configure rollout parameters",
        "Configure verification parameters",
        "Run verification (single spec)",
        "Run SAFEWORLD-BENCH (all specs)",
        "Browse specifications",
        "Quit",
    ]
    idx = _choose("Choose action", options)
    return options[idx]


def screen_model(state: UIState):
    _header("World Model Configuration")

    _section("Model type")
    idx = _choose("Model", ["RandomWorldModel (no ML deps)", "DreamerV3", "TD-MPC2 (stub)", "Custom endpoint"])
    state.model_type = ["random", "dreamerv3", "tdmpc2", "custom"][idx]

    if state.model_type != "random":
        state.checkpoint_path = _input("Checkpoint path", state.checkpoint_path or "path/to/checkpoint.pkl")

    _section("Environment")
    env_choices = [
        "SafetyPointGoal1-v0",
        "SafetyPointGoal2-v0",
        "SafetyCarGoal1-v0",
        "SafetyPointButton1-v0",
    ]
    ei = _choose("Environment", env_choices)
    state.env_name = env_choices[ei]

    _section("Latent space")
    state.latent_dim = _input_int("Latent dimension d", state.latent_dim, lo=1, hi=4096)

    if state.model_type == "dreamerv3":
        state.use_decoder = _input_bool("Use decoder for AP extraction", state.use_decoder)
        state.use_stoch   = _input_bool("Concatenate stochastic z_t to h_t", state.use_stoch)

    _section("Action source")
    ai = _choose("Action source", ["Uniform random (broad coverage)", "Trained policy π(z)", "Zero actions (debug)"])
    state.action_source = ["random", "policy", "zeros"][ai]

    if state.model_type == "random":
        state.fidelity = _input_float("Fidelity (0=unsafe, 1=safe)", state.fidelity, 0.0, 1.0)

    print(green("\n  ✓ Model configuration updated."))


def screen_rollout(state: UIState):
    _header("Rollout Parameters")
    state.n_rollouts = _input_int("Number of rollouts N", state.n_rollouts, lo=1, hi=10_000)
    state.horizon    = _input_int("Rollout horizon T",    state.horizon,    lo=1, hi=1_000)
    state.seed       = _input_int("Random seed",          state.seed,       lo=0, hi=2**31)
    print(green("\n  ✓ Rollout parameters updated."))


def screen_verify_params(state: UIState):
    _header("Verification Hyperparameters")

    _section("Transfer Calibrator (Section 4.2, Corollary 5.2)")
    print(dim("  ĉ_err: model/environment mismatch budget. Used when no paired rollouts are provided."))
    state.model_error_budget = _input_float("Model error budget ĉ_err", state.model_error_budget, 0.0, 1.0)
    state.delta_cp           = _input_float("Coverage failure prob δ_cp (model bound)", state.delta_cp, 0.001, 0.5)
    state.delta_err          = _input_float("Coverage failure prob δ_err (error bound)", state.delta_err, 0.001, 0.5)

    _section("LPPM Certificate (Section 4.3, Theorem 5.5)")
    state.gamma              = _input_float("LPPM failure prob γ",          state.gamma,              0.001, 0.5)
    state.eta                = _input_float("Descent margin η",              state.eta,                1e-6, 1.0)
    state.warrant_threshold  = _input_float("Warrant threshold (p̂_γ ≥ ?)", state.warrant_threshold,  0.0,  1.0)

    _section("LPPM training")
    state.fit_lppm_params    = _input_bool("Fit LPPM parameters (Eq. 5)?", state.fit_lppm_params)
    if state.fit_lppm_params:
        state.lppm_epochs    = _input_int("Training epochs", state.lppm_epochs, lo=1, hi=10_000)

    print(green("\n  ✓ Verification parameters updated."))


def screen_browse_specs():
    """Show all specs grouped by level."""
    from specs import ALL_SPECS

    _header("SAFEWORLD-BENCH Specifications")
    levels = sorted({s["level"] for s in ALL_SPECS})
    for lv in levels:
        specs_at_level = [s for s in ALL_SPECS if s["level"] == lv]
        print(f"\n  {bold(f'Level {lv}')}:")
        for s in specs_at_level:
            kind = "LTL" if s["id"].startswith("ltl") else "STL"
            print(f"    {dim(kind)}  {s['id']:<40} {s['mp_class']:<12}  {s['description'][:60]}…")

    input(dim("\n  Press Enter to return to menu..."))


def screen_run_single(state: UIState):
    """Let user pick a spec and run verification."""
    from specs import ALL_SPECS, get_spec_by_id

    _header("Run Verification – Single Spec")

    # Apply level/MP filter
    candidates = ALL_SPECS
    if state.filter_level:
        candidates = [s for s in candidates if s["level"] == state.filter_level]
    if state.filter_mp:
        candidates = [s for s in candidates if s["mp_class"].lower() == state.filter_mp.lower()]

    if not candidates:
        print(red("  No specs match current filter. Showing all."))
        candidates = ALL_SPECS

    spec_labels = [
        f"[L{s['level']} {s['mp_class'][:3]}] {s['id']}"
        for s in candidates
    ]
    idx = _choose("Select specification", spec_labels)
    spec = candidates[idx]

    print(f"\n  {bold(spec['name'])}")
    print(f"  {dim(spec['description'])}")
    print()

    _run_and_show(state, spec)


def screen_run_benchmark(state: UIState):
    """Run verification across all 23 specs."""
    from specs import ALL_SPECS

    _header("SAFEWORLD-BENCH – Full Run")
    print(f"  Running {len(ALL_SPECS)} specifications...")
    print(f"  N={state.n_rollouts}  T={state.horizon}  model={state.model_type}\n")

    wrapper = _build_wrapper(state)
    wrapper.load()

    from main import run_benchmark, VerifyConfig
    vcfg = _build_verify_config(state, verbose=False)

    t0 = time.perf_counter()
    bench = run_benchmark(wrapper, rollout_config=None, verify_config=vcfg)
    elapsed = time.perf_counter() - t0

    # Print results table
    print(f"\n{'Spec ID':<40} {'Lv':>3} {'MP':>12} {'Verdict':>14} "
          f"{'ρ*':>7} {'ρ_net':>7} {'p̂_γ':>5} {'T':>5}")
    print("─" * 100)

    warrant_count = stl_margin_count = violation_count = 0
    for sid, r in bench.results.items():
        p_str = f"{r.p_hat:.3f}" if r.lppm else "  —"
        v_str = _verdict_colored(f"{r.verdict:<14}")
        print(
            f"{sid:<40} {r.level:>3} {r.mp_class:>12} {v_str} "
            f"{r.rho_star:>+7.3f} {r.rho_net:>+7.3f} {p_str:>5} {r.wall_time:>4.1f}s"
        )
        if r.verdict == "WARRANT":    warrant_count += 1
        elif r.verdict == "VIOLATION": violation_count += 1
        else:                          stl_margin_count += 1

    print("─" * 100)
    n = len(bench.results)
    print(f"  {green(f'WARRANT: {warrant_count}/{n}')}  "
          f"{yellow(f'STL-MARGIN: {stl_margin_count}/{n}')}  "
          f"{red(f'VIOLATION: {violation_count}/{n}')}  "
          f"  Total: {elapsed:.1f}s")

    input(dim("\n  Press Enter to return to menu..."))
    wrapper.close()


# ── helpers for building objects from state ───────────────────────────────────

def _build_wrapper(state: UIState):
    from wrappers import RandomWorldModelWrapper, DreamerV3Wrapper, RolloutConfig

    extra = {
        "env_name":    state.env_name,
        "latent_dim":  state.latent_dim,
        "use_decoder": state.use_decoder,
        "use_stoch":   state.use_stoch,
        "fidelity":    state.fidelity,
        "spec_type":   "always_safe",   # overridden per spec in benchmark
    }
    cfg = RolloutConfig(
        horizon=state.horizon,
        n_rollouts=state.n_rollouts,
        seed=state.seed,
        action_source=state.action_source,
        extra=extra,
    )

    if state.model_type == "dreamerv3":
        return DreamerV3Wrapper(cfg)
    else:
        return RandomWorldModelWrapper(cfg)


def _build_verify_config(state: UIState, verbose: bool = True):
    from main import VerifyConfig
    return VerifyConfig(
        delta_cp=state.delta_cp,
        delta_err=state.delta_err,
        model_error_budget=state.model_error_budget,
        gamma=state.gamma,
        eta=state.eta,
        warrant_threshold=state.warrant_threshold,
        fit_lppm_params=state.fit_lppm_params,
        lppm_epochs=state.lppm_epochs,
        verbose=verbose,
    )


def _run_and_show(state: UIState, spec: dict):
    """Build wrapper, run verify, print result."""
    from main import verify_from_wrapper, VerifyConfig
    from wrappers import RolloutConfig

    wrapper  = _build_wrapper(state)
    vcfg     = _build_verify_config(state)

    # update spec_type hint for random wrapper
    spec_type = spec["id"].replace("ltl_", "").replace("stl_", "")
    wrapper.config.extra["spec_type"] = spec_type

    print(f"  Loading {state.model_type}... ", end="", flush=True)
    wrapper.load(checkpoint_path=state.checkpoint_path or None)
    print("done")
    print(f"  Sampling {state.n_rollouts} rollouts × T={state.horizon}... ", end="", flush=True)
    trajs = wrapper.sample_rollouts()
    print("done\n")

    from main import verify
    result = verify(trajs, spec, vcfg)

    # Detailed result display
    print()
    print(bold("  ── Result ──────────────────────────────────────────"))
    print(f"  Verdict:       {_verdict_colored(result.verdict)}")
    print(f"  ρ* (worst STL): {result.rho_star:+.4f}")
    print(f"  ĉ_err (budget): {result.c_hat_err:.4f}")
    print(f"  ρ_net:          {result.rho_net:+.4f}  "
          + (green("✓ transfers") if result.transfer.transfers() else red("✗ insufficient")))
    if result.lppm:
        print(f"  p̂_γ (LPPM):    {result.p_hat:.3f}  "
              + (green("✓ warranted") if result.lppm.is_warranted() else yellow("✗ not warranted")))
        print(f"  avg descent η: {result.lppm.avg_descent_margin:.4f}")
    print(f"  Wall time:      {result.wall_time:.2f}s")

    if result.verdict == "VIOLATION":
        print(f"\n  {red('⚠ Witness rollout index:')} {result.monitor.witness_idx}")

    # Robustness distribution mini-chart
    _print_margin_chart(result.monitor.margins)

    wrapper.close()
    input(dim("\n  Press Enter to return to menu..."))


def _print_margin_chart(margins: list[float], width: int = 40):
    """ASCII bar chart of per-rollout STL margins."""
    print(f"\n  {dim('STL robustness distribution:')}")
    lo = min(margins + [-1.0])
    hi = max(margins + [1.0])
    rng = max(hi - lo, 1e-6)
    for i, m in enumerate(margins):
        bar_len = int((m - lo) / rng * width)
        bar     = "█" * bar_len
        color   = green if m > 0 else red
        print(f"  [{i:>3}] {color(bar):<{width+10}} {m:+.3f}")


# ── main loop ─────────────────────────────────────────────────────────────────

def run_ui():
    state = UIState()

    while True:
        _clear()
        try:
            action = screen_main(state)
        except (KeyboardInterrupt, EOFError):
            break

        if action.startswith("Configure world"):
            screen_model(state)
        elif action.startswith("Configure rollout"):
            screen_rollout(state)
        elif action.startswith("Configure verif"):
            screen_verify_params(state)
        elif action.startswith("Run verification"):
            screen_run_single(state)
        elif action.startswith("Run SAFEWORLD-BENCH"):
            screen_run_benchmark(state)
        elif action.startswith("Browse"):
            screen_browse_specs()
        elif action.startswith("Quit"):
            break

    print(bold(cyan("\nGoodbye.\n")))


if __name__ == "__main__":
    run_ui()
