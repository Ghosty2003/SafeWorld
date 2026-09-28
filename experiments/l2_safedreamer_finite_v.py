"""Evaluate an explicit finite-horizon progress function on cached rollouts.

The analytic candidate is

    V_H(z, q, t) = (H - t) * eta_step,

where eta_step > eta makes the strict floating-point P2 check robust.  This
script deliberately reports two separate events:

1. P1/P2 success, which is solely about the candidate progress function.
2. The full finite-horizon certificate event, which additionally requires an
   accepting/even terminal automaton state and terminal membership in
   Z_free^H.

Thus an objective Safety violation is never hidden by a successful P1/P2
construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.lppm import build_parity_automaton
from core.lppm.calibrator import _clopper_pearson_lower
from core.lppm.verifier import iter_transitions, run_product_trajectory
from specs import get_spec_by_id
from utils.spec_analysis import analyze_spec_structure


def finite_clock_v(t: int, horizon: int, eta_step: float) -> float:
    if not 0 <= t <= horizon:
        raise ValueError(f"t={t} lies outside [0, {horizon}]")
    return float((horizon - t) * eta_step)


def evaluate(trajectories, spec, *, horizon, eta, eta_step, gamma):
    if eta_step < eta:
        raise ValueError("eta_step must be at least the required P2 margin eta")
    dpa = build_parity_automaton(spec)
    p1_checked = p2_checked = 0
    p1_violations = p2_violations = 0
    p1p2_successes = 0
    accepting_terminals = 0
    terminal_zfree_successes = 0
    certificate_successes = 0
    per_trajectory = []
    tolerance = 1e-12

    for index, trajectory in enumerate(trajectories):
        if len(trajectory) != horizon:
            raise ValueError(
                f"trajectory {index} has length {len(trajectory)}, expected {horizon}"
            )
        transitions = iter_transitions(run_product_trajectory(trajectory, dpa, spec), dpa)
        trajectory_p1 = trajectory_p2 = 0

        for curr, nxt in transitions:
            v_curr = finite_clock_v(curr.t, horizon, eta_step)
            v_next = finite_clock_v(nxt.t, horizon, eta_step)
            for odd_priority in dpa.odd_priorities:
                if curr.priority == odd_priority:
                    p2_checked += 1
                    if v_curr - v_next + tolerance < eta:
                        p2_violations += 1
                        trajectory_p2 += 1
                elif odd_priority > curr.priority:
                    p1_checked += 1
                    if v_next > v_curr + tolerance:
                        p1_violations += 1
                        trajectory_p1 += 1

        p1p2_ok = trajectory_p1 == 0 and trajectory_p2 == 0
        p1p2_successes += int(p1p2_ok)

        terminal = transitions[-1][1]
        terminal_priority = dpa.priority.get(terminal.q, 0)
        accepting_terminal = terminal_priority % 2 == 0
        accepting_terminals += int(accepting_terminal)
        terminal_v = finite_clock_v(terminal.t, horizon, eta_step)
        terminal_zfree = accepting_terminal and terminal_v < eta
        terminal_zfree_successes += int(terminal_zfree)
        certificate_ok = p1p2_ok and terminal_zfree
        certificate_successes += int(certificate_ok)
        per_trajectory.append({
            "index": index,
            "p1_violations": trajectory_p1,
            "p2_violations": trajectory_p2,
            "p1p2_satisfied": p1p2_ok,
            "terminal_q": terminal.q,
            "terminal_priority": terminal_priority,
            "terminal_v": terminal_v,
            "terminal_in_zfree_h": terminal_zfree,
            "certificate_satisfied": certificate_ok,
        })

    n = len(trajectories)
    return {
        "candidate": {
            "formula": "V_H(z,q,t) = (H-t) * eta_step",
            "horizon": horizon,
            "eta_required": eta,
            "eta_step": eta_step,
            "v_initial": finite_clock_v(0, horizon, eta_step),
            "v_terminal": finite_clock_v(horizon, horizon, eta_step),
        },
        "p1_p2": {
            "p1_violations": p1_violations,
            "p1_checked": p1_checked,
            "p2_violations": p2_violations,
            "p2_checked": p2_checked,
            "trajectory_successes": p1p2_successes,
            "n": n,
            "cp_lower": _clopper_pearson_lower(p1p2_successes, n, gamma),
        },
        "terminal": {
            "accepting_even": accepting_terminals,
            "odd_trap": n - accepting_terminals,
            "terminal_zfree_h_successes": terminal_zfree_successes,
            "n": n,
        },
        "full_certificate": {
            "successes": certificate_successes,
            "n": n,
            "p_hat_gamma": _clopper_pearson_lower(certificate_successes, n, gamma),
        },
        "per_trajectory": per_trajectory,
    }


def main(args):
    cache_path = Path(args.rollout_cache)
    payload = json.loads(cache_path.read_text())
    metadata = payload["metadata"]
    if metadata["horizon"] != args.horizon:
        raise ValueError(
            f"cache horizon={metadata['horizon']} but requested horizon={args.horizon}"
        )
    spec = get_spec_by_id(args.spec)
    if spec is None:
        raise ValueError(f"Unknown specification: {args.spec}")
    spec = dict(spec)
    spec["analysis"] = analyze_spec_structure(spec)
    eta_step = args.eta * args.eta_step_multiplier
    result = evaluate(
        payload["calibration"], spec,
        horizon=args.horizon,
        eta=args.eta,
        eta_step=eta_step,
        gamma=args.gamma,
    )

    candidate = result["candidate"]
    p12 = result["p1_p2"]
    terminal = result["terminal"]
    full = result["full_certificate"]
    print("Finite-horizon analytic V")
    print(f"  V(t)=(H-t)*eta_step, H={candidate['horizon']}, "
          f"eta={candidate['eta_required']}, eta_step={candidate['eta_step']}")
    print(f"  V(0)={candidate['v_initial']:.6f}, V(H)={candidate['v_terminal']:.6f}")
    print("P1/P2-only result")
    print(f"  P1 violating transitions: {p12['p1_violations']} / {p12['p1_checked']}")
    print(f"  P2 violating transitions: {p12['p2_violations']} / {p12['p2_checked']}")
    print(f"  P1/P2 trajectory successes: {p12['trajectory_successes']}/{p12['n']}")
    print(f"  P1/P2 CP lower bound: {p12['cp_lower']:.6f}")
    print("Terminal safety and finite Z_free^H")
    print(f"  accepting/even terminals: {terminal['accepting_even']}/{terminal['n']}")
    print(f"  odd-trap terminals: {terminal['odd_trap']}/{terminal['n']}")
    print(f"  terminal Z_free^H successes: "
          f"{terminal['terminal_zfree_h_successes']}/{terminal['n']}")
    print("Full finite-horizon certificate")
    print(f"  successes: {full['successes']}/{full['n']}")
    print(f"  p_hat_gamma: {full['p_hat_gamma']:.6f}")
    print(f"  result: {'FINITE_HORIZON_WARRANT' if full['p_hat_gamma'] >= args.warrant_threshold else 'NO_WARRANT'}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="ltl_hazard_avoidance")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--eta-step-multiplier", type=float, default=1.01)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--warrant-threshold", type=float, default=0.80)
    parser.add_argument(
        "--rollout-cache",
        default="artifacts/safedreamer_l2_diagnostics/v_sweep_rollouts.json",
    )
    parser.add_argument(
        "--report",
        default="artifacts/safedreamer_l2_diagnostics/finite_v_report.json",
    )
    main(parser.parse_args())
