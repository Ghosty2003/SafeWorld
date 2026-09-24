"""Diagnose the mechanism behind p_hat_gamma collapsing to exactly 0.0 on
the calibrated route: is every held-out pathwise failure noise-level (small
numeric residue, matching Finding 1's free-MLP-noise story), or are there
genuinely large/structural V jumps introduced by the diversity-fix
branches?

Must replay the FULL 140-system sequence in file order to keep the shared
`random.Random(seed)` stream synchronized with the actual batch run (every
system, whether or not selected for detailed diagnosis, consumes RNG calls
for train/held-out sampling) -- otherwise the reproduced rollouts and
selected-system indices would not match track1_full_result.json.

Does not touch calibration data or Track B; read-only diagnosis, no new
generator changes.
"""
from __future__ import annotations

import json
import pathlib
import random

import numpy as np
import torch

from tdmpc2.l2_efficacy_run_pipeline import (
    sample_rollouts, witnessed_violation, train_v,
    N_TRAIN_ROLLOUTS, N_HELD_OUT_ROLLOUTS, ROLLOUT_LEN, ETA,
)

SYSTEMS_DIR = pathlib.Path("artifacts/tdmpc2_l2_efficacy_study/systems")
OUT = pathlib.Path("artifacts/tdmpc2_l2_efficacy_study")

TARGETS = [
    "safe_unreachable_000", "safe_unreachable_003",
    "safe_recoverable_000",
    "unsafe_transient_cycle_001", "unsafe_transient_cycle_002",
]


def detailed_pathwise(model, seq, state_to_idx, bad_set, device="cpu"):
    idx = torch.tensor([state_to_idx[s] for s in seq], device=device)
    with torch.no_grad():
        v = model(idx).cpu().numpy()
    first_fail_step = None
    fail_kind = None
    fail_delta = None
    for i in range(len(seq) - 1):
        delta = float(v[i + 1] - v[i])
        is_bad = seq[i] in bad_set
        p1_fail = delta > 1e-9
        p2_fail = is_bad and delta > -ETA + 1e-9
        if p1_fail or p2_fail:
            first_fail_step = i
            if p1_fail and is_bad:
                fail_kind = "P1_violation_on_bad_source"
            elif p1_fail:
                fail_kind = "P1_violation_on_safe_source"
            else:
                fail_kind = "P2_insufficient_descent"
            fail_delta = delta
            break
    return dict(
        passed=first_fail_step is None,
        first_fail_step=first_fail_step, fail_kind=fail_kind, fail_delta=fail_delta,
        v_values=[float(x) for x in v],
    )


def run():
    rng = random.Random(42)
    files = sorted(SYSTEMS_DIR.glob("*.json"))
    diagnostics = {}

    for i, path in enumerate(files):
        system = json.loads(path.read_text())
        transitions = {k: tuple(v) for k, v in system["transitions"].items()}
        initial_states = tuple(system["initial_states"])
        bad_set = set(system["bad_states"])
        all_states = sorted(transitions.keys())
        state_to_idx = {s: idx for idx, s in enumerate(all_states)}

        torch.manual_seed(42 * 100000 + i)
        train_rollouts = sample_rollouts(rng, transitions, initial_states, N_TRAIN_ROLLOUTS, ROLLOUT_LEN)
        witness = witnessed_violation(train_rollouts, bad_set)
        if witness is not None:
            continue
        model = train_v(train_rollouts, state_to_idx, bad_set)
        held_out_rollouts = sample_rollouts(rng, transitions, initial_states, N_HELD_OUT_ROLLOUTS, ROLLOUT_LEN)

        if system["system_id"] not in TARGETS:
            continue

        per_rollout = []
        for seq in held_out_rollouts:
            d = detailed_pathwise(model, seq, state_to_idx, bad_set)
            per_rollout.append(d)
        n_pass = sum(d["passed"] for d in per_rollout)
        fail_steps = [d["first_fail_step"] for d in per_rollout if not d["passed"]]
        fail_deltas = [d["fail_delta"] for d in per_rollout if not d["passed"]]
        fail_kinds = [d["fail_kind"] for d in per_rollout if not d["passed"]]
        diagnostics[system["system_id"]] = dict(
            kind=system["kind"], ground_truth=system["ground_truth"],
            n_held_out=len(held_out_rollouts), n_pass=n_pass,
            fail_step_min=min(fail_steps) if fail_steps else None,
            fail_step_median=sorted(fail_steps)[len(fail_steps) // 2] if fail_steps else None,
            fail_step_max=max(fail_steps) if fail_steps else None,
            fail_delta_abs_min=float(min(abs(x) for x in fail_deltas)) if fail_deltas else None,
            fail_delta_abs_median=float(sorted(abs(x) for x in fail_deltas)[len(fail_deltas) // 2]) if fail_deltas else None,
            fail_delta_abs_max=float(max(abs(x) for x in fail_deltas)) if fail_deltas else None,
            fail_kind_counts={k: fail_kinds.count(k) for k in set(fail_kinds)} if fail_kinds else {},
            example_rollouts=[
                dict(seq=seq[:d["first_fail_step"] + 2] if d["first_fail_step"] is not None else seq[:5],
                     first_fail_step=d["first_fail_step"], fail_kind=d["fail_kind"], fail_delta=d["fail_delta"],
                     v_around_failure=(
                         d["v_values"][max(0, d["first_fail_step"] - 1): d["first_fail_step"] + 3]
                         if d["first_fail_step"] is not None else None
                     ))
                for seq, d in list(zip(held_out_rollouts, per_rollout))[:3]
            ],
        )
        print(system["system_id"], "n_pass=", n_pass, "/", len(held_out_rollouts),
              "fail_delta_median=", diagnostics[system["system_id"]]["fail_delta_abs_median"],
              "fail_delta_max=", diagnostics[system["system_id"]]["fail_delta_abs_max"], flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "zero_phat_diagnosis.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return diagnostics


if __name__ == "__main__":
    run()
