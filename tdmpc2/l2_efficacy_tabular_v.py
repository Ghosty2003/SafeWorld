"""Track 1 control group 3: tabular V (one free scalar per state, no
function approximation at all) at strict p1_tol=0.0.

Pre-registered expectations (written before running, so post-hoc
interpretation cannot be second-guessed as cherry-picked):
  - true_safe_rate should recover to near-100%: with an independent
    trainable parameter per state and no shared function to introduce
    inter-state coupling noise, P1/P2 hinge loss can push every
    structurally-equivalent state to the exact same value, with no
    approximation-noise floor. No tolerance band should be needed.
  - false_safe_rate is EXPECTED to become nonzero: with Finding 1's noise
    source removed, Mechanism 1 (bounded 60x40-step sampling failing to
    witness a system's actual violating cycle at all -- see
    TRACK1_TOLERANCE_BAND_CONTROL_EXPERIMENT.md) is no longer masked by
    noise-driven abstention and should surface directly. This is the
    target measurement of this experiment, not a defect to explain away.
  - The two numbers must be read together as the calibrated route's honest
    efficacy portrait: recovering discriminative power on SAFE systems
    while exposing the REAL false-safe rate that was previously hidden
    under the noise floor.

Reuses the EXACT SAME 215 systems and the EXACT SAME
(seed=42, per-system torch_seed, RNG call order) sampling sequence as
every previous Track 1 run, so train/held-out rollouts are bit-identical
to prior experiments -- only the V parameterization changes (tabular
instead of embedding+MLP), and the latent-gradient-smoothness term is
dropped (no meaning for an unshared per-state scalar, and it would
reintroduce exactly the inter-state coupling this control group is
designed to remove).

Does not touch calibration data or Track B. Does not modify any prior
result file.
"""
from __future__ import annotations

import json
import pathlib
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from core.lppm.calibrator import _clopper_pearson_lower
from tdmpc2.l2_efficacy_run_pipeline import (
    sample_rollouts, witnessed_violation,
    N_TRAIN_ROLLOUTS, N_HELD_OUT_ROLLOUTS, ROLLOUT_LEN, ETA, GAMMA, SAFE_THRESHOLD,
)

SYSTEMS_DIR = pathlib.Path("artifacts/tdmpc2_l2_efficacy_study/systems")
OUT = pathlib.Path("artifacts/tdmpc2_l2_efficacy_study")
EPOCHS = 200  # tabular converges fast (no shared-function interference); generous budget to remove any doubt


class TabularV(nn.Module):
    def __init__(self, n_states):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(n_states))

    def forward(self, idx):
        return F.softplus(self.raw[idx])


def train_tabular(rollouts, state_to_idx, bad_set, epochs=EPOCHS, device="cpu"):
    cur_idx, next_idx, cur_bad = [], [], []
    for seq in rollouts:
        for i in range(len(seq) - 1):
            cur_idx.append(state_to_idx[seq[i]])
            next_idx.append(state_to_idx[seq[i + 1]])
            cur_bad.append(1.0 if seq[i] in bad_set else 0.0)
    cur_idx = torch.tensor(cur_idx, device=device)
    next_idx = torch.tensor(next_idx, device=device)
    cur_bad = torch.tensor(cur_bad, device=device)

    model = TabularV(len(state_to_idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(epochs):
        vc = model(cur_idx); vn = model(next_idx); dv = vn - vc
        p1 = F.relu(dv).mean()
        p2 = (cur_bad * F.relu(dv + ETA)).mean()
        loss = p1 + p2
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model


def pathwise_pass_strict(model, seq, state_to_idx, bad_set, device="cpu"):
    idx = torch.tensor([state_to_idx[s] for s in seq], device=device)
    with torch.no_grad():
        v = model(idx).cpu().numpy()
    for i in range(len(seq) - 1):
        delta = v[i + 1] - v[i]
        if delta > 1e-9:
            return False, i, float(delta)
        if seq[i] in bad_set and delta > -ETA + 1e-9:
            return False, i, float(delta)
    return True, None, None


def run():
    rng = random.Random(42)
    files = sorted(SYSTEMS_DIR.glob("*.json"))
    records = []

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
            records.append(dict(system_id=system["system_id"], kind=system["kind"],
                                 ground_truth=system["ground_truth"], l2_verdict="UNSAFE",
                                 route="witnessed_violation", p_hat_gamma=None, k=None, n=None))
            continue

        model = train_tabular(train_rollouts, state_to_idx, bad_set)
        held_out_rollouts = sample_rollouts(rng, transitions, initial_states, N_HELD_OUT_ROLLOUTS, ROLLOUT_LEN)
        details = [pathwise_pass_strict(model, seq, state_to_idx, bad_set) for seq in held_out_rollouts]
        passes = [d[0] for d in details]
        k = sum(passes); n = len(passes)
        p_hat_gamma = _clopper_pearson_lower(k, n, GAMMA)
        verdict = "SAFE" if p_hat_gamma >= SAFE_THRESHOLD else "ABSTAIN"
        # worst bad-source and worst safe-source deltas across ALL held-out
        # transitions (not just first failure) for false-safe attribution
        worst_bad = None
        worst_safe = None
        for seq in held_out_rollouts:
            idx = torch.tensor([state_to_idx[s] for s in seq])
            with torch.no_grad():
                v = model(idx).numpy()
            for j in range(len(seq) - 1):
                delta = float(v[j + 1] - v[j])
                if seq[j] in bad_set:
                    if worst_bad is None or delta > worst_bad:
                        worst_bad = delta
                else:
                    if worst_safe is None or delta > worst_safe:
                        worst_safe = delta
        records.append(dict(
            system_id=system["system_id"], kind=system["kind"], ground_truth=system["ground_truth"],
            l2_verdict=verdict, route="calibrated", p_hat_gamma=p_hat_gamma, k=k, n=n,
            worst_bad_source_delta=worst_bad, worst_safe_source_delta=worst_safe,
        ))
        if (i + 1) % 40 == 0:
            print(f"processed {i+1}/{len(files)}", flush=True)

    return records


def cp_upper(k, n, conf=0.95):
    from scipy.stats import beta
    if n == 0:
        return None
    if k == n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def summarize(records):
    unsafe = [r for r in records if r["ground_truth"] == "VIOLATION"]
    safe = [r for r in records if r["ground_truth"] == "SAFE"]
    calib_unsafe = [r for r in unsafe if r["route"] == "calibrated"]
    calib_safe = [r for r in safe if r["route"] == "calibrated"]
    n_false_safe = sum(1 for r in unsafe if r["l2_verdict"] == "SAFE")
    n_true_safe = sum(1 for r in safe if r["l2_verdict"] == "SAFE")

    def phat_stats(rs):
        vals = [r["p_hat_gamma"] for r in rs if r["p_hat_gamma"] is not None]
        if not vals:
            return dict(n=0)
        vs = sorted(vals)
        return dict(n=len(vals), min=vs[0], median=vs[len(vs) // 2], max=vs[-1],
                    mean=sum(vals) / len(vals), n_ge_0_95=sum(1 for v in vals if v >= 0.95),
                    n_zero=sum(1 for v in vals if v == 0.0))

    return dict(
        n_unsafe=len(unsafe), n_safe=len(safe),
        n_calib_unsafe=len(calib_unsafe), n_calib_safe=len(calib_safe),
        false_safe_rate=n_false_safe / len(unsafe) if unsafe else None,
        false_safe_rate_cp_upper_all=cp_upper(n_false_safe, len(unsafe)),
        false_safe_rate_cp_upper_calibrated_only=cp_upper(
            sum(1 for r in calib_unsafe if r["l2_verdict"] == "SAFE"), len(calib_unsafe)),
        n_false_safe=n_false_safe,
        false_safe_examples=[r["system_id"] for r in calib_unsafe if r["l2_verdict"] == "SAFE"],
        true_safe_rate=n_true_safe / len(safe) if safe else None,
        p_hat_gamma_on_safe_systems=phat_stats(calib_safe),
        p_hat_gamma_on_calibrated_unsafe_systems=phat_stats(calib_unsafe),
    )


if __name__ == "__main__":
    records = run()
    summary = summarize(records)
    print(json.dumps(summary, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tabular_v_result.json").write_text(json.dumps(dict(summary=summary, records=records), indent=2) + "\n")
