"""L2-efficacy study, Track 1: run the sampled-access L2 pipeline (NOT the
exhaustive checker) on each generated finite system, and compare its
verdict against the exact ground truth from finite_graph.py.

Pipeline per system, using ONLY resettable-generative sampling (never the
full transition table):
  1. Sample TRAIN rollouts (random walks from a random initial state,
     uniformly choosing among the successors actually offered at each
     step -- treating multiple successors as the environment's own
     stochastic/nondeterministic behavior, not agent-chosen actions).
  2. WITNESSED-VIOLATION check on the train rollouts: if any single
     rollout revisits the SAME state twice with at least one bad state
     strictly between the two visits, that is an exact, sound, finite
     counterexample cycle (mirrors this project's established "bit-exact
     repeated state" methodology for real MuJoCo rollouts) -- if found,
     L2 verdict = UNSAFE immediately, no training needed.
  3. Otherwise, train a V (state embedding -> 128 ReLU -> 128 ReLU ->
     softplus, Appendix C.1 hinge loss: P1 = mean(relu(V(succ)-V(cur))),
     P2 = mean(bad(cur)*relu(V(succ)-V(cur)+eta)); no latent-smoothness
     term -- there is no continuous latent here, states are discrete
     symbols, so smoothness has no meaning and is omitted, not faked).
  4. Sample fresh HELD-OUT rollouts (disjoint resets). For each, check
     PATHWISE P1/P2 (the whole observed rollout satisfies P1 always and
     P2 whenever bad). k = number passing, n = held-out rollout count.
  5. p_hat_gamma = one-sided (1-gamma) Clopper-Pearson lower bound on
     k/n (core.lppm.calibrator._clopper_pearson_lower, reused verbatim --
     this project's established Theorem 5.4 estimator, not reinvented).
  6. Verdict mapping (frozen BEFORE seeing any results, see
     artifacts/tdmpc2_l2_efficacy_study/DEFINITIONS.md):
       - witnessed violation in step 2 -> L2 = UNSAFE
       - else p_hat_gamma >= SAFE_THRESHOLD -> L2 = SAFE
       - else -> L2 = ABSTAIN

Does not touch calibration data or Track B.
"""
from __future__ import annotations

import json
import pathlib
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from core.lppm.calibrator import _clopper_pearson_lower

SYSTEMS_DIR = pathlib.Path("artifacts/tdmpc2_l2_efficacy_study/systems")
OUT = pathlib.Path("artifacts/tdmpc2_l2_efficacy_study")
ETA = 0.01
GAMMA = 0.05
SAFE_THRESHOLD = 0.95
N_TRAIN_ROLLOUTS = 60
N_HELD_OUT_ROLLOUTS = 60
ROLLOUT_LEN = 40
EPOCHS = 60
HIDDEN_DIM = 64


def sample_rollouts(rng, transitions, initial_states, n_rollouts, rollout_len):
    rollouts = []
    states_list = list(transitions.keys())
    for _ in range(n_rollouts):
        s = rng.choice(initial_states)
        seq = [s]
        for _ in range(rollout_len):
            succs = transitions.get(s)
            if not succs:
                break
            s = rng.choice(succs)
            seq.append(s)
        rollouts.append(seq)
    return rollouts


def witnessed_violation(rollouts, bad_set):
    for seq in rollouts:
        first_seen = {}
        for i, s in enumerate(seq):
            if s in first_seen:
                j = first_seen[s]
                if any(x in bad_set for x in seq[j:i]):
                    return dict(rollout=seq, cycle_start=j, cycle_end=i,
                                cycle=seq[j:i + 1])
            else:
                first_seen[s] = i
    return None


class GraphV(nn.Module):
    def __init__(self, n_states, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embed = nn.Embedding(n_states, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, idx):
        return F.softplus(self.mlp(self.embed(idx))).squeeze(-1)


def train_v(rollouts, state_to_idx, bad_set, epochs=EPOCHS, device="cpu"):
    cur_idx, next_idx, cur_bad = [], [], []
    for seq in rollouts:
        for i in range(len(seq) - 1):
            cur_idx.append(state_to_idx[seq[i]])
            next_idx.append(state_to_idx[seq[i + 1]])
            cur_bad.append(1.0 if seq[i] in bad_set else 0.0)
    if not cur_idx:
        return None
    cur_idx = torch.tensor(cur_idx, device=device)
    next_idx = torch.tensor(next_idx, device=device)
    cur_bad = torch.tensor(cur_bad, device=device)

    model = GraphV(len(state_to_idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        vc = model(cur_idx); vn = model(next_idx); dv = vn - vc
        p1 = F.relu(dv).mean()
        p2 = (cur_bad * F.relu(dv + ETA)).mean()
        loss = p1 + p2
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model


def pathwise_pass(model, seq, state_to_idx, bad_set, device="cpu"):
    idx = torch.tensor([state_to_idx[s] for s in seq], device=device)
    with torch.no_grad():
        v = model(idx).cpu().numpy()
    for i in range(len(seq) - 1):
        delta = v[i + 1] - v[i]
        if delta > 1e-9:
            return False
        if seq[i] in bad_set and delta > -ETA + 1e-9:
            return False
    return True


def run_one_system(system, rng, torch_seed):
    torch.manual_seed(torch_seed)
    transitions = {k: tuple(v) for k, v in system["transitions"].items()}
    initial_states = tuple(system["initial_states"])
    bad_set = set(system["bad_states"])
    all_states = sorted(transitions.keys())
    state_to_idx = {s: i for i, s in enumerate(all_states)}

    started = time.time()
    train_rollouts = sample_rollouts(rng, transitions, initial_states, N_TRAIN_ROLLOUTS, ROLLOUT_LEN)
    witness = witnessed_violation(train_rollouts, bad_set)
    if witness is not None:
        return dict(
            system_id=system["system_id"], kind=system["kind"], ground_truth=system["ground_truth"],
            l2_verdict="UNSAFE", route="witnessed_violation", witness=witness,
            p_hat_gamma=None, k=None, n=None, elapsed_seconds=time.time() - started,
        )

    model = train_v(train_rollouts, state_to_idx, bad_set)
    held_out_rollouts = sample_rollouts(rng, transitions, initial_states, N_HELD_OUT_ROLLOUTS, ROLLOUT_LEN)
    # A held-out rollout that itself witnesses a violation is an automatic
    # fail for that rollout's pathwise check (not a separate global override
    # here -- consistent with the same pathwise-satisfaction definition used
    # for k/n everywhere else in this project's calibration machinery).
    passes = [pathwise_pass(model, seq, state_to_idx, bad_set) for seq in held_out_rollouts]
    k = sum(passes); n = len(passes)
    p_hat_gamma = _clopper_pearson_lower(k, n, GAMMA)
    verdict = "SAFE" if p_hat_gamma >= SAFE_THRESHOLD else "ABSTAIN"
    return dict(
        system_id=system["system_id"], kind=system["kind"], ground_truth=system["ground_truth"],
        l2_verdict=verdict, route="calibrated", p_hat_gamma=p_hat_gamma, k=k, n=n,
        elapsed_seconds=time.time() - started,
    )


def run(system_files, seed=0):
    rng = random.Random(seed)
    results = []
    for i, path in enumerate(system_files):
        system = json.loads(path.read_text())
        result = run_one_system(system, rng, torch_seed=seed * 100000 + i)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "witness"}), flush=True)
    return results


def summarize(results):
    confusion = {}
    for r in results:
        key = (r["kind"], r["ground_truth"], r["l2_verdict"])
        confusion[key] = confusion.get(key, 0) + 1
    n_gt_unsafe = sum(1 for r in results if r["ground_truth"] == "VIOLATION")
    n_false_safe = sum(1 for r in results if r["ground_truth"] == "VIOLATION" and r["l2_verdict"] == "SAFE")
    n_recall_unsafe = sum(1 for r in results if r["ground_truth"] == "VIOLATION" and r["l2_verdict"] == "UNSAFE")
    n_recall_abstain = sum(1 for r in results if r["ground_truth"] == "VIOLATION" and r["l2_verdict"] == "ABSTAIN")
    n_gt_safe = sum(1 for r in results if r["ground_truth"] == "SAFE")
    n_true_safe = sum(1 for r in results if r["ground_truth"] == "SAFE" and r["l2_verdict"] == "SAFE")
    n_safe_abstain = sum(1 for r in results if r["ground_truth"] == "SAFE" and r["l2_verdict"] == "ABSTAIN")
    n_safe_falseunsafe = sum(1 for r in results if r["ground_truth"] == "SAFE" and r["l2_verdict"] == "UNSAFE")
    return dict(
        n_systems=len(results),
        n_ground_truth_unsafe=n_gt_unsafe, n_ground_truth_safe=n_gt_safe,
        false_safe_rate=n_false_safe / n_gt_unsafe if n_gt_unsafe else None,
        recall_unsafe_correctly_flagged_unsafe_rate=n_recall_unsafe / n_gt_unsafe if n_gt_unsafe else None,
        recall_unsafe_correctly_abstained_rate=n_recall_abstain / n_gt_unsafe if n_gt_unsafe else None,
        n_false_safe=n_false_safe, n_recall_unsafe=n_recall_unsafe, n_recall_abstain=n_recall_abstain,
        true_safe_rate_on_safe_systems=n_true_safe / n_gt_safe if n_gt_safe else None,
        safe_system_wrongly_abstained_rate=n_safe_abstain / n_gt_safe if n_gt_safe else None,
        safe_system_wrongly_flagged_unsafe_rate=n_safe_falseunsafe / n_gt_safe if n_gt_safe else None,
        confusion_by_kind={f"{k[0]}|gt={k[1]}|l2={k[2]}": v for k, v in confusion.items()},
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only run the first N system files (feasibility check)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    files = sorted(SYSTEMS_DIR.glob("*.json"))
    if args.limit:
        # take a spread across kinds, not just the first N alphabetically
        by_kind = {}
        for f in files:
            kind = f.stem.rsplit("_", 1)[0]
            by_kind.setdefault(kind, []).append(f)
        per_kind = max(1, args.limit // len(by_kind))
        files = [f for lst in by_kind.values() for f in lst[:per_kind]]

    started = time.time()
    results = run(files, seed=args.seed)
    summary = summarize(results)
    summary["elapsed_seconds_total"] = time.time() - started
    summary["mean_seconds_per_system"] = summary["elapsed_seconds_total"] / max(1, len(results))
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = pathlib.Path(args.output) if args.output else OUT / "pipeline_feasibility_result.json"
    out_path.write_text(json.dumps(dict(summary=summary, results=results), indent=2) + "\n")
    print(json.dumps(summary, indent=2))
