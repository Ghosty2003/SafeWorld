"""Train-only function search with explicit observed/synthetic edge accounting.

This is a diagnostic, not a replacement certificate or warranting rule.
No time or trajectory index is supplied to the candidate functions.
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.lppm import build_parity_automaton
from core.lppm.model import get_or_build_lppm_model
from core.lppm.verifier import iter_transitions, run_product_trajectory
from specs import get_spec_by_id

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/safedreamer_l2_diagnostics"
ETA = 0.01
STEP = 0.011
KEYS = ["goal_dist", "hazard_dist", "velocity"]


def vector(s):
    return np.array([s.z[k] for k in KEYS], dtype=np.float64)


def identity(s):
    return (s.q, *vector(s))


def summarize(paths, predict, dpa, synthetic=False):
    counts = dict(p1_checked=0, p1_violations=0, p2_checked=0,
                  p2_violations=0, paths_p1p2_pass=0, terminal_event=0)
    margins, values = [], []
    for path in paths:
        edges = iter_transitions(path, dpa) if synthetic else list(zip(path, path[1:]))
        a = predict([s for s, _ in edges])
        b = predict([s for _, s in edges])
        odd = np.array([s.priority % 2 == 1 for s, _ in edges])
        p1 = int(np.sum(b[~odd] > a[~odd]))
        p2 = int(np.sum(a[odd] - b[odd] < ETA))
        counts['p1_checked'] += int((~odd).sum())
        counts['p2_checked'] += int(odd.sum())
        counts['p1_violations'] += p1
        counts['p2_violations'] += p2
        counts['paths_p1p2_pass'] += int(p1 == p2 == 0)
        terminal = edges[-1][1]
        counts['terminal_event'] += int(p1 == p2 == 0 and terminal.priority % 2 == 0 and b[-1] < ETA)
        margins.extend((a[odd] - b[odd]).tolist())
        values.extend(a.tolist() + b.tolist())
    counts.update(n_paths=len(paths), min_p2_margin=min(margins),
                  v_min=min(values), v_max=max(values))
    return counts


def main():
    torch.set_num_threads(1)
    cache = json.loads((ART / 'imagination_holdout_rollouts.json').read_text())
    spec = get_spec_by_id('ltl_goal_reach')
    dpa = build_parity_automaton(spec)
    train = [run_product_trajectory(t, dpa, spec) for t in cache['train']]
    edges = [e for p in train for e in zip(p, p[1:])]
    selfloops = []
    for i, p in enumerate(train):
        for a, b in iter_transitions(p, dpa):
            if a.priority % 2 and identity(a) == identity(b):
                selfloops.append(dict(trajectory=i, t=a.t, q=a.q,
                                      synthetic=(a.t == len(p)-1)))
    allstates = [s for p in train for s in p]
    x = np.array([vector(s) for s in allstates])
    mu, sd = x.mean(0), x.std(0)
    sd = np.maximum(sd, 1e-8)

    # Original model: inspect pre-softplus logits, without refitting it.
    saved = torch.load(ART / 'goal_reach_v_phi.pt', map_location='cpu', weights_only=False)
    original = get_or_build_lppm_model(saved)
    with torch.no_grad():
        z = torch.tensor([[s.z[k] for k in saved['feature_keys']] for s in allstates], dtype=torch.float32)
        q = torch.tensor([saved['state_to_idx'][s.q] for s in allstates])
        logits = original.head(original.mlp(torch.cat([z, original.q_embedding(q)], -1)))
    report = dict(spec='ltl_goal_reach', eta=ETA, fit_step=STEP,
                  features=KEYS, original_bad_selfloops=selfloops,
                  original_logits=dict(min=float(logits.min()), max=float(logits.max())),
                  selection='train only; calibration/test reused for exploratory evaluation, not a fresh confirmatory claim',
                  scope='observed adjacent states; no fabricated final dynamics edge; finite sampled constraints only')
    print('Original impossible P2 self-loops:', len(selfloops), flush=True)
    print('Original pre-softplus logits:', report['original_logits'], flush=True)

    # Exact graph ranking; accepting states have value zero. Every waiting
    # node receives STEP times one plus its largest successor waiting rank.
    nodes = {identity(s): s for s in allstates if s.priority % 2}
    succ = {k: set() for k in nodes}
    pred = {k: set() for k in nodes}
    for a, b in edges:
        if a.priority % 2 and b.priority % 2:
            ka, kb = identity(a), identity(b)
            succ[ka].add(kb)
            pred[kb].add(ka)
    degree = {k: len(v) for k, v in succ.items()}
    queue = deque(k for k in nodes if degree[k] == 0)
    ranks = {k: 1 for k in nodes}
    processed = 0
    while queue:
        k = queue.popleft()
        processed += 1
        for parent in pred[k]:
            ranks[parent] = max(ranks[parent], ranks[k]+1)
            degree[parent] -= 1
            if degree[parent] == 0:
                queue.append(parent)
    if processed != len(nodes):
        raise RuntimeError('Observed waiting-state graph contains a strict cycle')
    centers = np.array([vector(nodes[k]) for k in nodes])
    labels = np.array([STEP * ranks[k] for k in nodes])
    tree = cKDTree((centers-mu)/sd)

    def table_predict(states):
        _, indices = tree.query((np.array([vector(s) for s in states])-mu)/sd)
        return np.array([labels[i] if s.priority % 2 else 0.0 for i, s in zip(indices, states)])

    candidates = {'graph_nearest_neighbor': table_predict}
    report['graph'] = dict(nodes=len(nodes), max_rank=max(ranks.values()))
    report['function_search'] = []
    # Smooth low-order candidates with exact finite-sample linear constraints.
    def basis(states, quadratic):
        v = (np.array([vector(s) for s in states])-mu)/sd
        cols = [np.ones(len(v)), *v.T]
        if quadratic:
            cols.extend(v[:, i]*v[:, j] for i in range(3) for j in range(i, 3))
        out = np.stack(cols, 1)
        out[np.array([s.priority % 2 == 0 for s in states])] = 0
        return out

    a_states, b_states = zip(*edges)
    bad = np.array([a.priority % 2 == 1 for a in a_states])
    for quadratic in (False, True):
        ba, bb = basis(a_states, quadratic), basis(b_states, quadratic)
        bs = basis(allstates, quadratic)
        fit = linprog(np.zeros(ba.shape[1]), A_ub=np.concatenate([bb-ba, -bs]),
                      b_ub=np.concatenate([-STEP*bad, np.zeros(len(bs))]),
                      bounds=[(None, None)]*ba.shape[1], method='highs')
        name = 'quadratic_lp' if quadratic else 'affine_lp'
        report['function_search'].append(dict(name=name, feasible=bool(fit.success), message=fit.message))
        print(name, fit.message, flush=True)
        if fit.success:
            candidates[name] = lambda states, c=fit.x, quadratic=quadratic: np.maximum(0, basis(states, quadratic) @ c)

    # Normalized MLPs: accepting branch fixed to zero; a positive waiting
    # offset prevents numerical zero saturation. All hyperparameters fixed
    # here and evaluated/selected on TRAIN ONLY.
    za = torch.tensor((np.array([vector(s) for s in a_states])-mu)/sd, dtype=torch.float32)
    zb = torch.tensor((np.array([vector(s) for s in b_states])-mu)/sd, dtype=torch.float32)
    ma = torch.tensor([s.priority % 2 for s in a_states], dtype=torch.float32)
    mb = torch.tensor([s.priority % 2 for s in b_states], dtype=torch.float32)
    yt = torch.tensor(table_predict(a_states), dtype=torch.float32)
    trained = {}
    for seed, supervised in ((0, False), (1, False), (0, True)):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(3, 128), torch.nn.Tanh(),
                                  torch.nn.Linear(128, 128), torch.nn.Tanh(),
                                  torch.nn.Linear(128, 1))
        opt = torch.optim.Adam(net.parameters(), lr=0.001)
        for epoch in range(1200):
            va = (torch.nn.functional.softplus(net(za).squeeze(-1)) + STEP)*ma
            vb = (torch.nn.functional.softplus(net(zb).squeeze(-1)) + STEP)*mb
            gap = torch.relu(vb-va + STEP*ma)
            loss = gap.mean() + gap.max()
            if supervised:
                loss = loss + 0.1*(va-yt).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        name = f'normalized_mlp_seed{seed}_rank_target{supervised}'
        def predict(states, net=net):
            with torch.no_grad():
                inp = torch.tensor((np.array([vector(s) for s in states])-mu)/sd, dtype=torch.float32)
                out = torch.nn.functional.softplus(net(inp).squeeze(-1)).numpy()+STEP
            return out*np.array([s.priority % 2 for s in states])
        candidates[name] = predict
        trained[name] = net.state_dict()
        print(name, summarize(train, predict, dpa), flush=True)

    # Freeze before any heldout evaluation. Table ranking is selected on
    # train feasibility; it has no deductive/generalization guarantee.
    report['selected'] = 'graph_nearest_neighbor'
    report['candidates'] = {}
    for name, predict in candidates.items():
        report['candidates'][name] = {'train_observed': summarize(train, predict, dpa),
                                     'train_original': summarize(train, predict, dpa, synthetic=True)}
    for split in ('calibration', 'test'):
        paths = [run_product_trajectory(t, dpa, spec) for t in cache[split]]
        report['candidates']['graph_nearest_neighbor'][split] = summarize(paths, table_predict, dpa)
    np.savez(ART / 'goal_v_feasibility_table.npz', centers=centers, values=labels, mean=mu, scale=sd)
    torch.save(dict(models=trained, mean=mu.tolist(), scale=sd.tolist(), features=KEYS, offset=STEP),
               ART / 'goal_v_feasibility_mlps.pt')
    (ART / 'goal_v_feasibility_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report['candidates']['graph_nearest_neighbor'], indent=2), flush=True)
    print('Saved goal_v_feasibility_report.json', flush=True)


if __name__ == '__main__':
    main()
