"""Train-only representation search, with adaptive development-set selection.

No calibration/test loading or rollout collection is implemented here. Repeated
validation selection is NOT an independent confidence/accuracy assessment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.lppm.latent_reachability import audit_values, fit_scaler, normalize
from experiments.l2_latent_v_refine import combined
from experiments.l2_latent_v_ensemble import OUT as BASE, bundle_predictor
from experiments.l2_latent_v_search import net_for, FLOOR

OUT = BASE.parent / 'safedreamer_l2_validation_search'
ETA = .01
CONFIGS = [
    dict(name='pca16', rank=16, width=128, multistep=0., noise=0., seed=71),
    dict(name='pca32', rank=32, width=128, multistep=0., noise=0., seed=72),
    dict(name='pca64', rank=64, width=256, multistep=0., noise=0., seed=73),
    dict(name='pca128', rank=128, width=256, multistep=0., noise=0., seed=74),
    dict(name='pca32_multistep', rank=32, width=128, multistep=.3, noise=0., seed=75),
    dict(name='pca64_multistep', rank=64, width=256, multistep=.3, noise=0., seed=76),
    dict(name='deter_multistep', rank=0, width=256, multistep=.3, noise=0., seed=77),
    dict(name='deter_smooth', rank=0, width=256, multistep=.3, noise=.02, seed=78),
]


def development_audit(values, done):
    row = audit_values(values, done, eta=ETA)
    for value in row.values():
        if isinstance(value, dict):
            value.pop('confidence', None)
            value.pop('cp_lower', None)
    return row


def transform(x, model):
    z = normalize(x, model['mean'], model['scale'])
    latent = z[..., :256]
    if model['projection'] is not None:
        latent = latent @ model['projection']
    return np.concatenate([latent, z[..., 512:515]], axis=-1).astype(np.float32)


def predictor(model):
    if model['kind'] == 'baseline':
        return bundle_predictor(model['bundle'])
    if model['kind'] == 'combination':
        functions = [predictor(m) for m in model['members']]
        def predict_combined(x):
            values = np.stack([f(x) for f in functions]).astype(np.float64)
            return getattr(values, model['mode'])(axis=0)
        return predict_combined
    dim = (model['projection'].shape[1] if model['projection'] is not None else 256) + 3
    net = net_for(model['width'], dim, 'silu')
    net.load_state_dict(model['weights'])
    net.eval()
    def predict(x):
        with torch.no_grad():
            raw = net(torch.from_numpy(transform(x, model))).squeeze(-1)
            return model['gain'] * (FLOOR + torch.nn.functional.softplus(raw).numpy().astype(np.float64))
    return predict


def score(row):
    tr, va = row['train'], row['validation']
    # Never replace the baseline by a candidate that simply sacrifices train fit.
    return (tr['p2']['violations'] / max(1, tr['p2']['checked']) <= .01,
            -va['p2']['violations'], -tr['p2']['violations'])


def search(steps):
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    OUT.mkdir(parents=True)
    torch.set_num_threads(2)
    x, d = combined('train')
    xv, dv = combined('validation')
    mean, scale = fit_scaler(x)
    z = normalize(x, mean, scale)[..., :256].reshape(-1, 256)
    # PCA uses TRAIN states only. Keep natural PCA scales (no whitening).
    _, _, vh = np.linalg.svd(z, full_matrices=False)
    baseline = dict(kind='baseline', bundle=torch.load(BASE/'selected.pt', weights_only=False))
    rules = dict(configs=CONFIGS, updates=steps, eta=ETA, margin=.05,
                 train_paths=len(d), validation_paths=len(dv), test_access=False,
                 validation_used_for='hyperparameter/checkpoint selection only; no gradients',
                 selection='train P2 <=1%, then minimum validation P2, then training P2',
                 inference_inputs='current latent/AP and accepting automaton state; no clock/path id',
                 confidence_status='not estimated on repeatedly selected development data')
    (OUT/'plan.json').write_text(json.dumps(rules, indent=2))
    records, pool = [], []
    best = None

    def offer(model, name, a=None, b=None):
        nonlocal best
        if a is None:
            fn = predictor(model)
            a, b = fn(x), fn(xv)
        row = dict(name=name, train=development_audit(a, d), validation=development_audit(b, dv))
        records.append(row)
        if best is None or score(row) > score(best):
            best = row
            torch.save(model, OUT/'selected.pt')
            print('NEW BEST', name, 'train', row['train']['p2'], 'val', row['validation']['p2'], flush=True)
        (OUT/'progress.json').write_text(json.dumps(dict(best=best, records=records), indent=2))
        return row, a, b

    row, a0, b0 = offer(baseline, 'previous_frozen')
    pool.append((baseline, a0, b0, row))
    mask = torch.tensor(~d, dtype=torch.float32)
    for cfg in CONFIGS:
        torch.manual_seed(cfg['seed'])
        projection = vh[:cfg['rank']].T if cfg['rank'] else None
        meta = dict(kind='network', mean=mean, scale=scale, projection=projection,
                    width=cfg['width'], gain=1., config=cfg, eta=ETA)
        a = torch.from_numpy(transform(x, meta))
        net = net_for(cfg['width'], a.shape[-1], 'silu')
        opt = torch.optim.AdamW(net.parameters(), lr=.0005, weight_decay=.001)
        family_best = None
        for step in range(1, steps+1):
            ix = torch.randperm(len(a))[:32]
            inputs = a[ix]
            if cfg['noise']:
                inputs = inputs + cfg['noise'] * torch.randn_like(inputs)
            values = (FLOOR + torch.nn.functional.softplus(net(inputs).squeeze(-1))) * mask[ix]
            gaps = torch.relu(values[:, 1:] - values[:, :-1] + .05 * mask[ix, :-1])
            loss = gaps.mean() + torch.topk(gaps.flatten(), 100).values.mean()
            if cfg['multistep']:
                for lag in (3, 10):
                    # Sum of the source waiting indicators along REAL train edges.
                    required = .05 * mask[ix].unfold(1, lag, 1)[:, :-1].sum(-1)
                    long_gap = torch.relu(values[:, lag:] - values[:, :-lag] + required)
                    loss = loss + cfg['multistep'] * long_gap.mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
            opt.step()
            if step % 300 and step != steps:
                continue
            saved = dict(meta, weights={k: v.detach().clone() for k, v in net.state_dict().items()}, step=step)
            fn = predictor(saved)
            aa, bb = fn(x), fn(xv)
            for gain in (1., 8.):
                model = dict(saved, gain=gain)
                rec, av, bv = offer(model, f'{cfg["name"]}_{step}_x{gain:g}', aa*gain, bb*gain)
                if family_best is None or score(rec) > score(family_best[3]):
                    family_best = (model, av, bv, rec)
            print(cfg['name'], step, 'train P2', rec['train']['p2']['violations'],
                  'validation P2', rec['validation']['p2']['violations'], flush=True)
        pool.append(family_best)
        torch.save(family_best[0], OUT/f'{cfg["name"]}.pt')

    for i, (lm, la, lb, lr) in enumerate(pool):
        for rm, ra, rb, rr in pool[i+1:]:
            for mode in ('mean', 'min', 'max'):
                model = dict(kind='combination', members=[lm, rm], mode=mode, eta=ETA)
                aa = getattr(np.stack([la, ra]), mode)(axis=0)
                bb = getattr(np.stack([lb, rb]), mode)(axis=0)
                offer(model, f'{lr["name"]}+{rr["name"]}:{mode}', aa, bb)
    selected = torch.load(OUT/'selected.pt', weights_only=False)
    fn = predictor(selected)
    assert development_audit(fn(x), d) == best['train']
    assert development_audit(fn(xv), dv) == best['validation']
    report = dict(best=best, candidates=len(records), test_run=False,
                  validation_zero=best['validation']['p2']['violations'] == 0,
                  model_sha256=hashlib.sha256((OUT/'selected.pt').read_bytes()).hexdigest(),
                  inferential_status='development selection only; no independent guarantee')
    (OUT/'report.json').write_text(json.dumps(report, indent=2))
    print('FINISHED', json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=2400)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    search(args.steps)
