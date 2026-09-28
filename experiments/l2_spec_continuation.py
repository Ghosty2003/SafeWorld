"""Fit/internal-development L2 continuation; never loads calibration or test.

Eight fixed monitors, two neural families, strict pointwise P1/P2. Finite
window completion is not infinite persistence. No time/trajectory-ID inputs.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import safedreamer_l2_spec_search as base

SOURCE = ROOT / 'artifacts/safedreamer_l2_spec_search_v2'


def monitor(c, g, spec):
    if spec['kind'] != 'window':
        return base.monitor(c, g, spec)
    run = 0
    q = []
    for clear in c:
        if run < spec['length']:
            run = run + 1 if clear >= spec['threshold'] else 0
        q.append(run)
    q = np.asarray(q)
    done = q == spec['length']
    return q, ~done, done


def product(paths, spec):
    xs, qs, bs, ds = [], [], [], []
    for a in paths:
        c, g = base.signals(a)
        q, b, d = monitor(c, g, spec)
        xs.append(np.concatenate([base.physical(a), c[:, None], g[:, None], np.eye(spec['n_states'])[q]], 1))
        qs.append(q); bs.append(b); ds.append(d)
    return tuple(np.asarray(v) for v in (xs, qs, bs, ds))


class Value(nn.Module):
    def __init__(self, dim, family):
        super().__init__()
        self.family = family
        self.net = (nn.Linear(dim, 1) if family == 'linear' else
                    nn.Sequential(nn.Linear(dim, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1)))

    def forward(self, x, bad, done):
        value = nn.functional.softplus(self.net(x).squeeze(-1)) + base.ETA * bad
        return torch.where(done.bool(), torch.zeros_like(value), value)


def predict(model, x, bad, done):
    net = Value(x.shape[-1], model['family']).double()
    net.load_state_dict(model['weights']); net.eval()
    with torch.no_grad():
        return net(torch.as_tensor((x - model['mean']) / model['scale']),
                   torch.as_tensor(bad), torch.as_tensor(done)).numpy()


def score(s):
    return s['p1p2_paths'], -s['p2_violations'], -s['p1_violations']


def load_internal():
    paths, hashes = [], []
    for r in json.loads((SOURCE / 'internal_collection.json').read_text())['records']:
        if not r['eligible']:
            continue
        path = SOURCE / 'internal_raw' / r['file']
        assert base.digest(path) == r['sha256']
        with np.load(path, allow_pickle=False) as data:
            paths.append({k: data[k].copy() for k in data.files})
        hashes.append(r['sha256'])
    assert len(paths) == 20
    return paths, hashes


def qualification(stats, controls):
    numeric = all(s['p1_violations'] == 0 and s['p2_violations'] == 0 and s['bad_transitions'] > 0 for s in stats)
    negative = all(s['p1p2_paths'] < s['n'] for s in controls.values())
    return numeric and negative


def run(out, updates):
    torch.set_num_threads(2)
    old_plan = json.loads((SOURCE / 'plan.json').read_text())
    assert base.digest(base.__file__) == old_plan['code_sha256']
    fit, records = base.load_train()
    assert records == old_plan['train_records']
    specs = copy.deepcopy(old_plan['candidates'])
    for length in (24, 48):
        specs.append(dict(id='safe_window%d' % length, kind='window', length=length, threshold=.2,
            specification='F[0,%d] G[0,%d](clearance>=0.2)' % (301-length, length-1),
            n_states=length+1, accepting=length, bad_set=list(range(length)), persistence=False,
            mp_class='Guarantee (bounded)', completion='L consecutive safe observed states, accepting mode absorbing',
            automaton='Count consecutive safe states causally; reset on hazard until completion; t0 consumed'))
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(created=base.stamp(), source=str(SOURCE), source_plan_sha256=base.digest(SOURCE/'plan.json'),
        checkpoint=old_plan['checkpoint'], checkpoint_sha256=old_plan['checkpoint_sha256'], policy=old_plan['policy'],
        H=300, eta=base.ETA, tolerance=0., candidates=specs, families=['linear', 'elu128'], updates=updates,
        fit_records=records, internal_source=str(SOURCE/'internal_collection.json'),
        internal_scope='20 reused internal-development paths; no gradients; not fresh confirmation',
        data_scope='Only80 train_fit paths and20 internal-development paths; no calibration/test opened',
        selection='Whole-path P1/P2 on internal, then fewer P2/P1 violations, then train score',
        z_free='V<eta; pending bad>=eta; guarantee accepting V=0. Not a physical invariant region.',
        training='All active edges, hard replay, 1.5eta train margin, graph rank train-only supervision',
        region='Original fit-only standardized physical-state radius; not enlarged',
        inference_inputs='Full RSSM, planner carry, current clearance/goal distance, causal mode only',
        calibration_count=0, test_count=0, code_sha256=base.digest(__file__))
    base.write(out/'plan.json', plan)
    frozen = []
    started = time.monotonic()
    for si, spec in enumerate(specs):
        x, q, bad, done = product(fit, spec)
        feasible, rank = base.graph_feasibility(x, q, bad)
        base.write(out/(spec['id']+'_feasibility.json'), feasible)
        if not feasible['feasible']:
            continue
        mean = x.mean((0, 1)); scale = np.maximum(x.std((0, 1)), .05)
        # Omit only identically-zero accepting -> accepting edges from fitting.
        pi, ti = np.where(~(done[:, :-1] & done[:, 1:]))
        normalized = (x - mean) / scale
        a = torch.tensor(normalized[pi, ti], dtype=torch.float32)
        b = torch.tensor(normalized[pi, ti+1], dtype=torch.float32)
        ba, bb = torch.tensor(bad[pi, ti]), torch.tensor(bad[pi, ti+1])
        da, db = torch.tensor(done[pi, ti]), torch.tensor(done[pi, ti+1])
        y = torch.tensor(rank[pi, ti] + base.ETA * bad[pi, ti], dtype=torch.float32)
        completed = torch.tensor(done[pi, -1])
        for fi, family in enumerate(plan['families']):
            torch.manual_seed(156100000 + 100*si + fi)
            net = Value(x.shape[-1], family)
            optimizer = torch.optim.AdamW(net.parameters(), lr=.0005, weight_decay=1e-6)
            weights = torch.ones(len(a)); best = None; history = []
            for step in range(updates+1):
                if step:
                    ix = torch.multinomial(weights, 512, replacement=True)
                    va = net(a[ix], ba[ix], da[ix]); vb = net(b[ix], bb[ix], db[ix])
                    residual = (vb-va)/base.ETA + 1.5*ba[ix]
                    pos = torch.relu(residual)
                    err = va-y[ix]
                    aux = torch.where(completed[ix], err.square(), torch.relu(-err).square()).mean()
                    loss = pos.square().mean() + pos.topk(32).values.mean() + .03*aux + .00001*va.square().mean()
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 100.); optimizer.step()
                if step % 200 and step != updates:
                    continue
                model = dict(family=family, mean=mean, scale=scale, spec=spec, step=step,
                             weights={k: v.detach().clone() for k, v in net.state_dict().items()})
                s = base.statistics(predict(model, x, bad, done), bad, done, spec['persistence'])
                history.append(dict(step=step, **{k:v for k,v in s.items() if not isinstance(v,list)}))
                if best is None or score(s) > best[0]:
                    best = score(s), copy.deepcopy(model), s
                with torch.no_grad():
                    hard = (torch.relu((net(b,bb,db)-net(a,ba,da))/base.ETA+1.5*ba)+.1).clamp(max=20.)
                    weights = .25/len(a) + .75*hard/hard.sum()
            ident = spec['id']+'_'+family
            path = out/(ident+'.pt'); torch.save(best[1], path)
            rec = dict(id=ident, spec=spec, family=family, weights=path.name, sha256=base.digest(path),
                       frozen=base.stamp(), train=best[2], history=history)
            frozen.append(rec); base.write(out/(ident+'_training.json'), rec)
            print('FROZEN', ident, 'train', best[2]['p1p2_paths'], '/80 P1/P2 bad',
                  best[2]['p1_violations'], best[2]['p2_violations'], 'elapsed', round(time.monotonic()-started,1), flush=True)
    base.write(out/'frozen.json', dict(at=base.stamp(), records=frozen))
    internal, hashes = load_internal()
    assert not set(hashes) & {r['sha256'] for r in records}
    region = np.load(SOURCE/'region.npz')
    masks = [np.linalg.norm((np.stack([base.physical(a) for a in paths])-region['mean'])/region['scale'],axis=-1)
             <= float(region['radius'])+1e-10 for paths in (fit,internal)]
    rows = []
    for rec in frozen:
        spec = rec['spec']; model = torch.load(out/rec['weights'], map_location='cpu')
        assert base.digest(out/rec['weights']) == rec['sha256']
        datasets = [product(paths, spec) for paths in (fit, internal)]
        stats, values = [], []
        for (x,q,b,d), mask in zip(datasets, masks):
            v = predict(model,x,b,d); values.append(v)
            stats.append(base.statistics(v,b,d,spec['persistence'],mask))
        x,q,b,d = datasets[1]; v = values[1]
        controls = {}
        control_values = [('constant', np.ones_like(v)), ('mode_constant', np.where(d,0.,1.))]
        if spec['kind']=='window':
            control_values.append(('monitor_rank_only', .02*(spec['length']-q)))
        for seed in range(3):
            rng = np.random.RandomState(156200000+seed); sv = v.copy()
            for mode in np.unique(q):
                sv[q==mode] = rng.permutation(sv[q==mode])
            control_values.append(('within_mode_shuffle%d'%seed,sv))
            torch.manual_seed(156201000+seed)
            rm = copy.deepcopy(model); rm['weights'] = Value(x.shape[-1],rec['family']).state_dict()
            control_values.append(('random%d'%seed,predict(rm,x,b,d)))
        for name, cv in control_values:
            controls[name] = base.statistics(cv,b,d,spec['persistence'])
        primary_controls = {k:v for k,v in controls.items() if k != 'monitor_rank_only'}
        qualified = qualification(stats,primary_controls) and not spec['persistence']
        pending = ~d[:,0]
        row = dict(id=rec['id'], spec=spec, train=stats[0], internal=stats[1], controls=controls,
            internal_initially_pending_count=int(pending.sum()),
            internal_pending_path_pass=int(np.asarray(stats[1]['p1p2_per_path'])[pending].sum()),
            status='KEEP' if qualified else 'BORDERLINE',
            monitor_only_allpass=bool('monitor_rank_only' in controls and controls['monitor_rank_only']['p1p2_paths']==len(v)))
        rows.append(row)
        np.savez_compressed(out/(rec['id']+'_evaluated.npz'),fit_v=values[0],internal_v=values[1],
            fit_bad=datasets[0][2],internal_bad=b,fit_done=datasets[0][3],internal_done=d)
        print('INTERNAL',rec['id'],stats[1]['p1p2_paths'],'/20',row['status'],flush=True)
    best = max(rows, key=lambda r:(r['status']=='KEEP',*score(r['internal']),*score(r['train'])))
    report = dict(completed=base.stamp(),rows=rows,selected=best['id'],qualified=sum(r['status']=='KEEP' for r in rows),
        internal_hashes=hashes,calibration_count=0,test_count=0,
        scope='Reused internal-development only; KEEP requires fresh confirmation, no confidence claim')
    base.write(out/'report.json', report)
    fields=['candidate','train_P1_bad','train_P2_bad','train_paths','internal_P1_bad','internal_P2_bad','internal_paths',
            'internal_completion','internal_certificate','initially_complete','status','monitor_only_allpass']
    with (out/'screening_summary.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(fields)
        for r in rows:
            a,b=r['train'],r['internal']
            writer.writerow([r['id'],a['p1_violations'],a['p2_violations'],a['p1p2_paths'],b['p1_violations'],
                b['p2_violations'],b['p1p2_paths'],b['finite_completion_paths'],b['certificate_preview'],
                b['already_complete_at_t0'],r['status'],r['monitor_only_allpass']])
    print('FINISHED',report['selected'],'qualified',report['qualified'],flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--updates',type=int,default=2000)
    args=p.parse_args(); run(args.output.resolve(),args.updates)
