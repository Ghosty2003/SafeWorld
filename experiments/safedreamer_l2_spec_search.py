"""Independent L2 search: genuine pointwise V, no countdown/reset budget.

Stages are explicit. No calibration/test data are opened by this program.
Only a KEEP can authorize a subsequent frozen-candidate calibration run.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import deque

import numpy as np
import torch
from torch import nn
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.l3_safedreamer_clear_pipeline import digest, write, stamp

ETA = .01
H = 300
SOURCE = ROOT / 'artifacts/safedreamer_b_fit100_v1'


def signals(a):
    d = a['decoded'].astype(float)
    return np.linalg.norm(d[:, 9:25].reshape(-1, 8, 2), axis=-1).min(-1), np.linalg.norm(d[:, 7:9], axis=-1)


def physical(a):
    n = len(a['latent'])
    return np.concatenate([a['latent'], a['planner_action_mean'].reshape(n, -1),
        a['planner_action_std'].reshape(n, -1), a['planner_initialized'].reshape(n, 1)], axis=1).astype(np.float32)


def monitor(clearance, goal, spec):
    """q_t consumes observations through t, including t=0; no future access."""
    kind = spec['kind']; q = []; state = 0
    for c, g in zip(clearance, goal):
        if kind == 'high':
            state = int(state == 1 or c >= spec['high'])
        elif kind == 'goal':
            state = int(state == 1 or g < .3)
        elif kind == 'return':
            if state == 0 and c <= spec['low']: state = 1
            elif state == 1 and c >= spec['high']: state = 2
        elif kind == 'recovery8':
            if state == 0 and c <= spec['low']: state = 1
            elif 1 <= state < 9:
                state = min(9, state + 1) if c >= spec['high'] else 1
        elif kind == 'persist_safe': state = int(c >= .2)
        elif kind == 'persist_high': state = int(c >= spec['high'])
        else: raise ValueError(kind)
        q.append(state)
    q = np.asarray(q, dtype=np.int64)
    bad = np.isin(q, spec['bad_set'])
    done = np.zeros(len(q), bool) if spec['persistence'] else q == spec['accepting']
    return q, bad, done


def candidates(low, high):
    rows = [
        ('high', 'F(clearance >= high)', 2, 1, [0], False),
        ('return', 'F(LOW & F HIGH)', 3, 2, [0, 1], False),
        ('goal', 'F(goal_distance < 0.3)', 2, 1, [0], False),
        ('recovery8', 'F(LOW & F G[0,7](clearance >= high))', 10, 9, list(range(9)), False),
        ('persist_safe', 'F G(clearance >= 0.2)', 2, 1, [0], True),
        ('persist_high', 'F G(clearance >= high)', 2, 1, [0], True)]
    return [dict(id=k, kind=k, specification=s, n_states=n, accepting=a, bad_set=b,
        persistence=p, low=low, high=high, mp_class='Persistence / co-Buchi' if p else 'Guarantee',
        completion='No finite completion; good q is not absorbing' if p else 'Absorbing accepting q=%d' % a,
        automaton=('q=0 iff current predicate false; q=1 iff true; both can recur' if p else
            'Causal deterministic finite monitor, accepting state absorbing; see monitor() and tests'),
        monitor_clock='q_t after reading z_t, P2 uses q_t on the real edge z_t -> z_(t+1)')
        for k, s, n, a, b, p in rows]


def load_train():
    collection = json.loads((SOURCE / 'collection.json').read_text())
    paths = []; records = []
    for r in collection['records']:
        if r['role'] != 'train_fit' or not r['eligible']: continue
        f = SOURCE / 'raw' / r['file']; assert digest(f) == r['sha256']
        with np.load(f, allow_pickle=False) as a: paths.append({k: a[k].copy() for k in a.files})
        records.append(r)
    assert len(paths) == 80
    return paths, records


def product_data(paths, spec):
    xs = []; qs = []; bs = []; ds = []
    for a in paths:
        c, g = signals(a); q, b, d = monitor(c, g, spec)
        xs.append(np.concatenate([physical(a), c[:, None], g[:, None], np.eye(spec['n_states'])[q]], axis=1))
        qs.append(q); bs.append(b); ds.append(d)
    return np.asarray(xs), np.asarray(qs), np.asarray(bs), np.asarray(ds)


def graph_feasibility(xs, qs, bad, eta=ETA):
    """Exact product-node difference constraints, no artificial terminal edges.

    Every edge has required descent 0 or eta. Infeasible exactly when a
    positive-weight edge lies within an SCC. Otherwise reverse-DAG longest
    distance gives a finite, nonnegative feasible witness on sampled nodes.
    """
    ids = {}; nodes = []; sources = []; targets = []; weights = []
    for x, q, b in zip(xs, qs, bad):
        path = []
        for row, mode in zip(x, q):
            key = hashlib.sha256(np.asarray(row, '<f8').tobytes() + np.asarray(mode, '<i8').tobytes()).digest()
            if key not in ids: ids[key] = len(ids)
            path.append(ids[key])
        sources.extend(path[:-1]); targets.extend(path[1:]); weights.extend(b[:-1].astype(float) * eta); nodes.append(path)
    s = np.asarray(sources); t = np.asarray(targets); w = np.asarray(weights)
    graph = coo_matrix((np.ones(len(s)), (s, t)), shape=(len(ids), len(ids))).tocsr()
    n, labels = connected_components(graph, directed=True, connection='strong')
    conflict = (labels[s] == labels[t]) & (w > 0)
    result = dict(nodes=len(ids), edges=len(s), bad_edges=int((w > 0).sum()),
        positive_cycle_edges=int(conflict.sum()), feasible=not bool(conflict.any()),
        identity='exact full RSSM + planner carry + decoded features + automaton state; no approximate merge',
        scope='finite sampled graph only; unseen edges and infinite paths not established')
    if conflict.any(): return result, None
    adj = [[] for _ in range(n)]; indegree = np.zeros(n, int)
    for a, b, weight in zip(labels[s], labels[t], w):
        if a != b: adj[a].append((b, weight)); indegree[b] += 1
    queue = deque(np.flatnonzero(indegree == 0)); order = []
    while queue:
        a = queue.popleft(); order.append(a)
        for b, _ in adj[a]:
            indegree[b] -= 1
            if indegree[b] == 0: queue.append(b)
    assert len(order) == n
    v = np.zeros(n)
    for a in reversed(order):
        if adj[a]: v[a] = max(v[b] + weight for b, weight in adj[a])
    witness = v[labels]
    assert np.max(witness[t] - witness[s] + w) < 1e-10
    result['witness_max'] = float(witness.max())
    return result, witness[np.asarray(nodes)]


class Value(nn.Module):
    def __init__(self, dim, width):
        super().__init__()
        self.net = nn.Linear(dim, 1) if not width else nn.Sequential(
            nn.Linear(dim, width), nn.ELU(), nn.Linear(width, width), nn.ELU(), nn.Linear(width, 1))

    def forward(self, x, bad, done):
        v = nn.functional.softplus(self.net(x).squeeze(-1)) + ETA * bad
        return torch.where(done.bool(), torch.zeros_like(v), v)


def statistics(v, bad, done, persistence=False, region=None):
    """STRICT comparisons, float64 evaluation, all H observed transitions."""
    dv = np.diff(v, axis=1)
    p1 = dv <= 0.; p2 = (~bad[:, :-1]) | (dv <= -ETA)
    z = v < ETA
    sound = (~z | ~bad).all(axis=1)
    closure = (~z[:, :-1] | (z[:, 1:] & ~bad[:, 1:])).all(axis=1)
    # Persistence: endpoint membership is a sampled premise, NOT completion.
    endpoint = z[:, -1] if persistence else (done[:, -1] & z[:, -1])
    region = np.ones(v.shape, bool) if region is None else region
    path = p1.all(1) & p2.all(1)
    cert = path & endpoint & sound & closure & region.all(1) & np.isfinite(v).all(1) & (v >= 0).all(1)
    nbad = int(bad[:, :-1].sum())
    return dict(n=len(v), transitions=dv.size, bad_transitions=nbad,
        p1_violations=int((~p1).sum()), p2_violations=int((~p2).sum()),
        p1_pass=float(p1.mean()), p2_pass=float(p2[bad[:, :-1]].mean()) if nbad else None,
        p1p2_paths=int(path.sum()), certificate_preview=int(cert.sum()),
        certificate_per_path=cert.tolist(), p1p2_per_path=path.tolist(),
        endpoint_failures=int((~endpoint).sum()), region_failures=int((~region.all(1)).sum()),
        z_free_states=int(z.sum()), bad_in_z_free=int((z & bad).sum()),
        closure_failure_paths=int((~closure).sum()),
        finite_completion_paths=None if persistence else int(done[:, -1].sum()),
        already_complete_at_t0=None if persistence else int(done[:, 0].sum()),
        v_min=float(v.min()), v_max=float(v.max()),
        max_p1_residual=float(dv.max()), max_p2_residual=float((dv + ETA)[bad[:, :-1]].max()) if nbad else None)


def predict(model, x, bad, done):
    net = Value(x.shape[-1], model['width']).double(); net.load_state_dict(model['weights']); net.eval()
    with torch.no_grad():
        return net(torch.tensor((x-model['mean'])/model['scale'], dtype=torch.float64),
            torch.tensor(bad, dtype=torch.float64), torch.tensor(done)).numpy()


def prepare(out):
    paths, records = load_train(); old = json.loads((SOURCE / 'plan.json').read_text())
    assert digest(old['checkpoint']) == old['checkpoint_sha256']
    assert digest(ROOT/'wrappers/safedreamer_wrapper.py') == old['wrapper_sha256']
    low, high = np.quantile(np.concatenate([signals(a)[0] for a in paths]), [.25, .75])
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(created=stamp(), H=H, eta=ETA, numeric_tolerance=0., checkpoint=old['checkpoint'],
        checkpoint_sha256=old['checkpoint_sha256'], policy=old['policy'],
        wrapper_sha256=old['wrapper_sha256'], start_distribution=old['start_distribution'],
        train_source=str(SOURCE), train_records=records, candidates=candidates(float(low), float(high)),
        thresholds_from='80 train-fit paths only, pooled state quantiles .25/.75',
        architectures=[0, 128, 256], updates=1200, batch_paths=8, training_seed=142000001,
        train_checkpoint_selection='max full train P1P2 paths, then fewer P2, then fewer P1; every 200 updates',
        validation_selection='KEEP only if strict P1/P2 all pass train and unseen internal validation, nonvacuous bad edges, controls fail',
        region_rule='full RSSM+planner standardized L2 radius <= training maximum; frozen train-only',
        z_free='V(x)<eta; never equate a good persistence label with eternal completion',
        gates=['P1_all', 'P2_all', 'endpoint', 'Zfree_bad_exclusion', 'observed_Zfree_closure', 'region', 'finite_nonnegative'],
        scope='sampled model-only; persistence has no finite completion proof; no global deductive claim',
        roles=dict(train=80, internal_validation=dict(n=20, seed_base=141000000),
            calibration=dict(n=200, seed_base=143000000, status='LOCKED_UNTIL_KEEP'),
            test=dict(n=1000, seed_base=144000000, status='LOCKED_UNTIL_KEEP')),
        warrant_threshold=.95, confidence=.95, code_sha256=digest(__file__))
    write(out/'plan.json', plan)
    print('PREPARED', '6 specs, fit80, NEW internal20; cal/test locked', flush=True)


def train(out, device):
    plan = json.loads((out/'plan.json').read_text()); assert digest(__file__) == plan['code_sha256']
    paths, _ = load_train(); torch.set_num_threads(2)
    if device == 'cuda': assert torch.cuda.is_available()
    core = np.stack([physical(a) for a in paths]); mean = core.mean((0,1)); scale = np.maximum(core.std((0,1)), .05)
    np.savez_compressed(out/'region.npz', mean=mean, scale=scale, radius=np.linalg.norm((core-mean)/scale,axis=-1).max())
    records = []; start = time.monotonic()
    for si, spec in enumerate(plan['candidates']):
        x, q, bad, done = product_data(paths, spec)
        feasible, witness = graph_feasibility(x, q, bad); write(out/(spec['id']+'_feasibility.json'), feasible)
        if not feasible['feasible']: continue
        np.savez_compressed(out/(spec['id']+'_graph_witness.npz'), values=witness, q=q, bad=bad)
        mean=x.mean((0,1)); scale=np.maximum(x.std((0,1)),.05)
        xt=torch.tensor((x-mean)/scale,dtype=torch.float32,device=device)
        bt=torch.tensor(bad,dtype=torch.float32,device=device); dt=torch.tensor(done,device=device)
        target=torch.tensor(witness,dtype=torch.float32,device=device)
        for width in plan['architectures']:
            seed=plan['training_seed']+si*1000+width; torch.manual_seed(seed)
            net=Value(x.shape[-1],width).to(device); optim=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5)
            best=None; history=[]
            for step in range(1,plan['updates']+1):
                ix=torch.randperm(len(x),device=device)[:plan['batch_paths']]
                v=net(xt[ix],bt[ix],dt[ix]); delta=v[:,1:]-v[:,:-1]
                hinge=torch.relu(delta+2*ETA*bt[ix,:-1])
                # Graph-rank auxiliary supervision is train-only, never an input.
                residual=v-target[ix]
                # Incomplete traces only give lower-bound targets, not time-to-success labels.
                complete=dt[ix,-1,None].expand_as(v)
                aux=torch.where(complete,residual.square(),torch.relu(-residual).square()).mean()
                loss=10*hinge.mean()+4*hinge.amax(1).mean()+.03*aux+.0001*v.square().mean()
                optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10.);optim.step()
                if step%200 and step!=plan['updates']:continue
                model=dict(width=width,mean=mean,scale=scale,weights={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                    spec=spec,eta=ETA,step=step,seed=seed)
                pv=predict(model,x,bad,done); score=statistics(pv,bad,done,spec['persistence'])
                key=(score['p1p2_paths'],-score['p2_violations'],-score['p1_violations'])
                history.append(dict(step=step,loss=float(loss),**score))
                if best is None or key>best[0]:best=(key,model,score)
            file=out/('%s_w%d.pt'%(spec['id'],width));torch.save(best[1],file)
            record=dict(spec=spec['id'],width=width,file=file.name,sha256=digest(file),frozen=stamp(),train=best[2],history=history)
            records.append(record);write(out/('training_progress_%02d.json'%len(records)),records)
            print('FROZEN',spec['id'],width,'train paths',best[2]['p1p2_paths'],'/80',
                'P1/P2 violations',best[2]['p1_violations'],best[2]['p2_violations'],'elapsed',round(time.monotonic()-start,1),flush=True)
    write(out/'candidates_frozen.json',dict(at=stamp(),validation_not_collected=True,records=records))


def collect(out):
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    from experiments.safedreamer_finite_recurrence_pilot import SeededResetRecorder, extract
    plan=json.loads((out/'plan.json').read_text()); assert digest(__file__)==plan['code_sha256']
    assert (out/'candidates_frozen.json').exists()
    assert digest(plan['checkpoint'])==plan['checkpoint_sha256']
    assert digest(ROOT/'wrappers/safedreamer_wrapper.py')==plan['wrapper_sha256']
    raw=out/'internal_raw';raw.mkdir(exist_ok=False)
    extra=build_extra(argparse.Namespace(repo_root=None,checkpoint=plan['checkpoint']))
    extra['action_source']='policy';extra['config_overrides']['jax'].update(platform='gpu',logical_gpus=0)
    role=plan['roles']['internal_validation'];records=[];seen={r['fingerprint'] for r in plan['train_records']};count=0
    with SafeDreamerWrapper(RolloutConfig(horizon=H,n_rollouts=1,seed=role['seed_base']+1,action_source='policy',extra=extra)) as wrapper:
        wrapper.load()
        import jax
        assert jax.default_backend()=='gpu'
        actual=dict(action_source='policy',expl_behavior=wrapper._config.expl_behavior,
            planner={k:getattr(wrapper._config.planner,k) for k in plan['policy']['planner']},cost_limit=wrapper._config.cost_limit,
            planner_source_sha256=digest(Path(extra['repo_root'])/'SafeDreamer/behaviors.py'),note=plan['policy']['note'])
        assert actual==plan['policy']
        recorder=SeededResetRecorder(wrapper._env);wrapper._env=recorder
        try:
            for draw in range(role['n']*5):
                reset=role['seed_base']+2*draw;recorder.pending=reset
                data=wrapper.sample_latent_rollouts(RolloutConfig(horizon=H,n_rollouts=1,seed=reset+1,action_source='policy',extra=extra))
                assert recorder.pending is None and data['action_source']=='policy'
                a=extract(data,recorder.observation);assert a['latent'].shape==(301,512)
                fingerprint=hashlib.sha256(a['latent'].tobytes()).hexdigest();assert fingerprint not in seen;seen.add(fingerprint)
                initial_goal=float(np.linalg.norm(a['decoded'][0,7:9]));eligible=initial_goal>=1.
                file=raw/('draw_%04d.npz'%draw);np.savez_compressed(file,**a)
                records.append(dict(file=file.name,sha256=digest(file),reset_seed=reset,imagination_seed=reset+1,
                    eligible=eligible,initial_goal=initial_goal,fingerprint=fingerprint,time=stamp()))
                count+=int(eligible);write(out/('internal_collection_progress_%03d.json'%draw),records)
                print('INTERNAL',count,'/',role['n'],'draw',draw,flush=True)
                if count==role['n']:break
            assert count==role['n']
        finally:recorder.restore();wrapper._env=recorder.outer
    write(out/'internal_collection.json',dict(completed=stamp(),records=records,retained=count,calibration=0,test=0))


def controls(model,x,bad,done,persistence):
    v=predict(model,x,bad,done);results={}
    values=[('constant',np.ones_like(v)),('mode_constant',np.where(done,0.,1.))]
    for seed in range(5):
        rng=np.random.RandomState(142900+seed);shuffled=v.copy()
        # Shuffle within each q category to preserve legitimate automaton semantics.
        q=x[:,:,-model['spec']['n_states']:].argmax(-1)
        for mode in np.unique(q):shuffled[q==mode]=rng.permutation(v[q==mode])
        values.append(('shuffled_%d'%seed,shuffled))
        torch.manual_seed(142950+seed);random=Value(x.shape[-1],model['width']).double()
        with torch.no_grad():rv=random(torch.tensor((x-model['mean'])/model['scale'],dtype=torch.float64),torch.tensor(bad,dtype=torch.float64),torch.tensor(done)).numpy()
        values.append(('random_%d'%seed,rv))
    for name,val in values:
        row=statistics(val,bad,done,persistence)
        results[name]={k:row[k] for k in ('p1_violations','p2_violations','p1p2_paths','n')}
    return results


def evaluate(out):
    plan=json.loads((out/'plan.json').read_text());assert digest(__file__)==plan['code_sha256']
    train,_=load_train();val=[]
    provenance=json.loads((out/'internal_collection.json').read_text())
    for r in provenance['records']:
        if not r['eligible']:continue
        f=out/'internal_raw'/r['file'];assert digest(f)==r['sha256']
        with np.load(f,allow_pickle=False) as a:val.append({k:a[k].copy() for k in a.files})
    assert len(val)==20
    rows=[];allrows=[];region=np.load(out/'region.npz')
    region_masks=[np.linalg.norm((np.stack([physical(a) for a in p])-region['mean'])/region['scale'],axis=-1)<=float(region['radius'])+1e-10 for p in (train,val)]
    frozen=json.loads((out/'candidates_frozen.json').read_text())
    for spec in plan['candidates']:
        graph=json.loads((out/(spec['id']+'_feasibility.json')).read_text())
        if not graph['feasible']:
            rows.append(dict(spec=spec,status='REJECT',reason='sampled graph infeasible'));continue
        data=[product_data(p,spec) for p in (train,val)];variants=[]
        for r in frozen['records']:
            if r['spec']!=spec['id']:continue
            file=out/r['file'];assert digest(file)==r['sha256'];model=torch.load(file,map_location='cpu',weights_only=False)
            stats=[];predictions=[]
            for (x,q,b,d),reg in zip(data,region_masks):
                pred=predict(model,x,b,d);predictions.append(pred);stats.append(statistics(pred,b,d,spec['persistence'],reg))
            x,q,b,d=data[1];control=controls(model,x,b,d,spec['persistence'])
            strict=all(s['p1_violations']==0 and s['p2_violations']==0 and s['bad_transitions']>0 for s in stats)
            nondegenerate=all(c['p1_violations']+c['p2_violations']>0 for c in control.values())
            meaningful=all(s['z_free_states']>0 and s['bad_in_z_free']==0 for s in stats)
            # Persistence additionally needs an independently justified closure premise;
            # good labels / finite suffixes alone cannot license its infinite claim.
            keep=strict and nondegenerate and meaningful and not spec['persistence']
            row=dict(spec=spec,width=r['width'],weights=r['file'],sha256=r['sha256'],train=stats[0],internal_validation=stats[1],
                controls=control,status='KEEP' if keep else 'BORDERLINE',strict_sampled_100=strict,
                nondegenerate_controls=nondegenerate,z_free_observed=meaningful,
                reason='All qualification checks pass' if keep else 'No strict sampled 100% P1/P2 or no meaningful Zfree; persistence also lacks global closure premise')
            np.savez_compressed(out/(file.stem+'_evaluated.npz'),train_v=predictions[0],validation_v=predictions[1],
                train_bad=data[0][2],validation_bad=b,train_done=data[0][3],validation_done=d)
            variants.append(row);allrows.append(row)
        # Balance P1 and P2 instead of allowing many trivial accepting edges to dominate.
        def rank(r):
            s=r['internal_validation'];t=r['train']
            return (r['status']=='KEEP',s['p1p2_paths']/s['n'],min(s['p1_pass'],s['p2_pass'] or 0),
                t['p1p2_paths']/t['n'],-s['p2_violations'],-s['p1_violations'])
        best=max(variants,key=rank);rows.append(best)
    qualified=[r for r in rows if r['status']=='KEEP']
    report=dict(completed=stamp(),rows=rows,all_candidates=allrows,qualified_count=len(qualified),
        calibration_count=0,test_count=0,scope=plan['scope'],confidence_lower_bound=None,
        conclusion='QUALIFIED_PENDING_CALIBRATION' if qualified else 'ABSTAIN / no-qualified-spec')
    write(out/'report.json',report)
    fields=['spec','MP_class','width','train_P1_violations','train_P2_violations','train_P1P2_paths','internal_P1_violations',
        'internal_P2_violations','internal_P1P2_paths','internal_bad_transitions','internal_completion','certificate_preview','status']
    with (out/'screening_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            if 'train' not in r:continue
            t=r['train'];v=r['internal_validation']
            writer.writerow(dict(spec=r['spec']['specification'],MP_class=r['spec']['mp_class'],width=r['width'],
                train_P1_violations=t['p1_violations'],train_P2_violations=t['p2_violations'],train_P1P2_paths=t['p1p2_paths'],
                internal_P1_violations=v['p1_violations'],internal_P2_violations=v['p2_violations'],internal_P1P2_paths=v['p1p2_paths'],
                internal_bad_transitions=v['bad_transitions'],internal_completion=v['finite_completion_paths'],certificate_preview=v['certificate_preview'],status=r['status']))
    if qualified:
        write(out/'qualified_freeze.json',dict(frozen=stamp(),plan_sha256=digest(out/'plan.json'),candidates=qualified))
    print(json.dumps(dict(conclusion=report['conclusion'],qualified_count=len(qualified),
        rows=[dict(spec=r['spec']['id'],status=r['status'],train=r.get('train'),validation=r.get('internal_validation')) for r in rows]),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','train','collect','evaluate'])
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();out=args.output.resolve()
    if args.stage=='prepare':prepare(out)
    elif args.stage=='train':train(out,args.device)
    elif args.stage=='collect':collect(out)
    else:evaluate(out)
