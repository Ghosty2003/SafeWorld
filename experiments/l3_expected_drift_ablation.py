"""S24 development experiment: expected-only loss, nested kappa, candidate H2.

H2 core I=F is an unproved proposal, NOT an accepting-state proof exemption.
No fresh final validation, support-wide invariance, or L3 warrant is produced.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.l3_spec_ablation import SPECS, load_role, relabel, collect_shared
from experiments.l3_spec_ablation_audit import load_queries
from experiments.l3_margin_sweep import diagnose, distribution
from experiments.l3_wu_search import Candidate, prepare
from experiments.l3_wu_data_ablation import restore
from experiments.l3_safedreamer_clear_pipeline import digest, write, stamp, make_sampler
from core.lbsm.operator_model import assert_anchor_splits_disjoint


def required_mask(x, kind, retention):
    return x[:, -1] < 1 if kind == 'W' or retention == 'H2_candidate' else torch.ones(len(x), dtype=torch.bool)


def expected_loss(values, current, mask, margin):
    drift = values.mean(1) - current
    return torch.relu(drift[mask] + margin).mean() if mask.any() else drift.sum()*0


def fit(data, mean, scale, kind, margin, retention, path_weight, steps):
    torch.manual_seed(73 if kind == 'W' else 41)
    net = Candidate(mean, scale, 'wide' if kind == 'W' else 'sublevel')
    opt = torch.optim.Adam(net.parameters(), lr=.001)
    history = []
    for step in range(steps):
        b = torch.randperm(len(data['x']))[:32]
        x, y = data['x'][b], data['y'][b]
        c = data['core'][torch.randperm(len(data['core']))[:512]]
        # Draw matching indices even in expected-only arms: preserve RNG controls.
        i = torch.randperm(len(data['px']))[:512]
        px, py = data['px'][i], data['py'][i]
        values = net(y.flatten(0, 1)).reshape(y.shape[:2])
        branch = expected_loss(values, net(x), required_mask(x, kind, retention), margin)
        pl = values.sum()*0
        if path_weight:
            pd = net(py)-net(px); mask = required_mask(px, kind, retention)
            pl = torch.relu(pd[mask]+margin).mean() if mask.any() else pd.sum()*0
        if kind == 'W':
            shape = ((net(c)-.48*(1-c[:, -1]))**2).mean()
        else:
            # Same shaping as H1 controls. No forced accepting branch or artificial
            # absorbing monitor; H2 changes only the candidate loss domain.
            shell = c.clone()
            shell[:, :-2] = net.mean[:-2]+4*(c[:, :-2]-net.mean[:-2])
            shape = ((net(c)-.2)**2).mean()+torch.relu(.85-net(shell)).square().mean()
        loss = branch+path_weight*pl+.25*shape
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 0 or (step+1) % 100 == 0 or step+1 == steps:
            history.append(dict(step=step+1,loss=float(loss),branch=float(branch),
                                path=float(pl),shape=float(shape)))
    return net.eval(), history


def core_check(batches):
    selected = [b for b in batches if b.anchor[-1] >= 1]
    exits = [int((b.successors[:, -1] < 1).sum()) for b in selected]
    return dict(core='I_candidate = F (S24 counter=24)',anchors=len(selected),
                successors=sum(len(b.successors) for b in selected),exit_successors=sum(exits),
                anchors_with_exits=sum(n>0 for n in exits),
                sample_status=('COUNTEREXAMPLE' if sum(exits) else
                               'SAMPLED_NO_EXIT' if selected else 'UNOBSERVED'),
                support_wide_invariance='NOT_ESTABLISHED')


def evaluate(net, kind, retention, batches, paths, margin, slack):
    r = diagnose(net, kind, batches, paths, margin, slack)
    # Always retain the full H1 population for honest denominator comparison.
    required = [a for a in r['anchor_rows'] if kind == 'U' and retention == 'H1'
                or not a['accepting']]
    g = np.array([a['drift'] for a in required])
    threshold = -.01 if kind == 'W' else 0.
    r['candidate_domain'] = dict(required=len(g),raw_pass=int((g<=threshold).sum()),
        raw_rate=float((g<=threshold).mean()) if len(g) else None,
        fixed_slack_pass=int((g+slack<=threshold).sum()),distribution=distribution(g))
    if kind == 'U':
        r['H1_all_anchors'] = dict(required=r['required'],raw_pass=r['raw_pass'])
        core_values = [a['value'] for a in r['anchor_rows'] if a['accepting']]
        r['sampled_I_subset_C'] = bool(core_values and max(core_values)<=.8)
        r['global_I_subset_C'] = 'NOT_ESTABLISHED'
    return r


def load_source(source):
    plan = json.loads((source/'plan.json').read_text())
    frozen = json.loads((source/'frozen.json').read_text())
    if digest(source/'plan.json') != frozen['plan_sha256']: raise ValueError('Source plan changed')
    for name, sha in plan['source_sha256'].items():
        if digest(ROOT/name) != sha: raise ValueError(f'Source changed: {name}')
    train = load_role(Path(plan['source']), 'fit')
    dev = load_role(Path(plan['source']), 'development')
    tq, ts = load_queries(source, 'fit', train, frozen['time'])
    dq, ds = load_queries(source, 'development', dev, frozen['time'])
    if set(ts) & set(ds): raise ValueError('Cross-role RNG seed reuse')
    return plan, frozen, train, dev, tq, dq, ts, ds


def collect(args):
    source, out = args.source.resolve(), args.output.resolve()
    old, frozen, train, dev, tq, dq, ts, ds = load_source(source)
    new_seeds = [args.seed_base+pi*10000+ai*100+ci for pi,p in enumerate(train)
                 for ai,_ in enumerate(p['times']) for ci in range(3)]
    if (set(ts)|set(ds)) & set(new_seeds): raise ValueError('Repeated query seed')
    out.mkdir(parents=True, exist_ok=False)
    write(out/'plan.json', dict(time=stamp(),source=str(source),source_plan_sha256=digest(source/'plan.json'),
        source_frozen_sha256=digest(source/'frozen.json'),spec='B_S24',fit_paths=len(train),dev_paths=len(dev),
        fit_anchors=len(tq),dev_anchors=len(dq),fit_kappas=[32,128],extra_successors=96,dev_kappa=256,
        seed_base=args.seed_base,steps=args.steps,W_margins=[.01,.05,.10],U_margins=[.001,.03],
        W_family='wide',U_family='sublevel',B=1.,ell=.8,verify_W_epsilon=.01,verify_U_epsilon=0.,
        dev_gate=dict(W_raw=.9,U_raw=.9,W_q90_less_than=-.05),
        scope='ADAPTIVE_DEVELOPMENT_ONLY',H2_core='I=F candidate; no proof exemption',
        final_validation='NOT_RUN',source_signature=old['source_signature'],
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),
                     ROOT/'experiments/l3_margin_sweep.py']}))
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    signature = old['source_signature']
    extra = build_extra(argparse.Namespace(repo_root=None,checkpoint=signature['checkpoint']))
    extra['action_source'] = 'policy'
    extra['config_overrides']['jax'].update(platform='gpu',logical_gpus=0)
    ref = json.loads((ROOT/'artifacts/safedreamer_l3_paper_pilot192/plan.json').read_text())
    if digest(signature['checkpoint']) != ref['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    started = time.perf_counter()
    with SafeDreamerWrapper(RolloutConfig(horizon=192,n_rollouts=1,seed=args.seed_base,
                                        action_source='policy',extra=extra)) as wrapper:
        wrapper.load()
        import jax
        if jax.default_backend() != 'gpu': raise RuntimeError('GPU required')
        actual = dict(action_source='policy',expl_behavior=wrapper._config.expl_behavior,
            planner={k:getattr(wrapper._config.planner,k) for k in signature['policy']['planner']},
            cost_limit=wrapper._config.cost_limit,
            planner_source_sha256=digest(Path(extra['repo_root'])/'SafeDreamer/behaviors.py'),
            note=signature['policy']['note'])
        if actual != signature['policy']: raise ValueError('Policy changed')
        collect_shared(wrapper,make_sampler(wrapper,32),train,out/'fit_queries',96,args.seed_base)
    write(out/'collection.json',dict(time=stamp(),elapsed_seconds=time.perf_counter()-started,
          query_chunks=len(new_seeds),successors=32*len(new_seeds),sampling_device='JAX_GPU'))


def train(args):
    torch.set_num_threads(2)
    out = args.output.resolve(); plan = json.loads((out/'plan.json').read_text())
    source = Path(plan['source'])
    if digest(source/'plan.json') != plan['source_plan_sha256']: raise ValueError('Source changed')
    for name, sha in plan['code_sha256'].items():
        if digest(ROOT/name) != sha: raise ValueError(f'Changed experiment: {name}')
    _, frozen, tp0, dp0, tq, dq, ts, ds = load_source(source)
    extra, seeds = load_queries(out,'fit',tp0,'')
    if set(seeds)&(set(ts)|set(ds)): raise ValueError('Query seed overlap')
    combined = []
    for a,b in zip(tq,extra):
        if (a['path_index'],a['t']) != (b['path_index'],b['t']): raise ValueError('Anchor order changed')
        if len(a['successors']) != 32 or len(b['successors']) != 96: raise ValueError('Wrong kappa')
        combined.append(dict(path_index=a['path_index'],t=a['t'],
            successors=np.concatenate([a['successors'],b['successors']]),decoded=np.concatenate([a['decoded'],b['decoded']])))
    spec = next(s for s in SPECS if s['id']=='B_S24')
    tp,tb32 = relabel(tp0,tq,spec); _,tb128 = relabel(tp0,combined,spec)
    dp,db = relabel(dp0,dq,spec)
    assert_anchor_splits_disjoint(tb128,db,left_name='fit',right_name='development')
    core = tp.reshape(-1,tp.shape[-1]); mean=core.mean(0); scale=np.maximum(core.std(0),.05)
    mean[-2:]=0; scale[-2:]=1
    slack=json.loads((source/'report.json').read_text())['rows'][0]['sampling_radius']
    started=time.perf_counter(); fitted=[]
    for kappa,tb in [(32,tb32),(128,tb128)]:
        data=prepare(tp,np.zeros(tp.shape[:2],np.float32),tb)
        settings=[('W',m,'H1',0.) for m in plan['W_margins']]
        settings += [('U',m,r,0.) for m in plan['U_margins'] for r in ('H1','H2_candidate')]
        if kappa==32:
            settings += [('W',m,'H1',.5) for m in plan['W_margins']]
            settings += [('U',m,'H1',.5) for m in plan['U_margins']]
        for kind,margin,retention,pw in settings:
            name=f'{kind}_k{kappa}_m{margin:g}_{retention}_path{pw:g}'
            net,history=fit(data,mean,scale,kind,margin,retention,pw,plan['steps'])
            if kappa==32 and pw==.5 and margin==(.01 if kind=='W' else .001) and plan['steps']==400:
                file=source/f'B_S24_{kind}.pt'
                if digest(file)!=frozen['models']['B_S24'][kind]: raise ValueError('Baseline weights changed')
                original=restore(file)
                for key,value in original.state_dict().items():
                    if not torch.equal(value,net.state_dict()[key]): raise ValueError('Baseline mismatch')
            file=out/f'{name}.pt'; torch.save(dict(family=net.family,state=net.state_dict()),file)
            fitted.append(dict(name=name,kind=kind,kappa=kappa,margin=margin,retention=retention,
                path_weight=pw,sha256=digest(file),history=history))
            print('FITTED',name,flush=True)
    write(out/'frozen.json',dict(time=stamp(),plan_sha256=digest(out/'plan.json'),
                               models={r['name']:r['sha256'] for r in fitted}))
    # These outcomes predate this experiment. Freezing does NOT make them independent.
    rows=[]
    for row in fitted:
        net=restore(out/f'{row["name"]}.pt')
        for role,paths,batches in [('train',tp,tb32 if row['kappa']==32 else tb128),('development',dp,db)]:
            row[role]=evaluate(net,row['kind'],row['retention'],batches,paths,row['margin'],slack)
        d=row['development']['candidate_domain']; r=row['development']
        coverage=r['below_ell_anchor_count']==len(db) and r['initial_below_ell_count']==len(dp)
        row['development_numeric_gate']=bool(d['required'] and d['raw_rate']>=.9 and
            (d['distribution']['q90']<-.05 if row['kind']=='W' else coverage and not r['collapsed_sample_range']))
        rows.append(row)
        print('RESULT',row['name'],d,flush=True)
    write(out/'report.json',dict(completed=stamp(),training_evaluation_seconds=time.perf_counter()-started,
        rows=rows,core_checks={'train32':core_check(tb32),'train128':core_check(tb128),'development':core_check(db)},
        confidence=None,global_L3='ABSTAIN',deductive_status='NOT_ESTABLISHED',
        infinite_recurrence_probability_lower_bound=None,final_validation='NOT_RUN',
        limitations=['Previously inspected development data; no new confidence claim.',
          'I=F sampled closure does not establish support-wide invariance or global I subset C.',
          'No cell/Lipschitz extension, region coverage, initial expectation or collar probability proof.',
          'Large train margins are preferences, not assumptions proved feasible.'],
        numeric_gate_candidates={k:[r['name'] for r in rows if r['kind']==k and r['development_numeric_gate']]
                                 for k in ('W','U')}))
    with (out/'summary.csv').open('x',newline='') as f:
        writer=csv.writer(f); writer.writerow(['name','split','required','raw_pass','fixed_slack_pass','min','q10','median','q90','max'])
        for row in rows:
            for role in ('train','development'):
                d=row[role]['candidate_domain']
                writer.writerow([row['name'],role,d['required'],d['raw_pass'],d['fixed_slack_pass'],
                                 *[d['distribution'][k] for k in ('min','q10','median','q90','max')]])
    with (out/'anchor_drifts.csv').open('x',newline='') as f:
        writer=csv.writer(f); writer.writerow(['name','split','id','accepting','candidate_required','value','mean_successor','drift'])
        for row in rows:
            for role in ('train','development'):
                for a in row[role]['anchor_rows']:
                    required=not a['accepting'] or row['kind']=='U' and row['retention']=='H1'
                    writer.writerow([row['name'],role,a['id'],a['accepting'],required,a['value'],a['mean_successor'],a['drift']])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['collect','train'])
    parser.add_argument('--source',type=Path,default=ROOT/'artifacts/safedreamer_l3_spec_ablation_v1')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed-base',type=int,default=81500000)
    parser.add_argument('--steps',type=int,default=400)
    args=parser.parse_args()
    if args.steps<1: parser.error('Positive steps required')
    (collect if args.stage=='collect' else train)(args)
