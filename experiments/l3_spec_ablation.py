"""Shared-dynamics development comparison: GF S48/S16/S24/S32 and GF clearance.

Only monitors differ. Same underlying paths, union of anchor locations, fresh
shared successor queries, network families, update budget, epsilon and range.
No final validation or infinite-recurrence claim.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.l3_safedreamer_clear_pipeline import digest,write,stamp,make_sampler,query
from experiments.l3_wu_data_ablation import products,checked_array,fit,scan,restore
from experiments.l3_wu_search import prepare
from core.lbsm.operator_model import AnchorSuccessorBatch,assert_anchor_splits_disjoint

SPECS=[dict(id='A_S48',window=48,threshold=0.),
       dict(id='B_S16',window=16,threshold=0.),
       dict(id='B_S24',window=24,threshold=0.),
       dict(id='B_S32',window=32,threshold=0.),
       dict(id='C_clearance',window=1,threshold=.2)]


def margin(decoded):
    return np.linalg.norm(decoded[...,9:25].reshape(-1,8,2),axis=-1).min(-1).astype(float)-.2


def counters(predicate,window):
    count=0; result=[]
    for valid in predicate:
        count=min(count+1,window) if valid else 0
        result.append(count)
    return np.asarray(result,dtype=np.int64)


def next_counter(count,predicate,window):
    return np.where(predicate,min(count+1,window),0)/float(window)


def finite_behavior(predicate,window):
    c=counters(predicate,window)
    run=0; disjoint=0
    for valid in predicate:
        run=run+1 if valid else 0
        if run==window: disjoint+=1; run=0
    accept=c==window
    exits=accept[:-1]&~accept[1:]
    # True return after an observed exit, not merely adjacent accepting times.
    exited=False; returns=0
    for t in range(1,len(accept)):
        if accept[t-1] and not accept[t]: exited=True
        if exited and accept[t]: returns+=1; exited=False
    half=len(predicate)//2
    late=counters(predicate[half:],window)
    return dict(two_nonoverlapping_windows=disjoint>=2,nonoverlapping_windows=disjoint,
                accepting_fraction=float(accept.mean()),observed_exits=int(exits.sum()),
                return_after_exit=returns>0,return_count=returns,
                complete_window_in_second_half=bool((late==window).any()))


def load_role(source,role):
    paths=[]
    for file in sorted((source/role).glob('path_*.json')):
        r=json.loads(file.read_text())
        if not r['accepted']: continue
        a=checked_array(file.with_suffix('.npz'),r['sha256'])
        base,_,_=products(a)
        monitors={s['id']:counters(margin(a['decoded'])>=s['threshold'],s['window']) for s in SPECS}
        # Common union ensures every spec sees exactly the same physical anchors.
        times=set(); candidates=np.arange(1,len(base)-1)
        for i,s in enumerate(SPECS):
            rng=np.random.default_rng(r['seed']+800000+i)
            for nonf in (True,False):
                pool=candidates[(monitors[s['id']][candidates]<s['window'])==nonf]
                if len(pool): times.add(int(rng.choice(pool)))
        paths.append(dict(file=str(file.with_suffix('.npz').resolve()),record=r,arrays=a,
                          base=base,monitors=monitors,times=sorted(times)))
    return paths


def collect_shared(wrapper,sampler,paths,folder,kappa,seed_base):
    folder.mkdir()
    result=[]
    for pi,p in enumerate(paths):
        for ai,t in enumerate(p['times']):
            chunks=[]; decoded=[]
            for ci in range(kappa//32):
                # Counter does not affect policy/dynamics. Rebuild each monitor
                # from decoded successors below; never reuse the S48 q value.
                seed=seed_base+pi*10000+ai*100+ci
                start=time.perf_counter()
                successor,raw=query(wrapper,sampler,p['arrays'],t,0,seed)
                file=folder/f'p{pi:03d}_t{t:03d}_c{ci:02d}.npz'
                np.savez_compressed(file,anchor=p['base'][t],successors=successor,**raw)
                write(file.with_suffix('.json'),dict(time=stamp(),seed=seed,path_sha256=p['record']['sha256'],
                      t=t,sha256=digest(file),elapsed_seconds=time.perf_counter()-start))
                chunks.append(successor); decoded.append(raw['decoded'])
            result.append(dict(path_index=pi,t=t,successors=np.concatenate(chunks),decoded=np.concatenate(decoded)))
        print('QUERIED',folder.name,pi+1,'/',len(paths),'common anchors',len(result),flush=True)
    return result


def relabel(paths,queries,spec):
    sid=spec['id']; length=spec['window']; labeled=[]
    for p in paths:
        x=p['base'].copy(); x[:,-1]=p['monitors'][sid]/float(length); labeled.append(x)
    batches=[]
    for q in queries:
        pi,t=q['path_index'],q['t']
        successor=q['successors'].copy()
        successor[:,-1]=next_counter(paths[pi]['monitors'][sid][t],
                          margin(q['decoded'])>=spec['threshold'],length)
        batches.append(AnchorSuccessorBatch(labeled[pi][t],successor,
                       f'{sid}:{paths[pi]["record"]["seed"]}:{t}','COMMON_ANCHORS_DEVELOPMENT'))
    return np.stack(labeled),batches


def behavior_summary(paths,spec):
    per=[finite_behavior(margin(p['arrays']['decoded'])>=spec['threshold'],spec['window']) for p in paths]
    return dict(n_paths=len(per),two_windows=sum(r['two_nonoverlapping_windows'] for r in per),
                late_window=sum(r['complete_window_in_second_half'] for r in per),
                exit_and_return=sum(r['return_after_exit'] for r in per),per_path=per,
                note='Finite 192-step proxy only; not infinite recurrence satisfaction probability')


def run(args):
    torch.set_num_threads(2)
    source=args.source.resolve(); out=args.output.resolve()
    old=json.loads((source/'plan.json').read_text())['signature']
    train=load_role(source,'fit'); dev=load_role(source,'development')
    if {p['record']['path_fingerprint'] for p in train}&{p['record']['path_fingerprint'] for p in dev}:
        raise ValueError('Path leakage')
    out.mkdir(parents=True,exist_ok=False)
    write(out/'plan.json',dict(time=stamp(),scope='SPECIFICATION_DEVELOPMENT_ABLATION',specs=SPECS,
          source=str(source),source_plan_sha256=digest(source/'plan.json'),
          source_signature=old,horizon=192,fit_paths=len(train),dev_paths=len(dev),
          fit_kappa=32,dev_kappa=args.kappa,steps=args.steps,W='wide',U='sublevel',
          eps_W=.01,ell=.8,B=1.,seed_base=args.seed_base,
          W_shaping='Same .48*(1-normalized_monitor_progress) target for every spec; not a theorem.',
          C_choice='hazard margin>=0.2; chosen from training-only screening (11/12 paths with exits).',
          counters='Each monitor separately normalized by its own window length; no absorbing acceptance.',
          uncertainty='Sampling-only union bound across 5 frozen pairs and all common anchors; no cell correction.',
          caveats=['Reused fit/development paths, fresh conditional successors; no untouched final validation.',
                   'Common anchor union is enriched by all monitors, not a natural rollout sample.',
                   'Small pilot, one seed pair, no statistical significance or causal bottleneck conclusion.',
                   'Always-safe behavior satisfies recurrence; no mandatory hazard between windows.'],
          anchors={role:[dict(file=p['file'],sha256=p['record']['sha256'],times=p['times']) for p in data]
                   for role,data in [('fit',train),('development',dev)]},
          source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),
                        ROOT/'experiments/l3_wu_data_ablation.py',ROOT/'experiments/l3_wu_search.py',
                        ROOT/'experiments/l3_safedreamer_clear_pipeline.py',ROOT/'wrappers/safedreamer_wrapper.py']}))
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    extra=build_extra(argparse.Namespace(repo_root=None,checkpoint=old['checkpoint']))
    extra['action_source']='policy'; extra['config_overrides']['jax'].update(platform='gpu',logical_gpus=0)
    reference=json.loads((ROOT/'artifacts/safedreamer_l3_paper_pilot192/plan.json').read_text())
    if digest(old['checkpoint'])!=reference['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    started=time.perf_counter()
    with SafeDreamerWrapper(RolloutConfig(horizon=192,n_rollouts=1,seed=args.seed_base,
                                         action_source='policy',extra=extra)) as wrapper:
        wrapper.load()
        import jax
        if jax.default_backend()!='gpu': raise RuntimeError('GPU required')
        actual=dict(action_source='policy',expl_behavior=wrapper._config.expl_behavior,
                    planner={k:getattr(wrapper._config.planner,k) for k in old['policy']['planner']},
                    cost_limit=wrapper._config.cost_limit,
                    planner_source_sha256=digest(Path(extra['repo_root'])/'SafeDreamer/behaviors.py'),
                    note=old['policy']['note'])
        if actual!=old['policy']: raise ValueError('Policy changed')
        sampler=make_sampler(wrapper,32)
        tq=collect_shared(wrapper,sampler,train,out/'fit_queries',32,args.seed_base+1000000)
        models={}; training={}
        for spec in SPECS:
            sid=spec['id']; paths,batches=relabel(train,tq,spec)
            data=prepare(paths,np.zeros(paths.shape[:2],np.float32),batches)
            core=paths.reshape(-1,578); mean=core.mean(0); scale=np.maximum(core.std(0),.05)
            mean[-2:]=0; scale[-2:]=1
            nets=[]
            for kind in ('W','U'):
                net,history=fit(kind,data,mean,scale,args.steps)
                file=out/f'{sid}_{kind}.pt'
                torch.save(dict(family=net.family,state=net.state_dict()),file)
                write(file.with_suffix('.json'),dict(sha256=digest(file),history=history))
                nets.append(net)
            models[sid]=nets
            training[sid]=scan(*nets,batches,[32],len(SPECS))[0]
            # Training means only, never describe fitted-sample UCB as valid.
            for kind in ('W','U'):
                training[sid][kind].pop('sampling_corrected_pass')
                training[sid][kind].pop('sampling_corrected_rate')
            training[sid]['scope']='TRAINING_RAW_DIAGNOSTIC_ONLY'
            print('FROZEN',sid,'training W',training[sid]['W']['raw_pass'],'/',training[sid]['W']['required_in_C'],flush=True)
        write(out/'frozen.json',dict(time=stamp(),plan_sha256=digest(out/'plan.json'),
              models={sid:{k:digest(out/f'{sid}_{k}.pt') for k in ('W','U')} for sid in models}))
        dq=collect_shared(wrapper,sampler,dev,out/'development_queries',args.kappa,args.seed_base+2000000)
    rows=[]
    for spec in SPECS:
        sid=spec['id']; _,tb=relabel(train,tq,spec); _,db=relabel(dev,dq,spec)
        assert_anchor_splits_disjoint(tb,db,left_name='fit',right_name='development')
        row=scan(*models[sid],db,[args.kappa],len(SPECS))[0]
        row.update(spec=spec,behavior=behavior_summary(dev,spec),training=training[sid],
                   fit_behavior=behavior_summary(train,spec))
        rows.append(row)
        print('RESULT',sid,'finite-repeat',row['behavior']['two_windows'],'/',len(dev),
              'W',row['W']['raw_pass'],'/',row['W']['required_in_C'],'U',row['U']['raw_pass'],'/',row['U']['required_in_C'],
              'sampling-corrected',row['W']['sampling_corrected_pass'],row['U']['sampling_corrected_pass'],flush=True)
    write(out/'report.json',dict(completed=stamp(),elapsed_seconds=time.perf_counter()-started,rows=rows,
          scope='DEVELOPMENT_ONLY_FINITE_BEHAVIOR_AND_POINT_DRIFT',global_L3='ABSTAIN',
          infinite_recurrence_probability=None,final_validation='NOT_RUN'))
    with (out/'comparison.csv').open('x',newline='') as f:
        w=csv.writer(f); w.writerow(['spec','finite_2windows','dev_paths','exit_return_paths','W_raw','W_required',
                                    'U_raw','U_required','W_sampling_corrected','U_sampling_corrected'])
        for r in rows: w.writerow([r['spec']['id'],r['behavior']['two_windows'],len(dev),r['behavior']['exit_and_return'],
                                   r['W']['raw_pass'],r['W']['required_in_C'],r['U']['raw_pass'],r['U']['required_in_C'],
                                   r['W']['sampling_corrected_pass'],r['U']['sampling_corrected_pass']])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'artifacts/safedreamer_l3_data_kappa_timing')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kappa',type=int,default=256)
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--seed-base',type=int,default=67500000)
    a=p.parse_args()
    if a.kappa<32 or a.kappa%32 or a.steps<1: p.error('kappa must be a positive multiple of 32; steps positive')
    run(a)
