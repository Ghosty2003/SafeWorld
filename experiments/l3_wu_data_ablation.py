"""Fixed W/U architecture: nested fit-size x nested successor-count ablation.

Development only, not final validation or a global L3 proof. Paths and queries
are saved incrementally. --resume reuses completed units after hash checks.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.l3_safedreamer_clear_pipeline import digest,write,stamp,product,make_sampler,query
from experiments.l3_safedreamer_clear_recurrence import monitor,summarize
from experiments.l3_wu_search import Candidate,prepare,return_targets
from core.lbsm.operator_model import AnchorSuccessorBatch,assert_anchor_splits_disjoint


def anchor_times(counter, seed):
    """One uniform anchor per stratum per path; no ranking by W/U failures."""
    rng=np.random.default_rng(seed)
    times=np.arange(1,len(counter)-1)
    c=np.asarray(counter)
    result=[]
    for non_f in (True,False):
        candidates=times[(c[times]<48)==non_f]
        if len(candidates): result.append(int(rng.choice(candidates)))
    return result


def products(a):
    safe=np.linalg.norm(a['decoded'][:,9:25].reshape(-1,8,2),axis=-1).min(-1).astype(float)-.2>=0
    counter=monitor(safe)['counter']
    xs=np.stack([product(z,a['planner_action_mean'][t],a['planner_action_std'][t],
                         a['planner_initialized'][t].item(),counter[t]) for t,z in enumerate(a['latent'])])
    return xs,counter,safe


def checked_array(file,sha):
    if digest(file)!=sha: raise ValueError(f'Changed artifact: {file}')
    with np.load(file,allow_pickle=False) as a: return {k:a[k].copy() for k in a.files}


def collect_paths(wrapper,extra,out,role,count,base,seen):
    from configs.settings import RolloutConfig
    folder=out/role; folder.mkdir(exist_ok=True)
    records=[]; safes=[]
    for draw in range(count*5):
        if len(records)>=count: break
        meta=folder/f'path_{draw:04d}.json'; raw=meta.with_suffix('.npz')
        if meta.exists():
            row=json.loads(meta.read_text()); a=checked_array(raw,row['sha256'])
        else:
            if raw.exists(): raise RuntimeError(f'Uncommitted path file; inspect before resuming: {raw}')
            started=time.perf_counter(); seed=base+draw
            data=wrapper.sample_latent_rollouts(RolloutConfig(horizon=192,n_rollouts=1,seed=seed,
                                                             action_source='policy',extra=extra))
            if data['action_source']!='policy': raise ValueError('Policy required')
            a=dict(latent=data['latent'][0],decoded=data['decoded'][0],actions=data['actions'][0],
                   **{k:v[0] for k,v in data['policy_state'].items()},
                   **{'rssm_'+k:v[0] for k,v in data['rssm'].items()})
            xs,counter,safe=products(a)
            fp=hashlib.sha256(a['latent'].tobytes()).hexdigest()
            if fp in seen: raise ValueError('Duplicate raw path across roles')
            initial=float(np.linalg.norm(a['decoded'][0,7:9]))
            row=dict(seed=seed,accepted=initial>=1.,initial_goal=initial,path_fingerprint=fp,
                     anchor_times=anchor_times(counter,seed+700000),elapsed_seconds=time.perf_counter()-started)
            np.savez_compressed(raw,**a); row['sha256']=digest(raw); write(meta,row)
        if row['path_fingerprint'] in seen: raise ValueError('Path leakage')
        seen.add(row['path_fingerprint'])
        if row['accepted']:
            xs,counter,safe=products(a); safes.append(safe)
            records.append((meta,row,xs,counter))
            print('PATH',role,len(records),'/',count,'anchors',row['anchor_times'],flush=True)
    if len(records)!=count: raise RuntimeError('Initial-only rejection budget exhausted')
    summary=folder/'summary.json'
    if not summary.exists(): write(summary,summarize(np.stack(safes)))
    return records


def collect_queries(wrapper,sampler,out,role,records,kmax,base):
    """Chunks of 32, all reset to same saved RSSM + planner state.

    Each kappa uses a prefix of the very same 256 successor draws. Prefixes
    share randomness, but samples within each prefix are conditional draws.
    """
    folder=out/(role+'_queries'); folder.mkdir(exist_ok=True)
    batches=[]; counts=[]
    for pi,(meta,row,xs,counter) in enumerate(records):
        a=checked_array(meta.with_suffix('.npz'),row['sha256'])
        n=0
        for ai,t in enumerate(row['anchor_times']):
            chunks=[]
            for chunk in range(math.ceil(kmax/32)):
                file=folder/f'p{pi:04d}_a{ai}_c{chunk:02d}.npz'; manifest=file.with_suffix('.json')
                if manifest.exists():
                    m=json.loads(manifest.read_text()); saved=checked_array(file,m['sha256'])
                    np.testing.assert_array_equal(saved['anchor'],xs[t]); succ=saved['successors']
                    if m['path_sha256']!=row['sha256'] or m['t']!=t: raise ValueError('Anchor changed')
                else:
                    if file.exists(): raise RuntimeError(f'Uncommitted query file: {file}')
                    seed=base+pi*1000+ai*100+chunk
                    start=time.perf_counter()
                    succ,raw=query(wrapper,sampler,a,t,counter[t],seed)
                    np.savez_compressed(file,anchor=xs[t],successors=succ,**raw)
                    write(manifest,dict(time=stamp(),seed=seed,t=t,path_sha256=row['sha256'],
                                        sha256=digest(file),elapsed_seconds=time.perf_counter()-start))
                chunks.append(succ)
            successors=np.concatenate(chunks)[:kmax]
            batches.append(AnchorSuccessorBatch(xs[t],successors,f'{role}:{row["seed"]}:{t}','STRATIFIED_DEVELOPMENT'))
            n+=1
        counts.append(n)
        print('QUERIES',role,pi+1,'/',len(records),'anchors',len(batches),'kappa',kmax,flush=True)
    return batches,counts


def fit(kind,data,mean,scale,steps):
    seed=73 if kind=='W' else 41
    torch.manual_seed(seed)
    net=Candidate(mean,scale,'wide' if kind=='W' else 'sublevel')
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    history=[]
    for step in range(steps):
        # Fixed optimizer budget across fit sizes; minibatching avoids a large
        # full-batch RAM/compute increase at N_fit=200.
        b=torch.randperm(len(data['x']))[:32]
        x,y=data['x'][b],data['y'][b]
        idx=torch.randperm(len(data['core']))[:512]; c=data['core'][idx]
        idx=torch.randperm(len(data['px']))[:512]; px,py=data['px'][idx],data['py'][idx]
        drift=net(y.flatten(0,1)).reshape(y.shape[:2]).mean(1)-net(x)
        pd=net(py)-net(px)
        if kind=='W':
            nf=x[:,-1]<1; pnf=px[:,-1]<1
            branch=torch.relu(drift[nf]+.01).mean() if nf.any() else drift.sum()*0
            pl=torch.relu(pd[pnf]+.01).mean() if pnf.any() else pd.sum()*0
            shape=((net(c)-.48*(1-c[:,-1]))**2).mean()
        else:
            branch=torch.relu(drift+.001).mean(); pl=torch.relu(pd+.001).mean()
            shell=c.clone(); shell[:,:-2]=net.mean[:-2]+4*(c[:,:-2]-net.mean[:-2])
            shape=((net(c)-.2)**2).mean()+torch.relu(.85-net(shell)).square().mean()
        loss=branch+.5*pl+.25*shape
        opt.zero_grad(); loss.backward(); opt.step()
        if step==0 or (step+1)%100==0 or step+1==steps:
            history.append(dict(step=step+1,loss=float(loss),branch=float(branch),shape=float(shape)))
    return net.eval(),history


def restore(file):
    ck=torch.load(file,map_location='cpu',weights_only=True)
    net=Candidate(ck['state']['mean'].numpy(),ck['state']['scale'].numpy(),ck['family'])
    net.load_state_dict(ck['state']); return net.eval()


@torch.no_grad()
def scan(W,U,batches,kappas,model_count,delta=.05):
    m=len(batches)
    # Union across W/U, all pre-frozen candidate fits, anchors and kappa values.
    # Correlation between prefixes/anchors is allowed by the union bound.
    delta_each=delta/(2*model_count*m*len(kappas))
    x=torch.tensor(np.stack([b.anchor for b in batches]))
    w=W(x).numpy(); u=U(x).numpy(); in_c=u<=.8; nonf=x[:,-1].numpy()<1
    # Batched by anchor to keep RAM bounded for large grids.
    ws=np.stack([W(torch.tensor(b.successors)).numpy() for b in batches])
    us=np.stack([U(torch.tensor(b.successors)).numpy() for b in batches])
    result=[]
    for k in kappas:
        if k<1 or any(len(b.successors)<k for b in batches): raise ValueError('Insufficient successors')
        gw=ws[:,:k].astype(float).mean(1)-w
        gu=us[:,:k].astype(float).mean(1)-u
        radius=math.sqrt(math.log(1/delta_each)/(2*k))
        row=dict(kappa=k,anchors=m,anchors_in_C=int(in_c.sum()),nonaccepting=int(nonf.sum()),
                 sampling_radius=radius,delta_per_check=delta_each,cell_correction=None,
                 scope='SAMPLED_ANCHORS_ONLY_NO_CELL_LIFT')
        for kind,g,mask,threshold in (('W',gw,nonf,-.01),('U',gu,np.ones(m,dtype=bool),0.)):
            raw=g<=threshold; corrected=g+radius<=threshold
            required=mask&in_c
            row[kind]=dict(required_in_C=int(required.sum()),all_stratum_count=int(mask.sum()),
                           raw_pass=int((raw&required).sum()),sampling_corrected_pass=int((corrected&required).sum()),
                           raw_rate=float(raw[required].mean()) if required.any() else None,
                           sampling_corrected_rate=float(corrected[required].mean()) if required.any() else None,
                           raw_pass_all_anchors=int((raw&mask).sum()),
                           mean_drift=float(g[mask].mean()) if mask.any() else None,
                           drifts=g.tolist())
        result.append(row)
    return result


def run(args):
    torch.set_num_threads(2)
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    out=args.output.resolve()
    source=ROOT/'artifacts/safedreamer_l3_paper_pilot192'
    original=json.loads((source/'plan.json').read_text())
    code=[Path(__file__).resolve(),ROOT/'experiments/l3_wu_search.py',
          ROOT/'experiments/l3_safedreamer_clear_pipeline.py',ROOT/'wrappers/safedreamer_wrapper.py']
    signature=dict(fit_counts=args.fit_counts,dev_paths=args.dev_paths,kappas=args.kappas,steps=args.steps,
                   seed_base=args.seed_base,checkpoint=original['checkpoint'],policy=original['policy'],
                   code_sha256={str(p.relative_to(ROOT)):digest(p) for p in code})
    if digest(original['checkpoint'])!=original['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    if args.resume:
        plan=json.loads((out/'plan.json').read_text())
        if plan['signature']!=signature: raise ValueError('Resume configuration/source mismatch')
    else:
        out.mkdir(parents=True,exist_ok=False)
        plan=dict(time=stamp(),signature=signature,spec=original['spec'],horizon=192,
                  fit_kappa=32,W_family='wide',U_family='sublevel',eps_W=.01,ell=.8,B=1.,
                  scope='DEVELOPMENT_ABLATION_NOT_FINAL_VALIDATION',
                  anchors='One uniform nonaccepting and one uniform accepting t in [1,191] per path when available.',
                  sampling='All kappa prefixes use same saved max-kappa successors; full RSSM and planner carry reset per draw.',
                  freeze='All compared weights and dev anchors frozen before fresh dev successor queries.',
                  independence='Conditional successor draws; path anchors are clustered, not iid rollout success trials.',
                  initial_condition='decoded initial goal distance>=1.0 only; no success filtering',
                  caveats=['Simulator reset RNG not fully controlled by explicit imagination seeds.',
                           'No final holdout touched; development outcomes may inform future choices.',
                           'Sampling-only UCB is not cellwise correction or an infinite-horizon warrant.',
                           'Failure to certify does not prove no W/U exists.'])
        write(out/'plan.json',plan)
    if (out/'report.json').exists():
        print('Already complete:',out/'report.json'); return
    extra=build_extra(argparse.Namespace(repo_root=None,checkpoint=original['checkpoint']))
    extra['action_source']='policy'; extra['config_overrides']['jax'].update(platform='gpu',logical_gpus=0)
    start=time.perf_counter()
    with SafeDreamerWrapper(RolloutConfig(horizon=192,n_rollouts=1,seed=args.seed_base,
                                         action_source='policy',extra=extra)) as wrapper:
        wrapper.load()
        import jax
        if jax.default_backend()!='gpu': raise RuntimeError('GPU required, no silent fallback')
        actual=dict(action_source='policy',expl_behavior=wrapper._config.expl_behavior,
                    planner={k:getattr(wrapper._config.planner,k) for k in original['policy']['planner']},
                    cost_limit=wrapper._config.cost_limit,
                    planner_source_sha256=digest(Path(extra['repo_root'])/'SafeDreamer/behaviors.py'),
                    note=original['policy']['note'])
        if actual!=original['policy']: raise ValueError('Changed controller')
        print('GPU',jax.devices(),flush=True)
        sampler=make_sampler(wrapper,32); seen=set()
        records=collect_paths(wrapper,extra,out,'fit',max(args.fit_counts),args.seed_base,seen)
        train,counts=collect_queries(wrapper,sampler,out,'fit',records,32,args.seed_base+1000000)
        paths=np.stack([r[2] for r in records]); targets=np.stack([return_targets(r[3]) for r in records])
        models={}; fits={}
        for n in args.fit_counts:
            num=sum(counts[:n]); batches=train[:num]
            data=prepare(paths[:n],targets[:n],batches)
            core=paths[:n].reshape(-1,578); mean=core.mean(0); scale=np.maximum(core.std(0),.05)
            mean[-2:]=0; scale[-2:]=1
            nets=[]
            for kind in ('W','U'):
                file=out/f'{kind}_fit{n}.pt'; meta=file.with_suffix('.json')
                if meta.exists():
                    m=json.loads(meta.read_text())
                    if digest(file)!=m['sha256']: raise ValueError('Frozen weights changed')
                    net=restore(file)
                else:
                    if file.exists(): raise RuntimeError(f'Uncommitted checkpoint: {file}')
                    net,history=fit(kind,data,mean,scale,args.steps)
                    torch.save(dict(family=net.family,state=net.state_dict()),file)
                    write(meta,dict(time=stamp(),sha256=digest(file),history=history))
                nets.append(net)
            models[n]=nets
            fits[n]=dict(fit_paths=n,W_nonaccepting_anchors=sum(b.anchor[-1]<1 for b in batches),
                         U_anchors=len(batches),fit_kappa=32)
            fits[n]['W_nonaccepting_anchors']=int(fits[n]['W_nonaccepting_anchors'])
            print('FIT_FROZEN',fits[n],flush=True)
        devrecords=collect_paths(wrapper,extra,out,'development',args.dev_paths,args.seed_base+100000,seen)
        # Persist exact anchors and all compared model hashes BEFORE successors.
        freeze=out/'development_frozen.json'
        frozen=dict(models={str(n):{k:digest(out/f'{k}_fit{n}.pt') for k in ('W','U')} for n in models},
                    anchors=[dict(path=str(r[0].relative_to(out)),sha=r[1]['sha256'],times=r[1]['anchor_times']) for r in devrecords])
        if freeze.exists():
            if json.loads(freeze.read_text())['frozen']!=frozen: raise ValueError('Development freeze mismatch')
        else: write(freeze,dict(time=stamp(),frozen=frozen))
        dev,_=collect_queries(wrapper,sampler,out,'development',devrecords,max(args.kappas),args.seed_base+2000000)
        assert_anchor_splits_disjoint(train,dev,left_name='fit',right_name='development')
    grid=[]
    for n,(W,U) in models.items():
        for row in scan(W,U,dev,args.kappas,len(models)):
            row.update(fits[n]); grid.append(row)
            print('GRID',n,row['kappa'],'W',row['W']['raw_pass'],'/',row['W']['required_in_C'],
                  'corrected',row['W']['sampling_corrected_pass'],'U',row['U']['raw_pass'],'/',row['U']['required_in_C'],
                  'corrected',row['U']['sampling_corrected_pass'],flush=True)
    timing=[]
    for role in ('fit','development','fit_queries','development_queries'):
        items=[json.loads(p.read_text()) for p in (out/role).glob('*.json')]
        seconds=[r['elapsed_seconds'] for r in items if 'elapsed_seconds' in r]
        timing.append(dict(role=role,units=len(seconds),total_seconds=sum(seconds),
                           median_seconds=float(np.median(seconds)) if seconds else None))
    write(out/'report.json',dict(completed=stamp(),scope=plan['scope'],grid=grid,timing=timing,
          this_invocation_seconds=time.perf_counter()-start,
          sampling_only_union_confidence=.95,
          note='Simultaneous conditional mean UCB for this fixed family only, assuming conditionally independent successor draws. Not recurrence confidence.',
          final_validation='NOT_RUN',global_L3='ABSTAIN',recurrence_probability_lower_bound=None))
    with (out/'grid.csv').open('x',newline='') as f:
        w=csv.writer(f); w.writerow(['fit_paths','kappa','W_anchors','W_raw','W_sampling_corrected','U_anchors','U_raw','U_sampling_corrected','radius'])
        for r in grid: w.writerow([r['fit_paths'],r['kappa'],r['W']['required_in_C'],r['W']['raw_rate'],r['W']['sampling_corrected_rate'],
                                  r['U']['required_in_C'],r['U']['raw_rate'],r['U']['sampling_corrected_rate'],r['sampling_radius']])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fit-counts',type=int,nargs='+',default=[12,50,100,200])
    p.add_argument('--dev-paths',type=int,default=100)
    p.add_argument('--kappas',type=int,nargs='+',default=[8,16,32,64,128,256])
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--seed-base',type=int,default=47500000)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if min(a.fit_counts+a.kappas+[a.dev_paths,a.steps])<1 or max(a.kappas)>256: p.error('Positive counts and kappa<=256 required')
    a.fit_counts=sorted(set(a.fit_counts)); a.kappas=sorted(set(a.kappas))
    run(a)
