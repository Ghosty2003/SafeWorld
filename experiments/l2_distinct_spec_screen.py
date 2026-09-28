"""Eight non-window, causal L2 tasks. Fit/internal only; no CP or Test1.

All thresholds are set from80 train-fit trajectories before internal replay.
Three function families and a fixed positive scale grid per task. Independent
sampling is required after this repeatedly-used development set selects a V.
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import l2_spec_continuation as exp
from experiments.l2_spec_continuation_audit import independent
base=exp.base


def signals(a):
    decoded=a['decoded'].astype(np.float64)
    clearance,goal=base.signals(a)
    position=decoded[:,27:29]
    return dict(clearance=clearance,goal=goal,speed=np.linalg.norm(decoded[:,:2],axis=-1),
                displacement=np.linalg.norm(position-position[0],axis=-1))


def candidates(fit):
    pooled={k:np.concatenate([signals(a)[k] for a in fit]) for k in signals(fit[0])}
    def qt(k,p):return float(np.quantile(pooled[k],p))
    thresholds=dict(clear_low=qt('clearance',.25),clear_high=qt('clearance',.60),
                    speed_low=qt('speed',.25),speed_high=qt('speed',.75),
                    goal_near=qt('goal',.35),goal_far=qt('goal',.75),away=qt('displacement',.5))
    t=thresholds
    def pred(key,op,level):return dict(signal=key,op=op,threshold=level)
    low=pred('clearance','le',t['clear_low']);high=pred('clearance','ge',t['clear_high'])
    slow=pred('speed','le',t['speed_low']);fast=pred('speed','ge',t['speed_high'])
    far=pred('goal','ge',t['goal_far']);near=pred('goal','le',t['goal_near'])
    away=pred('displacement','ge',t['away']);home=pred('displacement','le',t['away']/2)
    tasks=[('recover','LOW后恢复到HIGH',[low,high],False),
           ('motion_burst','低速后进入高速',[slow,fast],False),
           ('slow_down','高速后减速到低速',[fast,slow],False),
           ('goal_approach','远离目标后接近目标',[far,near],False),
           ('goal_revisit','接近、远离、再接近目标',[near,far,near],False),
           ('leave_start','离开起始位置',[away],False),
           ('return_start','离开起始区域后返回',[away,home],False),
           ('safe_recover','恢复到HIGH前不进入hazard',[low,high],True)]
    result=[]
    for ident,name,stages,safe in tasks:
        accepting=len(stages);reject=accepting+1 if safe else None
        n=accepting+1+int(safe)
        result.append(dict(id=ident,name=name,stages=stages,safety_until_completion=safe,
            n_states=n,accepting=accepting,reject=reject,bad_set=[i for i in range(n) if i!=accepting],
            mp_class='Guarantee (bounded)',H=300,
            specification='ordered events within [0,300]: '+json.dumps(stages)+(' AND clearance>=0.2 until completion' if safe else ''),
            automaton='q counts completed predicates; at most one stage per state; accept absorbing; safety rejection absorbing if enabled',
            completion='all ordered stages consumed by t=300',
            interpretation='Decoded-model event only, not real-world task achievement or infinite recurrence'))
    return result,thresholds


def monitor(sig,spec):
    state=0;q=[];acc=spec['accepting'];reject=spec['reject']
    for i in range(len(sig['clearance'])):
        if state!=acc and state!=reject:
            if spec['safety_until_completion'] and sig['clearance'][i]<.2:
                state=reject
            else:
                pred=spec['stages'][state];v=sig[pred['signal']][i]
                hit=v<=pred['threshold'] if pred['op']=='le' else v>=pred['threshold']
                if hit:state+=1
        q.append(state)
    q=np.asarray(q);done=q==acc
    return q,~done,done


def products(paths,spec):
    xs,qs,bs,ds,ps=[],[],[],[],[]
    for a in paths:
        sig=signals(a);q,b,d=monitor(sig,spec)
        extra=np.stack([sig[k] for k in ('clearance','goal','speed','displacement')],-1)
        initial=np.broadcast_to(a['decoded'][0,27:29],(len(q),2))
        # Initial reference is a fixed task parameter, not a path identifier.
        x=np.concatenate([base.physical(a),extra,initial,np.eye(spec['n_states'])[q]],-1)
        prior=np.zeros(len(q))
        for i,mode in enumerate(q):
            if d[i]:continue
            if mode==spec['reject']:
                prior[i]=1.;continue
            pred=spec['stages'][mode];level=pred['threshold'];v=sig[pred['signal']][i]
            gap=max(0.,v-level) if pred['op']=='le' else max(0.,level-v)
            prior[i]=spec['accepting']-mode+min(gap/max(abs(level),.1),10.)
        xs.append(x);qs.append(q);bs.append(b);ds.append(d);ps.append(prior)
    return tuple(np.asarray(v) for v in (xs,qs,bs,ds,ps))


class Value(nn.Module):
    def __init__(self,dim,family):
        super().__init__();self.family=family
        self.net=nn.Linear(dim,1) if family=='linear' else nn.Sequential(
            nn.Linear(dim,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,1))
    def forward(self,x,done,prior):
        raw=nn.functional.softplus(self.net(x).squeeze(-1))
        v=.01*prior*(1.5+raw) if self.family=='signal_progress' else .01+raw
        return torch.where(done.bool(),torch.zeros_like(v),v)


def predict(m,x,d,p):
    net=Value(x.shape[-1],m['family']).double();net.load_state_dict(m['weights']);net.eval()
    with torch.no_grad():return net(torch.tensor((x-m['mean'])/m['scale']),torch.tensor(d),torch.tensor(p)).numpy()


def outcome(d,q,spec):
    first=[int(np.flatnonzero(row)[0]) if row.any() else None for row in d]
    return dict(first_completion=first,initially_complete=int(d[:,0].sum()),completed=int(d[:,-1].sum()),
                safety_rejected=int((q[:,-1]==spec['reject']).sum()) if spec['reject'] is not None else 0)


def run(out,updates):
    torch.set_num_threads(2);fit,fit_records=base.load_train()
    source_plan=json.loads((exp.SOURCE/'plan.json').read_text())
    assert fit_records==source_plan['train_records'] and base.digest(base.__file__)==source_plan['code_sha256']
    assert base.digest(source_plan['checkpoint'])==source_plan['checkpoint_sha256']
    specs,thresholds=candidates(fit);out.mkdir(parents=True,exist_ok=False)
    plan=dict(created=base.stamp(),backend='SafeDreamer',checkpoint=source_plan['checkpoint'],
        checkpoint_sha256=source_plan['checkpoint_sha256'],policy=source_plan['policy'],H=300,eta=.01,tolerance=0.,
        source_plan_sha256=base.digest(exp.SOURCE/'plan.json'),fit_records=fit_records,
        threshold_source='Only pooled states of80train_fit paths; fixed quantiles, no internal quantile fitting',
        thresholds=thresholds,specs=specs,families=['linear','elu128','signal_progress'],scales=[1.,2.,4.],updates=updates,
        training_seed=158000000,internal_scope='20reused internal-development paths; no gradients; no confidence claim',
        selection='Whole-path P1/P2 then certificate preview then fewer P2/P1; train check every200 updates',
        gates='all300P1/P2 AND completion AND endpoint V<eta AND bad-exclusion AND observed closure AND frozen region AND finite/nonnegative',
        region='Same original fit-only RSSM/planner standardized radius; not expanded on internal data',
        zfree='V<eta in product state; accepting monitor absorbing, not physical-state invariance',
        caveats=['Goal is decoded current goal vector, not goal_met; target may change',
                 'Position displacement uses current decoded robot XY relative to decoded t0; model-only meaning',
                 'One-time ordered events are L2 reachability, not infinite recurrence'],
        code_sha256=base.digest(__file__),calibration_count=0,test_count=0)
    base.write(out/'plan.json',plan);frozen=[];started=time.monotonic()
    for si,spec in enumerate(specs):
        x,q,b,d,p=products(fit,spec);feasible,rank=base.graph_feasibility(x,q,b)
        base.write(out/(spec['id']+'_feasibility.json'),feasible)
        if not feasible['feasible']:continue
        mean=x.mean((0,1));scale=np.maximum(x.std((0,1)),.05);xx=(x-mean)/scale
        pi,ti=np.where(b[:,:-1]);a=torch.tensor(xx[pi,ti],dtype=torch.float32);aa=torch.tensor(xx[pi,ti+1],dtype=torch.float32)
        da=torch.tensor(d[pi,ti]);db=torch.tensor(d[pi,ti+1])
        pa=torch.tensor(p[pi,ti],dtype=torch.float32);pb=torch.tensor(p[pi,ti+1],dtype=torch.float32)
        y=torch.tensor(rank[pi,ti]+.01,dtype=torch.float32);completed=torch.tensor(d[pi,-1])
        for fi,family in enumerate(plan['families']):
            torch.manual_seed(plan['training_seed']+100*si+fi);net=Value(x.shape[-1],family)
            opt=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=1e-6);weights=torch.ones(len(a));best=None
            history=[]
            for step in range(updates+1):
                if step:
                    ix=torch.multinomial(weights,512,replacement=True);va=net(a[ix],da[ix],pa[ix]);vb=net(aa[ix],db[ix],pb[ix])
                    pos=torch.relu((vb-va)/.01+1.5);err=va-y[ix]
                    aux=torch.where(completed[ix],err.square(),torch.relu(-err).square()).mean()
                    loss=pos.square().mean()+pos.topk(32).values.mean()+.03*aux+.00001*va.square().mean()
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),100.);opt.step()
                if step%200 and step!=updates:continue
                m=dict(family=family,spec=spec,mean=mean,scale=scale,step=step,weights=copy.deepcopy(net.state_dict()))
                s=base.statistics(predict(m,x,d,p),b,d);key=exp.score(s)
                history.append(dict(step=step,p1=s['p1_violations'],p2=s['p2_violations'],paths=s['p1p2_paths']))
                if best is None or key>best[0]:best=key,m,s
                with torch.no_grad():
                    hard=(torch.relu((net(aa,db,pb)-net(a,da,pa))/.01+1.5)+.1).clamp(max=30.)
                    weights=.25/len(a)+.75*hard/hard.sum()
            ident=spec['id']+'_'+family;torch.save(best[1],out/(ident+'.pt'))
            rec=dict(id=ident,spec=spec,weights=ident+'.pt',sha256=base.digest(out/(ident+'.pt')),family=family,
                     train=best[2],train_events=outcome(d,q,spec),history=history,frozen=base.stamp())
            frozen.append(rec);base.write(out/(ident+'_training.json'),rec)
            print('FIT',ident,best[2]['p1p2_paths'],'/80 P1/P2 bad',best[2]['p1_violations'],best[2]['p2_violations'],
                  'task',int(d[:,-1].sum()),'elapsed',round(time.monotonic()-started,1),flush=True)
    base.write(out/'frozen.json',dict(at=base.stamp(),records=frozen))
    internal,hashes=exp.load_internal();assert not set(hashes)&{r['sha256'] for r in fit_records}
    region=np.load(exp.SOURCE/'region.npz')
    masks=[np.linalg.norm((np.stack([base.physical(a) for a in paths])-region['mean'])/region['scale'],axis=-1)
           <=float(region['radius'])+1e-10 for paths in (fit,internal)]
    rows=[]
    for rec in frozen:
        assert base.digest(out/rec['weights'])==rec['sha256'];m=torch.load(out/rec['weights'],map_location='cpu')
        datasets=[products(paths,rec['spec']) for paths in (fit,internal)]
        vs=[predict(m,x,d,p) for x,q,b,d,p in datasets]
        x,q,b,d,p=datasets[1];v=vs[1]
        cvs=[('constant',np.ones_like(v)),('mode_constant',np.where(d,0.,1.)),('signal_prior_only',.02*p)]
        for seed in range(3):
            rng=np.random.RandomState(158100000+seed);sv=v.copy()
            for mode in np.unique(q):sv[q==mode]=rng.permutation(v[q==mode])
            cvs.append(('shuffle%d'%seed,sv));torch.manual_seed(158101000+seed)
            rm=copy.deepcopy(m);rm['weights']=Value(x.shape[-1],m['family']).state_dict()
            cvs.append(('random%d'%seed,predict(rm,x,d,p)))
        for gain in plan['scales']:
            stats=[]
            for (x,q,b,d,p),v,mask in zip(datasets,vs,masks):
                val=gain*v;s=base.statistics(val,b,d,region=mask)
                pp,cc=independent(val,b,d,mask)
                np.testing.assert_array_equal(pp,s['p1p2_per_path']);np.testing.assert_array_equal(cc,s['certificate_per_path'])
                stats.append(s)
            controls={name:base.statistics(gain*cv,b,d) for name,cv in cvs}
            ok=exp.qualification(stats,{k:v for k,v in controls.items() if k!='signal_prior_only'})
            status='KEEP_DEV' if ok else 'BORDERLINE'
            if not stats[1]['finite_completion_paths']:status='REJECT_NO_INTERNAL_COMPLETION'
            row=dict(id=rec['id'],spec=rec['spec'],gain=gain,train=stats[0],internal=stats[1],controls=controls,
                     train_events=rec['train_events'],internal_events=outcome(d,q,rec['spec']),status=status)
            rows.append(row)
        np.savez_compressed(out/(rec['id']+'_values.npz'),train=vs[0],internal=vs[1])
        print('DEV',rec['id'],'gain1/2/4 paths',[r['internal']['p1p2_paths'] for r in rows[-3:]],flush=True)
    bests=[]
    for spec in specs:
        choices=[r for r in rows if r['spec']['id']==spec['id']]
        if not choices:continue
        best=max(choices,key=lambda r:(r['status']=='KEEP_DEV',r['internal']['p1p2_paths'],r['internal']['certificate_preview'],
                                      -r['internal']['p2_violations'],-r['internal']['p1_violations'],*exp.score(r['train']),-r['gain']))
        bests.append(best)
    base.write(out/'report.json',dict(at=base.stamp(),rows=rows,best_per_spec=bests,internal_hashes=hashes,
        qualified=sum(r['status']=='KEEP_DEV' for r in rows),independent_AND_replay=True,calibration_count=0,test_count=0))
    fields=['spec','function','gain','fit_completion','internal_completion','fit_paths','internal_paths',
            'internal_P1_bad','internal_P2_bad','internal_P2_checked','certificate_preview','status']
    with (out/'all_candidates.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(fields)
        for r in rows:
            a,b=r['train'],r['internal'];w.writerow([r['spec']['id'],r['id'],r['gain'],a['finite_completion_paths'],
                b['finite_completion_paths'],a['p1p2_paths'],b['p1p2_paths'],b['p1_violations'],b['p2_violations'],b['bad_transitions'],b['certificate_preview'],r['status']])
    print('FINISHED',len(frozen),'networks;',len(rows),'scaled candidates; KEEP_DEV',sum(r['status']=='KEEP_DEV' for r in rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--updates',type=int,default=3000)
    a=p.parse_args();run(a.output.resolve(),a.updates)
