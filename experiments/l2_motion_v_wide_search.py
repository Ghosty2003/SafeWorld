"""48 fixed motion-certificate candidates; fit gradients, internal selection only."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import l2_motion_v_refine as prev
from experiments import l2_motion_v_ensemble as ens
base=prev.base;exp=prev.exp;screen=prev.screen


def configs():
    templates=[('elu_original','mlp','elu',128,'original','all'),
        ('elu256','mlp','elu',256,'full','pending'),('elu_deter','mlp','elu',128,'deter','pending'),
        ('tanh','mlp','tanh',128,'full','all'),('silu','mlp','silu',128,'full','pending'),
        ('softplus','mlp','softplus',128,'deter','all'),('residual','residual','silu',128,'full','pending'),
        ('mode_heads','heads','elu',128,'full','pending'),('max8','max','elu',8,'deter','pending'),
        ('quadratic','quadratic','elu',64,'deter','pending'),('fourier','fourier','elu',128,'deter','all'),
        ('latent','mlp','elu',128,'latent','pending')]
    return [dict(id=name+'_s%d'%seed,family=family,activation=act,width=width,features=features,
        normalization=norm,seed=160100000+seed*100+i,margin=[.015,.025,.04,.06][seed],
        auxiliary=[.03,.1,.3,.1][seed],pretrain=[0,400,400,0][seed],long_weight=[0.,.5,1.,1.5][seed])
        for i,(name,family,act,width,features,norm) in enumerate(templates) for seed in range(4)]


def products(paths,spec,kind):
    if kind=='original':return screen.products(paths,spec)[:4]
    if kind!='latent':return prev.products(paths,spec,kind)
    x,q,b,d=prev.products(paths,spec,'full')
    return np.concatenate([x[...,:512],x[...,-4:]],-1),q,b,d


class Value(nn.Module):
    def __init__(self,dim,cfg):
        super().__init__();self.cfg=cfg;w=cfg['width'];family=cfg['family']
        act={'elu':nn.ELU,'tanh':nn.Tanh,'silu':nn.SiLU,'softplus':nn.Softplus}[cfg['activation']]
        if family=='max':self.net=nn.Linear(dim,w)
        elif family=='residual':
            self.input=nn.Linear(dim,w);self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(w,w),nn.SiLU(),nn.Linear(w,w)) for _ in range(2)])
            self.output=nn.Linear(w,1)
        elif family in ('quadratic','fourier'):
            self.input=nn.Linear(dim,w);self.output=nn.Linear(w,1)
        else:self.net=nn.Sequential(nn.Linear(dim,w),act(),nn.Linear(w,w),act(),nn.Linear(w,3 if family=='heads' else 1))
    def forward(self,x,q):
        family=self.cfg['family']
        if family=='residual':
            h=nn.functional.silu(self.input(x))
            for block in self.blocks:h=h+.5*block(h)
            raw=self.output(nn.functional.silu(h)).squeeze(-1)
        elif family=='max':raw=self.net(x).max(-1).values
        elif family=='quadratic':raw=self.output(self.input(x).square()).squeeze(-1)
        elif family=='fourier':raw=self.output(torch.sin(self.input(x))).squeeze(-1)
        elif family=='heads':raw=self.net(x).gather(-1,q.long().unsqueeze(-1)).squeeze(-1)
        else:raw=self.net(x).squeeze(-1)
        v=.01+nn.functional.softplus(raw)
        return torch.where(q==2,torch.zeros_like(v),v)


def predict(m,x,q):
    # Each network is pointwise, no batchnorm/dropout. Accepted q is exactly
    # zero by its definition, so skip its neural evaluation (not its gates).
    pending=q!=2;v=np.zeros(q.shape,dtype=np.float64)
    if pending.any():
        net=Value(x.shape[-1],m['config']).double();net.load_state_dict(m['weights']);net.eval()
        with torch.no_grad():v[pending]=net(torch.tensor((x[pending]-m['mean'])/m['scale']),torch.tensor(q[pending])).numpy()
    return v


def run(out,updates):
    torch.set_num_threads(2)
    source=ROOT/'artifacts/safedreamer_l2_distinct_specs_v1'
    p=json.loads((source/'plan.json').read_text());spec=next(s for s in p['specs'] if s['id']=='motion_burst')
    assert base.digest(screen.__file__)==p['code_sha256'];fit,records=base.load_train();assert records==p['fit_records']
    out.mkdir(exist_ok=False,parents=True)
    plan=dict(at=base.stamp(),spec=spec,checkpoint=p['checkpoint'],checkpoint_sha256=p['checkpoint_sha256'],policy=p['policy'],
        H=300,eta=.01,tolerance=0.,configs=configs(),updates=updates,gains=[1.,2.,4.,8.],fit_records=records,
        internal_scope='20reused development paths; no gradients on internal; no confidence guarantee',
        code_sha256=base.digest(__file__),source_plan_sha256=base.digest(source/'plan.json'),
        training='Only actual fit pending edges; nonnegative pointwise V; long-wait path replay, hard edges, optional fit rank supervision',
        train_selection='Most full fit paths, fewer P2/P1 failures; every400 updates',
        internal_selection='Train strict100 first, then most full internal paths, fewer P2/P1 failures; fixed gain grid',
        ensembles='Per new model: .25/.5/.75 old19baseline mixture at gain4, pointwise min/max; no continuous optimization on internal',
        frozen_gates='All300P1/P2 AND endpoint/completion AND Zfree/bad exclusion/observed closure AND originalfit region AND finite/nonnegative',
        inference_prohibited=['time','trajectory ID','future labels','running-min overwrite'],calibration_count=0,test_count=0)
    base.write(out/'plan.json',plan);cache={};frozen=[];started=time.monotonic()
    for ci,cfg in enumerate(plan['configs']):
        kind=cfg['features']
        if kind not in cache:
            data=products(fit,spec,kind);graph,rank=base.graph_feasibility(data[0],data[1],data[2])
            assert graph['feasible'];cache[kind]=(data,rank);base.write(out/(kind+'_feasibility.json'),graph)
        (x,q,b,d),rank=cache[kind]
        norm=x[b] if cfg['normalization']=='pending' else x.reshape(-1,x.shape[-1])
        mean=norm.mean(0);scale=np.maximum(norm.std(0),.05);pi,ti=np.where(b[:,:-1]);xx=(x-mean)/scale
        a=torch.tensor(xx[pi,ti],dtype=torch.float32);aa=torch.tensor(xx[pi,ti+1],dtype=torch.float32)
        qa=torch.tensor(q[pi,ti]);qb=torch.tensor(q[pi,ti+1])
        target=torch.tensor(2*cfg['margin']*(rank[pi,ti]/.01+1),dtype=torch.float32);observed=torch.tensor(d[pi,-1])
        times=np.array([int(np.flatnonzero(row)[0]) if row.any() else 301 for row in d])
        prior=torch.tensor(np.clip((times/np.median(times))**cfg['long_weight'],1,8)[pi],dtype=torch.float32)
        weights=prior.clone();torch.manual_seed(cfg['seed']);net=Value(x.shape[-1],cfg)
        opt=torch.optim.AdamW(net.parameters(),lr=.0004,weight_decay=1e-6);best=None;history=[]
        for step in range(updates+cfg['pretrain']+1):
            if step:
                ix=torch.multinomial(weights,512,replacement=True);va=net(a[ix],qa[ix]);vb=net(aa[ix],qb[ix])
                pos=torch.relu((vb-va+cfg['margin'])/.01);err=va-target[ix]
                aux=torch.where(observed[ix],err.square(),torch.relu(-err).square()).mean()
                loss=aux if step<=cfg['pretrain'] else pos.square().mean()+pos.topk(32).values.mean()+cfg['auxiliary']*aux+.00001*va.square().mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),100.);opt.step()
            if step%400 and step!=updates+cfg['pretrain']:continue
            m=dict(config=cfg,mean=mean,scale=scale,spec=spec,step=step,weights=copy.deepcopy(net.state_dict()))
            stat=base.statistics(predict(m,x,q),b,d)
            history.append(dict(step=step,paths=stat['p1p2_paths'],p1=stat['p1_violations'],p2=stat['p2_violations']))
            if best is None or exp.score(stat)>best[0]:best=exp.score(stat),m,stat
            with torch.no_grad():
                hard=(torch.relu((net(aa,qb)-net(a,qa)+cfg['margin'])/.01)+.1).clamp(max=30.)
                weights=.25*prior/prior.sum()+.75*prior*hard/(prior*hard).sum()
        path=out/(cfg['id']+'.pt');torch.save(best[1],path)
        rec=dict(id=cfg['id'],file=path.name,config=cfg,sha256=base.digest(path),at=base.stamp(),train=best[2],history=history)
        frozen.append(rec);base.write(out/(cfg['id']+'_training.json'),rec)
        print('FIT',ci+1,'/48',cfg['id'],'paths',best[2]['p1p2_paths'],'P1/P2bad',best[2]['p1_violations'],best[2]['p2_violations'],
              'elapsed',round(time.monotonic()-started,1),flush=True)
    base.write(out/'frozen.json',dict(at=base.stamp(),records=frozen))
    evaluate(out,fit)


def evaluate(out,fit=None):
    torch.set_num_threads(2);plan=json.loads((out/'plan.json').read_text());spec=plan['spec']
    assert base.digest(__file__)==plan['code_sha256']
    if fit is None:fit,_=base.load_train()
    internal,hashes=exp.load_internal();region=np.load(exp.SOURCE/'region.npz')
    masks=[np.linalg.norm((np.stack([base.physical(a) for a in paths])-region['mean'])/region['scale'],axis=-1)<=float(region['radius'])+1e-10 for paths in (fit,internal)]
    baseline_dir=ROOT/'artifacts/safedreamer_l2_motion_v_ensemble_v1'
    baseline=json.loads((baseline_dir/'selected_manifest.json').read_text());baseline_values=[]
    for paths in (fit,internal):
        for primitive in baseline['primitives']:assert base.digest(primitive['path'])==primitive['sha256']
        baseline_values.append(ens.aggregate(np.stack([ens.primitive_values(p,paths,spec) for p in baseline['primitives']]),baseline['aggregation']))
    base.write(out/'baseline_manifest.json',baseline);rows=[];cached={}
    baseline_stats=[]
    for paths,v,mask in zip((fit,internal),baseline_values,masks):
        _,q,b,d=products(paths,spec,'full')
        baseline_stats.append(base.statistics(v,b,d,region=mask))
    rows.append(dict(config=dict(id='previous19_baseline',kind='baseline'),network=None,sha256=None,
        train=baseline_stats[0],internal=baseline_stats[1],numeric100=False))
    for rec in json.loads((out/'frozen.json').read_text())['records']:
        assert base.digest(out/rec['file'])==rec['sha256'];m=torch.load(out/rec['file'],map_location='cpu');kind=m['config']['features']
        if kind not in cached:cached[kind]=[products(paths,spec,kind) for paths in (fit,internal)]
        datasets=cached[kind];vs=[predict(m,x,q) for x,q,b,d in datasets]
        candidates=[dict(id=rec['id']+'_g%g'%g,kind='single',gain=g) for g in plan['gains']]
        candidates += [dict(id=rec['id']+'_mix%g'%w,kind='blend',weight=w) for w in [.25,.5,.75]]
        candidates += [dict(id=rec['id']+'_'+op,kind=op) for op in ['min','max']]
        for cfg in candidates:
            stats=[]
            for (x,q,b,d),v,old,mask in zip(datasets,vs,baseline_values,masks):
                value=combine(v,old,cfg);stat=base.statistics(value,b,d,region=mask)
                pp,cc=screen.independent(value,b,d,mask)
                np.testing.assert_array_equal(pp,stat['p1p2_per_path']);np.testing.assert_array_equal(cc,stat['certificate_per_path'])
                stats.append(stat)
            rows.append(dict(config=cfg,network=rec['file'],sha256=rec['sha256'],train=stats[0],internal=stats[1],
                numeric100=all(s['p1_violations']==0 and s['p2_violations']==0 and s['bad_transitions']>0 for s in stats)))
        np.savez_compressed(out/(rec['id']+'_values.npz'),train=vs[0],internal=vs[1])
        best=max(rows[-len(candidates):],key=key)
        print('DEV',rec['id'],'best',best['config']['id'],best['internal']['p1p2_paths'],'P1/P2bad',best['internal']['p1_violations'],best['internal']['p2_violations'],flush=True)
    base.write(out/'numeric_report.json',dict(at=base.stamp(),rows=rows,selected=max(rows,key=key),
        internal_hashes=hashes,numeric100=sum(r['numeric100'] for r in rows),calibration_count=0,test_count=0))
    print('NUMERIC SELECTED',max(rows,key=key)['config']['id'],'double100',sum(r['numeric100'] for r in rows),flush=True)


def combine(v,old,cfg):
    if cfg['kind']=='baseline':return old
    if cfg['kind']=='single':return cfg['gain']*v
    if cfg['kind']=='blend':return cfg['weight']*old+(1-cfg['weight'])*4*v
    if cfg['kind']=='min':return np.minimum(old,4*v)
    if cfg['kind']=='max':return np.maximum(old,4*v)
    raise ValueError(cfg)


def key(r):
    a,b=r['train'],r['internal']
    return (r['numeric100'],a['p1p2_paths']==80,b['p1p2_paths'],b['certificate_preview'],-b['p2_violations'],-b['p1_violations'],*exp.score(a))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=4000)
    a=p.parse_args();run(a.output.resolve(),a.updates)
