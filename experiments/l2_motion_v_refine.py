"""Fixed motion-burst detector: fit-only V-family/long-wait refinement."""
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
from experiments import l2_distinct_spec_screen as screen
from experiments.l2_distinct_spec_audit import manual_completion
base=screen.base;exp=screen.exp
SOURCE=ROOT/'artifacts/safedreamer_l2_distinct_specs_v1'
CONFIGS=[
    dict(id='full128',features='full',family='elu',width=128,margin=.02,aux=.10),
    dict(id='full256',features='full',family='elu',width=256,margin=.03,aux=.10),
    dict(id='deter128',features='deter',family='elu',width=128,margin=.03,aux=.10),
    dict(id='kinematic64',features='kinematic',family='elu',width=64,margin=.03,aux=.10),
    dict(id='full_max4',features='full',family='max4',width=4,margin=.03,aux=.05),
    dict(id='full_residual',features='full',family='residual',width=128,margin=.03,aux=.20),
    dict(id='full128_rank',features='full',family='elu',width=128,margin=.05,aux=1.),
    dict(id='deter128_rank',features='deter',family='elu',width=128,margin=.05,aux=.20),
]


def products(paths,spec,kind):
    xs,qs,bs,ds=[],[],[],[]
    for path in paths:
        sig=screen.signals(path);q,b,d=screen.monitor(sig,spec)
        dec=path['decoded'].astype(np.float64);phys=base.physical(path)
        if kind=='full':x=np.concatenate([phys,dec],-1)
        elif kind=='deter':x=np.concatenate([phys[:,:256],phys[:,512:],dec],-1)
        elif kind=='kinematic':x=dec[:,:7]
        else:raise ValueError(kind)
        x=np.concatenate([x,sig['speed'][:,None],np.eye(3)[q]],-1)
        xs.append(x);qs.append(q);bs.append(b);ds.append(d)
    return tuple(np.asarray(a) for a in (xs,qs,bs,ds))


class Value(nn.Module):
    def __init__(self,dim,cfg):
        super().__init__();self.cfg=cfg;w=cfg['width']
        if cfg['family']=='max4':self.net=nn.Linear(dim,4)
        elif cfg['family']=='residual':
            self.input=nn.Linear(dim,w)
            self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(w,w),nn.SiLU(),nn.Linear(w,w)) for _ in range(2)])
            self.output=nn.Linear(w,1)
        else:self.net=nn.Sequential(nn.Linear(dim,w),nn.ELU(),nn.Linear(w,w),nn.ELU(),nn.Linear(w,1))
    def forward(self,x,done):
        if self.cfg['family']=='residual':
            h=nn.functional.silu(self.input(x))
            for block in self.blocks:h=h+.5*block(h)
            raw=self.output(nn.functional.silu(h)).squeeze(-1)
        elif self.cfg['family']=='max4':raw=self.net(x).max(-1).values
        else:raw=self.net(x).squeeze(-1)
        v=.01+nn.functional.softplus(raw)
        return torch.where(done.bool(),torch.zeros_like(v),v)


def predict(m,x,d):
    net=Value(x.shape[-1],m['config']).double();net.load_state_dict(m['weights']);net.eval()
    with torch.no_grad():return net(torch.tensor((x-m['mean'])/m['scale']),torch.tensor(d)).numpy()


def run(out,updates):
    torch.set_num_threads(2)
    p=json.loads((SOURCE/'plan.json').read_text());spec=next(s for s in p['specs'] if s['id']=='motion_burst')
    assert base.digest(screen.__file__)==p['code_sha256']
    fit,records=base.load_train();assert records==p['fit_records']
    out.mkdir(exist_ok=False,parents=True)
    plan=dict(at=base.stamp(),source=str(SOURCE),source_plan_sha256=base.digest(SOURCE/'plan.json'),
        checkpoint=p['checkpoint'],checkpoint_sha256=p['checkpoint_sha256'],policy=p['policy'],spec=spec,
        H=300,eta=.01,tolerance=0.,configs=CONFIGS,updates=updates,pretrain=400,scales=[1.,2.,4.],
        fit_records=records,training_seed=159100000,
        training='fit-only return-time rank supervision; pending-state normalization; long-latency path weighting plus hard edges',
        long_wait_weight='sqrt(first completion step / median fit completion step), clipped [1,4]; all pending edges included',
        train_selection='max whole fit P1/P2, then fewer P2/P1 failures',
        internal_selection='reused20 internal only; no gradients; max full P1/P2, then certificate, then fewer P2/P1',
        fixed_gates='All300P1/P2 AND completion AND endpointV<eta AND exclusion AND sampled closure AND original fit-only region AND finite/nonnegative',
        zfree='V<eta; accepting0, pending>=eta',
        inference_excludes='time index, time remaining, future return label, path identity; no initial position feature for this motion-only task',
        controls='constant, mode constant, within-mode shuffle3, random networks3',
        code_sha256=base.digest(__file__),calibration_count=0,test_count=0)
    base.write(out/'plan.json',plan);frozen=[];started=time.monotonic()
    for ci,cfg in enumerate(CONFIGS):
        x,q,b,d=products(fit,spec,cfg['features']);graph,rank=base.graph_feasibility(x,q,b)
        base.write(out/(cfg['id']+'_feasibility.json'),graph)
        if not graph['feasible']:continue
        mean=x[b].mean(0);scale=np.maximum(x[b].std(0),.05)
        pi,ti=np.where(b[:,:-1]);xx=(x-mean)/scale
        a=torch.tensor(xx[pi,ti],dtype=torch.float32);aa=torch.tensor(xx[pi,ti+1],dtype=torch.float32)
        da=torch.tensor(d[pi,ti]);db=torch.tensor(d[pi,ti+1])
        observed=torch.tensor(d[pi,-1])
        y=torch.tensor(2*cfg['margin']*(rank[pi,ti]/.01+1),dtype=torch.float32)
        completions=np.array([int(np.flatnonzero(row)[0]) if row.any() else 301 for row in d])
        pathweight=np.clip(np.sqrt(completions/np.median(completions)),1,4)
        prior=torch.tensor(pathweight[pi],dtype=torch.float32);weights=prior.clone()
        torch.manual_seed(plan['training_seed']+ci);net=Value(x.shape[-1],cfg)
        opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=1e-6);best=None;history=[]
        for step in range(updates+plan['pretrain']+1):
            if step:
                ix=torch.multinomial(weights,512,replacement=True);va=net(a[ix],da[ix]);vb=net(aa[ix],db[ix])
                pos=torch.relu((vb-va+cfg['margin'])/.01);err=va-y[ix]
                aux=torch.where(observed[ix],err.square(),torch.relu(-err).square()).mean()
                loss=aux if step<=plan['pretrain'] else pos.square().mean()+pos.topk(32).values.mean()+cfg['aux']*aux+.00001*va.square().mean()
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),100.);opt.step()
            if step%200 and step!=updates+plan['pretrain']:continue
            m=dict(config=cfg,mean=mean,scale=scale,spec=spec,step=step,weights=copy.deepcopy(net.state_dict()))
            s=base.statistics(predict(m,x,d),b,d)
            history.append(dict(step=step,p1=s['p1_violations'],p2=s['p2_violations'],paths=s['p1p2_paths']))
            if best is None or exp.score(s)>best[0]:best=exp.score(s),m,s
            with torch.no_grad():
                hard=(torch.relu((net(aa,db)-net(a,da)+cfg['margin'])/.01)+.1).clamp(max=30.)
                weights=.25*prior/prior.sum()+.75*prior*hard/(prior*hard).sum()
        file=out/(cfg['id']+'.pt');torch.save(best[1],file)
        rec=dict(id=cfg['id'],config=cfg,weights=file.name,sha256=base.digest(file),at=base.stamp(),
                 train=best[2],step=best[1]['step'],history=history,fit_pending_edges=len(pi),fit_completion_steps=completions.tolist())
        frozen.append(rec);base.write(out/(cfg['id']+'_training.json'),rec)
        print('FIT',cfg['id'],'paths',best[2]['p1p2_paths'],'P1/P2 bad',best[2]['p1_violations'],best[2]['p2_violations'],
              'elapsed',round(time.monotonic()-started,1),flush=True)
    base.write(out/'frozen.json',dict(at=base.stamp(),records=frozen))
    internal,hashes=exp.load_internal();region=np.load(exp.SOURCE/'region.npz');rows=[]
    masks=[np.linalg.norm((np.stack([base.physical(a) for a in paths])-region['mean'])/region['scale'],axis=-1)
           <=float(region['radius'])+1e-10 for paths in (fit,internal)]
    for rec in frozen:
        m=torch.load(out/rec['weights'],map_location='cpu');assert base.digest(out/rec['weights'])==rec['sha256']
        datasets=[products(paths,spec,m['config']['features']) for paths in (fit,internal)]
        vs=[predict(m,x,d) for x,q,b,d in datasets];x,q,b,d=datasets[1];v=vs[1]
        cvs=[('constant',np.ones_like(v)),('mode_constant',np.where(d,0.,1.))]
        for seed in range(3):
            rng=np.random.RandomState(159200000+seed);sv=v.copy()
            for mode in np.unique(q):sv[q==mode]=rng.permutation(v[q==mode])
            cvs.append(('shuffle%d'%seed,sv));torch.manual_seed(159201000+seed)
            rm=copy.deepcopy(m);rm['weights']=Value(x.shape[-1],m['config']).state_dict();cvs.append(('random%d'%seed,predict(rm,x,d)))
        for gain in plan['scales']:
            stats=[]
            for (x,q,b,d),v,mask in zip(datasets,vs,masks):
                val=gain*v;s=base.statistics(val,b,d,region=mask)
                pp,cc=screen.independent(val,b,d,mask)
                np.testing.assert_array_equal(pp,s['p1p2_per_path']);np.testing.assert_array_equal(cc,s['certificate_per_path'])
                stats.append(s)
            controls={name:base.statistics(gain*cv,b,d) for name,cv in cvs}
            ok=exp.qualification(stats,controls)
            rows.append(dict(id=rec['id'],gain=gain,train=stats[0],internal=stats[1],controls=controls,status='KEEP_DEV' if ok else 'BORDERLINE'))
        np.savez_compressed(out/(rec['id']+'_values.npz'),train=vs[0],internal=vs[1])
        print('DEV',rec['id'],'gain1/2/4 paths',[r['internal']['p1p2_paths'] for r in rows[-3:]],flush=True)
    # Preserve original baseline as an eligible fallback; never overwrite it.
    oldrow=next(r for r in json.loads((SOURCE/'report.json').read_text())['best_per_spec'] if r['spec']['id']=='motion_burst')
    baseline=dict(id='original_motion_burst_elu128',gain=oldrow['gain'],train=oldrow['train'],internal=oldrow['internal'],status='BORDERLINE')
    choices=rows+[baseline]
    key=lambda r:(r['status']=='KEEP_DEV',r['internal']['p1p2_paths'],r['internal']['certificate_preview'],
                  -r['internal']['p2_violations'],-r['internal']['p1_violations'],*exp.score(r['train']),-r['gain'])
    best=max(choices,key=key)
    if best['id']!=baseline['id']:
        m=torch.load(out/(best['id']+'.pt'),map_location='cpu')
        torch.save(dict(base_model=m,output_scale=best['gain'],predictor='output_scale*l2_motion_v_refine.predict(base_model,x,done)'),out/'selected.pt')
    base.write(out/'report.json',dict(at=base.stamp(),rows=rows,baseline=baseline,selected=best,internal_hashes=hashes,
        qualified=sum(r['status']=='KEEP_DEV' for r in rows),independent_gate_AND=True,calibration_count=0,test_count=0))
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['function','gain','fit_paths','internal_paths','internal_P1_bad','internal_P2_bad','certificate','status'])
        for r in choices:
            b=r['internal'];w.writerow([r['id'],r['gain'],r['train']['p1p2_paths'],b['p1p2_paths'],b['p1_violations'],b['p2_violations'],b['certificate_preview'],r['status']])
    print('SELECTED',best['id'],best['gain'],best['status'],'paths',best['internal']['p1p2_paths'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=4000)
    a=p.parse_args();run(a.output.resolve(),a.updates)
