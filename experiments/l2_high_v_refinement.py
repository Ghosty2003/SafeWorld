"""Fixed high-clearance L2 refinement. Only fit and reused internal dev data.

No world model/planner edits, no changed spec/eta, no calibration or Test1.
Accepted-mode transitions are identically 0->0 by construction; training
focuses on ALL source-pending edges, but evaluation checks ALL real edges.
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
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import safedreamer_l2_spec_search as base

SOURCE=ROOT/'artifacts/safedreamer_l2_spec_search_v2'
CONFIGS=[
    dict(id='warm_elu_hard',family='elu',width=256,gain=1.,features='full',warm=True,pretrain=0),
    dict(id='elu_gain10_hard',family='elu',width=256,gain=10.,features='full',warm=False,pretrain=0),
    dict(id='tanh_gain10_hard',family='tanh',width=256,gain=10.,features='full',warm=False,pretrain=0),
    dict(id='deter_elu_hard',family='elu',width=256,gain=10.,features='deter',warm=False,pretrain=0),
    dict(id='elu_rank_supervision',family='elu',width=256,gain=10.,features='full',warm=False,pretrain=1000),
    dict(id='residual_hard',family='residual',width=128,gain=10.,features='full',warm=False,pretrain=0),
]


class ProgressValue(nn.Module):
    def __init__(self,dim,cfg):
        super().__init__();self.cfg=cfg;w=cfg['width']
        if cfg['family']=='residual':
            self.input=nn.Linear(dim,w)
            self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(w,w),nn.SiLU(),nn.Linear(w,w)) for _ in range(2)])
            self.output=nn.Linear(w,1)
        else:
            act=nn.ELU if cfg['family']=='elu' else nn.Tanh
            self.net=nn.Sequential(nn.Linear(dim,w),act(),nn.Linear(w,w),act(),nn.Linear(w,1))
    def forward(self,x,done):
        if self.cfg['family']=='residual':
            h=nn.functional.silu(self.input(x))
            for block in self.blocks:h=h+.5*block(h)
            raw=self.output(nn.functional.silu(h)).squeeze(-1)
        else:raw=self.net(x).squeeze(-1)
        pending=self.cfg['gain']*nn.functional.softplus(raw)+base.ETA
        return torch.where(done.bool(),torch.zeros_like(pending),pending)


def feature_indices(dim,cfg):
    # Full sufficient state remains available; this candidate deliberately
    # ignores its stochastic block, not a claim that reduced inputs are Markov.
    return np.r_[0:256,512:dim] if cfg['features']=='deter' else np.arange(dim)


def predict(model,x,done):
    idx=model['indices'];net=ProgressValue(len(idx),model['config']).double()
    net.load_state_dict(model['weights']);net.eval()
    with torch.no_grad():
        return net(torch.tensor((x[...,idx]-model['mean'])/model['scale'],dtype=torch.float64),torch.tensor(done)).numpy()


def linear_diagnosis(x,bad,done,mean,scale):
    """Restricted affine family on source AND successor pending states.

    Maximum-margin inequalities; an intercept can ensure nonnegativity on
    finite nodes afterwards. This diagnoses only affine V, not arbitrary V.
    """
    mask=bad[:,:-1]&~done[:,1:]
    dx=((x[:,1:]-x[:,:-1])/scale)[mask]
    result=linprog(np.zeros(x.shape[-1]),A_ub=dx,b_ub=np.full(len(dx),-base.ETA),
        bounds=[(None,None)]*x.shape[-1],method='highs',options={'time_limit':20.})
    return dict(scope='Affine V finite sampled pending-to-pending edges only',n_edges=len(dx),
        status=int(result.status),message=str(result.message),feasible=True if result.status==0 else False if result.status==2 else None,
        max_residual=None if result.x is None else float(np.max(dx@result.x+base.ETA)),
        warning='Affine infeasibility does not imply nonlinear V infeasibility; near-coincident states may require steep functions')


def score_key(s):
    return (s['p1p2_paths'],-s['p2_violations'],-s['p1_violations'])


def train(out,updates):
    torch.set_num_threads(2)
    source_plan=json.loads((SOURCE/'plan.json').read_text());spec=source_plan['candidates'][0]
    assert spec['id']=='high' and source_plan['eta']==.01 and source_plan['H']==300
    assert base.digest(Path(base.__file__))==source_plan['code_sha256']
    paths,records=base.load_train();assert records==source_plan['train_records']
    x,q,bad,done=base.product_data(paths,spec)
    mean=x.mean((0,1));scale=np.maximum(x.std((0,1)),.05)
    feasible,witness=base.graph_feasibility(x,q,bad);assert feasible['feasible']
    out.mkdir(parents=True,exist_ok=False)
    plan=dict(created=base.stamp(),scope='Fixed-spec V development; no calibration/test',
        source=str(SOURCE),source_plan_sha256=base.digest(SOURCE/'plan.json'),spec=spec,
        checkpoint=source_plan['checkpoint'],checkpoint_sha256=source_plan['checkpoint_sha256'],policy=source_plan['policy'],
        H=300,eta=.01,tolerance=0.,fit_n=80,dev_n=20,
        internal_validation='Reused 20 internal-development paths; never used for gradients; NOT fresh final validation',
        inputs='Full RSSM+planner+current decoded features+causal monitor; optional omission of stochastic block',
        inference_prohibitions=['time index','remaining horizon','trajectory ID','future labels'],
        configs=CONFIGS,updates=updates,training_seed=145100000,
        training='All pending edges; eta-normalized squared hinge with training margin1.5eta; hard-edge replay updated each200 steps; train-only rank auxiliary',
        hard_sampling='25% uniform +75% proportional min(positive residual/eta+.1,20) across all pending edges',
        selection='maximize dev whole-path P1/P2, then fewer P2/P1 violations; train checkpoints chosen by same train-only rule',
        qualification='Both splits strict zero P1/P2 violations, non-vacuous bad transitions, constant/shuffled/random controls do not all-pass',
        region='unchanged source region',z_free='V<eta; accepting q absorbing with V=0, pending V>=eta',
        code_sha256=base.digest(__file__),calibration_count=0,test_count=0)
    base.write(out/'plan.json',plan)
    lp=linear_diagnosis(x,bad,done,mean,scale);base.write(out/'affine_diagnosis.json',lp)
    print('AFFINE',lp,flush=True)
    mask=bad[:,:-1];pi,ti=np.where(mask)
    target=witness[pi,ti]+2*base.ETA
    complete=done[pi,-1]
    source_diagnostic=dict(pending_edges=int(mask.sum()),all_edges=int(mask.size),train_only_graph=feasible,
        incomplete_paths=int((~done[:,-1]).sum()),initially_complete=int(done[:,0].sum()))
    base.write(out/'fit_diagnostics.json',source_diagnostic)
    frozen=[];started=time.monotonic()
    for ci,cfg in enumerate(CONFIGS):
        torch.manual_seed(plan['training_seed']+ci)
        idx=feature_indices(x.shape[-1],cfg)
        cm,cs=mean[idx],scale[idx]
        normalized=(x[...,idx]-cm)/cs
        a=torch.tensor(normalized[pi,ti],dtype=torch.float32);b=torch.tensor(normalized[pi,ti+1],dtype=torch.float32)
        ad=torch.zeros(len(pi),dtype=torch.bool);bd=torch.tensor(done[pi,ti+1])
        y=torch.tensor(target,dtype=torch.float32);observed=torch.tensor(complete)
        net=ProgressValue(len(idx),cfg)
        if cfg['warm']:
            old=torch.load(SOURCE/'high_w256.pt',map_location='cpu',weights_only=False)
            np.testing.assert_array_equal(old['mean'],mean);np.testing.assert_array_equal(old['scale'],scale)
            net.load_state_dict(old['weights'])
        optimizer=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=1e-6)
        weights=torch.ones(len(a));history=[];best=None
        for step in range(updates+cfg['pretrain']+1):
            if step>0:
                ix=torch.multinomial(weights,512,replacement=True)
                va=net(a[ix],ad[ix]);vb=net(b[ix],bd[ix])
                residual=(vb-va)/base.ETA+1.5
                positive=torch.relu(residual)
                rank_loss=positive.square().mean()+positive.topk(32).values.mean()
                target_error=va-y[ix]
                aux=torch.where(observed[ix],target_error.square(),torch.relu(-target_error).square()).mean()
                loss=aux if step<=cfg['pretrain'] else rank_loss+.05*aux+.00001*va.square().mean()
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),100.);optimizer.step()
            if step%200 and step!=updates+cfg['pretrain']:continue
            model=dict(config=cfg,indices=idx,mean=cm,scale=cs,weights={k:v.detach().clone() for k,v in net.state_dict().items()},step=step,spec=spec)
            values=predict(model,x,done);s=base.statistics(values,bad,done)
            history.append(dict(step=step,**s))
            if best is None or score_key(s)>best[0]:best=(score_key(s),copy.deepcopy(model),s)
            with torch.no_grad():
                residual=torch.relu((net(b,bd)-net(a,ad))/base.ETA+1.5)
                priority=(residual+.1).clamp(max=20.)
                weights=.25/len(a)+.75*priority/priority.sum()
            if step%1000==0:
                print('FIT',cfg['id'],step,'paths',s['p1p2_paths'],'/80','P1/P2',s['p1_violations'],s['p2_violations'],flush=True)
        file=out/(cfg['id']+'.pt');torch.save(best[1],file)
        rec=dict(config=cfg,file=file.name,sha256=base.digest(file),step=best[1]['step'],frozen=base.stamp(),train=best[2],history=history)
        frozen.append(rec);base.write(out/(cfg['id']+'_training.json'),rec)
        print('FROZEN',cfg['id'],best[1]['step'],'train paths',best[2]['p1p2_paths'],'elapsed',round(time.monotonic()-started,1),flush=True)
    base.write(out/'all_candidates_frozen.json',dict(at=base.stamp(),records=frozen))


def evaluate(out):
    torch.set_num_threads(2)
    plan=json.loads((out/'plan.json').read_text());assert base.digest(__file__)==plan['code_sha256']
    train,_=base.load_train();val=[]
    collection=json.loads((SOURCE/'internal_collection.json').read_text())
    for r in collection['records']:
        if not r['eligible']:continue
        file=SOURCE/'internal_raw'/r['file'];assert base.digest(file)==r['sha256']
        with np.load(file,allow_pickle=False) as a:val.append({k:a[k].copy() for k in a.files})
    data=[base.product_data(p,plan['spec']) for p in (train,val)]
    region=np.load(SOURCE/'region.npz')
    masks=[np.linalg.norm((np.stack([base.physical(a) for a in p])-region['mean'])/region['scale'],axis=-1)<=float(region['radius'])+1e-10 for p in (train,val)]
    old=torch.load(SOURCE/'high_w256.pt',map_location='cpu',weights_only=False)
    rows=[];control_rows=[]
    for rec in json.loads((out/'all_candidates_frozen.json').read_text())['records']:
        file=out/rec['file'];assert base.digest(file)==rec['sha256'];model=torch.load(file,map_location='cpu',weights_only=False)
        stats=[];vs=[]
        for (x,q,b,d),mask in zip(data,masks):
            v=predict(model,x,d);stats.append(base.statistics(v,b,d,region=mask));vs.append(v)
        x,q,b,d=data[1];v=vs[1];controls={}
        values=[('constant',np.ones_like(v)),('mode_constant',np.where(d,0.,1.))]
        for seed in range(5):
            rng=np.random.RandomState(145200000+seed);sv=v.copy();sv[~d]=rng.permutation(v[~d]);values.append(('shuffled_%d'%seed,sv))
            torch.manual_seed(145200100+seed)
            random=copy.deepcopy(model);random['weights']=ProgressValue(len(model['indices']),model['config']).state_dict()
            values.append(('random_%d'%seed,predict(random,x,d)))
        for name,cv in values:
            s=base.statistics(cv,b,d);controls[name]={k:s[k] for k in ('p1_violations','p2_violations','p1p2_paths','n')}
        qualified=all(s['p1_violations']==0 and s['p2_violations']==0 and s['bad_transitions']>0 for s in stats) and all(c['p1p2_paths']<c['n'] for c in controls.values())
        row=dict(id=rec['config']['id'],step=rec['step'],weights=rec['file'],sha256=rec['sha256'],train=stats[0],dev=stats[1],controls=controls,status='KEEP' if qualified else 'BORDERLINE')
        rows.append(row)
        np.savez_compressed(out/(row['id']+'_values.npz'),train_v=vs[0],dev_v=vs[1],train_bad=data[0][2],dev_bad=data[1][2],train_done=data[0][3],dev_done=data[1][3])
        print('DEV',row['id'],'paths',stats[1]['p1p2_paths'],'/20','P1/P2',stats[1]['p1_violations'],stats[1]['p2_violations'],flush=True)
    baseline=[]
    for (x,q,b,d),mask in zip(data,masks):baseline.append(base.statistics(base.predict(old,x,b,d),b,d,region=mask))
    # Retain old candidate if refinement worsens internal development results.
    all_choices=rows+[dict(id='original_high_w256',train=baseline[0],dev=baseline[1],status='BORDERLINE')]
    best=max(all_choices,key=lambda r:(r['status']=='KEEP',*score_key(r['dev']),*score_key(r['train'])))
    result=dict(completed=base.stamp(),rows=rows,baseline=dict(train=baseline[0],dev=baseline[1]),selected=best,
        qualified_count=sum(r['status']=='KEEP' for r in rows),calibration_count=0,test_count=0,
        conclusion='QUALIFIED_PENDING_FRESH_CONFIRMATION' if best['status']=='KEEP' else 'ABSTAIN / no-qualified-V',
        validation_scope='20 reused internal-development paths, no gradients on these paths, not a final confidence claim')
    base.write(out/'report.json',result)
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['candidate','train_P1_bad','train_P2_bad','train_paths','dev_P1_bad','dev_P2_bad','dev_paths','dev_certificate','status'])
        for r in all_choices:
            t=r['train'];v=r['dev'];w.writerow([r['id'],t['p1_violations'],t['p2_violations'],t['p1p2_paths'],v['p1_violations'],v['p2_violations'],v['p1p2_paths'],v['certificate_preview'],r['status']])
    print('SELECTED',best['id'],result['conclusion'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['train','evaluate']);p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=4000)
    a=p.parse_args()
    if a.stage=='train':train(a.output.resolve(),a.updates)
    else:evaluate(a.output.resolve())
