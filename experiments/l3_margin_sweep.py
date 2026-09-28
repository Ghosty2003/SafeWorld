"""Training-margin-only development sweep on S48, S24 and clearance recurrence.

All development successor outcomes have already been inspected. Fixed-slack
counts below are sensitivity diagnostics, NOT fresh confidence certificates.
Verification thresholds stay W<=-0.01 and U<=0 for every candidate.
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
from experiments.l3_spec_ablation import SPECS,load_role,relabel
from experiments.l3_spec_ablation_audit import load_queries
from experiments.l3_wu_search import Candidate,prepare
from experiments.l3_wu_data_ablation import restore
from experiments.l3_safedreamer_clear_pipeline import digest,write,stamp

MARGINS={'W':[.01,.03,.05,.10,.15],'U':[0.,.01,.03,.05]}
SPEC_IDS=('A_S48','B_S24','C_clearance')


def fit_margin(kind,data,mean,scale,steps,margin):
    if margin<0: raise ValueError('Nonnegative training margin required')
    torch.manual_seed(73 if kind=='W' else 41)
    net=Candidate(mean,scale,'wide' if kind=='W' else 'sublevel')
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    history=[]
    for step in range(steps):
        b=torch.randperm(len(data['x']))[:32]; x,y=data['x'][b],data['y'][b]
        idx=torch.randperm(len(data['core']))[:512]; c=data['core'][idx]
        idx=torch.randperm(len(data['px']))[:512]; px,py=data['px'][idx],data['py'][idx]
        drift=net(y.flatten(0,1)).reshape(y.shape[:2]).mean(1)-net(x)
        pd=net(py)-net(px)
        if kind=='W':
            nf=x[:,-1]<1; pnf=px[:,-1]<1
            branch=torch.relu(drift[nf]+margin).mean() if nf.any() else drift.sum()*0
            pl=torch.relu(pd[pnf]+margin).mean() if pnf.any() else pd.sum()*0
            shape=((net(c)-.48*(1-c[:,-1]))**2).mean()
        else:
            # H1 still includes accepting states. A positive U train margin is
            # only a stress-test preference, not a new global theorem premise.
            branch=torch.relu(drift+margin).mean(); pl=torch.relu(pd+margin).mean()
            shell=c.clone(); shell[:,:-2]=net.mean[:-2]+4*(c[:,:-2]-net.mean[:-2])
            shape=((net(c)-.2)**2).mean()+torch.relu(.85-net(shell)).square().mean()
        loss=branch+.5*pl+.25*shape
        opt.zero_grad(); loss.backward(); opt.step()
        if step==0 or (step+1)%100==0 or step+1==steps:
            history.append(dict(step=step+1,loss=float(loss),branch=float(branch),path=float(pl),shape=float(shape)))
    return net.eval(),history


def distribution(values):
    if not len(values): return None
    q=np.quantile(np.asarray(values,dtype=float),[0,.1,.5,.9,1])
    return dict(zip(('min','q10','median','q90','max'),map(float,q)))


@torch.no_grad()
def diagnose(net,kind,batches,paths,train_margin,reference_slack):
    x=torch.tensor(np.stack([b.anchor for b in batches]))
    vx=net(x).numpy().astype(float)
    means=np.array([net(torch.tensor(b.successors)).numpy().astype(float).mean() for b in batches])
    g=means-vx
    required=x[:,-1].numpy()<1 if kind=='W' else np.ones(len(x),dtype=bool)
    threshold=-.01 if kind=='W' else 0.
    core=torch.tensor(paths.reshape(-1,paths.shape[-1])); values=net(core).numpy()
    return dict(required=int(required.sum()),raw_pass=int(((g<=threshold)&required).sum()),
                train_margin_pass=int(((g<=-train_margin)&required).sum()),
                fixed_slack_pass=int(((g+reference_slack<=threshold)&required).sum()),
                verification_threshold=threshold,reference_sampling_slack=reference_slack,
                distribution=distribution(g[required]),all_anchor_distribution=distribution(g),
                values=distribution(values),value_std=float(values.std()),
                below_ell_anchor_count=int((vx<=.8).sum()),n_anchors=len(x),
                initial_below_ell_count=int((net(torch.tensor(paths[:,0])).numpy()<=.8).sum()),
                n_paths=len(paths),collapsed_sample_range=bool(float(values.max()-values.min())<.02),
                anchor_rows=[dict(id=b.sample_id,accepting=bool(x[i,-1]>=1),required=bool(required[i]),
                                 value=float(vx[i]),mean_successor=float(means[i]),drift=float(g[i]),
                                 raw_pass=bool(g[i]<=threshold),fixed_slack_pass=bool(g[i]+reference_slack<=threshold))
                             for i,b in enumerate(batches)],
                confidence=None,scope='REUSED_DATA_DIAGNOSTIC_NO_NEW_CONFIDENCE')


def ranking(row):
    r=row['development']; kind=row['kind']
    eligible=r['required']>0
    if kind=='U':
        eligible=eligible and r['below_ell_anchor_count']==r['n_anchors'] and not r['collapsed_sample_range']
        eligible=eligible and r['initial_below_ell_count']==r['n_paths']
    # Primary population is fixed (not shrunk by each U sublevel).
    return (eligible,r['fixed_slack_pass'],r['raw_pass'],
            -r['distribution']['median'] if r['distribution'] else -float('inf'))


def run(args):
    torch.set_num_threads(2)
    source=args.source.resolve(); out=args.output.resolve()
    plan=json.loads((source/'plan.json').read_text()); frozen=json.loads((source/'frozen.json').read_text())
    if digest(source/'plan.json')!=frozen['plan_sha256']: raise ValueError('Source plan changed')
    for name,sha in plan['source_sha256'].items():
        if digest(ROOT/name)!=sha: raise ValueError(f'Source changed: {name}')
    train=load_role(Path(plan['source']),'fit'); dev=load_role(Path(plan['source']),'development')
    tq,ts=load_queries(source,'fit',train,frozen['time'])
    dq,ds=load_queries(source,'development',dev,frozen['time'])
    if set(ts)&set(ds): raise ValueError('Cross-role query seed reuse')
    previous=json.loads((source/'report.json').read_text())
    slack=previous['rows'][0]['sampling_radius']
    out.mkdir(parents=True,exist_ok=False)
    write(out/'plan.json',dict(time=stamp(),source=str(source),source_report_sha256=digest(source/'report.json'),
          selected_specs=list(SPEC_IDS),training_margins=MARGINS,steps=args.steps,
          W_family='wide',U_family='sublevel',B=1.,ell=.8,W_verification_epsilon=.01,U_verification_epsilon=0.,
          fit_kappa=32,dev_kappa=256,fit_paths=len(train),dev_paths=len(dev),
          fixed_reference_slack=slack,scope='ADAPTIVE_DEVELOPMENT_MARGIN_SWEEP',
          caveats=['No new samples or final validation; nominal fixed-slack passes are NOT confidence-valid after adaptive reuse.',
                   'Global strict U decrease on a forever-retained set is impossible for bounded nonnegative U.',
                   'Increasing W margin also restricts return-time bounds; it need not be feasible.',
                   'W and U independently fitted: no Cartesian training sweep, no region-exclusion reward.'],
          code_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),
                       ROOT/'experiments/l3_wu_search.py',ROOT/'experiments/l3_spec_ablation.py',
                       ROOT/'experiments/l3_spec_ablation_audit.py']}))
    started=time.perf_counter(); rows=[]; selected={}
    for spec in [s for s in SPECS if s['id'] in SPEC_IDS]:
        sid=spec['id']; tp,tb=relabel(train,tq,spec); dp,db=relabel(dev,dq,spec)
        data=prepare(tp,np.zeros(tp.shape[:2],np.float32),tb)
        core=tp.reshape(-1,578); mean=core.mean(0); scale=np.maximum(core.std(0),.05)
        mean[-2:]=0; scale[-2:]=1
        for kind in ('W','U'):
            candidates=[]
            file=source/f'{sid}_{kind}.pt'
            if digest(file)!=frozen['models'][sid][kind]: raise ValueError('Source weights changed')
            original=restore(file)
            baseline_margin=.01 if kind=='W' else .001
            baseline=dict(spec=sid,kind=kind,train_margin=baseline_margin,name=f'{sid}_{kind}_baseline',
                          checkpoint=str(file),baseline=True,
                          train=diagnose(original,kind,tb,tp,baseline_margin,slack),
                          development=diagnose(original,kind,db,dp,baseline_margin,slack))
            rows.append(baseline); candidates.append(baseline)
            for margin in MARGINS[kind]:
                net,history=fit_margin(kind,data,mean,scale,args.steps,margin)
                # Numerical regression control: W=.01 reproduces the source
                # trainer exactly when the update budget is unchanged.
                if kind=='W' and margin==.01 and args.steps==plan['steps']:
                    for k,v in original.state_dict().items():
                        if not torch.equal(v,net.state_dict()[k]): raise ValueError('Baseline training not reproduced')
                name=f'{sid}_{kind}_m{margin:.3f}'; file=out/f'{name}.pt'
                torch.save(dict(family=net.family,state=net.state_dict()),file)
                row=dict(spec=sid,kind=kind,train_margin=margin,name=name,checkpoint=file.name,
                         sha256=digest(file),baseline=False,history=history,
                         train=diagnose(net,kind,tb,tp,margin,slack),
                         development=diagnose(net,kind,db,dp,margin,slack))
                write(out/f'{name}.json',row); rows.append(row); candidates.append(row)
                r=row['development']
                print(name,'raw',r['raw_pass'],'/',r['required'],'fixed-slack',r['fixed_slack_pass'],
                      'median',r['distribution']['median'],'train-target',row['train']['train_margin_pass'],flush=True)
            best=max(candidates,key=ranking)
            selected[f'{sid}_{kind}']=best['name'] if ranking(best)[0] else None
    write(out/'report.json',dict(completed=stamp(),elapsed_seconds=time.perf_counter()-started,rows=rows,
          selected_development_only=selected,scope='DEVELOPMENT_ONLY',confidence=None,
          global_L3='ABSTAIN',recurrence_probability_lower_bound=None,final_validation='NOT_RUN'))
    with (out/'summary.csv').open('x',newline='') as f:
        w=csv.writer(f); w.writerow(['spec','function','train_margin','split','required','raw_pass','fixed_slack_pass',
                                    'train_margin_pass','min','q10','median','q90','max','baseline'])
        for r in rows:
            for split in ('train','development'):
                d=r[split]; q=d['distribution']
                w.writerow([r['spec'],r['kind'],r['train_margin'],split,d['required'],d['raw_pass'],d['fixed_slack_pass'],
                            d['train_margin_pass'],*[q[k] for k in ('min','q10','median','q90','max')],r['baseline']])
    with (out/'anchor_drifts.csv').open('x',newline='') as f:
        w=csv.writer(f); w.writerow(['candidate','split','anchor_id','required','accepting','value','mean_successor','drift'])
        for r in rows:
            for split in ('train','development'):
                for a in r[split]['anchor_rows']:
                    w.writerow([r['name'],split,a['id'],a['required'],a['accepting'],a['value'],a['mean_successor'],a['drift']])
    print('SELECTED',selected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'artifacts/safedreamer_l3_spec_ablation_v1')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=400)
    args=p.parse_args()
    if args.steps<1: p.error('Positive steps required')
    run(args)
