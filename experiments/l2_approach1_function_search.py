"""Compare six V function families using only fixed train/development data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.lppm.value_families import FAMILIES, ValueFamily, predict_family
from core.lppm.latent_reachability import fit_scaler
from experiments import l2_achievement64 as base
from experiments.l2_approach1_outside_v import summarize, selection_key, write
from experiments.l2_policy_v_refine import objective
from experiments.l2_latent_prediction_check import digest, now


def predict(model,z,done):
    if 'family' in model: return predict_family(model,z,done)
    return base.predict(model,z,done)


def prepare(source,out,steps):
    plan=json.loads((source/'plan.json').read_text())
    frozen=json.loads((source/'frozen.json').read_text())
    if (plan['radius'],plan['horizon'],plan['eta'],plan['action_source'])!=(1.,64,.01,'policy'):
        raise ValueError('Expected unchanged approach1/outside source')
    if plan['initial_condition']!='norm(decoded initial goal vector)>=1.0':
        raise ValueError('Wrong initial-state population')
    if digest(source/'selected.pt')!=frozen['model_sha256']:
        raise ValueError('Baseline changed')
    if digest(source/'plan.json')!=frozen['plan_sha256']:
        raise ValueError('Source plan changed')
    if digest(Path(plan['checkpoint']))!=plan['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    if digest(source/'policy_config.json')!=plan['policy_config_sha256']:
        raise ValueError('Policy changed')
    out.mkdir(exist_ok=False)
    candidates=[dict(name=f'{family}_v{variant}',family=family,width=128,
                     seed=24510+2*i+variant,path_weight=0. if variant==0 else 4.)
                for i,family in enumerate(FAMILIES) for variant in (0,1)]
    plan.update(created_at=now(),source_run=str(source),source_model_sha256=frozen['model_sha256'],
                candidates=candidates,updates_per_candidate=steps,
                search='Original V plus six alternative state-only families, two fits each',
                extension='All alternative waiting functions wrapped in softplus; learned constant '
                          'accepting branch retained, no forced goal zero. RBF centers only from '
                          'training waiting states; Fourier frequencies random and fixed.',
                inference='Development-only selection; no holdouts or inferential bounds used')
    write(out/'plan.json',plan)
    shutil.copyfile(source/'selected.pt',out/'baseline.pt')
    shutil.copyfile(source/'policy_config.json',out/'policy_config.json')
    datasets={}; seen=set()
    for split in ('train','validation'):
        provenance=json.loads((source/f'{split}_provenance.json').read_text())
        if digest(source/f'{split}.npz')!=provenance['sha256']:
            raise ValueError('Source data changed')
        ids=provenance['fingerprints']
        if len(set(ids))!=len(ids) or seen.intersection(ids):
            raise ValueError('Duplicate trajectories')
        seen.update(ids)
        for suffix in ('.npz','_provenance.json'):
            shutil.copyfile(source/f'{split}{suffix}',out/f'{split}{suffix}')
        with np.load(out/f'{split}.npz') as a:
            z=base.value_features(a,'policy'); d=base.done_for_radius(a['aps'],1.)
        if d[:,0].any() or z.shape[1:]!=(65,577):
            raise ValueError('Initial condition or feature schema changed')
        datasets[split]=(z,d)
    return datasets,candidates


def train(out,datasets,candidates,steps,refit_scaler=False,pending_penalty=1.):
    if pending_penalty<1: raise ValueError('Do not weaken the pending-state penalty')
    z,d=datasets['train']; zv,dv=datasets['validation']
    baseline=torch.load(out/'baseline.pt',weights_only=False)
    mean,scale=fit_scaler(z) if refit_scaler else (baseline['mean'],baseline['scale'])
    x=torch.tensor((z-mean)/scale,dtype=torch.float32)
    dt=torch.tensor(d,dtype=torch.float32)
    best=None; records=[]; per_candidate={}

    def consider(model,name,step):
        nonlocal best
        tr=summarize(predict(model,z,d),d)
        va=summarize(predict(model,zv,dv),dv)
        row=dict(candidate=name,step=step,train=tr,validation=va)
        key=selection_key(va); records.append(row)
        if name not in per_candidate or key>selection_key(per_candidate[name]['validation']):
            per_candidate[name]=row
            torch.save(model,out/f'{name}.pt')
        if best is None or key>best[0]:
            best=(key,row); torch.save(model,out/'selected.pt')
        with (out/'progress.json').open('w') as f:
            json.dump(dict(best=best[1],per_candidate=per_candidate,records=records),f,indent=2,allow_nan=False)
        print('FIT',name,step,'train goal coverage',tr['p1p2_given_goal']['successes'],'/',int(d[:,-1].sum()),
              'dev goal coverage',va['p1p2_given_goal']['successes'],'/',int(dv[:,-1].sum()),
              'dev C',va['finite_goal_and_certificate']['successes'],
              'dev P1/P2 violations',va['p1']['violations'],va['p2']['violations'],flush=True)

    consider(baseline,'original_V',baseline['step'])
    for cfg in candidates:
        torch.manual_seed(cfg['seed'])
        centers=None
        if cfg['family']=='rbf':
            waiting=x[~dt.bool()]
            centers=waiting[torch.randperm(len(waiting))[:cfg['width']]]
        net=ValueFamily(cfg['family'],z.shape[-1],cfg['width'],centers)
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-4)
        for step in range(1,steps+1):
            ix=torch.randperm(len(x))[:32]
            values=net(x[ix],dt[ix])
            loss=objective(values,dt[ix],margin=2.,path_weight=cfg['path_weight'])
            waiting=~dt[ix].bool()
            if pending_penalty>1 and waiting.any():
                loss=loss+(pending_penalty-1)*torch.relu(.02-values[waiting]).mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
            opt.step()
            if step%200 and step!=steps: continue
            model=dict(family=cfg['family'],width=cfg['width'],
                       weights={k:v.detach().clone() for k,v in net.state_dict().items()},
                       mean=mean,scale=scale,radius=1.,eta=.01,step=step,
                       training_config=cfg,initial_condition='decoded distance>=1.0')
            consider(model,cfg['name'],step)
    write(out/'frozen.json',dict(frozen_at=now(),model_sha256=digest(out/'selected.pt'),
          plan_sha256=digest(out/'plan.json'),best=best[1],per_candidate=per_candidate,
          calibration_status='NOT_RUN',test_status='NOT_RUN'))
    write(out/'report.json',dict(result='TRAINED_NOT_CALIBRATED',warrant_status='NOT_ASSESSED',
          confidence_lower_bound=None,radius=1.,horizon=64,eta=.01,
          initial_condition='decoded initial distance>=1.0',baseline=records[0],**best[1]))
    print('FROZEN',best[1]['candidate'],best[1]['step'],
          'dev goal coverage',best[1]['validation']['p1p2_given_goal'],flush=True)


def evaluate(out):
    frozen=json.loads((out/'frozen.json').read_text())
    report=json.loads((out/'report.json').read_text())
    if digest(out/'selected.pt')!=frozen['model_sha256'] or digest(out/'plan.json')!=frozen['plan_sha256']:
        raise ValueError('Model or plan changed')
    model=torch.load(out/'selected.pt',weights_only=False)
    for split in ('train','validation'):
        prov=json.loads((out/f'{split}_provenance.json').read_text())
        if digest(out/f'{split}.npz')!=prov['sha256']: raise ValueError('Data changed')
        with np.load(out/f'{split}.npz') as a:
            z=base.value_features(a,'policy'); d=base.done_for_radius(a['aps'],1.)
        row=summarize(predict(model,z,d),d)
        if row!=report[split] or row!=frozen['best'][split]:
            raise ValueError('Report does not reproduce')
        print('VERIFIED',split,row['p1p2_given_goal'],row['finite_goal_and_certificate'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=base.ROOT/'artifacts/safedreamer_l2_approach1_outside_v')
    p.add_argument('--output',type=Path,default=base.ROOT/'artifacts/safedreamer_l2_approach1_function_search')
    p.add_argument('--steps',type=int,default=1600)
    p.add_argument('--evaluate-only',action='store_true')
    args=p.parse_args()
    if args.steps<1: p.error('Positive steps required')
    torch.set_num_threads(2)
    if args.evaluate_only: evaluate(args.output.resolve()); return
    data,candidates=prepare(args.source.resolve(),args.output.resolve(),args.steps)
    train(args.output.resolve(),data,candidates,args.steps)
    evaluate(args.output.resolve())


if __name__=='__main__': main()
