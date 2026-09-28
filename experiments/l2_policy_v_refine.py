"""Refine V only; reuse development data, then collect fresh model-only holdouts.

This is an explicitly extended training objective, not the original paper loss.
Evaluation retains the original exact P1/P2 inequalities and eta=0.01.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import l2_achievement64 as base
from experiments.l2_latent_prediction_check import digest, now


CANDIDATES = (
    dict(name='warm_max4', width=128, warm=True, lr=.0003, path_weight=4., margin=2.),
    dict(name='warm_max12', width=128, warm=True, lr=.0003, path_weight=12., margin=2.),
    dict(name='cold128_max4', width=128, warm=False, lr=.001, path_weight=4., margin=2.),
    dict(name='cold256_max4', width=256, warm=False, lr=.001, path_weight=4., margin=2.),
)


def objective(v, done, margin=2., path_weight=4.):
    """Optimize worst edge per path as well as mean; no path filtering."""
    waiting = ~done[:, :-1].bool()
    increase = v[:, 1:] - v[:, :-1]
    # Stronger TRAINING margin only; evaluation always uses the unchanged eta.
    residual = torch.relu(increase + margin * base.ETA * waiting)
    loss = residual.mean() + path_weight * residual.amax(dim=1).mean()
    mask = done.bool()
    if mask.any():
        loss = loss + .1 * torch.log(v[mask] / (base.ETA / 4)).square().mean()
    if (~mask).any():
        loss = loss + torch.relu(2 * base.ETA - v[~mask]).mean()
    return loss


def rank(row):
    # Development only; primary objective is COMPLETE-path P1 AND P2.
    return (row['p1p2_paths']['successes'],
            row['finite_goal_and_certificate']['successes'],
            -row['pending_sublevel_states'],
            -row['p2']['violations'], -row['p1']['violations'],
            -row['mean_paper_transition_loss'])


def prepare(source, output, steps):
    plan = json.loads((source/'plan.json').read_text())
    if (plan['action_source'], plan['horizon'], plan['eta'], plan['radii']) != ('policy',64,.01,[.3]):
        raise ValueError('Expected unchanged policy64 original-goal experiment')
    if plan['device'] != 'gpu':
        raise ValueError('Keep original GPU imagination configuration')
    if digest(Path(plan['checkpoint'])) != plan['checkpoint_sha256']:
        raise ValueError('World model checkpoint changed')
    frozen = json.loads((source/'frozen.json').read_text())
    if digest(source/'selected.pt') != frozen['model_sha256']:
        raise ValueError('Baseline V changed')
    output.mkdir(exist_ok=False)
    # Deliberately never load old calibration/test trajectories or their outcomes.
    for split in ('train','validation'):
        provenance = json.loads((source/f'{split}_provenance.json').read_text())
        if digest(source/f'{split}.npz') != provenance['sha256']:
            raise ValueError('Development data changed')
        for suffix in ('.npz','_provenance.json'):
            shutil.copyfile(source/f'{split}{suffix}', output/f'{split}{suffix}')
    for name in ('policy_config.json','selected.pt'):
        shutil.copyfile(source/name, output/('baseline.pt' if name=='selected.pt' else name))
    plan.update(created_at=now(), source_run=str(source),
                seeds={**plan['seeds'], 'calibration':22503, 'test':22504},
                updates_per_candidate=steps, refinement_candidates=CANDIDATES,
                variants=['anchored_accepting_scalar'],
                paper_objective='Reference only; this refinement uses refinement_objective below',
                source_model_sha256=frozen['model_sha256'],
                selection='max(dev complete-path P1P2, C&goal, -pending sublevel states, '
                          '-P2 violations, -P1 violations, -mean transition loss)',
                candidate_updates='Original V plus four variants, dev evaluation every 200 updates',
                refinement_objective='mean edge hinge + weighted mean per-path maximum hinge; '
                    'training descent margin 2*eta, accepting anchor and pending sublevel penalty; '
                    'AdamW weight decay 1e-4; all training paths retained; no time input',
                inference='Fresh calibration/test only after selection and freeze; '
                          'finite model-only evidence, not support-wide proof')
    (output/'plan.json').write_text(json.dumps(plan,indent=2))
    base.configure(output=output,restore=True)


def train(steps):
    z,d = base.load('train',.3)
    zv,dv = base.load('validation',.3)
    baseline = torch.load(base.OUT/'baseline.pt',weights_only=False)
    mean,scale = baseline['mean'],baseline['scale']
    x = torch.tensor((z-mean)/scale,dtype=torch.float32)
    dt = torch.tensor(d,dtype=torch.float32)
    records = []
    best = None

    def consider(model, name, step):
        nonlocal best
        tr = base.summary(base.predict(model,z,d),d)
        va = base.summary(base.predict(model,zv,dv),dv)
        row = dict(candidate=name,step=step,train=tr,validation=va)
        records.append(row)
        key = rank(va)
        if best is None or key>best[0]:
            best = (key,row)
            torch.save(model,base.OUT/'selected.pt')
        print('FIT',name,step,'train paths',tr['p1p2_paths']['successes'],'/',len(z),
              'dev paths',va['p1p2_paths']['successes'],'/',len(zv),
              'dev C',va['finite_goal_and_certificate']['successes'],
              'dev P1/P2',va['p1']['violations'],va['p2']['violations'],flush=True)
        base.write('progress.json',dict(best=best[1],records=records),replace=True)

    consider(baseline,'original_V',baseline['step'])
    for i,cfg in enumerate(CANDIDATES):
        torch.manual_seed(22510+i)
        net = base.AchievementValue('anchored_accepting_scalar',cfg['width'],z.shape[-1])
        if cfg['warm']:
            net.load_state_dict(baseline['weights'])
        optimizer = torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=1e-4)
        for step in range(1,steps+1):
            ix = torch.randperm(len(x))[:32]
            loss = objective(net(x[ix],dt[ix]),dt[ix],cfg['margin'],cfg['path_weight'])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite objective')
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
            optimizer.step()
            if step%200 and step!=steps: continue
            model = dict(weights={k:v.detach().clone() for k,v in net.state_dict().items()},
                         mean=mean,scale=scale,width=cfg['width'],variant='anchored_accepting_scalar',
                         radius=.3,step=step,eta=base.ETA,refinement=cfg)
            consider(model,cfg['name'],step)
    base.write('frozen.json',dict(frozen_at=now(),model_sha256=digest(base.OUT/'selected.pt'),
                                best=best[1],candidates=CANDIDATES))
    print('FROZEN',best[1]['candidate'],best[1]['step'],'dev',best[1]['validation']['p1p2_paths'],flush=True)


def compare_frozen():
    """Paired descriptive comparison; never select or retrain using holdouts."""
    plan=json.loads((base.OUT/'plan.json').read_text())
    frozen=json.loads((base.OUT/'frozen.json').read_text())
    if digest(base.OUT/'baseline.pt')!=plan['source_model_sha256']:
        raise ValueError('Original candidate changed')
    if digest(base.OUT/'selected.pt')!=frozen['model_sha256']:
        raise ValueError('Selected candidate changed')
    old=torch.load(base.OUT/'baseline.pt',weights_only=False)
    new=torch.load(base.OUT/'selected.pt',weights_only=False)
    rows={}
    for split in ('calibration','test'):
        prov=json.loads((base.OUT/f'{split}_provenance.json').read_text())
        if digest(base.OUT/f'{split}.npz')!=prov['sha256']:
            raise ValueError('Holdout changed')
        z,d=base.load(split,.3)
        a=base.summary(base.predict(old,z,d),d)
        b=base.summary(base.predict(new,z,d),d)
        ac=np.array(a['per_path_candidate_event'],dtype=bool)
        bc=np.array(b['per_path_candidate_event'],dtype=bool)
        rows[split]=dict(original=a,refined=b,
                        newly_passing=int((~ac & bc).sum()),
                        newly_failing=int((ac & ~bc).sum()),
                        both_passing=int((ac & bc).sum()))
    result=dict(compared_at=now(),scope='Same fresh paths, descriptive paired comparison only; '
                'no post-test selection or refitting',**rows)
    base.write('paired_v_comparison.json',result)
    for split,row in rows.items():
        print('PAIRED',split,'original',row['original']['p1p2_paths'],
              'refined',row['refined']['p1p2_paths'],flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=base.ROOT/'artifacts/safedreamer_l2_policy64_gpu')
    p.add_argument('--output',type=Path,default=base.ROOT/'artifacts/safedreamer_l2_policy64_gpu_vrefine')
    p.add_argument('--steps',type=int,default=1600)
    p.add_argument('--compare-only',action='store_true')
    args = p.parse_args()
    if args.steps<1: p.error('Positive steps required')
    torch.set_num_threads(2)
    if args.compare_only:
        base.configure(output=args.output.resolve(),restore=True)
        compare_frozen()
        return
    prepare(args.source.resolve(),args.output.resolve(),args.steps)
    train(args.steps)


if __name__=='__main__': main()
