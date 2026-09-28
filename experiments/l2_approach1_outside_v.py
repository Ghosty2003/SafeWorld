"""Train a finite approach-goal V conditional on decoded initial distance >= 1.

Train/development only. No old calibration or test split is loaded. A newly
frozen candidate still requires fresh conditional calibration and testing.
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
from core.lppm.latent_reachability import fit_scaler
from experiments import l2_achievement64 as base
from experiments.l2_policy_v_refine import objective
from experiments.l2_latent_prediction_check import digest, now

RADIUS = 1.0
CANDIDATES = (
    dict(name='mean128', width=128, seed=23510, path_weight=0.),
    dict(name='max4_128', width=128, seed=23511, path_weight=4.),
    dict(name='max12_128', width=128, seed=23512, path_weight=12.),
    dict(name='max4_256', width=256, seed=23513, path_weight=4.),
)


def write(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def conditional_labels(aps):
    """Keep by t=0 only, never by whether the future reaches the goal."""
    done = base.done_for_radius(aps, RADIUS)
    keep = ~done[:, 0]
    return keep, done[keep]


def summarize(v, done):
    row = base.summary(v, done, inferential=False)
    delta = v[:, :-1] - v[:, 1:]
    pp = ~((delta < 0) | (~done[:, :-1] & (delta < base.ETA))).any(1)
    success = done[:, -1]
    n = int(success.sum())
    k = int((pp & success).sum())
    row['p1p2_given_goal'] = dict(successes=k, trials=n, rate=k/n if n else None)
    return row


def selection_key(row):
    # Successful-path P1/P2 coverage first, then endpoint sublevel and all paths.
    return (row['p1p2_given_goal']['successes'],
            row['finite_goal_and_certificate']['successes'],
            row['p1p2_paths']['successes'],
            -row['pending_sublevel_states'],
            -row['p2']['violations'], -row['p1']['violations'],
            -row['mean_paper_transition_loss'])


def prepare(source, out, steps):
    original = json.loads((source/'plan.json').read_text())
    if original['horizon'] != 64 or original['action_source'] != 'policy' or original['eta'] != .01:
        raise ValueError('Expected fixed-policy 64-step source with eta=.01')
    if digest(Path(original['checkpoint'])) != original['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    if steps < 1:
        raise ValueError('Positive update count required')
    out.mkdir(exist_ok=False)
    policy_hash = digest(source/'policy_config.json')
    plan = dict(created_at=now(), source_run=str(source),
                checkpoint=original['checkpoint'], checkpoint_sha256=original['checkpoint_sha256'],
                policy_config_sha256=policy_hash, action_source='policy',
                source_imagination_device=original['device'], training_device='cpu',
                horizon=64, eta=.01, radius=RADIUS,
                spec='F_[0,64](norm(decoded goal vector)<1.0)',
                initial_condition='norm(decoded initial goal vector)>=1.0',
                condition_scope='Decoded model state, not verified real environment distance',
                value_inputs='577 latent and planner-state features plus goal-history q; no time input',
                candidates=CANDIDATES, updates_per_candidate=steps, evaluate_every=200,
                optimizer=dict(name='AdamW',lr=.001,weight_decay=1e-4,batch_size=32),
                loss='mean transition hinge + path_weight*mean worst transition hinge; '
                     'training margin 2*eta; learned accepting-scalar anchor + pending-sublevel penalty',
                selection='max(dev P1P2 among goal paths, full C&goal, all-path P1P2, '
                          '-pending sublevel, -P2 count, -P1 count, -mean transition loss)',
                warrant_threshold=.95, future_calibration_confidence=.95,
                inferential_status='TRAIN_DEVELOPMENT_ONLY; no probability confidence claim',
                source_spec_screening_note='Radius selected after earlier exploratory screening. '
                     'Old calibration/test data are not independent confirmatory evidence for this spec.')
    write(out/'plan.json',plan)
    shutil.copyfile(source/'policy_config.json',out/'policy_config.json')
    datasets = {}
    seen = set()
    for split in ('train','validation'):
        prov=json.loads((source/f'{split}_provenance.json').read_text())
        if digest(source/f'{split}.npz') != prov['sha256'] or prov['policy_config_sha256'] != policy_hash:
            raise ValueError('Source data or policy provenance changed')
        with np.load(source/f'{split}.npz') as data:
            keep,done = conditional_labels(data['aps'])
            if not keep.any(): raise ValueError('No eligible start states')
            retained = {k:data[k][keep] for k in data.files}
        z=base.value_features(retained,'policy')
        if z.shape[1:]!=(65,577): raise ValueError('Unexpected state/feature schema')
        ids=np.flatnonzero(keep)
        fingerprints=[prov['fingerprints'][i] for i in ids]
        if len(set(fingerprints)) != len(fingerprints) or seen.intersection(fingerprints):
            raise ValueError('Duplicate trajectory across training/development')
        seen.update(fingerprints)
        np.savez_compressed(out/f'{split}.npz',**retained)
        write(out/f'{split}_provenance.json',dict(source=str(source/f'{split}.npz'),
              source_sha256=prov['sha256'],sha256=digest(out/f'{split}.npz'),
              policy_config_sha256=policy_hash,source_seed=prov['seed'],
              selected_indices=ids.tolist(),fingerprints=fingerprints,
              original_count=len(keep),retained_count=len(ids),excluded_initial_success=int((~keep).sum()),
              eventual_successes=int(done[:,-1].sum()),eventual_failures=int((~done[:,-1]).sum())))
        datasets[split]=(z,done)
        print('DATA',split,'retained',len(ids),'goal',int(done[:,-1].sum()),
              'non-goal',int((~done[:,-1]).sum()),'excluded initially inside',int((~keep).sum()),flush=True)
    return datasets


def train(out, datasets, steps):
    z,d=datasets['train']; zv,dv=datasets['validation']
    mean,scale=fit_scaler(z)
    inputs=torch.tensor((z-mean)/scale,dtype=torch.float32)
    dt=torch.tensor(d,dtype=torch.float32)
    best=None; records=[]
    for cfg in CANDIDATES:
        torch.manual_seed(cfg['seed'])
        net=base.AchievementValue('anchored_accepting_scalar',cfg['width'],z.shape[-1])
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-4)
        family_best=None
        for step in range(1,steps+1):
            ix=torch.randperm(len(inputs))[:32]
            loss=objective(net(inputs[ix],dt[ix]),dt[ix],margin=2.,path_weight=cfg['path_weight'])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite objective')
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
            opt.step()
            if step%200 and step!=steps: continue
            model=dict(weights={k:v.detach().clone() for k,v in net.state_dict().items()},
                       mean=mean,scale=scale,width=cfg['width'],variant='anchored_accepting_scalar',
                       radius=RADIUS,eta=base.ETA,step=step,initial_condition='decoded distance>=1.0',
                       training_config=cfg)
            tr=summarize(base.predict(model,z,d),d)
            va=summarize(base.predict(model,zv,dv),dv)
            row=dict(candidate=cfg['name'],step=step,train=tr,validation=va)
            records.append(row); key=selection_key(va)
            if family_best is None or key>family_best:
                family_best=key; torch.save(model,out/f"{cfg['name']}.pt")
            if best is None or key>best[0]:
                best=(key,row); torch.save(model,out/'selected.pt')
            with (out/'progress.json').open('w') as f:
                json.dump(dict(best=best[1],records=records),f,indent=2,allow_nan=False)
            print('FIT',cfg['name'],step,'train P1P2',tr['p1p2_paths']['successes'],'/',len(z),
                  'dev P1P2',va['p1p2_paths']['successes'],'/',len(zv),
                  'dev goal coverage',va['p1p2_given_goal']['successes'],'/',int(dv[:,-1].sum()),
                  'dev C',va['finite_goal_and_certificate']['successes'],
                  'dev violations',va['p1']['violations'],va['p2']['violations'],flush=True)
    write(out/'frozen.json',dict(frozen_at=now(),model_sha256=digest(out/'selected.pt'),
          plan_sha256=digest(out/'plan.json'),best=best[1],calibration_status='NOT_RUN',test_status='NOT_RUN'))
    write(out/'report.json',dict(result='TRAINED_NOT_CALIBRATED',warrant_status='NOT_ASSESSED',
          confidence_lower_bound=None,radius=RADIUS,horizon=64,eta=base.ETA,
          initial_condition='decoded initial distance>=1.0',**best[1]))
    print('FROZEN',best[1]['candidate'],best[1]['step'],
          'dev goal P1P2 coverage',best[1]['validation']['p1p2_given_goal'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=base.ROOT/'artifacts/safedreamer_l2_policy64_gpu')
    p.add_argument('--output',type=Path,default=base.ROOT/'artifacts/safedreamer_l2_approach1_outside_v')
    p.add_argument('--steps',type=int,default=2400)
    args=p.parse_args()
    if args.steps<1: p.error('Positive update count required')
    torch.set_num_threads(2)
    data=prepare(args.source.resolve(),args.output.resolve(),args.steps)
    train(args.output.resolve(),data,args.steps)


if __name__=='__main__': main()
