"""Expand conditional train/development data and search more V functions.

No calibration or test outcomes are used. All eligible failures are retained.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.lppm.value_families import FAMILIES,EXTRA_FAMILIES
from experiments import l2_achievement64 as base
from experiments import l2_approach1_function_search as search
from experiments.l2_approach1_outside_v import write
from experiments.l2_approach1_holdout import take_eligible,prior_fingerprints
from experiments.l2_latent_prediction_check import digest,now
from experiments.l2_paper_sublevel_finite import hashes

TOTALS=dict(train=300,validation=100)
SEEDS=dict(train=26501000,validation=26502000)
BATCH=20


def candidates():
    items=[]
    for family in FAMILIES+EXTRA_FAMILIES:
        for variant in (0,1):
            items.append(dict(name=f'{family}_v{variant}',family=family,width=128,
                              seed=26510+len(items),path_weight=0. if variant==0 else 4.))
    for family in ('residual','elu'):
        for variant in (0,1):
            items.append(dict(name=f'{family}256_v{variant}',family=family,width=256,
                              seed=26510+len(items),path_weight=0. if variant==0 else 4.))
    return items


def prepare(source,out,steps):
    p=json.loads((source/'plan.json').read_text())
    f=json.loads((source/'frozen.json').read_text())
    if (p['horizon'],p['eta'],p['radius'],p['action_source'])!=(64,.01,1.,'policy'):
        raise ValueError('Expected fixed approach1 policy64 scope')
    if p['initial_condition']!='norm(decoded initial goal vector)>=1.0':
        raise ValueError('Initial-state condition changed')
    if digest(source/'selected.pt')!=f['model_sha256'] or digest(source/'plan.json')!=f['plan_sha256']:
        raise ValueError('Source V/plan changed')
    if digest(Path(p['checkpoint']))!=p['checkpoint_sha256'] or digest(source/'policy_config.json')!=p['policy_config_sha256']:
        raise ValueError('World model or policy changed')
    old_counts={}
    for split in TOTALS:
        prov=json.loads((source/f'{split}_provenance.json').read_text())
        if digest(source/f'{split}.npz')!=prov['sha256']: raise ValueError('Source data changed')
        with np.load(source/f'{split}.npz') as a:
            if base.done_for_radius(a['aps'],1.)[:,0].any(): raise ValueError('Wrong initial states')
            old_counts[split]=len(a['latent'])
        if old_counts[split]>=TOTALS[split]: raise ValueError('Expected new data to add')
    out.mkdir(exist_ok=False)
    for name in ('selected.pt','policy_config.json'):
        shutil.copyfile(source/name,out/('baseline.pt' if name=='selected.pt' else name))
    p.update(created_at=now(),source_run=str(source),source_model_sha256=f['model_sha256'],
             source_plan_sha256=f['plan_sha256'],counts=TOTALS,old_counts=old_counts,
             added_counts={s:TOTALS[s]-old_counts[s] for s in TOTALS},seeds=SEEDS,
             batch_size=BATCH,max_batches_per_split=100,device='gpu',
             candidates=candidates(),updates_per_candidate=steps,
             search='20 new candidate fits on enlarged training data; frozen old V is reference',
             normalization='Fit only on ALL enlarged training states; old V keeps its own scaler',
             loss='Mean plus weighted worst-edge hinge, training margin 2*eta, accepting anchor; '
                  'pending-state sublevel penalty coefficient 10 (was 1). No output floor.',
             sampling='Append fresh conditional paths, first eligible in draw order. Eligibility '
                      'depends only on decoded initial distance >=1.0. Keep all future failures.',
             inferential_status='TRAIN_DEVELOPMENT_ONLY; prior holdout outcomes not used or pooled',
             inference='No calibration or test performed. Repeated development selection needs '
                       'new held-out confirmation after freeze.',
             reset_rng_note='Imagination seeds explicit; simulator reset RNG is not fully specified by them.')
    write(out/'plan.json',p)
    for split in TOTALS:
        for suffix in ('.npz','_provenance.json'):
            shutil.copyfile(source/f'{split}{suffix}',out/f'old_{split}{suffix}')


def collect(source,out):
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    p=json.loads((out/'plan.json').read_text())
    extra=build_extra(argparse.Namespace(repo_root=None,checkpoint=p['checkpoint']))
    extra['action_source']='policy'
    extra['config_overrides']['jax'].update(platform='gpu',logical_gpus=0)
    seen=prior_fingerprints(source)
    # Identity metadata only; never read old calibration/test outcomes as training data.
    old_holdout=base.ROOT/'artifacts/safedreamer_l2_approach1_linear_holdout'
    for path in old_holdout.glob('*_provenance.json'):
        prov=json.loads(path.read_text()); seen.update(prov['fingerprints'])
        for batch in prov.get('batches',[]): seen.update(batch['fingerprints'])
    log=logging.getLogger('wrappers.safedreamer_wrapper'); log.setLevel(logging.INFO)
    if not log.handlers: log.addHandler(logging.StreamHandler(sys.stdout))
    cfg=RolloutConfig(horizon=64,n_rollouts=BATCH,seed=26500,action_source='policy',extra=extra)
    with SafeDreamerWrapper(cfg) as wrapper:
        wrapper.load()
        import jax
        if jax.default_backend()!='gpu': raise RuntimeError('GPU required')
        print('IMAGINATION DEVICE',jax.devices(),flush=True)
        actual=dict(action_source='policy',expl_behavior=wrapper._config.expl_behavior,
            planner={k:getattr(wrapper._config.planner,k) for k in
                     ('horizon','num_samples','num_elites','iterations','mixture_coef','momentum','init_std')},
            cost_limit=wrapper._config.cost_limit,
            planner_source_sha256=digest(Path(extra['repo_root'])/'SafeDreamer/behaviors.py'),
            note='Local reconstructed evaluation planner; no claim of exact historical training config.')
        if actual!=json.loads((out/'policy_config.json').read_text()): raise ValueError('Policy mismatch')
        for split in TOTALS:
            needed=p['added_counts'][split]; kept=0; pieces=[]; records=[]; start=now()
            raw_dir=out/f'new_{split}_raw'; raw_dir.mkdir()
            for batch in range(p['max_batches_per_split']):
                seed=SEEDS[split]+batch
                print('COLLECTING',split,'new',kept,'/',needed,'batch',batch,flush=True)
                data=wrapper.sample_latent_rollouts(RolloutConfig(
                    horizon=64,n_rollouts=BATCH,seed=seed,action_source='policy',extra=extra))
                if data['action_source']!='policy' or data['latent'].shape!=(BATCH,65,512):
                    raise ValueError('Wrong rollout schema')
                aps=np.array([[[s[k] for k in ('goal_dist','hazard_dist','velocity')] for s in path]
                              for path in data['aps']],dtype=np.float32)
                arrays=dict(latent=data['latent'],aps=aps,decoded=data['decoded'],actions=data['actions'],
                            **data['policy_state'],**{f'rssm_{k}':v for k,v in data['rssm'].items()})
                fp=hashes(arrays['latent'])
                if len(set(fp))!=BATCH or seen.intersection(fp): raise ValueError('Duplicate new trajectory')
                seen.update(fp)
                eligible,ids=take_eligible(aps,needed-kept)
                raw=raw_dir/f'batch_{batch:03d}.npz'; np.savez_compressed(raw,**arrays)
                record=dict(seed=seed,raw_file=str(raw.relative_to(out)),sha256=digest(raw),
                            fingerprints=fp,eligible_indices=np.flatnonzero(eligible).tolist(),
                            selected_indices=ids.tolist(),initial_rejected=int((~eligible).sum()),
                            eligible_not_needed=int(eligible.sum()-len(ids)))
                write(raw_dir/f'batch_{batch:03d}.json',record); records.append(record)
                if len(ids): pieces.append({k:v[ids] for k,v in arrays.items()})
                kept+=len(ids)
                print('PROGRESS',split,'new',kept,'/',needed,'drawn',len(records)*BATCH,flush=True)
                if kept==needed: break
            if kept!=needed: raise RuntimeError('Draw budget exhausted; no smaller-data substitute')
            fresh={k:np.concatenate([part[k] for part in pieces],axis=0) for k in pieces[0]}
            np.savez_compressed(out/f'new_{split}.npz',**fresh)
            with np.load(out/f'old_{split}.npz') as old:
                combined={k:np.concatenate([old[k],fresh[k]],axis=0) for k in old.files}
            np.savez_compressed(out/f'{split}.npz',**combined)
            write(out/f'{split}_provenance.json',dict(started_at=start,ended_at=now(),
                  sha256=digest(out/f'{split}.npz'),fingerprints=hashes(combined['latent']),
                  policy_config_sha256=p['policy_config_sha256'],old_sha256=digest(out/f'old_{split}.npz'),
                  new_sha256=digest(out/f'new_{split}.npz'),old_count=p['old_counts'][split],
                  new_count=kept,total=TOTALS[split],drawn=len(records)*BATCH,batches=records))
            d=base.done_for_radius(combined['aps'],1.)
            print('COLLECTED',split,'total',len(d),'reached',int(d[:,-1].sum()),
                  'not_reached',int((~d[:,-1]).sum()),flush=True)


def verify_data(out):
    p=json.loads((out/'plan.json').read_text()); datasets={}; seen=set()
    for split in TOTALS:
        prov=json.loads((out/f'{split}_provenance.json').read_text())
        if digest(out/f'{split}.npz')!=prov['sha256']: raise ValueError('Combined data changed')
        if digest(out/f'old_{split}.npz')!=prov['old_sha256'] or digest(out/f'new_{split}.npz')!=prov['new_sha256']:
            raise ValueError('Component data changed')
        selected=[]; remaining=prov['new_count']
        for batch in prov['batches']:
            path=out/batch['raw_file']
            if digest(path)!=batch['sha256']: raise ValueError('Raw batch changed')
            with np.load(path) as a:
                eligible,ids=take_eligible(a['aps'],remaining)
                if ids.tolist()!=batch['selected_indices'] or np.flatnonzero(eligible).tolist()!=batch['eligible_indices']:
                    raise ValueError('Invalid conditional selection')
                if len(ids): selected.append({k:a[k][ids] for k in a.files})
                remaining-=len(ids)
        if remaining: raise ValueError('Incomplete new quota')
        with np.load(out/f'{split}.npz') as a, np.load(out/f'old_{split}.npz') as old, np.load(out/f'new_{split}.npz') as new:
            for k in a.files:
                if not np.array_equal(new[k],np.concatenate([part[k] for part in selected],axis=0)):
                    raise ValueError('Raw new data mismatch')
                if not np.array_equal(a[k],np.concatenate([old[k],new[k]],axis=0)):
                    raise ValueError('Combined arrays mismatch')
            z=base.value_features(a,'policy'); d=base.done_for_radius(a['aps'],1.)
            fp=hashes(a['latent'])
        if fp!=prov['fingerprints'] or len(set(fp))!=len(fp) or seen.intersection(fp):
            raise ValueError('Repeated trajectory in train/development')
        seen.update(fp)
        if len(z)!=p['counts'][split] or z.shape[1:]!=(65,577) or d[:,0].any():
            raise ValueError('Population/schema changed')
        datasets[split]=(z,d)
    print('VERIFIED expanded data: hashes, selection and split disjointness',flush=True)
    return datasets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=base.ROOT/'artifacts/safedreamer_l2_approach1_function_search')
    p.add_argument('--output',type=Path,default=base.ROOT/'artifacts/safedreamer_l2_approach1_more_data')
    p.add_argument('--steps',type=int,default=1600)
    p.add_argument('--phase',choices=['all','train','evaluate'],default='all')
    args=p.parse_args(); torch.set_num_threads(2)
    if args.steps<1: p.error('Positive steps required')
    source,out=args.source.resolve(),args.output.resolve()
    if args.phase=='all': prepare(source,out,args.steps); collect(source,out)
    data=verify_data(out)
    if args.phase in ('all','train'):
        if (out/'frozen.json').exists(): raise FileExistsError('Already frozen')
        plan=json.loads((out/'plan.json').read_text())
        search.train(out,data,plan['candidates'],plan['updates_per_candidate'],refit_scaler=True,pending_penalty=10.)
    search.evaluate(out)


if __name__=='__main__': main()
