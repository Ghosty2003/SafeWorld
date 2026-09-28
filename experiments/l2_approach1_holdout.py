"""Fresh conditional calibration, then prospective independent imagination test.

Uses the frozen approach1 function-search winner without training or selection.
Rejection is based ONLY on decoded initial distance, never on future success.
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
from experiments import l2_achievement64 as base
from experiments.l2_approach1_function_search import predict
from experiments.l2_approach1_outside_v import write
from experiments.l2_latent_prediction_check import digest, now, compare
from experiments.l2_paper_sublevel_finite import hashes

N, BATCH, MAX_BATCHES = 100, 20, 50
SEEDS = dict(calibration=25503000, test=25504000)


def take_eligible(aps,remaining):
    if remaining < 1: raise ValueError('Positive remaining quota required')
    done=base.done_for_radius(aps,1.)
    eligible=~done[:,0]
    return eligible,np.flatnonzero(eligible)[:remaining]


def audit(model,data):
    z=base.value_features(data,'policy')
    done=base.done_for_radius(data['aps'],1.)
    if done[:,0].any(): raise ValueError('Initial condition violated')
    v=predict(model,z,done)
    row=base.summary(v,done,True)
    delta=v[:,:-1]-v[:,1:]
    passes=~((delta<0)|(~done[:,:-1] & (delta<.01))).any(1)
    row['p1p2_given_goal']=base.rate(passes[done[:,-1]],True)
    return row


def verdict(row,threshold=.95):
    lower=row['finite_goal_and_certificate']['cp_lower']
    return dict(required_probability=threshold,confidence=.95,lower_bound=lower,
                result='FINITE_EVENT_THRESHOLD_MET' if lower>=threshold else 'NO_WARRANT',
                scope='Conditional 64-step full certificate AND actual decoded goal event; '
                      'not an infinite-horizon or support-wide warrant')


def prepare(source,out):
    source_plan=json.loads((source/'plan.json').read_text())
    frozen=json.loads((source/'frozen.json').read_text())
    if (source_plan['horizon'],source_plan['eta'],source_plan['radius'],source_plan['action_source'])!=(64,.01,1.,'policy'):
        raise ValueError('Source scope differs')
    if source_plan['initial_condition']!='norm(decoded initial goal vector)>=1.0':
        raise ValueError('Source initial distribution differs')
    if digest(source/'selected.pt')!=frozen['model_sha256'] or digest(source/'plan.json')!=frozen['plan_sha256']:
        raise ValueError('Source model/plan changed')
    if digest(Path(source_plan['checkpoint']))!=source_plan['checkpoint_sha256']:
        raise ValueError('World model changed')
    if digest(source/'policy_config.json')!=source_plan['policy_config_sha256']:
        raise ValueError('Policy changed')
    if source_plan['warrant_threshold']!=.95 or source_plan['future_calibration_confidence']!=.95:
        raise ValueError('Predeclared threshold/confidence changed')
    out.mkdir(exist_ok=False)
    for name in ('selected.pt','policy_config.json'):
        shutil.copyfile(source/name,out/name)
    write(out/'frozen.json',dict(frozen_at=frozen['frozen_at'],source_run=str(source),
          source_plan_sha256=frozen['plan_sha256'],model_sha256=frozen['model_sha256'],
          selected_candidate=frozen['best']['candidate'],step=frozen['best']['step']))
    write(out/'plan.json',dict(created_at=now(),source_run=str(source),
          model_sha256=frozen['model_sha256'],checkpoint=source_plan['checkpoint'],
          checkpoint_sha256=source_plan['checkpoint_sha256'],policy_config_sha256=source_plan['policy_config_sha256'],
          horizon=64,eta=.01,radius=1.,action_source='policy',device='gpu',
          initial_condition=source_plan['initial_condition'],
          spec='F_[0,64](norm(decoded goal vector)<1.0)',counts=dict(calibration=N,test=N),
          batch_size=BATCH,max_batches_per_split=MAX_BATCHES,seeds=SEEDS,
          sampling='Take first 100 eligible paths in chronological draw order per split; '
                   'eligibility uses ONLY decoded t=0 distance>=1.0. Keep raw batches and exclusions.',
          reset_rng_note='New simulator resets per draw; imagination RNG explicit, simulator reset '
                         'RNG not completely controlled by recorded imagination seeds.',
          primary_event='All 64 real model transitions satisfy P1/P2, endpoint V<eta, '
                        'and decoded goal reached within 64 steps',
          warrant_threshold=.95,confidence=.95,
          decision='Compare individual one-sided 95% CP lower bound of the primary event with .95',
          inference='Assumes iid same-distribution draws conditional on initial eligibility; '
                    'other reported bounds are individual, not simultaneous. No real-world guarantee.',
          sequencing='V frozen before calibration; prediction persisted before test; no refitting'))


def prior_fingerprints(source):
    seen=set()
    # Read identity metadata only, not old holdout outcomes or trajectories.
    sources=[source,base.ROOT/'artifacts/safedreamer_l2_policy64_gpu',
             base.ROOT/'artifacts/safedreamer_l2_policy64_gpu_vrefine']
    for directory in sources:
        for p in directory.glob('*_provenance.json'):
            seen.update(json.loads(p.read_text()).get('fingerprints',[]))
    return seen


def collect(source,out,*,audit_fn=None,verdict_fn=None,extra_seen=()):
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    plan=json.loads((out/'plan.json').read_text())
    audit_fn = audit if audit_fn is None else audit_fn
    verdict_fn = verdict if verdict_fn is None else verdict_fn
    event_key=plan.get('certificate_key','finite_goal_and_certificate')
    task_key=plan.get('task_key','goal_reached')
    model=torch.load(out/'selected.pt',weights_only=False)
    extra=build_extra(argparse.Namespace(repo_root=None,checkpoint=plan['checkpoint']))
    extra['action_source']='policy'
    extra['config_overrides']['jax'].update(platform='gpu',logical_gpus=0)
    logger=logging.getLogger('wrappers.safedreamer_wrapper')
    logger.setLevel(logging.INFO)
    if not logger.handlers: logger.addHandler(logging.StreamHandler(sys.stdout))
    seen=prior_fingerprints(source)
    seen.update(extra_seen)
    cfg=RolloutConfig(horizon=64,n_rollouts=BATCH,seed=plan.get('initialization_seed',25500),action_source='policy',extra=extra)
    with SafeDreamerWrapper(cfg) as wrapper:
        wrapper.load()
        import jax
        if jax.default_backend()!='gpu': raise RuntimeError('GPU required; no silent CPU fallback')
        print('IMAGINATION DEVICE',jax.devices(),flush=True)
        actual=dict(action_source='policy',expl_behavior=wrapper._config.expl_behavior,
            planner={k:getattr(wrapper._config.planner,k) for k in
                     ('horizon','num_samples','num_elites','iterations','mixture_coef','momentum','init_std')},
            cost_limit=wrapper._config.cost_limit,
            planner_source_sha256=digest(Path(extra['repo_root'])/'SafeDreamer/behaviors.py'),
            note='Local reconstructed evaluation planner; no claim of exact historical training config.')
        if actual!=json.loads((out/'policy_config.json').read_text()):
            raise ValueError('Runtime planner differs from training data')
        for split in ('calibration','test'):
            if digest(out/'selected.pt')!=plan['model_sha256']: raise ValueError('V changed')
            if split=='test':
                pred=json.loads((out/'prediction.json').read_text())
                if pred['model_sha256']!=plan['model_sha256']: raise ValueError('Prediction uses different V')
                write(out/'test_started.json',dict(started_at=now(),prediction_sha256=digest(out/'prediction.json')))
            start=now(); kept=0; parts=[]; records=[]; initial_rejected=0
            raw_dir=out/f'{split}_raw'; raw_dir.mkdir()
            for batch in range(MAX_BATCHES):
                seed=plan['seeds'][split]+batch
                print('COLLECTING',split,'batch',batch,'seed',seed,'retained',kept,'/',N,flush=True)
                data=wrapper.sample_latent_rollouts(RolloutConfig(
                    horizon=64,n_rollouts=BATCH,seed=seed,action_source='policy',extra=extra))
                if data['action_source']!='policy' or data['latent'].shape!=(BATCH,65,512):
                    raise ValueError('Incorrect rollout schema')
                aps=np.asarray([[[s[k] for k in ('goal_dist','hazard_dist','velocity')] for s in path]
                                for path in data['aps']],dtype=np.float32)
                arrays=dict(latent=data['latent'],aps=aps,decoded=data['decoded'],actions=data['actions'],
                            **data['policy_state'],**{f'rssm_{k}':v for k,v in data['rssm'].items()})
                fingerprints=hashes(arrays['latent'])
                if len(set(fingerprints))!=BATCH or seen.intersection(fingerprints):
                    raise ValueError('Duplicate draw; abort rather than silently resample')
                seen.update(fingerprints)
                eligible,indices=take_eligible(aps,N-kept)
                raw=raw_dir/f'batch_{batch:03d}.npz'; np.savez_compressed(raw,**arrays)
                record=dict(batch=batch,seed=seed,raw_file=str(raw.relative_to(out)),sha256=digest(raw),
                            eligible_indices=np.flatnonzero(eligible).tolist(),selected_indices=indices.tolist(),
                            fingerprints=fingerprints,draw_count=BATCH,
                            initial_rejected=int((~eligible).sum()),
                            eligible_not_needed=int(eligible.sum()-len(indices)))
                write(raw_dir/f'batch_{batch:03d}.json',record); records.append(record)
                if len(indices): parts.append({k:v[indices] for k,v in arrays.items()})
                kept+=len(indices); initial_rejected+=int((~eligible).sum())
                print('PROGRESS',split,'retained',kept,'/',N,'drawn',(batch+1)*BATCH,
                      'initially_inside_excluded',initial_rejected,flush=True)
                if kept==N: break
            if kept!=N: raise RuntimeError('Predeclared draw budget exhausted; no partial-N confidence claim')
            accepted={k:np.concatenate([part[k] for part in parts],axis=0) for k in parts[0]}
            np.savez_compressed(out/f'{split}.npz',**accepted)
            write(out/f'{split}_provenance.json',dict(started_at=start,ended_at=now(),
                  sha256=digest(out/f'{split}.npz'),fingerprints=hashes(accepted['latent']),
                  model_sha256=plan['model_sha256'],policy_config_sha256=plan['policy_config_sha256'],
                  accepted=N,drawn=len(records)*BATCH,initial_rejected=initial_rejected,
                  eligible_not_needed=sum(r['eligible_not_needed'] for r in records),batches=records))
            row=audit_fn(model,accepted)
            print('COLLECTED',split,'C',row[event_key],'task',row[task_key],flush=True)
            if split=='calibration':
                write(out/'prediction.json',dict(saved_at=now(),model_sha256=plan['model_sha256'],
                      plan_sha256=digest(out/'plan.json'),calibration=row,
                      next_n_paths=N,decision=verdict_fn(row)))
                print('FORECAST',verdict_fn(row),flush=True)


def evaluate(out,*,audit_fn=None,verdict_fn=None,extra_seen=()):
    plan=json.loads((out/'plan.json').read_text())
    audit_fn = audit if audit_fn is None else audit_fn
    verdict_fn = verdict if verdict_fn is None else verdict_fn
    frozen=json.loads((out/'frozen.json').read_text())
    pred=json.loads((out/'prediction.json').read_text())
    started=json.loads((out/'test_started.json').read_text())
    if not digest(out/'selected.pt')==plan['model_sha256']==frozen['model_sha256']==pred['model_sha256']:
        raise ValueError('Frozen weights changed')
    if digest(out/'plan.json')!=pred['plan_sha256'] or digest(out/'prediction.json')!=started['prediction_sha256']:
        raise ValueError('Prospective forecast changed')
    if digest(out/'policy_config.json')!=plan['policy_config_sha256']:
        raise ValueError('Policy metadata changed')
    if not frozen['frozen_at']<=pred['saved_at']<=started['started_at']:
        raise ValueError('Prediction must precede test')
    model=torch.load(out/'selected.pt',weights_only=False)
    rows={}; seen=set(extra_seen)
    for split in ('calibration','test'):
        prov=json.loads((out/f'{split}_provenance.json').read_text())
        if digest(out/f'{split}.npz')!=prov['sha256']: raise ValueError('Saved data changed')
        with np.load(out/f'{split}.npz') as a: data={k:a[k] for k in a.files}
        if len(data['latent'])!=N: raise ValueError('Wrong accepted count')
        pieces=[]; remaining=N
        for record in prov['batches']:
            if record['seed'] != plan['seeds'][split]+record['batch']:
                raise ValueError('Recorded seed differs from frozen sampling plan')
            raw=out/record['raw_file']
            if digest(raw)!=record['sha256']: raise ValueError('Raw draw changed')
            with np.load(raw) as a:
                fp=hashes(a['latent'])
                if fp!=record['fingerprints'] or len(set(fp))!=len(fp) or seen.intersection(fp):
                    raise ValueError('Duplicate/changed raw draw')
                seen.update(fp)
                eligible,indices=take_eligible(a['aps'],remaining)
                if indices.tolist()!=record['selected_indices'] or np.flatnonzero(eligible).tolist()!=record['eligible_indices']:
                    raise ValueError('Selection must use initial state only and chronological order')
                if len(indices): pieces.append({k:a[k][indices] for k in a.files})
                remaining-=len(indices)
        if remaining: raise ValueError('Missing selected draws')
        for k in data:
            if not np.array_equal(data[k],np.concatenate([p[k] for p in pieces],axis=0)):
                raise ValueError('Accepted data differ from raw selected draws')
        rows[split]=audit_fn(model,data)
    if rows['calibration']!=pred['calibration'] or verdict_fn(rows['calibration'])!=pred['decision']:
        raise ValueError('Forecast does not reproduce')
    spec_fields=dict(radius=1.) if 'certificate_key' not in plan else dict(spec=plan['spec'])
    report=dict(horizon=64,**spec_fields,eta=.01,initial_condition=plan['initial_condition'],
                **rows,decision=pred['decision'],result=pred['decision']['result'],
                comparisons={k:compare(rows['calibration'][k],rows['test'][k],N) for k in
                    (plan.get('certificate_key','finite_goal_and_certificate'),
                     plan.get('task_key','goal_reached'),'zfree_entered','p1p2_paths')},
                support_wide_validity='NOT_ESTABLISHED',infinite_horizon_warrant='NOT_ESTABLISHED',
                confidence_note=plan['inference'])
    if (out/'report.json').exists():
        if json.loads((out/'report.json').read_text())!=report: raise ValueError('Report differs')
    else: write(out/'report.json',report)
    print('RESULT',json.dumps(dict(decision=report['decision'],comparisons=report['comparisons'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=base.ROOT/'artifacts/safedreamer_l2_approach1_function_search')
    p.add_argument('--output',type=Path,default=base.ROOT/'artifacts/safedreamer_l2_approach1_linear_holdout')
    p.add_argument('--evaluate-only',action='store_true')
    args=p.parse_args(); torch.set_num_threads(2)
    source,out=args.source.resolve(),args.output.resolve()
    if not args.evaluate_only: prepare(source,out); collect(source,out)
    evaluate(out)


if __name__=='__main__': main()
