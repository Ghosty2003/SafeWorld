"""SafeDreamer L3 H1 candidate experiment with explicit paper-premise audit.

This is an executable training/independent-point-validation experiment,
NOT a completed SafeDreamer global proof backend. Unknown global premises
force ABSTAIN even when every sampled transition is safe or drift-valid.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.l3_safedreamer_clear_pipeline import digest,write,stamp,make_sampler,collect
from core.lbsm.sampled_recurrence import (fit_h1,drift_diagnostics,formal_status,
                                        NormalizedBoundedCertificate)


def load_batches(out,role):
    from core.lbsm.operator_model import AnchorSuccessorBatch
    result=[]
    for meta in sorted((out/role).glob('path_*.json')):
        record=json.loads(meta.read_text())
        if digest(meta.with_suffix('.npz')) != record['sha256']: raise ValueError('Path hash changed')
        if not record['accepted']: continue
        file=meta.parent/record['query']
        if digest(file)!=record['query_sha256']: raise ValueError('Successor batch hash changed')
        with np.load(file,allow_pickle=False) as a:
            result.append(AnchorSuccessorBatch(a['anchor'].copy(),a['successors'].copy(),
                         f"{role}:{record['seed']}:{record['anchor_time']}",'paper_H1'))
    return result


def evaluate(out):
    import torch
    plan=json.loads((out/'plan.json').read_text())
    freeze=json.loads((out/'frozen.json').read_text())
    if digest(out/'certificates.pt')!=freeze['certificate_sha256'] or digest(out/'plan.json')!=freeze['plan_sha256']:
        raise ValueError('Frozen model/plan changed')
    for rel,sha in plan['code_sha256'].items():
        if digest(ROOT/rel)!=sha: raise ValueError(f'Frozen source changed: {rel}')
    started=json.loads((out/'validation/started.json').read_text())
    if started['time']<=freeze['time']: raise ValueError('Validation did not start after freeze')
    state=torch.load(out/'certificates.pt',map_location='cpu',weights_only=True)
    W=NormalizedBoundedCertificate(state['W']['mean'].numpy(),state['W']['scale'].numpy())
    U=NormalizedBoundedCertificate(state['U']['mean'].numpy(),state['U']['scale'].numpy())
    W.load_state_dict(state['W']); U.load_state_dict(state['U']); W.eval(); U.eval()
    train=load_batches(out,'certificate'); val=load_batches(out,'validation')
    if len(train)!=plan['train_paths'] or len(val)!=plan['validation_paths']:
        raise ValueError('Incomplete data roles')
    train_stats=drift_diagnostics(W,U,train,fit_batches=train,independent=False)
    val_stats=drift_diagnostics(W,U,val,fit_batches=train)
    report=dict(spec=plan['spec'],scope=plan['scope'],smoke_only=plan['smoke_only'],
                train_drift=train_stats,independent_point_validation=val_stats,
                paper_L3=formal_status(val_stats,independent_frozen=True),
                validation_finite_behavior=json.loads((out/'validation/summary.json').read_text()))
    if (out/'report.json').exists():
        if json.loads((out/'report.json').read_text())!=report: raise ValueError('Report replay mismatch')
    else: write(out/'report.json',report)
    print('POINT_VALIDATION',json.dumps({k:v for k,v in val_stats.items() if k!='rows'}),flush=True)
    print('PAPER_L3',json.dumps(report['paper_L3']),flush=True)
    behavior=report['validation_finite_behavior']
    print('FINITE_BEHAVIOR',json.dumps({k:v for k,v in behavior.items() if k!='per_path'}),flush=True)
    return report


def run(args):
    import torch
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    torch.set_num_threads(2)
    source=ROOT/'artifacts/safedreamer_l2_clear_window_holdout'
    old=json.loads((source/'plan.json').read_text())
    if digest(old['checkpoint'])!=old['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    if digest(source/'policy_config.json')!=old['policy_config_sha256']: raise ValueError('Policy changed')
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    horizon,kappa,n_train,n_val,epochs=(60,4,2,3,5) if args.smoke else (192,32,12,20,500)
    base=35500000 if args.smoke else 36500000
    plan=dict(created=stamp(),spec='GF G[0,47](decoded_hazard_margin>=0)',
              scope='MODEL_ONLY_PAPER_H1_CANDIDATE_AND_POINT_VALIDATION_NOT_GLOBAL_PROOF',
              smoke_only=args.smoke,horizon=horizon,kappa=kappa,train_paths=n_train,validation_paths=n_val,
              epochs=epochs,seed_base=base,checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],
              policy=json.loads((source/'policy_config.json').read_text()),
              retention='H1: U checked at ALL sampled source states in C, including accepting states',
              region='C={U<=0.8}; candidate only, no certified covering/containment',
              ell=.8,eps_W=.01,training_eps_U=.001,validation_U_margin=0.,
              delta_pointwise=.05,initial_distribution='decoded initial goal distance>=1.0',
              anchor_distribution='uniform t in [1,H-1], one anchor per retained path',
              labels='decoder minimum hazard-vector norm minus 0.20 >=0',
              independence='freeze W/U before validation paths; fresh path/branch seeds; no tuning on validation',
              caveats=['Simulator reset RNG is not fully controlled by the explicit imagination seed.',
                       'Sampled successors include exits; no successor truncation/rejection.',
                       'No global closed-loop envelope, covering, containment, collar or initial-support certificate.',
                       'Pointwise confidence cannot be substituted for recurrence probability.',
                       'Small pilot sample counts measure interface/diagnostics, not benchmark performance.'])
    code=[Path(__file__).resolve(),ROOT/'experiments/l3_safedreamer_clear_pipeline.py',
          ROOT/'experiments/l3_safedreamer_clear_recurrence.py',ROOT/'wrappers/safedreamer_wrapper.py',
          ROOT/'core/lbsm/sampled_recurrence.py',ROOT/'core/lbsm/operator_model.py']
    plan['code_sha256']={str(p.relative_to(ROOT)):digest(p) for p in code}
    write(out/'plan.json',plan)
    extra=build_extra(argparse.Namespace(repo_root=None,checkpoint=old['checkpoint']))
    extra['action_source']='policy'; extra['config_overrides']['jax'].update(platform='gpu',logical_gpus=0)
    logging.basicConfig(level=logging.INFO)
    with SafeDreamerWrapper(RolloutConfig(horizon=horizon,n_rollouts=1,seed=base,action_source='policy',extra=extra)) as wrapper:
        wrapper.load()
        import jax
        if jax.default_backend()!='gpu': raise RuntimeError('GPU required')
        actual=dict(action_source='policy',expl_behavior=wrapper._config.expl_behavior,
                    planner={k:getattr(wrapper._config.planner,k) for k in plan['policy']['planner']},
                    cost_limit=wrapper._config.cost_limit,
                    planner_source_sha256=digest(Path(extra['repo_root'])/'SafeDreamer/behaviors.py'),
                    note=plan['policy']['note'])
        if actual!=plan['policy']: raise ValueError('Policy configuration changed')
        print('GPU',jax.devices(),'HORIZON',horizon,'TRAIN/VAL',n_train,n_val,'KAPPA',kappa,flush=True)
        sampler=make_sampler(wrapper,kappa); seen=set()
        train,core=collect(wrapper,extra,sampler,out,'certificate',n_train,base+1000,horizon,seen)
        W,U,history=fit_h1(train,core,epochs=epochs,seed=base+31)
        torch.save(dict(W=W.state_dict(),U=U.state_dict()),out/'certificates.pt')
        write(out/'frozen.json',dict(time=stamp(),certificate_sha256=digest(out/'certificates.pt'),
                                    plan_sha256=digest(out/'plan.json'),loss_history=history))
        print('CERTIFICATES_FROZEN',out/'certificates.pt',flush=True)
        collect(wrapper,extra,sampler,out,'validation',n_val,base+2000,horizon,seen)
    evaluate(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--evaluate-only',action='store_true')
    args=p.parse_args()
    if args.evaluate_only: evaluate(args.output.resolve())
    else: run(args)


if __name__=='__main__': main()
