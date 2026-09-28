"""Real-checkpoint CCEPlanner smoke test; NOT certificate calibration."""
import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from configs.settings import RolloutConfig
from experiments.l2_safedreamer import build_extra
from experiments.l2_achievement64 import value_features
from wrappers.safedreamer_wrapper import SafeDreamerWrapper


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--horizon',type=int,default=2)
    p.add_argument('--device',choices=['cpu','gpu'],default='cpu')
    p.add_argument('--repeat',type=int,default=1)
    args=p.parse_args()
    if args.horizon<1: p.error('positive horizon required')
    if args.repeat<1: p.error('positive repeat count required')
    suffix='' if args.device=='cpu' else '_gpu'
    out=Path(__file__).resolve().parents[1]/'artifacts'/f'safedreamer_policy_smoke{suffix}_h{args.horizon}'
    out.mkdir(exist_ok=False)
    logging.basicConfig(level=logging.INFO)
    extra=build_extra(argparse.Namespace(repo_root=None,checkpoint=None))
    extra['action_source']='policy'
    extra['config_overrides']['jax']['platform']=args.device
    if args.device=='gpu': extra['config_overrides']['jax']['logical_gpus']=0
    cfg=RolloutConfig(horizon=args.horizon,n_rollouts=1,seed=19501,action_source='policy',extra=extra)
    started=time.monotonic()
    with SafeDreamerWrapper(cfg) as w:
        w.load()
        import jax
        devices=[str(d) for d in jax.devices()]
        if jax.default_backend()!=args.device:
            raise RuntimeError(f'Requested {args.device}, got {jax.default_backend()}')
        datasets=[]; timings=[]
        for i in range(args.repeat):
            loaded=time.monotonic()
            datasets.append(w.sample_latent_rollouts(RolloutConfig(horizon=args.horizon,n_rollouts=1,
                seed=19501+i,action_source='policy',extra=extra)))
            timings.append(time.monotonic()-loaded)
            print('TIMING',i+1,timings[-1],flush=True)
        data=datasets[0]; elapsed=timings[0]
        planner={k:getattr(w._config.planner,k) for k in
                 ('horizon','num_samples','num_elites','iterations','mixture_coef','momentum','init_std')}
        cost_limit=w._config.cost_limit
    h=args.horizon
    for i,trial in enumerate(datasets):
        assert trial['action_source']=='policy'
        assert trial['latent'].shape==(1,h+1,512)
        assert np.isfinite(trial['latent']).all() and np.isfinite(trial['actions']).all()
        assert np.array_equal(trial['actions'],trial['policy_state']['planner_action_mean'][:,1:,0,:])
        assert np.abs(trial['actions']).max()<=1.00001
        assert value_features(dict(latent=trial['latent'],**trial['policy_state']),'policy').shape==(1,h+1,577)
        if i:
            np.savez_compressed(out/f'rollout_{i+1}.npz',latent=trial['latent'],decoded=trial['decoded'],
                actions=trial['actions'],**trial['policy_state'],**{f'rssm_{k}':v for k,v in trial['rssm'].items()})
    assert data['action_source']=='policy'
    assert data['latent'].shape==(1,h+1,512)
    assert data['actions'].shape==(1,h,2)
    assert np.isfinite(data['latent']).all() and np.isfinite(data['actions']).all()
    assert (np.abs(data['actions'])<=1.00001).all()
    state=data['policy_state']
    assert not state['planner_initialized'][:,0].any()
    assert state['planner_initialized'][:,1:].all()
    assert np.array_equal(data['actions'],state['planner_action_mean'][:,1:,0,:])
    features=value_features(dict(latent=data['latent'],**state),'policy')
    assert features.shape==(1,h+1,577)
    np.savez_compressed(out/'rollout.npz',latent=data['latent'],decoded=data['decoded'],
                        actions=data['actions'],**state,**{f'rssm_{k}':v for k,v in data['rssm'].items()})
    report=dict(horizon=h,checkpoint=extra['checkpoint_path'],action_source=data['action_source'],
        planner=planner,cost_limit=cost_limit,features_shape=list(features.shape),
        device=args.device,jax_devices=devices,per_path_seconds=timings,
        actual_actions_match_each_planned_first_action=True,planner_memory_preserved=True,
        first_actions=data['actions'][0,:min(h,4)].tolist(),imagination_and_compile_seconds=elapsed,
        total_seconds=time.monotonic()-started,scope='Integration smoke only; not success-rate or certificate evidence')
    with (out/'report.json').open('x') as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
