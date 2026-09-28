"""Fresh FIT ONLY SafeDreamer reset-budget pilot; no calibration or Test1.

Independent entry point. Never imports TD-MPC2 or changes old L2/L3 artifacts.
M_MIN remains unset; 3/4/5/6 are fit-only sensitivity views, not selected specs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.finite_recurrence import safe_window_monitor,reference_events,segments,path_score,budget_diagnostics,combine_gates
from experiments.l3_safedreamer_clear_pipeline import digest,write,stamp

ROLE_BASES={'fit':121000000,'cal_delta':122000000,'cal_CP':123000000,'test':124000000}
CHECKPOINT=Path('/home/sunyhg/Documents/SafeDreamer/checkpoint/20240307-010600_osrp_vector_safetygymcoor_SafetyPointGoal1-v0_0.ckpt')


def seed_pair(role,index):
    if role not in ROLE_BASES or not 0<=index<100000: raise ValueError('Invalid role/draw')
    base=ROLE_BASES[role]+2*index
    return base,base+1


class SeededResetRecorder:
    """Instance-only shim. Preserve policy/model and seed the ACTUAL gym reset."""
    def __init__(self,outer):
        self.outer=outer; node=outer; seen=set()
        while type(node).__name__!='SafetyGymCoor':
            children=vars(node)
            key='env' if 'env' in children else '_env'
            if id(node) in seen or key not in children: raise RuntimeError('Unknown env wrapper chain')
            seen.add(id(node)); node=children[key]
        self.gym=node._dmenv; self.original_reset=self.gym.reset; self.pending=None
        self.observation=None; self.reset_calls=0
        def reset(*args,**kwargs):
            if self.pending is None: raise RuntimeError('Unseeded reset prohibited')
            kwargs['seed']=self.pending; self.pending=None; self.reset_calls+=1
            return self.original_reset(*args,**kwargs)
        self.gym.reset=reset

    def __getattr__(self,name): return getattr(self.outer,name)

    def step(self,action):
        if not bool(action['reset']): raise RuntimeError('Pilot may not advance real environment')
        obs=self.outer.step(action); self.observation=np.asarray(obs['observation']).copy()
        return obs

    def restore(self): self.gym.reset=self.original_reset


def extract(data,obs0):
    return dict(latent=data['latent'][0],decoded=data['decoded'][0],actions=data['actions'][0],
        initial_observation=obs0,**{k:v[0] for k,v in data['policy_state'].items()},
        **{'rssm_'+k:v[0] for k,v in data['rssm'].items()})


def hazard_mask(decoded):
    decoded=np.asarray(decoded)
    distances=np.linalg.norm(decoded[:,9:25].reshape(-1,8,2),axis=-1)
    return np.isfinite(distances).all(1)&(distances.min(1)>=.2)


def features(arrays,counter):
    n=len(counter)
    return np.concatenate([arrays['latent'],arrays['planner_action_mean'].reshape(n,-1),
        arrays['planner_action_std'].reshape(n,-1),arrays['planner_initialized'].reshape(n,1),
        np.asarray(counter)[:,None]/48],axis=1).astype(np.float32)


class BudgetNet(nn.Module):
    def __init__(self,mean,scale):
        super().__init__(); self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(len(mean),64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1),nn.Softplus())
    def forward(self,x): return self.net((x-self.mean)/self.scale).squeeze(-1)


def train_budget(paths,n_train,updates):
    core=np.concatenate([p['x'] for p in paths[:n_train]])
    mean=core.mean(0); scale=np.maximum(core.std(0),.05); mean[-2:]=0; scale[-2:]=1
    xs=[]; ys=[]; censored=[]
    for p in paths[:n_train]:
        for s,e,completed in segments(p['monitor']['event_times'],len(p['x'])-1):
            xs.extend(p['x'][s:e]); ys.extend(.01*np.arange(e-s,0,-1)); censored.extend([not completed]*(e-s))
    torch.manual_seed(917)
    net=BudgetNet(mean,scale); opt=torch.optim.Adam(net.parameters(),lr=.001)
    x=torch.tensor(np.stack(xs)); y=torch.tensor(ys,dtype=torch.float32); c=torch.tensor(censored)
    history=[]
    for step in range(updates):
        idx=torch.randperm(len(x))[:256]; pred=net(x[idx]); residual=pred-y[idx]
        loss=torch.where(c[idx],torch.relu(-residual).square(),residual.square()).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step==0 or (step+1)%100==0: history.append(dict(step=step+1,loss=float(loss)))
    standardized=(core-mean)/scale
    radius=float(np.linalg.norm(standardized,axis=1).max())
    return net.eval(),dict(mean=mean,scale=scale,radius=radius),dict(history=history,
        completed_targets=int((~c).sum()),censored_lower_targets=int(c.sum()),
        note='Right-censored targets bound observed tail only, not an unobserved return time')


def region_membership(x,region):
    score=np.linalg.norm((x-region['mean'])/region['scale'],axis=1)
    return np.isfinite(x).all(1)&(score<=region['radius']+1e-5)&(x[:,-1]>=0)&(x[:,-1]<1)


def require_fit_stage(role):
    if role!='fit': raise RuntimeError('M_MIN and final certificate definition are not frozen; calibration/test forbidden')


def run(args):
    require_fit_stage(args.role); torch.set_num_threads(2)
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    n_train=args.n_fit*3//4
    reference=json.loads((ROOT/'artifacts/safedreamer_l2_clear_window_holdout/policy_config.json').read_text())
    files=[Path(__file__).resolve(),ROOT/'core/finite_recurrence.py',ROOT/'wrappers/safedreamer_wrapper.py',
           Path('/home/sunyhg/Documents/SafeDreamer/SafeDreamer/behaviors.py')]
    plan=dict(created=stamp(),stage='FIT_ONLY',backend='SafeDreamer',checkpoint=str(CHECKPOINT),
        checkpoint_sha256=digest(CHECKPOINT),policy=reference,horizon=300,window=48,eta=.01,
        detector='Decoded nearest-hazard distance >=0.2; consume states1..300; nonoverlapping completion pulses',
        budget_reset='Only completed events; hazard resets counter but NOT progress budget',
        start_distribution='fresh independently seeded environment reset, encoded initial goal distance>=1.0',
        fit_count=args.n_fit,fit_internal_train=n_train,fit_internal_check=args.n_fit-n_train,updates=args.updates,
        M_MIN=None,M_MIN_candidates=[3,4,5,6],CV='DISABLED_FOR_PILOT',sliding='DISABLED_FOR_PILOT',
        false_positive_deduction='NOT_APPLICABLE: event defined on decoded observation; no real-world detector claim',
        region='FIT_TRAIN_ENVELOPE: normalized full-state L2 ball containing training states; NOT support-wide validated',
        delta='NOT_CALIBRATED: show zero and max training-path residual as explicitly biased fit previews only',
        certificate_event=None,confidence=None,roles={r:dict(seed_base=b,status='FIT_ONLY' if r=='fit' else 'LOCKED_NOT_COLLECTED',
            count=args.n_fit if r=='fit' else 0) for r,b in ROLE_BASES.items()},
        scope='FINITE_MODEL_ONLY_RESET_BUDGET_PILOT_NOT_THEOREM_5_6',
        code_sha256={str(p):digest(p) for p in files})
    write(out/'plan.json',plan)
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    extra=build_extra(argparse.Namespace(repo_root=None,checkpoint=str(CHECKPOINT)))
    extra['action_source']='policy'; extra['config_overrides']['jax'].update(platform='gpu',logical_gpus=0)
    raw=out/'fit'; raw.mkdir(); paths=[]; seen=set(); started=time.perf_counter(); replay=None
    with SafeDreamerWrapper(RolloutConfig(horizon=300,n_rollouts=1,seed=seed_pair('fit',0)[1],action_source='policy',extra=extra)) as wrapper:
        wrapper.load()
        import jax
        if jax.default_backend()!='gpu': raise RuntimeError('GPU required')
        actual=dict(action_source='policy',expl_behavior=wrapper._config.expl_behavior,
            planner={k:getattr(wrapper._config.planner,k) for k in reference['planner']},
            cost_limit=wrapper._config.cost_limit,planner_source_sha256=digest(Path(extra['repo_root'])/'SafeDreamer/behaviors.py'),
            note=reference['note'])
        if actual!=reference: raise RuntimeError('Current policy changed')
        recorder=SeededResetRecorder(wrapper._env); wrapper._env=recorder
        try:
            for draw in range(args.n_fit*5):
                reset_seed,imagination_seed=seed_pair('fit',draw)
                def sample():
                    recorder.pending=reset_seed
                    d=wrapper.sample_latent_rollouts(RolloutConfig(horizon=300,n_rollouts=1,
                        seed=imagination_seed,action_source='policy',extra=extra))
                    if recorder.pending is not None or d['action_source']!='policy': raise RuntimeError('Wrong reset/action source')
                    return extract(d,recorder.observation)
                a=sample()
                if a['latent'].shape!=(301,512) or a['actions'].shape!=(300,2): raise RuntimeError('Wrong rollout schema')
                if draw==0:
                    repeated=sample(); exact={k:bool(np.array_equal(v,repeated[k])) for k,v in a.items()}
                    replay=dict(reset_seed=reset_seed,imagination_seed=imagination_seed,exact=exact,all_exact=all(exact.values()))
                    write(out/'seed_replay.json',replay)
                    if not replay['all_exact']: raise RuntimeError('Seed replay failed; stop before fitting')
                fp=hashlib.sha256(a['latent'].tobytes()).hexdigest()
                if fp in seen: raise RuntimeError('Duplicate draw')
                seen.add(fp)
                initial_goal=float(np.linalg.norm(a['decoded'][0,7:9])); eligible=initial_goal>=1.
                file=raw/f'path_{draw:04d}.npz'; np.savez_compressed(file,**a)
                record=dict(draw=draw,reset_seed=reset_seed,imagination_seed=imagination_seed,
                    sha256=digest(file),fingerprint=fp,initial_goal=initial_goal,eligible=bool(eligible),
                    fit_index=len(paths) if eligible else None,role='fit',time=stamp())
                write(file.with_suffix('.json'),record)
                if eligible:
                    mon=safe_window_monitor(hazard_mask(a['decoded']))
                    paths.append(dict(arrays=a,monitor=mon,x=features(a,mon['counter']),record=record,file=file.name))
                print('FIT DRAW',draw,'retained',len(paths),'/',args.n_fit,
                      'events',paths[-1]['monitor']['event_count'] if eligible else 'initial-rejected',flush=True)
                if len(paths)==args.n_fit: break
        finally:
            recorder.restore(); wrapper._env=recorder.outer
    if len(paths)!=args.n_fit: raise RuntimeError('Fit-only draw budget exhausted')
    net,region,training=train_budget(paths,n_train,args.updates)
    model=out/'g_init.pt'; torch.save(dict(state=net.state_dict(),region_radius=region['radius']),model)
    predictions=[]
    with torch.no_grad():
        for p in paths: predictions.append(net(torch.tensor(p['x'])).numpy())
    delta_fit=max(path_score(pred,p['monitor']['event_times']) for p,pred in zip(paths[:n_train],predictions[:n_train]))
    write(out/'fit_model.json',dict(time=stamp(),sha256=digest(model),training=training,
        region_radius=region['radius'],delta_fit_preview=delta_fit,calibrated_delta=None,
        M_MIN=None,not_a_final_freeze=True))
    rows=[]
    for i,(p,pred) in enumerate(zip(paths,predictions)):
        mon=p['monitor']; ref=reference_events(hazard_mask(p['arrays']['decoded']))
        row=dict(fit_index=i,file=p['file'],fit_partition='train' if i<n_train else 'internal_check',
            **{k:v for k,v in mon.items() if k!='counter'},previews={})
        for label,delta in [('delta_zero',0.),('delta_fit_only',delta_fit)]:
            flags=budget_diagnostics(pred,mon['event_times'],ref,delta,region_membership(p['x'],region))
            row['previews'][label]=dict(delta=delta,flags=flags,by_M_MIN={str(m):combine_gates(flags,mon['event_count'],m) for m in (3,4,5,6)})
        # Honest sanity baseline: constant H*eta budget trivially covers each
        # remaining finite segment. Same event/region gates; no learning claim.
        constant=budget_diagnostics(np.full(301,3.),mon['event_times'],ref,0.,region_membership(p['x'],region))
        row['constant_H_eta_baseline']={str(m):combine_gates(constant,mon['event_count'],m) for m in (3,4,5,6)}
        rows.append(row)
    summary={}
    for partition in ('all','train','internal_check'):
        selected=[r for r in rows if partition=='all' or r['fit_partition']==partition]
        summary[partition]=dict(n=len(selected),event_count_distribution=dict(Counter(r['event_count'] for r in selected)),
            previews={label:{str(m):dict(successes=sum(r['previews'][label]['by_M_MIN'][str(m)]['fit_preview_C_rec'] for r in selected),
                n=len(selected),failure_breakdown=dict(Counter(reason for r in selected for reason in r['previews'][label]['by_M_MIN'][str(m)]['failure_reasons'])))
                for m in (3,4,5,6)} for label in ('delta_zero','delta_fit_only')})
    write(out/'report.json',dict(completed=stamp(),elapsed_seconds=time.perf_counter()-started,rows=rows,summary=summary,
        M_MIN=None,calibrated_delta=None,confidence=None,certificate_probability_lower_bound=None,
        formal_status='PENDING_USER_M_MIN_CHOICE_AND_INDEPENDENT_CALIBRATION',
        split_counts={'fit':len(paths),'cal_delta':0,'cal_CP':0,'test':0},seed_replay=replay,
        limitations=['All rates are fit-only diagnostics, not warrants.',
                    'Region is an empirical fit envelope, not a proved invariant set.',
                    'CV/sliding disabled for this pilot; their future definitions require freeze.',
                    'Budget drift is by construction, not learned pointwise g descent.']))
    print('FIT SUMMARY',json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--role',default='fit',choices=list(ROLE_BASES))
    p.add_argument('--n-fit',type=int,default=16); p.add_argument('--updates',type=int,default=400)
    args=p.parse_args()
    if args.n_fit<4 or args.updates<1: p.error('At least4 fit paths and positive updates')
    run(args)
