"""Fresh model-only, distribution-scoped L3 pilot based on main's W/U core.

Does NOT establish infinite recurrence. A fixed independent path supplies
one uniform-time anchor; repeated independent one-step queries reset BOTH
RSSM and planner carry. Four path-disjoint roles, no model selection on cal.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.l3_safedreamer_clear_recurrence import monitor, summarize


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(path, obj):
    def clean(x):
        if isinstance(x, dict): return {k:clean(v) for k,v in x.items()}
        if isinstance(x, (tuple,list,set,frozenset)): return [clean(v) for v in x]
        if isinstance(x, float) and not np.isfinite(x): return str(x)
        return x
    with Path(path).open('x') as f: json.dump(clean(obj), f, indent=2, allow_nan=False)


def stamp(): return datetime.now(timezone.utc).isoformat()


def product(latent, mean, std, initialized, count):
    # Full 512 RSSM coordinates + 32 mean + 32 std + 1 initialized + 1 counter.
    x = np.concatenate([np.asarray(latent).reshape(-1), np.asarray(mean).reshape(-1),
                        np.asarray(std).reshape(-1), [initialized, count/48.]])
    if x.shape != (578,) or not np.isfinite(x).all() or not 0 <= count <= 48:
        raise ValueError('Invalid complete product state')
    return x.astype(np.float32)


def accepting(x): return bool(x[-1] >= 1.)
def retained(x): return bool(x[-1] >= .5)  # I: 24 consecutive safe states; F subset I.


def make_sampler(wrapper, kappa):
    import jax
    import jax.numpy as jnp
    import SafeDreamer.ninjax as nj
    raw = wrapper._agent.agent

    def sample(latent, planner_state):
        def one(carry, unused):
            # Crucial: each query starts from SAME latent/carry, not the last successor.
            out, state = raw.expl_behavior.policy(latent, planner_state)
            z = raw.wm.rssm.img_step(latent, out['action'])
            obs = raw.wm.heads['decoder'](z)['observation'].mode()
            return carry, (z, state, obs, out['action'])
        return nj.scan(one, jnp.zeros((), dtype=jnp.int32), jnp.arange(kappa))[1]

    return nj.jit(nj.pure(sample))


def query(wrapper, sampler, arrays, t, count, seed):
    import jax
    latent = {k[5:]:jax.device_put(v[t][None]) for k,v in arrays.items() if k.startswith('rssm_')}
    carry = {k:jax.device_put(arrays['planner_'+k][t]) for k in ('action_mean','action_std')}
    key = jax.device_put(np.array([seed, seed ^ 0x13579], dtype=np.uint32))
    (z, state, obs, actions), _ = sampler(wrapper._agent.varibs, key, latent, carry)
    z, state, obs, actions = jax.device_get((z, state, obs, actions))
    obs = np.asarray(obs)[:,0]
    margin = np.linalg.norm(obs[:,9:25].reshape(-1,8,2), axis=-1).min(-1).astype(float)-.2
    next_count = np.where(margin >= 0, min(count+1,48), 0)
    latents = np.concatenate([z['deter'][:,0], z['stoch'][:,0].reshape(len(obs),-1)], axis=-1)
    products = np.stack([product(latents[i], state['action_mean'][i],state['action_std'][i],1,next_count[i])
                         for i in range(len(obs))])
    return products, dict(decoded=obs, actions=np.asarray(actions)[:,0],
                          **{'rssm_'+k:np.asarray(v)[:,0] for k,v in z.items()},
                          **{'planner_'+k:np.asarray(v) for k,v in state.items()})


def collect(wrapper, extra, sampler, out, name, n, base, horizon, seen):
    from configs.settings import RolloutConfig
    from core.lbsm.operator_model import AnchorSuccessorBatch
    batches, cores, safes = [], [], []
    folder = out/name; folder.mkdir()
    scope = f'SafeDreamer_CCE_goal_initial_ge1_H{horizon}_uniform_anchor_t1_to{horizon-1}_counter48'
    write(folder/'started.json', dict(time=stamp(), role=name, requested=n, seed_base=base))
    for draw in range(n*5):
        if len(batches) == n: break
        seed = base + draw
        data = wrapper.sample_latent_rollouts(RolloutConfig(
            horizon=horizon, n_rollouts=1, seed=seed, action_source='policy', extra=extra))
        arrays = dict(latent=data['latent'][0], decoded=data['decoded'][0], actions=data['actions'][0],
                      **{k:v[0] for k,v in data['policy_state'].items()},
                      **{'rssm_'+k:v[0] for k,v in data['rssm'].items()})
        if arrays['latent'].shape != (horizon+1,512): raise ValueError('Invalid path shape')
        path_hash = hashlib.sha256(arrays['latent'].tobytes()).hexdigest()
        if path_hash in seen: raise ValueError('Repeated path across data roles')
        seen.add(path_hash)
        rawfile = folder/f'path_{draw:04d}.npz'; np.savez_compressed(rawfile, **arrays)
        initial_goal = float(np.linalg.norm(arrays['decoded'][0,7:9]))
        if initial_goal < 1.:
            write(rawfile.with_suffix('.json'), dict(seed=seed, accepted=False, initial_goal=initial_goal,
                                                   sha256=digest(rawfile)))
            continue
        margin = np.linalg.norm(arrays['decoded'][:,9:25].reshape(-1,8,2), axis=-1).min(-1).astype(float)-.2
        safe = margin >= 0
        counter = monitor(safe)['counter']
        # Independent PRNG stream, not selected on future labels or certificate quality.
        t = int(np.random.default_rng(seed+700000).integers(1,horizon))
        xs = np.stack([product(z, arrays['planner_action_mean'][j],arrays['planner_action_std'][j],
                               arrays['planner_initialized'][j].item(), counter[j])
                       for j,z in enumerate(arrays['latent'])])
        succ, rawsucc = query(wrapper, sampler, arrays, t, counter[t], seed+900000)
        if not batches and name == 'certificate':
            replay, _ = query(wrapper, sampler, arrays, t, counter[t], seed+900000)
            np.testing.assert_array_equal(succ,replay)
        queryfile = folder/f'query_{len(batches):04d}.npz'
        np.savez_compressed(queryfile, anchor=xs[t], successors=succ, **rawsucc)
        write(rawfile.with_suffix('.json'), dict(seed=seed, accepted=True, initial_goal=initial_goal,
             sha256=digest(rawfile), path_fingerprint=path_hash, anchor_time=t, counter=counter[t],
             query=queryfile.name, query_sha256=digest(queryfile), query_seed=seed+900000))
        batches.append(AnchorSuccessorBatch(xs[t],succ, f'{name}:{seed}:{t}',scope))
        if name == 'certificate': cores.append(xs)
        safes.append(safe)
        print('COLLECTED', name, len(batches), '/', n, 'draw',draw+1,'anchor_t',t,flush=True)
    if len(batches) != n: raise RuntimeError('Initial-only rejection budget exhausted')
    write(folder/'summary.json', dict(completed=stamp(), **summarize(np.stack(safes))))
    return batches, np.concatenate(cores) if cores else None


def run(args):
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    from core.lbsm.distributional_trainer import fit_distributional_certificates
    from core.lbsm.operator_model import fit_post_expectation_models
    from core.lbsm.operator_calibrator import calibrate_post_expectation_models
    from core.lbsm.distributional_verifier import verify_distributional_l3
    import torch
    torch.set_num_threads(2)
    source = ROOT/'artifacts/safedreamer_l2_clear_window_holdout'
    old = json.loads((source/'plan.json').read_text())
    if digest(old['checkpoint']) != old['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    if digest(source/'policy_config.json') != old['policy_config_sha256']: raise ValueError('Policy changed')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    counts = (2,2,2,2) if args.smoke else (40,40,100,120)
    horizon, kappa = (60,2) if args.smoke else (192,32)
    seeds = (32501000,32502000,32503000,32504000) if args.smoke else (33501000,33502000,33503000,33504000)
    epochs = 3 if args.smoke else 500
    plan = dict(created=stamp(), spec='GF G[0,47](decoded_hazard_margin>=0)',
                main_reference='055ab637d35114e7f81099ba17eabadd407996ad',
                scope='DISTRIBUTIONAL_DRIFT_ONLY_NOT_INFINITE_RECURRENCE_PROOF',
                horizon=horizon,kappa=kappa,counts=dict(zip(('certificate','operator','residual','warrant'),counts)),
                seeds=list(seeds),checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],
                policy=json.loads((source/'policy_config.json').read_text()),
                initial_condition='decoded goal distance >= 1.0; no future filtering',
                anchor='one uniform time in [1,H-1] per independent rollout',
                acceptance='counter=48; hazard resets counter; nonabsorbing',
                retention_I='counter>=24; explicit product-state set, not proven invariant',
                eps_W=.01,eps_U=.01,B_W=1.,B_U=1.,gamma=.05,theta=.95,
                alpha_W=.01,alpha_U=.01,delta_mc_W=.001,delta_mc_U=.001,
                training_epochs=epochs,hidden_dim=64,core_inside_upper=.2,core_boundary_lower=.8,
                limitations=['Simulator reset randomness not fully controlled by imagination seeds.',
                             'Monte Carlo correction may be conservative at kappa=32.',
                             'Independent roles; no tuning after residual calibration.',
                             'Retention/acceptance membership bypasses respective drift checks in main.',
                             'Finite empirical window rates are not infinite GF probabilities.'])
    plan['code_sha256'] = {str(p.relative_to(ROOT)):digest(p) for p in
                          [Path(__file__),ROOT/'wrappers/safedreamer_wrapper.py',
                           ROOT/'experiments/l3_safedreamer_clear_recurrence.py',
                           *sorted((ROOT/'core/lbsm').glob('*.py'))]}
    write(out/'plan.json',plan)
    extra = build_extra(argparse.Namespace(repo_root=None,checkpoint=old['checkpoint']))
    extra['action_source']='policy'; extra['config_overrides']['jax'].update(platform='gpu',logical_gpus=0)
    logging.basicConfig(level=logging.INFO)
    with SafeDreamerWrapper(RolloutConfig(horizon=horizon,n_rollouts=1,seed=seeds[0],action_source='policy',extra=extra)) as wrapper:
        wrapper.load()
        import jax
        if jax.default_backend() != 'gpu': raise RuntimeError('GPU required')
        actual=dict(action_source='policy',expl_behavior=wrapper._config.expl_behavior,
            planner={k:getattr(wrapper._config.planner,k) for k in plan['policy']['planner']},
            cost_limit=wrapper._config.cost_limit,
            planner_source_sha256=digest(Path(extra['repo_root'])/'SafeDreamer/behaviors.py'),
            note=plan['policy']['note'])
        if actual != plan['policy']: raise ValueError('Policy config mismatch')
        print('DEVICE',jax.devices(), 'PLAN',plan['counts'],flush=True)
        sampler=make_sampler(wrapper,kappa); seen=set()
        train, core=collect(wrapper,extra,sampler,out,'certificate',counts[0],seeds[0],horizon,seen)
        inside=core[core[:,-1]>=.5]; outside=core[core[:,-1]<.5]
        if len(inside)==0 or len(outside)==0: raise RuntimeError('Missing U core shaping examples')
        print('TRAIN W/U',len(inside),len(outside),flush=True)
        cert=fit_distributional_certificates(train_batches=train,is_accepting=accepting,in_retention_set=retained,
            core_inside=inside,core_boundary=outside,core_inside_upper=.2,core_boundary_lower=.8,
            seed=34001,B_W=1.,B_U=1.,eps_W=.01,eps_U=.01,n_epochs=epochs)
        torch.save(dict(W=cert.W.state_dict(),U=cert.U.state_dict()),out/'certificates.pt')
        write(out/'certificate_frozen.json',dict(time=stamp(),sha256=digest(out/'certificates.pt'),loss=cert.loss_history,
              n_anchors=len(train),accepting_anchors=sum(accepting(b.anchor) for b in train)))
        op,_=collect(wrapper,extra,sampler,out,'operator',counts[1],seeds[1],horizon,seen)
        operators=fit_post_expectation_models(W=cert.W,U=cert.U,certificate_train_batches=train,train_batches=op,
            seed=34002,B_W=1.,B_U=1.,n_epochs=3 if args.smoke else 300)
        torch.save(dict(H_W=operators.H_W.state_dict(),H_U=operators.H_U.state_dict()),out/'operators.pt')
        write(out/'operators_frozen.json',dict(time=stamp(),sha256=digest(out/'operators.pt'),loss=operators.loss_history))
        residual,_=collect(wrapper,extra,sampler,out,'residual',counts[2],seeds[2],horizon,seen)
        calibration=calibrate_post_expectation_models(training=operators,W=cert.W,U=cert.U,
            calibration_batches=residual,alpha_W=.01,alpha_U=.01,delta_mc_W=.001,delta_mc_U=.001)
        write(out/'calibration_frozen.json',dict(time=stamp(),**asdict(calibration)))
        warrant,_=collect(wrapper,extra,sampler,out,'warrant',counts[3],seeds[3],horizon,seen)
        result=verify_distributional_l3(training=operators,calibration=calibration,W=cert.W,U=cert.U,
            warrant_batches=warrant,is_accepting=accepting,in_retention_set=retained,
            eps_W=.01,eps_U=.01,gamma=.05,warrant_threshold=.95)
        report=dict(completed=stamp(),smoke_only=args.smoke,scope=plan['scope'],**asdict(result),
                    n_accepting_anchors=sum(accepting(b.anchor) for b in warrant),
                    n_retained_anchors=sum(retained(b.anchor) for b in warrant),
                    global_infinite_status='NOT_ESTABLISHED')
        write(out/'report.json',report)
        print('RESULT',result.summary(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args())
