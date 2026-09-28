"""Prospective 64-transition achievement-spec comparison, model imagination only.

Original Eq.12 versus an explicitly extended accepting-branch architecture/loss.
Neither sublevel membership nor certificate events are overridden by goal labels.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.lppm.learned_sublevel import ProductValue, paper_loss, audit, rate
from core.lppm.latent_reachability import fit_scaler
from experiments.l2_latent_prediction_check import digest, now, compare
from experiments.l2_paper_sublevel_finite import hashes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/safedreamer_l2_policy64'
H, ETA = 64, .01
RADII = (.3,)
COUNTS = dict(train=120, validation=60, calibration=100, test=100)
SEEDS = dict(train=16401, validation=16402, calibration=16403, test=16404)
VARIANTS = ('paper', 'anchored_accepting_scalar')
CHECKPOINT = None
ACTION_SOURCE = 'policy'
DEVICE = 'cpu'


def configure(output=None, checkpoint=None, radii=None, seed_base=None, restore=False, action_source=None, device=None):
    """Configure an isolated run; evaluation/resume uses its recorded plan."""
    global OUT, CHECKPOINT, RADII, SEEDS, ACTION_SOURCE, DEVICE
    if output is not None:
        OUT = Path(output).resolve()
    if restore:
        plan = json.loads((OUT/'plan.json').read_text())
        if plan['horizon'] != H or plan['eta'] != ETA or plan['counts'] != COUNTS:
            raise ValueError('Incompatible saved experiment schema')
        recorded = Path(plan['checkpoint']).resolve()
        if checkpoint is not None and Path(checkpoint).resolve() != recorded:
            raise ValueError('Cannot change checkpoint when resuming')
        if radii is not None and tuple(radii) != tuple(plan['radii']):
            raise ValueError('Cannot change specification when resuming')
        if seed_base is not None and {s:seed_base+i for i,s in enumerate(COUNTS,1)} != plan['seeds']:
            raise ValueError('Cannot change collection seeds when resuming')
        CHECKPOINT = str(recorded)
        RADII = tuple(plan['radii'])
        SEEDS = plan['seeds']
        if action_source is not None and action_source != plan['action_source']:
            raise ValueError('Cannot change policy when resuming')
        ACTION_SOURCE = plan['action_source']
        if device is not None and device != plan.get('device','cpu'):
            raise ValueError('Cannot change imagination device when resuming')
        DEVICE = plan.get('device','cpu')
    else:
        if device is not None:
            if device not in ('cpu','gpu'): raise ValueError('Unknown device')
            DEVICE=device
        if action_source is not None:
            if action_source not in ('random','policy'): raise ValueError('Unknown action source')
            ACTION_SOURCE = action_source
        if checkpoint is not None:
            CHECKPOINT = str(Path(checkpoint).resolve())
        if radii is not None:
            if not radii or any(not np.isfinite(r) or r <= 0 for r in radii):
                raise ValueError('Positive finite goal radii required')
            RADII = tuple(radii)
        if seed_base is not None:
            if seed_base < 0:
                raise ValueError('Nonnegative seed base required')
            SEEDS = {s:seed_base+i for i,s in enumerate(COUNTS,1)}


def write(name, obj, replace=False):
    with (OUT/name).open('w' if replace else 'x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def done_for_radius(aps, radius):
    # Wrapper reports norm(decoded goal vector) - 0.30, not the raw distance.
    if not np.isfinite(aps).all() or radius <= 0:
        raise ValueError('Finite APs and positive radius required')
    return np.maximum.accumulate(aps[..., 0] < radius-.30, axis=1)


class AchievementValue(ProductValue):
    def __init__(self, variant, width=128, latent_dim=512):
        super().__init__(latent_dim, width)
        self.variant = variant
        if variant == 'anchored_accepting_scalar':
            # Initially softplus(-4) = .01815 > eta: membership is learned.
            self.accepting_raw = torch.nn.Parameter(torch.tensor(-4.))
        elif variant != 'paper':
            raise ValueError(variant)

    def forward(self, z, done):
        if self.variant == 'paper':
            return super().forward(z, done)
        waiting = super().forward(z, torch.zeros_like(done))
        accepting = torch.nn.functional.softplus(self.accepting_raw)
        return torch.where(done.bool(), accepting, waiting)


def predict(model, z, done):
    net = AchievementValue(model['variant'], model['width'], z.shape[-1])
    net.load_state_dict(model['weights'])
    with torch.no_grad():
        return net(torch.tensor((z-model['mean'])/model['scale'], dtype=torch.float32),
                   torch.tensor(done, dtype=torch.float32)).numpy().astype(np.float64)


def summary(v, d, inferential=False):
    row = audit(v, d, ETA, inferential)
    source = v[:, :-1] < ETA
    delta = v[:, :-1]-v[:, 1:]
    bad = (delta < 0) | (~d[:, :-1] & (delta < ETA)) | (v[:, 1:] >= ETA)
    clean = source.any(1) & ~(source & bad).any(1) & d[:, -1] & ~((v < ETA) & ~d).any(1)
    row['clean_observed_region_paths'] = rate(clean, inferential)
    row['noninitial_certificate'] = rate(np.array(row['per_path_candidate_event']) & d[:, -1] & ~d[:, 0], inferential)
    row['arrival_given_initially_pending'] = rate(d[~d[:, 0], -1], inferential)
    return row


def selection_key(row):
    # Development only, fixed before collecting any calibration/test data.
    return (row['noninitial_certificate']['successes'],
            row['clean_observed_region_paths']['successes'],
            row['finite_goal_and_certificate']['successes'],
            -row['pending_sublevel_states'], -row['mean_paper_transition_loss'])


def load(split, radius):
    with np.load(OUT/f'{split}.npz') as a:
        return value_features(a, ACTION_SOURCE), done_for_radius(a['aps'], radius)


def value_features(data, action_source):
    """Planner memory is part of closed-loop state, not discarded from V."""
    z=np.asarray(data['latent'],dtype=np.float64)
    if action_source=='random': return z
    if action_source!='policy': raise ValueError('Unknown action source')
    keys=('planner_action_mean','planner_action_std','planner_initialized')
    fields=[z]
    for key in keys:
        a=np.asarray(data[key],dtype=np.float64)
        if a.shape[:2]!=z.shape[:2] or not np.isfinite(a).all():
            raise ValueError('Missing/misaligned policy state')
        fields.append(a.reshape(*z.shape[:2],-1))
    return np.concatenate(fields,axis=-1)


def collect(splits):
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    extra = build_extra(argparse.Namespace(repo_root=None, checkpoint=CHECKPOINT))
    extra['action_source']=ACTION_SOURCE
    extra['config_overrides']['jax']['platform']=DEVICE
    if DEVICE=='gpu': extra['config_overrides']['jax']['logical_gpus']=0
    if digest(Path(extra['checkpoint_path'])) != json.loads((OUT/'plan.json').read_text())['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    if any((OUT/f'{s}.npz').exists() for s in splits):
        raise FileExistsError('Existing raw splits will not be overwritten')
    seen = set()
    for p in OUT.glob('*.npz'):
        with np.load(p) as a:
            if 'latent' in a.files:
                seen.update(hashes(a['latent']))
    logger = logging.getLogger('wrappers.safedreamer_wrapper')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    cfg = RolloutConfig(horizon=H, n_rollouts=1, seed=16400, action_source=ACTION_SOURCE, extra=extra)
    with SafeDreamerWrapper(cfg) as wrapper:
        wrapper.load()
        import jax
        if jax.default_backend()!=DEVICE:
            raise RuntimeError(f'Requested {DEVICE}, got {jax.default_backend()}')
        print('IMAGINATION DEVICE',DEVICE,[str(d) for d in jax.devices()],flush=True)
        if ACTION_SOURCE=='policy':
            policy_config=dict(action_source=ACTION_SOURCE,
                expl_behavior=wrapper._config.expl_behavior,
                planner={k:getattr(wrapper._config.planner,k) for k in
                    ('horizon','num_samples','num_elites','iterations','mixture_coef','momentum','init_std')},
                cost_limit=wrapper._config.cost_limit,
                planner_source_sha256=digest(Path(extra['repo_root'])/'SafeDreamer/behaviors.py'),
                note='Local reconstructed evaluation planner; no claim of exact historical training config.')
            if (OUT/'policy_config.json').exists():
                if json.loads((OUT/'policy_config.json').read_text())!=policy_config:
                    raise ValueError('Policy changed between data splits')
            else: write('policy_config.json',policy_config)
        for split in splits:
            if split == 'test':
                forecast = json.loads((OUT/'prediction.json').read_text())
                write('test_started.json', dict(started_at=now(), prediction_sha256=digest(OUT/'prediction.json')))
                assert digest(OUT/'selected.pt') == forecast['model_sha256']
            started = now()
            print('COLLECTING',split,'paths',COUNTS[split],'horizon',H,'action_source',ACTION_SOURCE,flush=True)
            data = wrapper.sample_latent_rollouts(RolloutConfig(
                horizon=H, n_rollouts=COUNTS[split], seed=SEEDS[split], action_source=ACTION_SOURCE, extra=extra))
            assert data['action_source']==ACTION_SOURCE
            z = data['latent']
            assert z.shape == (COUNTS[split], H+1, 512)
            fp = hashes(z)
            assert len(set(fp)) == len(fp) and not seen.intersection(fp)
            seen.update(fp)
            aps = np.array([[[s[k] for k in ('goal_dist', 'hazard_dist', 'velocity')]
                             for s in path] for path in data['aps']], dtype=np.float32)
            np.savez_compressed(OUT/f'{split}.npz', latent=z, aps=aps, decoded=data['decoded'],
                                actions=data['actions'],**data['policy_state'],
                                **{f'rssm_{k}':v for k,v in data['rssm'].items()})
            provenance=dict(started_at=started, ended_at=now(),
                sha256=digest(OUT/f'{split}.npz'), fingerprints=fp, seed=SEEDS[split],action_source=ACTION_SOURCE)
            if ACTION_SOURCE=='policy': provenance['policy_config_sha256']=digest(OUT/'policy_config.json')
            write(f'{split}_provenance.json',provenance)
            print('COLLECTED', split, {r:dict(reached=int(done_for_radius(aps,r)[:,-1].sum()),
                  initial=int(done_for_radius(aps,r)[:,0].sum())) for r in RADII}, flush=True)
            if split == 'calibration':
                model = torch.load(OUT/'selected.pt', weights_only=False)
                d = done_for_radius(aps, model['radius'])
                features=value_features(dict(latent=z,**data['policy_state']),ACTION_SOURCE)
                result = summary(predict(model,features,d),d,True)
                write('prediction.json', dict(saved_at=now(), model_sha256=digest(OUT/'selected.pt'),
                    radius=model['radius'], variant=model['variant'], calibration=result,
                    next_n_paths=COUNTS['test'], scope='64-step model-only, individual 95% CP bounds'))
                print('FORECAST', result['finite_goal_and_certificate'], result['goal_reached'], flush=True)


def initialize(steps):
    from experiments.l2_safedreamer import build_extra
    OUT.mkdir(exist_ok=False)
    extra = build_extra(argparse.Namespace(repo_root=None, checkpoint=CHECKPOINT))
    write('plan.json', dict(created_at=now(), horizon=H, eta=ETA, radii=RADII,
        counts=COUNTS, seeds=SEEDS, updates_per_candidate=steps, variants=VARIANTS,
        checkpoint=extra['checkpoint_path'], checkpoint_sha256=digest(Path(extra['checkpoint_path'])),
        spec='F_[0,64](norm(decoded goal vector) < radius), label includes t=0 and t=64',
        action_source=ACTION_SOURCE,device=DEVICE,certificate_training_device='cpu',
        initial_distribution='encoded independent simulator resets',
        value_inputs='latent + planner mean/std/initialized + q' if ACTION_SOURCE=='policy' else 'latent + q',
        policy_scope='Local CCEPlanner in latent imagination, not real-observation-feedback deployment',
        reset_rng_note='Imagination seeds explicit; simulator reset RNG is not fully specified by these seeds.',
        paper_objective='Eq12 with gradient regularization 1e-6, original ProductValue',
        extension='Learned latent-independent accepting scalar; add 0.1*mean_done(log(V/(eta/4))^2) '
                  '+ mean_waiting(ReLU(2*eta-V)); no output floor, no forced zero, no time input',
        selection='max(validation noninitial C&goal count, clean observed region path count, C&goal count, '
                  '-pending sublevel states, -mean paper transition loss)',
        candidate_updates='evaluate every 300 updates; radius and variant selected on development only',
        inference='Only selected candidate calibrated and tested after freeze; no infinite-horizon proof',
        baseline='0.3 is original goal, 0.6 and 1.0 are relaxed approach tasks, NOT original success'))


def train():
    steps = json.loads((OUT/'plan.json').read_text())['updates_per_candidate']
    if (OUT/'frozen.json').exists():
        raise FileExistsError('Candidate already frozen')
    z,_ = load('train',.3); zv,_ = load('validation',.3)
    mean,scale=fit_scaler(z)
    inputs=torch.tensor((z-mean)/scale,dtype=torch.float32)
    scales=torch.tensor(scale,dtype=torch.float32)
    best=None; records=[]; family=[]
    for radius in RADII:
        _,d=load('train',radius); _,dv=load('validation',radius)
        dt=torch.tensor(d,dtype=torch.float32)
        for variant in VARIANTS:
            torch.manual_seed(16410)
            net=AchievementValue(variant,latent_dim=z.shape[-1])
            opt=torch.optim.Adam(net.parameters(),lr=.001)
            family_best=None
            for step in range(1,steps+1):
                ix=torch.randperm(len(inputs))[:32]
                a=inputs[ix].detach().clone().requires_grad_(True)
                v=net(a,dt[ix]); loss,_=paper_loss(v,dt[ix],a,scales,ETA,1e-6)
                if variant != 'paper':
                    mask=dt[ix].bool()
                    if mask.any(): loss=loss+.1*torch.log(v[mask]/(ETA/4)).square().mean()
                    if (~mask).any(): loss=loss+torch.relu(2*ETA-v[~mask]).mean()
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
                opt.step()
                if step%300 and step!=steps: continue
                model=dict(weights={k:v.detach().clone() for k,v in net.state_dict().items()},
                           mean=mean,scale=scale,width=128,variant=variant,radius=radius,step=step,eta=ETA)
                tr=summary(predict(model,z,d),d); va=summary(predict(model,zv,dv),dv)
                row=dict(radius=radius,variant=variant,step=step,train=tr,validation=va)
                records.append(row); key=selection_key(va)
                if family_best is None or key>family_best[0]:
                    family_best=(key,row); torch.save(model,OUT/f'r{radius}_{variant}.pt')
                if best is None or key>best[0]:
                    best=(key,row); torch.save(model,OUT/'selected.pt')
                print('FIT',radius,variant,step,'val goal/Z/C/clean',va['goal_reached']['successes'],
                      va['zfree_entered']['successes'],va['finite_goal_and_certificate']['successes'],
                      va['clean_observed_region_paths']['successes'],'P1/P2',va['p1']['violations'],
                      va['p2']['violations'],flush=True)
                write('progress.json',dict(best=best[1],records=records),replace=True)
            family.append(family_best[1])
    write('frozen.json',dict(frozen_at=now(),model_sha256=digest(OUT/'selected.pt'),best=best[1],families=family))


def evaluate():
    frozen=json.loads((OUT/'frozen.json').read_text()); pred=json.loads((OUT/'prediction.json').read_text())
    started=json.loads((OUT/'test_started.json').read_text())
    assert digest(OUT/'selected.pt')==frozen['model_sha256']==pred['model_sha256']
    assert digest(OUT/'prediction.json')==started['prediction_sha256']
    assert frozen['frozen_at'] <= pred['saved_at'] <= started['started_at']
    model=torch.load(OUT/'selected.pt',weights_only=False); rows={}
    for split in ('calibration','test'):
        provenance=json.loads((OUT/f'{split}_provenance.json').read_text())
        if ACTION_SOURCE=='policy':
            assert provenance['action_source']=='policy'
            assert provenance['policy_config_sha256']==digest(OUT/'policy_config.json')
        assert digest(OUT/f'{split}.npz')==provenance['sha256']
        z,d=load(split,model['radius']); rows[split]=summary(predict(model,z,d),d,True)
    assert rows['calibration']==pred['calibration']
    report=dict(radius=model['radius'],variant=model['variant'],step=model['step'],horizon=H,eta=ETA,
        **rows, comparisons={k:compare(rows['calibration'][k],rows['test'][k],COUNTS['test']) for k in
            ('goal_reached','finite_goal_and_certificate','zfree_entered','clean_observed_region_paths')},
        result='SAMPLED_DIAGNOSTICS_ONLY', region_status=rows['test']['sampled_region_status'],
        support_wide_validity='NOT_ESTABLISHED', infinite_horizon_warrant='NOT_ESTABLISHED',
        confidence_note='Individual 95% lower bounds on fixed events under iid same-distribution sampling; '
                        'not 95% confidence in V or joint coverage. No arbitrary warrant threshold.')
    if ACTION_SOURCE=='policy':
        report['policy_scope']='Fixed local CCEPlanner in world-model imagination; not real deployment'
        report['policy_config_sha256']=digest(OUT/'policy_config.json')
        report['value_input_dimension']=len(model['mean'])
    saved_plan=json.loads((OUT/'plan.json').read_text())
    if 'device' in saved_plan: report['imagination_device']=saved_plan['device']
    if (OUT/'report.json').exists(): assert json.loads((OUT/'report.json').read_text())==report
    else: write('report.json',report)
    print('RESULT',json.dumps({k:report[k] for k in ('radius','variant','step','region_status','comparisons')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['all','train','holdout','evaluate'],default='all')
    p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--output',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--radii',type=float,nargs='+')
    p.add_argument('--seed-base',type=int)
    p.add_argument('--action-source',choices=['policy','random'])
    p.add_argument('--device',choices=['cpu','gpu'])
    args=p.parse_args()
    if args.steps<1:p.error('steps must be positive')
    configure(args.output,args.checkpoint,args.radii,args.seed_base,restore=args.phase!='all',action_source=args.action_source,device=args.device)
    torch.set_num_threads(2)
    if args.phase=='all': initialize(args.steps); collect(('train','validation'))
    if args.phase in ('all','train'): train()
    if args.phase in ('all','holdout'): collect(('calibration','test'))
    if args.phase in ('all','holdout','evaluate'): evaluate()
