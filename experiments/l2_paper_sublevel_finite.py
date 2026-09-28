"""Paper-style learned V(z,q) and Z_free, evaluated on 50-step prefixes.

Uses Eq. (10)-(12) to fit a nonnegative network on real product transitions.
Z_free is ONLY {V<eta}. No accepting-state clamp, waiting floor, artificial
terminal edge, time input, or target-set shaping loss. This is not an
infinite-horizon verification implementation: all audits below are sampled.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.lppm.learned_sublevel import ProductValue, paper_loss, predict, audit
from core.lppm.latent_reachability import fit_scaler
from experiments.l2_latent_v_refine import combined, OUT as REFINE
from experiments.l2_latent_v_search import OLD, read
from experiments.l2_latent_prediction_check import digest, now, compare

OUT = OLD.parent/'safedreamer_l2_paper_sublevel_finite'
ETA, H, N = .01, 50, 100
CONFIGS = [
    dict(name='paper_w128_s101', width=128, seed=101, lr=.0003, regularization=1e-6),
    dict(name='paper_w256_s102', width=256, seed=102, lr=.0003, regularization=1e-6),
    dict(name='paper_w128_reg_s103', width=128, seed=103, lr=.0003, regularization=1e-4),
]
SEEDS = dict(calibration=15301, test=15302)


def write_json(name, data, replace=False):
    with (OUT/name).open('w' if replace else 'x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def train(steps):
    OUT.mkdir(exist_ok=False)
    x, d = combined('train'); xv, dv = combined('validation')
    z, zv = x[..., :512], xv[..., :512]
    mean, scale = fit_scaler(z)
    # Only existing training/validation paths enter fitting/selection.
    inputs = torch.tensor((z-mean)/scale, dtype=torch.float32)
    done = torch.tensor(d, dtype=torch.float32)
    scales = torch.tensor(scale, dtype=torch.float32)
    sources = [root/f'{split}.npz' for root, splits in
               ((OLD, ('train', 'validation')), (REFINE, ('extra_train', 'extra_validation')))
               for split in splits]
    write_json('plan.json', dict(
        created_at=now(), paper='/home/sunyhg/Documents/_ICLR_27_SafeWorld.pdf',
        paper_equations=[3,4,10,11,12], configs=CONFIGS, updates_per_config=steps,
        data_sources={str(p): digest(p) for p in sources},
        training_paths=len(d), development_paths=len(dv),
        spec_automaton='F(goal), q consumes current label; done absorbing',
        evaluated_property='F_[0,50](decoded goal_dist < 0)',
        horizon=H, latent_dim=512, q_input='one-hot waiting/done', eta=ETA,
        learned_sublevel='V(z,q) < eta, WITHOUT a done gate',
        constraints='P1 on all transitions; P2 additionally on waiting-source transitions',
        objective='mean(l1+l2) + lambda*mean(||grad_raw_z V||^2), paper Eq.12',
        selection='minimum development mean(l1+l2), training residual breaks ties; no calibration/test selection',
        no_goal_shaping_loss=True, no_time_input=True, no_output_floor=True,
        no_forced_accepting_zero=True, no_synthetic_tail=True,
        calibration_paths=N, test_paths=N, seeds=SEEDS, action_source='random',
        claims='finite-prefix sampled evidence only; no support-wide validity or infinite-horizon guarantee',
    ))
    records, best = [], None
    for cfg in CONFIGS:
        torch.manual_seed(cfg['seed'])
        net = ProductValue(512, cfg['width'])
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg['lr'])
        family_best = None
        for step in range(1, steps+1):
            ix = torch.randperm(len(inputs))[:32]
            a = inputs[ix].detach().clone().requires_grad_(True)
            v = net(a, done[ix])
            loss, parts = paper_loss(v, done[ix], a, scales, ETA, cfg['regularization'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{cfg["name"]} at {step}')
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
            optimizer.step()
            if step % 300 and step != steps:
                continue
            model = dict(weights={k:v.detach().clone() for k,v in net.state_dict().items()},
                         width=cfg['width'], mean=mean, scale=scale, eta=ETA,
                         name=cfg['name'], step=step, config=cfg,
                         schema='all_512_latent_dimensions_plus_q_onehot')
            tr = audit(predict(model,z,d),d,ETA)
            va = audit(predict(model,zv,dv),dv,ETA)
            row = dict(name=cfg['name'], step=step, train=tr, validation=va,
                       minibatch_loss=float(loss.detach()),
                       minibatch_parts=[float(p.detach()) for p in parts])
            key = (va['mean_paper_transition_loss'],tr['mean_paper_transition_loss'])
            records.append(row)
            if best is None or key < best[0]:
                best = (key,row)
                torch.save(model,OUT/'selected.pt')
            if family_best is None or key < family_best[0]:
                family_best = (key,row)
                torch.save(model,OUT/f'{cfg["name"]}.pt')
            print(cfg['name'],step,'train P1/P2',tr['p1']['violations'],tr['p2']['violations'],
                  'val P1/P2',va['p1']['violations'],va['p2']['violations'],
                  'val sublevel paths',va['zfree_entered']['successes'],
                  'pending members',va['pending_sublevel_states'],
                  'val C',va['candidate_certificate_event']['successes'],flush=True)
            write_json('progress.json',dict(best=best[1],records=records),replace=True)
    model = torch.load(OUT/'selected.pt',weights_only=False)
    assert audit(predict(model,z,d),d,ETA) == best[1]['train']
    assert audit(predict(model,zv,dv),dv,ETA) == best[1]['validation']
    write_json('frozen.json',dict(frozen_at=now(),best=best[1],
                                model_sha256=digest(OUT/'selected.pt'),candidates=len(records)))
    print('FROZEN',best[1]['name'],best[1]['step'],flush=True)


def hashes(latent):
    return [hashlib.sha256(z.tobytes()).hexdigest() for z in latent]


def collect():
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    frozen = json.loads((OUT/'frozen.json').read_text())
    assert digest(OUT/'selected.pt') == frozen['model_sha256']
    if any((OUT/name).exists() for name in ('prediction.json','calibration.npz','test.npz')):
        raise FileExistsError('Do not replace calibration/test data')
    model = torch.load(OUT/'selected.pt',weights_only=False)
    seen, fingerprint_sources = set(), []
    # Read old paths for exact-duplicate checking ONLY, never for model selection.
    for path in sorted(OLD.parent.glob('safedreamer_l2_*/*.npz')):
        if path.parent == OUT:
            continue
        with np.load(path) as data:
            if 'latent' in data.files:
                seen.update(hashes(data['latent']))
                fingerprint_sources.append(str(path))
    extra = build_extra(argparse.Namespace(repo_root=None,checkpoint=None))
    extra['config_overrides']['seed'] = 15300
    write_json('collection_plan.json',dict(
        created_at=now(),model_sha256=frozen['model_sha256'],
        checkpoint=extra['checkpoint_path'],checkpoint_sha256=digest(Path(extra['checkpoint_path'])),
        horizon=H,n_calibration=N,n_test=N,seeds=SEEDS,
        action_source='random',config_seed=15300,prior_fingerprint_sources=fingerprint_sources,
        note='Reset observations seed the imagination; no real environment trajectories evaluated.'))
    logger = logging.getLogger('wrappers.safedreamer_wrapper')
    logger.setLevel(logging.INFO);logger.addHandler(logging.StreamHandler(sys.stdout));logger.propagate=False
    fps, data_hashes = {}, {}
    cfg = RolloutConfig(horizon=H,n_rollouts=N,seed=SEEDS['calibration'],action_source='random',extra=extra)
    with SafeDreamerWrapper(cfg) as wrapper:
        wrapper.load()
        for split,seed in SEEDS.items():
            assert digest(OUT/'selected.pt') == frozen['model_sha256']
            if split == 'test':
                forecast_hash = digest(OUT/'prediction.json')
                test_started = now()
                print('Prediction frozen. Starting independent finite test.',flush=True)
            data = wrapper.sample_latent_rollouts(RolloutConfig(
                horizon=H,n_rollouts=N,seed=seed,action_source='random',extra=extra))
            fp = hashes(data['latent'])
            assert len(set(fp))==N and not set(fp)&seen
            seen.update(fp);fps[split]=fp
            aps = np.array([[[s[k] for k in ('goal_dist','hazard_dist','velocity')]
                             for s in tr] for tr in data['aps']],dtype=np.float32)
            np.savez_compressed(OUT/f'{split}.npz',latent=data['latent'],aps=aps,decoded=data['decoded'],
                                **{f'rssm_{k}':v for k,v in data['rssm'].items()})
            data_hashes[split]=digest(OUT/f'{split}.npz')
            x,d=read(OUT/f'{split}.npz');values=predict(model,x[...,:512],d)
            np.savez_compressed(OUT/f'{split}_values.npz',values=values,done=d,zfree=values<ETA)
            if split=='calibration':
                summary=audit(values,d,ETA,inferential=True)
                write_json('prediction.json',dict(saved_at=now(),model_sha256=frozen['model_sha256'],
                    calibration=summary,next_n_paths=N,
                    expected_counts={key:N*summary[key]['rate'] for key in
                        ('candidate_certificate_event','finite_goal_and_certificate','goal_reached','zfree_entered')},
                    scope='Finite 50-step events. Candidate sublevel membership is not a validity proof.'))
                print('FORECAST',json.dumps({key:summary[key] for key in
                    ('candidate_certificate_event','finite_goal_and_certificate','goal_reached','zfree_entered')}),flush=True)
    write_json('fingerprints.json',fps)
    write_json('collection.json',dict(test_started_at=test_started,completed_at=now(),
        prediction_sha256_before_test=forecast_hash,dataset_sha256=data_hashes))


def evaluate():
    frozen=json.loads((OUT/'frozen.json').read_text())
    forecast=json.loads((OUT/'prediction.json').read_text())
    collection=json.loads((OUT/'collection.json').read_text())
    assert digest(OUT/'selected.pt')==frozen['model_sha256']==forecast['model_sha256']
    assert digest(OUT/'prediction.json')==collection['prediction_sha256_before_test']
    assert forecast['saved_at']<=collection['test_started_at']
    model=torch.load(OUT/'selected.pt',weights_only=False)
    summaries={}
    for split in SEEDS:
        assert digest(OUT/f'{split}.npz')==collection['dataset_sha256'][split]
        x,d=read(OUT/f'{split}.npz');v=predict(model,x[...,:512],d)
        with np.load(OUT/f'{split}_values.npz') as a:
            assert np.array_equal(v,a['values']) and np.array_equal(v<ETA,a['zfree'])
        summaries[split]=audit(v,d,ETA,inferential=True)
    cal,test=summaries['calibration'],summaries['test']
    assert cal==forecast['calibration']
    comparisons={key:compare(cal[key],test[key],N) for key in
        ('candidate_certificate_event','finite_goal_and_certificate','goal_reached','zfree_entered','p1p2_paths')}
    report=dict(model_sha256=frozen['model_sha256'],candidate=frozen['best']['name'],step=frozen['best']['step'],
        paper_equations=[3,4,10,11,12],scope='Finite 50-step model-only audit, random actions',
        eta=ETA,calibration=cal,test=test,comparisons=comparisons,
        prediction_saved_at=forecast['saved_at'],test_started_at=collection['test_started_at'],
        support_wide_validity='NOT_ESTABLISHED',infinite_horizon_warrant='NOT_ESTABLISHED',
        region_audit_status=test['sampled_region_status'],
        notes=[
            'P1 includes ALL transitions; P2 is an additional bad-source constraint.',
            'Z_free membership is exactly V<eta; accepting values are learned, not forced to zero.',
            'Candidate C is Eq.4 on finite data, and is not automatically a property-satisfaction event.',
            'finite_goal_and_certificate explicitly intersects C with observed goal satisfaction.',
            'Terminal-only entry has no observed successor; no closure claim is made from it.',
            'Conditional observed-suffix closure is NOT an infinite-horizon invariant proof.',
            'Individual 95% CP bounds require fixed model and iid same-distribution paths; not confidence in V itself.',
            'No 80% threshold is used to turn these diagnostics into a warrant.'
        ])
    if (OUT/'report.json').exists():
        assert json.loads((OUT/'report.json').read_text())==report
        print('Report reproduced exactly.',flush=True)
    else:
        write_json('report.json',report)
    for key,row in comparisons.items():print(key,json.dumps(row),flush=True)
    print('REGION AUDIT',test['sampled_region_status'],'P1',test['p1'],'P2',test['p2'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['train','collect','evaluate','all'],default='all')
    p.add_argument('--steps',type=int,default=2400)
    args=p.parse_args()
    if args.steps<1:p.error('--steps must be positive')
    torch.set_num_threads(2)
    if args.phase in ('train','all'):train(args.steps)
    if args.phase in ('collect','all'):collect()
    if args.phase in ('evaluate','all'):evaluate()
