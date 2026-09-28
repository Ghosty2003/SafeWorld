"""Additional state-only ranking candidates; select on validation, then freeze.

No future labels, time index, test paths, or calibration paths are training
inputs. Fresh confirmation data is generated only after selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.lppm.latent_reachability import make_features, normalize, audit_values, kernel_predict

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'artifacts/safedreamer_l2_latent_trial'
OUT = ROOT / 'artifacts/safedreamer_l2_v_search'
ETA, FLOOR = 0.01, 0.011


def read(path):
    with np.load(path) as d:
        return make_features(d['latent'], d['aps'])


def net_for(width, dim, activation):
    act = torch.nn.Tanh if activation == 'tanh' else torch.nn.SiLU
    return torch.nn.Sequential(torch.nn.Linear(dim, width), act(),
                               torch.nn.Linear(width, width), act(), torch.nn.Linear(width, 1))


def predictor(saved):
    net = net_for(saved['width'], len(saved['indices']), saved['activation'])
    net.load_state_dict(saved['weights'])
    net.eval()
    def pred(x):
        with torch.no_grad():
            a = normalize(x, saved['mean'], saved['scale'])[..., saved['indices']]
            t = torch.tensor(a, dtype=torch.float32)
            raw = net(t).squeeze(-1)
            return FLOOR + torch.nn.functional.softplus(raw).numpy()
    return pred


def search():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'frozen.json').exists():
        raise FileExistsError('Frozen search exists; use --phase confirm/evaluate')
    torch.set_num_threads(2)
    x, done = read(OLD/'train.npz')
    xv, dv = read(OLD/'validation.npz')
    old = torch.load(OLD/'mlp_seed0_target0.0.pt', weights_only=False)
    mean, scale = old['mean'], old['scale']
    configs = [
        dict(name='continued_tanh', width=128, activation='tanh', view='all', seed=0, lr=0.0002, decay=0., init=True),
        dict(name='silu_full', width=256, activation='silu', view='all', seed=11, lr=0.0005, decay=1e-5, init=False),
        dict(name='silu_deter', width=256, activation='silu', view='deter', seed=12, lr=0.0005, decay=1e-4, init=False),
        dict(name='tanh_deter', width=128, activation='tanh', view='deter', seed=13, lr=0.0005, decay=1e-5, init=False),
    ]
    rules = dict(configs=configs, max_epochs=2000, evaluate_every=200,
                 eta=ETA, margin_target=0.015,
                 selection='prefer <=1% training P2 violations; then validation noninitial certificate successes, validation P2 pass rate, train P2 pass rate',
                 validation_reused=True, inference='no time or trajectory id',
                 fresh_test=dict(n=100, horizon=50, seed=9307))
    (OUT/'plan.json').write_text(json.dumps(rules, indent=2))
    records, best = [], None
    def score(tr, val):
        bad = tr['p2']['violations']/max(tr['p2']['checked'], 1)
        return (bad<=0.01, val['certificate_successes_after_start'],
                -val['p2']['violations']/max(val['p2']['checked'], 1), -bad)

    for cfg in configs:
        torch.manual_seed(cfg['seed'])
        # RSSM deter is stored explicitly; infer its width from the actual data.
        with np.load(OLD/'train.npz') as arr:
            deter_dim = arr['rssm_deter'].shape[-1]
        indices = list(range(x.shape[-1])) if cfg['view']=='all' else list(range(deter_dim))+list(range(512,515))
        a = torch.tensor(normalize(x,mean,scale)[...,indices], dtype=torch.float32)
        mask = torch.tensor(~done, dtype=torch.float32)
        net = net_for(cfg['width'], len(indices), cfg['activation'])
        if cfg['init']:
            net.load_state_dict(old['weights'])
        opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['decay'])
        for epoch in range(1,2001):
            values = (FLOOR+torch.nn.functional.softplus(net(a).squeeze(-1)))*mask
            gaps = torch.relu(values[:,1:]-values[:,:-1]+0.015*mask[:,:-1])
            loss = gaps.mean()+torch.topk(gaps.flatten(),150).values.mean()
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
            opt.step()
            if epoch%200:
                continue
            saved = dict(weights={k:v.detach().clone() for k,v in net.state_dict().items()},
                         mean=mean,scale=scale,indices=indices,width=cfg['width'],activation=cfg['activation'],
                         name=cfg['name'],epoch=epoch,eta=ETA,floor=FLOOR)
            pred=predictor(saved)
            tr=audit_values(pred(x),done);val=audit_values(pred(xv),dv)
            record=dict(name=cfg['name'],epoch=epoch,train=tr,validation=val)
            records.append(record)
            print(cfg['name'],epoch,'train P2',tr['p2']['violations'],'val P2',val['p2']['violations'],
                  'val C',val['certificate_event']['successes'],flush=True)
            if best is None or score(tr,val)>score(best['train'],best['validation']):
                best=record
                torch.save(saved,OUT/'selected.pt')
            (OUT/'search_progress.json').write_text(json.dumps(dict(best=best,records=records),indent=2))
    frozen=dict(best=best,records=records,selection=rules['selection'],
                model_sha256=hashlib.sha256((OUT/'selected.pt').read_bytes()).hexdigest())
    (OUT/'frozen.json').write_text(json.dumps(frozen,indent=2))
    print('FROZEN',best['name'],best['epoch'],flush=True)


def confirm():
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    if not (OUT/'frozen.json').exists():
        raise ValueError('Freeze candidate first')
    if (OUT/'fresh_test.npz').exists():
        raise FileExistsError('Fresh confirmation already collected; evaluate existing frozen candidate')
    extra=build_extra(argparse.Namespace(repo_root=None,checkpoint=None))
    cfg=RolloutConfig(horizon=50,n_rollouts=100,seed=9307,action_source='random',extra=extra)
    with SafeDreamerWrapper(cfg) as wrapper:
        wrapper.load()
        data=wrapper.sample_latent_rollouts(cfg)
    aps=np.array([[[z[k] for k in ('goal_dist','hazard_dist','velocity')] for z in tr] for tr in data['aps']],dtype=np.float32)
    np.savez_compressed(OUT/'fresh_test.npz',latent=data['latent'],aps=aps,decoded=data['decoded'],
                        **{f'rssm_{k}':v for k,v in data['rssm'].items()})
    print('Fresh confirmation saved',flush=True)


def evaluate():
    torch.set_num_threads(2)
    frozen=json.loads((OUT/'frozen.json').read_text())
    assert hashlib.sha256((OUT/'selected.pt').read_bytes()).hexdigest()==frozen['model_sha256']
    saved=torch.load(OUT/'selected.pt',weights_only=False)
    pred=predictor(saved)
    with np.load(OLD/'gaussian_bandwidth0.5.npz') as a:
        base={k:a[k] for k in a.files}
    result=dict(selected=frozen['best']['name'],epoch=saved['epoch'],scope='F(goal); random actions; 50 actual model transitions',
                frozen_sha256=frozen['model_sha256'],splits={})
    fingerprints=set()
    for split in ('train','validation','calibration','test','fresh_test'):
        path=OUT/'fresh_test.npz' if split=='fresh_test' else OLD/f'{split}.npz'
        x,d=read(path)
        with np.load(path) as a:
            hashes={hashlib.sha256(z.tobytes()).hexdigest() for z in a['latent']}
        assert not hashes&fingerprints
        fingerprints.update(hashes)
        current=audit_values(pred(x),d); baseline=audit_values(kernel_predict(x,base),d)
        if split=='train':
            assert current==frozen['best']['train']
        if split=='validation':
            assert current==frozen['best']['validation']
        if split!='fresh_test':
            for summary in (current,baseline):
                for v in summary.values():
                    if isinstance(v,dict):
                        v.pop('cp_lower',None);v.pop('confidence',None)
        result['splits'][split]=dict(candidate=current,baseline=baseline,
                                    inferential_bounds=(split=='fresh_test'))
        print(split,'candidate P2',current['p2'],'C',current['certificate_event'],
              'baseline P2',baseline['p2'],'C',baseline['certificate_event'],flush=True)
    lower=result['splits']['fresh_test']['candidate']['certificate_event']['cp_lower']
    result.update(result='NO_WARRANT' if lower<0.8 else 'SAMPLED_EVENT_THRESHOLD_MET',
                  confidence=0.95,threshold=0.8,deductive_status='NOT_ESTABLISHED')
    (OUT/'report.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['search','confirm','evaluate','all'],default='all')
    args=p.parse_args()
    if args.phase in ('search','all'):search()
    if args.phase in ('confirm','all'):confirm()
    if args.phase in ('evaluate','all'):evaluate()
