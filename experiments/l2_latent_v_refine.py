"""Expand training coverage and refine the frozen reachability network.

All paths, including non-goal paths, contribute training constraints.
Validation chooses the checkpoint; fresh confirmation follows freezing.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.l2_latent_v_search import OLD,OUT as PREVIOUS,read,predictor,net_for,FLOOR
from core.lppm.latent_reachability import normalize,audit_values

OUT=OLD.parent/'safedreamer_l2_v_refine'
CONFIGS=[
    dict(name='path_max',lr=0.0002,margin=0.015,decay=1e-4,objective='path',seed=31),
    dict(name='larger_margin',lr=0.0002,margin=0.03,decay=1e-4,objective='path',seed=32),
    dict(name='transition_replay',lr=0.0002,margin=0.015,decay=1e-3,objective='top',seed=33),
]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(kind):
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    OUT.mkdir(parents=True,exist_ok=True)
    if kind=='training':
        splits=[('extra_train',120,10301),('extra_validation',60,10302)]
    else:
        if not (OUT/'frozen.json').exists():raise ValueError('Freeze before confirmation')
        splits=[('fresh_test',100,10303)]
    if any((OUT/f'{s}.npz').exists() for s,_,_ in splits):
        raise FileExistsError('Refusing to overwrite collected split')
    extra=build_extra(argparse.Namespace(repo_root=None,checkpoint=None))
    cfg=RolloutConfig(horizon=50,n_rollouts=1,seed=10301,action_source='random',extra=extra)
    with SafeDreamerWrapper(cfg) as w:
        w.load()
        for name,n,seed in splits:
            print('Collecting',name,n,'paths',flush=True)
            c=RolloutConfig(horizon=50,n_rollouts=n,seed=seed,action_source='random',extra=extra)
            data=w.sample_latent_rollouts(c)
            aps=np.array([[[z[k] for k in ('goal_dist','hazard_dist','velocity')] for z in tr] for tr in data['aps']],dtype=np.float32)
            np.savez_compressed(OUT/f'{name}.npz',latent=data['latent'],aps=aps,decoded=data['decoded'],
                                **{f'rssm_{k}':v for k,v in data['rssm'].items()})
            print('Saved',name,data['latent'].shape,flush=True)


def combined(split):
    a,b=read(OLD/f'{split}.npz');c,d=read(OUT/f'extra_{split}.npz')
    return np.concatenate([a,c]),np.concatenate([b,d])


def search():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'frozen.json').exists():raise FileExistsError('Selection already frozen')
    torch.set_num_threads(2)
    old=torch.load(PREVIOUS/'selected.pt',weights_only=False)
    x,done=combined('train');xv,dv=combined('validation')
    mean,scale,indices=old['mean'],old['scale'],old['indices']
    a=torch.tensor(normalize(x,mean,scale)[...,indices],dtype=torch.float32)
    mask=torch.tensor(~done,dtype=torch.float32)
    pred0=predictor(old)
    baseline=dict(name='previous_frozen',epoch=0,train=audit_values(pred0(x),done),validation=audit_values(pred0(xv),dv))
    best=baseline;records=[baseline]
    torch.save(old,OUT/'selected.pt')

    def score(r):
        t,v=r['train'],r['validation']
        frac=t['p2']['violations']/t['p2']['checked']
        return (frac<=0.01,v['certificate_successes_after_start'],
                -v['p2']['violations']/v['p2']['checked'],-frac)

    for cfg in CONFIGS:
        torch.manual_seed(cfg['seed'])
        net=net_for(old['width'],len(indices),old['activation'])
        net.load_state_dict(old['weights'])
        optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['decay'])
        for step in range(1,3601):
            ix=torch.randperm(len(a))[:32]
            values=(FLOOR+torch.nn.functional.softplus(net(a[ix]).squeeze(-1)))*mask[ix]
            gaps=torch.relu(values[:,1:]-values[:,:-1]+cfg['margin']*mask[ix,:-1])
            if cfg['objective']=='path':
                loss=gaps.mean()+gaps.max(dim=1).values.mean()
            else:
                loss=gaps.mean()+torch.topk(gaps.flatten(),80).values.mean()
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
            optimizer.step()
            if step%300:continue
            saved={k:copy.deepcopy(v) for k,v in old.items() if k!='weights'}
            saved.update(weights={k:v.detach().clone() for k,v in net.state_dict().items()},name=cfg['name'],epoch=step)
            pred=predictor(saved)
            record=dict(name=cfg['name'],epoch=step,train=audit_values(pred(x),done),validation=audit_values(pred(xv),dv))
            records.append(record)
            print(cfg['name'],step,'train P2',record['train']['p2'],'val P2',record['validation']['p2'],
                  'val C',record['validation']['certificate_event']['successes'],flush=True)
            if score(record)>score(best):
                best=record;torch.save(saved,OUT/'selected.pt')
            (OUT/'progress.json').write_text(json.dumps(dict(best=best,records=records),indent=2))
    frozen=dict(best=best,baseline=baseline,records=records,model_sha256=sha(OUT/'selected.pt'),
                baseline_sha256=sha(PREVIOUS/'selected.pt'),eta=0.01,
                selection='<=1% train P2 gate, validation noninitial event successes, validation P2 pass rate, train P2 pass rate')
    (OUT/'frozen.json').write_text(json.dumps(frozen,indent=2))
    print('FROZEN',best['name'],best['epoch'],flush=True)


def evaluate():
    torch.set_num_threads(2)
    frozen=json.loads((OUT/'frozen.json').read_text())
    assert sha(OUT/'selected.pt')==frozen['model_sha256']
    assert sha(PREVIOUS/'selected.pt')==frozen['baseline_sha256']
    new=predictor(torch.load(OUT/'selected.pt',weights_only=False))
    old=predictor(torch.load(PREVIOUS/'selected.pt',weights_only=False))
    sources=[OLD/f'{s}.npz' for s in ('train','validation','calibration','test')]
    sources +=[PREVIOUS/'fresh_test.npz']+[OUT/f'{s}.npz' for s in ('extra_train','extra_validation','fresh_test')]
    seen=set()
    for path in sources:
        with np.load(path) as data:
            hashes=[hashlib.sha256(z.tobytes()).hexdigest() for z in data['latent']]
        assert len(set(hashes))==len(hashes) and not set(hashes)&seen
        seen.update(hashes)
    report=dict(selected=frozen['best']['name'],epoch=frozen['best']['epoch'],eta=0.01,splits={},
                model_sha256=frozen['model_sha256'],scope='50-step imagined F(goal) event; random actions')
    for split in ('train','validation','reused_test','fresh_test'):
        if split in ('train','validation'):x,d=combined(split)
        elif split=='reused_test':x,d=read(PREVIOUS/'fresh_test.npz')
        else:x,d=read(OUT/'fresh_test.npz')
        c=audit_values(new(x),d);b=audit_values(old(x),d)
        if split in ('train','validation'):assert c==frozen['best'][split]
        if split!='fresh_test':
            for row in (c,b):
                for value in row.values():
                    if isinstance(value,dict):value.pop('cp_lower',None);value.pop('confidence',None)
        report['splits'][split]=dict(candidate=c,baseline=b)
        print(split,'NEW P2',c['p2'],'C',c['certificate_event'],'OLD P2',b['p2'],'C',b['certificate_event'],flush=True)
    report['result']='NO_WARRANT' if report['splits']['fresh_test']['candidate']['certificate_event']['cp_lower']<0.8 else 'SAMPLED_EVENT_THRESHOLD_MET'
    report['deductive_status']='NOT_ESTABLISHED'
    (OUT/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['collect','search','confirm','evaluate','all'],default='all')
    args=p.parse_args()
    if args.phase in ('collect','all'):
        OUT.mkdir(parents=True,exist_ok=True)
        if not (OUT/'plan.json').exists():
            (OUT/'plan.json').write_text(json.dumps(dict(configs=CONFIGS,updates=3600,batch_paths=32,
                extra_train=dict(n=120,seed=10301),extra_validation=dict(n=60,seed=10302),
                fresh_test=dict(n=100,seed=10303),eta=0.01,spec='ltl_goal_reach',
                selection='train P2 <=1%, then validation noninitial event and P2 pass rate',
                data_rule='all new training paths retained; previous tests excluded from fit and selection'),indent=2))
        collect('training')
    if args.phase in ('search','all'):search()
    if args.phase in ('confirm','all'):collect('confirmation')
    if args.phase in ('evaluate','all'):evaluate()
