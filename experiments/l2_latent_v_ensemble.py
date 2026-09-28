"""Further state-only V search: continued fitting, scale, and ensembles.

Train/validation only for selection. All certificate tests keep eta=0.01.
Fresh test paths are collected only after the candidate bundle is frozen.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.l2_latent_v_refine import OUT as REFINE,PREVIOUS,combined
from experiments.l2_latent_v_search import OLD,read,predictor,net_for,FLOOR
from core.lppm.latent_reachability import audit_values,normalize

OUT=OLD.parent/'safedreamer_l2_v_ensemble'
SCALES=(1.,2.,4.,8.)
CONFIGS=[
    dict(name='fullbatch_polish',lr=0.00008,margin=0.02,batch=180,loss='top',seed=51,steps=1200),
    dict(name='smooth_path',lr=0.0001,margin=0.05,batch=48,loss='path',seed=52,steps=2400),
    dict(name='large_margin',lr=0.0001,margin=0.10,batch=48,loss='top',seed=53,steps=2400),
]


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def combine_values(values,mode,gain):
    stack=np.stack(values).astype(np.float64)
    if mode=='mean':base=stack.mean(0)
    elif mode=='min':base=stack.min(0)
    elif mode=='max':base=stack.max(0)
    else:raise ValueError(mode)
    return float(gain)*base


def bundle_predictor(bundle):
    functions=[predictor(m) for m in bundle['members']]
    return lambda x:combine_values([f(x) for f in functions],bundle['mode'],bundle['gain'])


def search():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'frozen.json').exists():raise FileExistsError('Search already frozen')
    torch.set_num_threads(2)
    x,d=combined('train');xv,dv=combined('validation')
    base=torch.load(REFINE/'selected.pt',weights_only=False)
    prior=torch.load(PREVIOUS/'selected.pt',weights_only=False)
    plan=dict(configs=CONFIGS,scales=SCALES,eta=.01,train_paths=180,validation_paths=100,
              combination_modes=['mean','min','max'],no_time_feature=True,
              selection='train P2 <=1%; validation noninitial event; validation P2 violations; training P2 violations',
              baseline_sha256=digest(REFINE/'selected.pt'),test_seed=12307,test_paths=100)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    records=[];best=None;pool={}

    def score(r):
        tr,v=r['train'],r['validation']
        return (tr['p2']['violations']/tr['p2']['checked']<=.01,
                v['certificate_successes_after_start'],-v['p2']['violations'],-tr['p2']['violations'])

    def offer(name,members,train_values,val_values,mode='mean'):
        nonlocal best
        local_best=None
        for gain in SCALES:
            a=combine_values(train_values,mode,gain);b=combine_values(val_values,mode,gain)
            rec=dict(name=name,mode=mode,gain=gain,train=audit_values(a,d),validation=audit_values(b,dv))
            records.append(rec)
            if local_best is None or score(rec)>score(local_best):local_best=rec
            if best is None or score(rec)>score(best):
                best=rec
                torch.save(dict(members=members,mode=mode,gain=gain,name=name,eta=.01),OUT/'selected.pt')
        print(name,'best gain',local_best['gain'],'train P2',local_best['train']['p2']['violations'],
              'val P2',local_best['validation']['p2']['violations'],'val C',local_best['validation']['certificate_event']['successes'],flush=True)
        (OUT/'progress.json').write_text(json.dumps(dict(best=best,records=records),indent=2))
        return local_best

    for name,model in (('previous',prior),('refined',base)):
        f=predictor(model);a,b=f(x),f(xv);pool[name]=(model,a,b)
        offer(name,[model],[a],[b])

    a=torch.tensor(normalize(x,base['mean'],base['scale'])[...,base['indices']],dtype=torch.float32)
    mask=torch.tensor(~d,dtype=torch.float32)
    for cfg in CONFIGS:
        torch.manual_seed(cfg['seed'])
        net=net_for(base['width'],len(base['indices']),base['activation']);net.load_state_dict(base['weights'])
        opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=1e-4)
        family_best=None
        for step in range(1,cfg['steps']+1):
            ix=torch.randperm(len(a))[:cfg['batch']]
            vals=(FLOOR+torch.nn.functional.softplus(net(a[ix]).squeeze(-1)))*mask[ix]
            gaps=torch.relu(vals[:,1:]-vals[:,:-1]+cfg['margin']*mask[ix,:-1])
            worst=gaps.max(dim=1).values.mean() if cfg['loss']=='path' else torch.topk(gaps.flatten(),min(150,gaps.numel())).values.mean()
            loss=gaps.mean()+worst
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),10.);opt.step()
            if step%300:continue
            model={k:copy.deepcopy(v) for k,v in base.items() if k!='weights'}
            model.update(weights={k:v.detach().clone() for k,v in net.state_dict().items()},name=cfg['name'],epoch=step)
            f=predictor(model);aa,bb=f(x),f(xv)
            rec=offer(f'{cfg["name"]}_{step}',[model],[aa],[bb])
            if family_best is None or score(rec)>score(family_best):
                family_best=rec;pool[cfg['name']]=(model,aa,bb)

    # Combine per-family winners chosen above with validation only.
    names=list(pool)
    for i,left in enumerate(names):
        for right in names[i+1:]:
            lm,la,lb=pool[left];rm,ra,rb=pool[right]
            for mode in ('mean','min','max'):
                offer(f'{left}+{right}',[lm,rm],[la,ra],[lb,rb],mode)
    offer('all_families',[v[0] for v in pool.values()],[v[1] for v in pool.values()],[v[2] for v in pool.values()])
    frozen=dict(best=best,records=records,model_sha256=digest(OUT/'selected.pt'),baseline_sha256=digest(REFINE/'selected.pt'))
    (OUT/'frozen.json').write_text(json.dumps(frozen,indent=2))
    print('FROZEN',best['name'],best['mode'],'gain',best['gain'],flush=True)


def confirm():
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    if not (OUT/'frozen.json').exists():raise ValueError('Freeze first')
    if (OUT/'fresh_test.npz').exists():raise FileExistsError('Do not overwrite test paths')
    logger=logging.getLogger('wrappers.safedreamer_wrapper');logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout));logger.propagate=False
    extra=build_extra(argparse.Namespace(repo_root=None,checkpoint=None))
    cfg=RolloutConfig(horizon=50,n_rollouts=100,seed=12307,action_source='random',extra=extra)
    with SafeDreamerWrapper(cfg) as w:
        w.load();data=w.sample_latent_rollouts(cfg)
    aps=np.array([[[z[k] for k in ('goal_dist','hazard_dist','velocity')] for z in tr] for tr in data['aps']],dtype=np.float32)
    np.savez_compressed(OUT/'fresh_test.npz',latent=data['latent'],aps=aps,decoded=data['decoded'],
                        **{f'rssm_{k}':v for k,v in data['rssm'].items()})


def evaluate():
    from experiments.l2_latent_prediction_check import existing_fingerprints
    torch.set_num_threads(2)
    frozen=json.loads((OUT/'frozen.json').read_text())
    assert digest(OUT/'selected.pt')==frozen['model_sha256']
    assert digest(REFINE/'selected.pt')==frozen['baseline_sha256']
    new=bundle_predictor(torch.load(OUT/'selected.pt',weights_only=False))
    old=predictor(torch.load(REFINE/'selected.pt',weights_only=False))
    seen=existing_fingerprints()
    for split in ('calibration','test'):
        with np.load(OLD.parent/f'safedreamer_l2_prediction_check/{split}.npz') as a:
            seen.update(hashlib.sha256(z.tobytes()).hexdigest() for z in a['latent'])
    with np.load(OUT/'fresh_test.npz') as a:
        hashes=[hashlib.sha256(z.tobytes()).hexdigest() for z in a['latent']]
        assert len(set(hashes))==100 and not set(hashes)&seen
    report=dict(selected=frozen['best']['name'],mode=frozen['best']['mode'],gain=frozen['best']['gain'],eta=.01,
                model_sha256=frozen['model_sha256'],splits={},scope='50-step F(goal), random-action model imagination')
    for split in ('train','validation','previous_confirmation','previous_prediction_test','fresh_test'):
        if split in ('train','validation'):x,d=combined(split)
        elif split=='previous_confirmation':x,d=read(REFINE/'fresh_test.npz')
        elif split=='previous_prediction_test':x,d=read(OLD.parent/'safedreamer_l2_prediction_check/test.npz')
        else:x,d=read(OUT/'fresh_test.npz')
        aa,bb=audit_values(new(x),d),audit_values(old(x),d)
        if split in ('train','validation'):assert aa==frozen['best'][split]
        if split!='fresh_test':
            for row in (aa,bb):
                for v in row.values():
                    if isinstance(v,dict):v.pop('cp_lower',None);v.pop('confidence',None)
        report['splits'][split]=dict(candidate=aa,baseline=bb)
        print(split,'new P2',aa['p2'],'event',aa['certificate_event'],'old P2',bb['p2'],'event',bb['certificate_event'],flush=True)
    lower=report['splits']['fresh_test']['candidate']['certificate_event']['cp_lower']
    report.update(result='NO_WARRANT' if lower<.8 else 'SAMPLED_EVENT_THRESHOLD_MET',deductive_status='NOT_ESTABLISHED')
    (OUT/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['search','confirm','evaluate','all'],default='all')
    a=p.parse_args()
    if a.phase in ('search','all'):search()
    if a.phase in ('confirm','all'):confirm()
    if a.phase in ('evaluate','all'):evaluate()
