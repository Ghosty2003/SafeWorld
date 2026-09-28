"""Development-only W/U candidate matrix; no fresh validation or L3 warrant.

Reuses pilot certificate paths for fitting and reclassifies previously examined
validation paths as development data. Never modifies frozen pilot artifacts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.l3_safedreamer_clear_pipeline import digest, write, stamp, product
from experiments.l3_safedreamer_clear_recurrence import monitor
from experiments.l3_safedreamer_paper import load_batches
from core.lbsm.sampled_recurrence import NormalizedBoundedCertificate
from core.lbsm.operator_model import assert_anchor_splits_disjoint


class Candidate(nn.Module):
    """Structurally bounded [0,1] functions, recomputed at every full state."""
    def __init__(self, mean, scale, family):
        super().__init__()
        self.B = 1.
        self.family = family
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))
        d = len(mean)
        if family == 'quadratic':
            self.weight = nn.Parameter(torch.full((d,), -3.))
            self.bias = nn.Parameter(torch.tensor(-1.))
        else:
            width, depth = (128, 3) if family == 'wide' else (64, 2)
            layers = []
            for _ in range(depth):
                layers.extend([nn.Linear(d, width), nn.Tanh()]); d = width
            layers.append(nn.Linear(width, 49 if family == 'mode' else 1))
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        z = (x-self.mean)/self.scale
        if self.family == 'quadratic':
            # Positive diagonal quadratic; initialization bit/counter excluded
            # from radial distance, not mistaken for continuous coordinates.
            logits = (torch.nn.functional.softplus(self.weight[:-2])*z[..., :-2]**2).mean(-1)+self.bias
        else:
            logits = self.net(z)
            if self.family == 'mode':
                mode = (x[..., -1]*48).round().long().clamp(0,48)
                logits = logits.gather(-1, mode[..., None]).squeeze(-1)
            else:
                logits = logits.squeeze(-1)
        return torch.sigmoid(logits)


def return_targets(counter):
    """Time to next acceptance, inclusive; right-censored suffix stays NaN."""
    result = np.full(len(counter), np.nan, dtype=np.float32)
    next_hit = None
    for t in reversed(range(len(counter))):
        if counter[t] == 48:
            next_hit = t
        if next_hit is not None:
            result[t] = next_hit-t
    return result


def load_paths(source, role):
    paths, targets, identities = [], [], set()
    for meta in sorted((source/role).glob('path_*.json')):
        record = json.loads(meta.read_text())
        if digest(meta.with_suffix('.npz')) != record['sha256']:
            raise ValueError('Raw path hash changed')
        if not record['accepted']:
            continue
        with np.load(meta.with_suffix('.npz'), allow_pickle=False) as a:
            margin = np.linalg.norm(a['decoded'][:,9:25].reshape(-1,8,2),axis=-1).min(-1).astype(float)-.2
            counter = monitor(margin >= 0)['counter']
            paths.append(np.stack([product(z,a['planner_action_mean'][t],a['planner_action_std'][t],
                                            a['planner_initialized'][t].item(),counter[t])
                                   for t,z in enumerate(a['latent'])]))
            targets.append(return_targets(counter))
        identities.add(record['path_fingerprint'])
    if not paths:
        raise ValueError('Missing paths')
    return np.stack(paths), np.stack(targets), identities


def prepare(paths, targets, batches):
    return dict(core=torch.tensor(paths.reshape(-1,paths.shape[-1])),
                x=torch.tensor(np.stack([b.anchor for b in batches])),
                y=torch.tensor(np.stack([b.successors for b in batches])),
                px=torch.tensor(paths[:,:-1].reshape(-1,paths.shape[-1])),
                py=torch.tensor(paths[:,1:].reshape(-1,paths.shape[-1])),
                target=torch.tensor(targets.reshape(-1)))


def train_candidate(kind, family, data, mean, scale, seed, epochs):
    torch.manual_seed(seed)
    net = Candidate(mean,scale,family)
    opt = torch.optim.Adam(net.parameters(),lr=.001)
    x,y,c = data['x'],data['y'],data['core']
    history = []
    for epoch in range(epochs):
        # Training data only: all real imagined transitions, not padded tails.
        idx = torch.randperm(len(c))[:512]
        ci = c[idx]
        tidx = torch.randperm(len(data['px']))[:512]
        px,py = data['px'][tidx],data['py'][tidx]
        drift = net(y.flatten(0,1)).reshape(y.shape[:2]).mean(1)-net(x)
        path_drift = net(py)-net(px)
        if kind == 'W':
            nonf = x[:,-1]<1
            pnonf = px[:,-1]<1
            branch_loss = torch.relu(drift[nonf]+.01).mean() if nonf.any() else drift.sum()*0
            path_loss = torch.relu(path_drift[pnonf]+.01).mean() if pnonf.any() else path_drift.sum()*0
            if family == 'progress':
                labels = data['target'][idx]
                valid = torch.isfinite(labels)
                # A supervision heuristic, not a countdown certificate or proof.
                target = torch.clamp(.01*labels[valid],max=.95)
                shape = ((net(ci[valid])-target)**2).mean() if valid.any() else drift.sum()*0
            else:
                shape = ((net(ci)-.48*(1-ci[:,-1]))**2).mean()
            loss = branch_loss + .5*path_loss + .25*shape
        else:
            # H1: NO acceptance bypass. eps_U is a fit preference only;
            # evaluated theorem condition is nonincrease (<=0).
            branch_loss = torch.relu(drift+.001).mean()
            path_loss = torch.relu(path_drift+.001).mean()
            if family in ('sublevel','quadratic'):
                target = torch.full((len(ci),),.2)
                shape = ((net(ci)-target)**2).mean()
                # Synthetic training shell is NOT a certified region boundary.
                shell = ci.clone()
                shell[:,:-2] = net.mean[:-2]+4*(ci[:,:-2]-net.mean[:-2])
                shape = shape + torch.relu(.85-net(shell)).square().mean()
            else:
                shape = ((net(ci)-(.1+.4*(1-ci[:,-1])))**2).mean()
            loss = branch_loss + .5*path_loss + .25*shape
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch == 0 or (epoch+1)%100 == 0 or epoch+1 == epochs:
            history.append(dict(epoch=epoch+1,loss=float(loss),branch_loss=float(branch_loss),
                                single_successor_loss=float(path_loss),shaping_loss=float(shape)))
    return net.eval(), history


@torch.no_grad()
def metrics(net, kind, data, paths):
    x,y,c = data['x'],data['y'],data['core']
    values = net(c)
    drift = net(y.flatten(0,1)).reshape(y.shape[:2]).mean(1)-net(x)
    required = x[:,-1]<1 if kind=='W' else torch.ones(len(x),dtype=torch.bool)
    threshold = -.01 if kind=='W' else 0.
    passing = drift<=threshold
    v = values.reshape(paths.shape[:2])
    diff = v[:,1:]-v[:,:-1]
    req_path = torch.tensor(paths[:,:-1,-1]<1) if kind=='W' else torch.ones_like(diff,dtype=torch.bool)
    path_ok = (~req_path | (diff<=threshold)).all(1)
    shell = c.clone(); shell[:,:-2] = net.mean[:-2]+4*(c[:,:-2]-net.mean[:-2])
    return dict(required_anchors=int(required.sum()),passing_anchors=int((required & passing).sum()),
                anchor_rate=float(passing[required].float().mean()) if required.any() else None,
                mean_required_drift=float(drift[required].mean()) if required.any() else None,
                initial_in_C=int((v[:,0]<=.8).sum()),n_initial=len(v),
                all_path_state_in_C_rate=float((values<=.8).float().mean()),
                complete_single_successor_paths=int(path_ok.sum()),n_paths=len(v),
                value_min=float(values.min()),value_max=float(values.max()),value_std=float(values.std()),
                synthetic_shell_mean=float(net(shell).mean()),value_mean=float(values.mean()),
                drift=drift.tolist(),anchor_pass=passing.tolist(),required=required.tolist(),
                anchor_in_C=(net(x)<=.8).tolist(),
                note='DEVELOPMENT_ONLY: raw sample means, no confidence claim; single-successor path checks are not expected-drift proofs')


def pair_score(w, u):
    """No credit for discarding difficult anchors from C or empty W tests."""
    eligible = (w['required_anchors']>0 and u['initial_in_C']==u['n_initial']
                and u['all_path_state_in_C_rate']>=.95 and u['value_max']-u['value_min']>.02)
    joint = [ic and up and (not wr or wp) for ic,up,wr,wp in
             zip(u['anchor_in_C'],u['anchor_pass'],w['required'],w['anchor_pass'])]
    wr,ur = w['anchor_rate'] or 0., u['anchor_rate'] or 0.
    return eligible, min(wr,ur), sum(joint)/len(joint), (wr+ur)/2


def run(args):
    torch.set_num_threads(2)
    source=args.source.resolve(); out=args.output.resolve()
    plan=json.loads((source/'plan.json').read_text())
    freeze=json.loads((source/'frozen.json').read_text())
    if digest(source/'plan.json')!=freeze['plan_sha256'] or digest(source/'certificates.pt')!=freeze['certificate_sha256']:
        raise ValueError('Source frozen artifact changed')
    if plan['spec']!='GF G[0,47](decoded_hazard_margin>=0)' or plan['eps_W']!=.01 or plan['ell']!=.8:
        raise ValueError('Unexpected source specification/parameters')
    train=load_batches(source,'certificate'); dev=load_batches(source,'validation')
    assert_anchor_splits_disjoint(train,dev,left_name='fit',right_name='development')
    tp,tt,ti=load_paths(source,'certificate'); dp,dt,di=load_paths(source,'validation')
    if ti & di: raise ValueError('Path leakage')
    td=prepare(tp,tt,train); dd=prepare(dp,dt,dev)
    mean=tp.reshape(-1,tp.shape[-1]).mean(0); scale=np.maximum(tp.reshape(-1,tp.shape[-1]).std(0),.05)
    mean[-2:]=0; scale[-2:]=1
    out.mkdir(parents=True,exist_ok=False)
    matrix={'W':['plain','wide','mode','progress'],'U':['plain','sublevel','mode','quadratic']}
    write(out/'plan.json',dict(time=stamp(),source=str(source),source_plan_sha256=digest(source/'plan.json'),
          spec=plan['spec'],horizon=plan['horizon'],eps_W=.01,ell=.8,structural_bound=1.,
          matrix=matrix,seeds=args.seeds,epochs=args.epochs,train_paths=len(tp),development_paths=len(dp),
          source_code_sha256=digest(Path(__file__).resolve()),
          roles='Original validation is now development; no fresh calibration or final test in this search.',
          selection='Require initial C coverage, >=95% dev state C coverage, noncollapsed U; maximize minimum W/U raw anchor pass rate, then joint pass rate.',
          caveats=['Only four nonaccepting development anchors in this pilot; ranking is unstable.',
                   'Right-censored return targets omitted, not labeled as never returning.',
                   'No H2 bypass or certified invariant core; synthetic U shaping is a heuristic.',
                   'Even all sampled passes do not establish global recurrence.']))
    rows={'W':{},'U':{}}
    old=torch.load(source/'certificates.pt',map_location='cpu',weights_only=True)
    for kind in ('W','U'):
        baseline=NormalizedBoundedCertificate(old[kind]['mean'].numpy(),old[kind]['scale'].numpy())
        baseline.load_state_dict(old[kind]); baseline.eval()
        rows[kind]['pilot_baseline']=dict(train=metrics(baseline,kind,td,tp),development=metrics(baseline,kind,dd,dp),
                                        source=str(source/'certificates.pt'))
        for family in matrix[kind]:
            for seed in args.seeds:
                name=f'{kind}_{family}_{seed}'
                net,history=train_candidate(kind,family,td,mean,scale,seed,args.epochs)
                torch.save(dict(family=family,kind=kind,state=net.state_dict()),out/f'{name}.pt')
                row=dict(train=metrics(net,kind,td,tp),development=metrics(net,kind,dd,dp),history=history,
                         checkpoint=f'{name}.pt',sha256=digest(out/f'{name}.pt'))
                rows[kind][name]=row
                write(out/f'{name}.json',row)
                print(name,'train',row['train']['passing_anchors'],'/',row['train']['required_anchors'],
                      'dev',row['development']['passing_anchors'],'/',row['development']['required_anchors'],flush=True)
    pairs=[]
    for wn,wr in rows['W'].items():
        for un,ur in rows['U'].items():
            score=pair_score(wr['development'],ur['development'])
            pairs.append(dict(W=wn,U=un,eligible=score[0],score=list(score[1:])))
    pairs.sort(key=lambda p:(p['eligible'],*p['score']),reverse=True)
    winner=pairs[0] if pairs[0]['eligible'] else None
    write(out/'report.json',dict(completed=stamp(),scope='CANDIDATE_SEARCH_DEVELOPMENT_ONLY',
          candidates=rows,pair_ranking=pairs,selected=winner,
          global_L3='ABSTAIN',probability_lower_bound=None,confidence=None,
          reason='No independent successor validation after selection; no certified global coverage/retention backend.'))
    print('SELECTED',winner,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'artifacts/safedreamer_l3_paper_pilot192')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=400)
    p.add_argument('--seeds',type=int,nargs='+',default=[41,73])
    a=p.parse_args()
    if a.epochs<1: p.error('Positive epochs required')
    run(a)
