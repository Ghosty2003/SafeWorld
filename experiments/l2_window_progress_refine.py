"""Small fit-only continuous-monitor progress family, no final-set access."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import l2_spec_continuation as exp
from experiments.l2_spec_continuation_audit import independent


class Value(nn.Module):
    def __init__(self,dim,family):
        super().__init__();self.family=family
        self.net=nn.Sequential(nn.Linear(dim,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,1))
    def forward(self,x,remaining):
        raw=nn.functional.softplus(self.net(x).squeeze(-1))
        if self.family=='coefficient':v=.01*remaining*(1.5+raw)
        elif self.family=='additive':v=.02*remaining+raw
        else:v=.01*remaining+raw
        return torch.where(remaining>0,v,torch.zeros_like(v))


def data(paths,spec):
    x,q,b,d=exp.product(paths,spec)
    # Replace one-hot mode with continuous causal safe-run progress.
    x=np.concatenate([x[...,:-spec['n_states']],q[...,None]/spec['length']],-1)
    return x,q,b,d


def predict(m,x,q):
    net=Value(x.shape[-1],m['family']).double();net.load_state_dict(m['weights']);net.eval()
    with torch.no_grad():return net(torch.tensor((x-m['mean'])/m['scale']),torch.tensor(m['spec']['length']-q)).numpy()


def run(out):
    torch.set_num_threads(2);fit,records=exp.base.load_train()
    source=ROOT/'artifacts/safedreamer_l2_spec_continuation_v1'
    p=json.loads((source/'plan.json').read_text());specs=[s for s in p['candidates'] if s['kind']=='window']
    out.mkdir(exist_ok=False,parents=True)
    exp.base.write(out/'plan.json',dict(at=exp.base.stamp(),source=str(source),source_plan_sha256=exp.base.digest(source/'plan.json'),
        code_sha256=exp.base.digest(__file__),specs=specs,eta=.01,updates=3000,
        families=['coefficient','additive','soft_progress'],fit_records=records,
        selection='train checkpoint by whole-path P1/P2 then fewer P2/P1; internal selects family only',
        scope='80fit/20reused internal; no calibration/test; no global guarantee'))
    models=[]
    for si,spec in enumerate(specs):
        x,q,b,d=data(fit,spec);mean=x.mean((0,1));scale=np.maximum(x.std((0,1)),.05)
        normalized=(x-mean)/scale;pi,ti=np.where(b[:,:-1])
        a=torch.tensor(normalized[pi,ti],dtype=torch.float32);aa=torch.tensor(normalized[pi,ti+1],dtype=torch.float32)
        r=torch.tensor(spec['length']-q[pi,ti]);rr=torch.tensor(spec['length']-q[pi,ti+1])
        for fi,family in enumerate(['coefficient','additive','soft_progress']):
            torch.manual_seed(156300000+si*100+fi);net=Value(x.shape[-1],family)
            opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=1e-6)
            best=None;weights=torch.ones(len(a))
            for step in range(3001):
                if step:
                    ix=torch.multinomial(weights,512,replacement=True)
                    va=net(a[ix],r[ix]);vb=net(aa[ix],rr[ix]);pos=torch.relu((vb-va)/.01+1.5)
                    loss=pos.square().mean()+pos.topk(32).values.mean()+.0001*va.square().mean()
                    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),100.);opt.step()
                if step%200:continue
                model=dict(spec=spec,family=family,mean=mean,scale=scale,step=step,weights=copy.deepcopy(net.state_dict()))
                stat=exp.base.statistics(predict(model,x,q),b,d)
                if best is None or exp.score(stat)>best[0]:best=(exp.score(stat),model,stat)
                with torch.no_grad():
                    hard=(torch.relu((net(aa,rr)-net(a,r))/.01+1.5)+.1).clamp(max=50.)
                    weights=.25/len(a)+.75*hard/hard.sum()
            ident=spec['id']+'_'+family;torch.save(best[1],out/(ident+'.pt'))
            models.append(dict(id=ident,spec=spec,weights=ident+'.pt',sha256=exp.base.digest(out/(ident+'.pt')),train=best[2],at=exp.base.stamp()))
            print('FROZEN',ident,'train',best[2]['p1p2_paths'],'P1/P2',best[2]['p1_violations'],best[2]['p2_violations'],flush=True)
    exp.base.write(out/'frozen.json',dict(at=exp.base.stamp(),models=models))
    internal,hashes=exp.load_internal();region=np.load(exp.SOURCE/'region.npz');rows=[]
    for rec in models:
        model=torch.load(out/rec['weights'],map_location='cpu');assert exp.base.digest(out/rec['weights'])==rec['sha256']
        stats=[]
        for paths in (fit,internal):
            x,q,b,d=data(paths,rec['spec']);v=predict(model,x,q)
            mask=np.linalg.norm((np.stack([exp.base.physical(a) for a in paths])-region['mean'])/region['scale'],axis=-1)<=float(region['radius'])+1e-10
            s=exp.base.statistics(v,b,d,region=mask);stats.append(s)
            pp,cc=independent(v,b,d,mask);np.testing.assert_array_equal(pp,s['p1p2_per_path']);np.testing.assert_array_equal(cc,s['certificate_per_path'])
        controls={}
        cvs=[('constant',np.ones_like(v)),('mode_constant',np.where(d,0.,1.)),('monitor_rank',.02*(rec['spec']['length']-q))]
        for seed in range(3):
            sv=v.copy();rng=np.random.RandomState(156400000+seed)
            for mode in np.unique(q):sv[q==mode]=rng.permutation(v[q==mode])
            cvs.append(('shuffle%d'%seed,sv));torch.manual_seed(156401000+seed)
            m=copy.deepcopy(model);m['weights']=Value(x.shape[-1],m['family']).state_dict();cvs.append(('random%d'%seed,predict(m,x,q)))
        for name,cv in cvs:controls[name]=exp.base.statistics(cv,b,d)
        ok=exp.qualification(stats,{k:v for k,v in controls.items() if k!='monitor_rank'})
        rows.append(dict(id=rec['id'],train=stats[0],internal=stats[1],controls=controls,status='KEEP' if ok else 'BORDERLINE'))
        np.savez_compressed(out/(rec['id']+'_internal.npz'),v=v,q=q,bad=b,done=d)
        print('INTERNAL',rec['id'],stats[1]['p1p2_paths'],'P1/P2',stats[1]['p1_violations'],stats[1]['p2_violations'],'KEEP',ok,flush=True)
    exp.base.write(out/'report.json',dict(rows=rows,internal_hashes=hashes,calibration_count=0,test_count=0,independent_gate_replay=True))
    lines=['# 窗口连续进度函数追加对照','','仅80 fit/20复用internal，H=300；无正式校准或Test1访问。',
        '| 函数 | train整路径 | internal整路径 | internal P1违反 | internal P2违反 | 状态 |','|---|---|---|---|---|---|']
    for r in rows:
        a,b=r['train'],r['internal'];lines.append('| %s | %d/80 | %d/20 | %d/%d | %d/%d | %s |'%(r['id'],a['p1p2_paths'],b['p1p2_paths'],b['p1_violations'],b['transitions'],b['p2_violations'],b['bad_transitions'],r['status']))
    lines+=['','remaining指还需多少个连续safe状态，由当前因果monitor决定，不是未来return time或剩余rollout时间。',
        '三种结构都允许latent影响V；当前mode提供progress先验。接受态吸收且V=0，不代表物理区域永久安全。',
        '每个模型均重算全部真实转移，并用独立逐路径AND复核；对照详见report.json。']
    (out/'RESULT_CN.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output.resolve())
