"""Fixed finite ensemble search on fit/reused internal only; no new labels."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import l2_motion_v_refine as new
from experiments import l2_distinct_spec_screen as old
base=new.base;exp=new.exp


def aggregate(values,cfg):
    a=values[cfg['members']]
    if cfg['op']=='weighted':return np.einsum('i,ijk->jk',np.array(cfg['weights']),a)
    if cfg['op']=='min':return a.min(0)
    if cfg['op']=='max':return a.max(0)
    if cfg['op']=='mean':return a.mean(0)
    if cfg['op']=='median':return np.median(a,axis=0)
    raise ValueError(cfg)


def primitive_values(primitive,paths,spec,random_seed=None):
    model=torch.load(primitive['path'],map_location='cpu')
    if primitive['kind']=='old':
        x,q,b,d,p=old.products(paths,spec)
        if random_seed is not None:
            torch.manual_seed(random_seed);model['weights']=old.Value(x.shape[-1],model['family']).state_dict()
        return 4*old.predict(model,x,d,p)
    x,q,b,d=new.products(paths,spec,model['config']['features'])
    if random_seed is not None:
        torch.manual_seed(random_seed);model['weights']=new.Value(x.shape[-1],model['config']).state_dict()
    return 4*new.predict(model,x,d)


def run(out):
    torch.set_num_threads(2)
    source=ROOT/'artifacts/safedreamer_l2_motion_v_refine_v1'
    plan0=json.loads((source/'plan.json').read_text());spec=plan0['spec']
    primitive_paths=[('old',new.SOURCE/'motion_burst_elu128.pt')]
    primitive_paths += [('new',source/(c['id']+'.pt')) for c in plan0['configs'] if c['id']!='kinematic64']
    primitives=[dict(kind=k,path=str(p),sha256=base.digest(p),scale=4.) for k,p in primitive_paths]
    configs=[dict(id='original',op='weighted',members=[0],weights=[1.])]
    for i in range(1,len(primitives)):
        for weight in [.25,.5,.75]:
            configs.append(dict(id='blend%d_%g'%(i,weight),op='weighted',members=[0,i],weights=[weight,1-weight]))
        for op in ['min','max']:configs.append(dict(id=op+str(i),op=op,members=[0,i]))
    configs += [dict(id='all_'+op,op=op,members=list(range(len(primitives)))) for op in ['mean','median','min','max']]
    out.mkdir(exist_ok=False,parents=True)
    plan=dict(at=base.stamp(),spec=spec,H=300,eta=.01,primitives=primitives,configs=configs,
        code_sha256=base.digest(__file__),new_predictor_sha256=base.digest(new.__file__),old_predictor_sha256=base.digest(old.__file__),
        scope='Fixed40combinations; no gradient fitting to internal, reused internal selects combination; not independent confirmation',
        selection='whole internal P1/P2 then certificate then fewer P2/P1; train must remain100%; controls must not all pass',
        calibration_count=0,test_count=0)
    base.write(out/'plan.json',plan)
    fit,records=base.load_train();internal,hashes=exp.load_internal()
    allvalues=[np.stack([primitive_values(p,paths,spec) for p in primitives]) for paths in (fit,internal)]
    region=np.load(exp.SOURCE/'region.npz');datasets=[]
    for paths in (fit,internal):
        x,q,b,d,_=old.products(paths,spec)
        mask=np.linalg.norm((np.stack([base.physical(a) for a in paths])-region['mean'])/region['scale'],axis=-1)<=float(region['radius'])+1e-10
        datasets.append((q,b,d,mask))
    randoms=[np.stack([primitive_values(p,internal,spec,159301000+seed*100+i) for i,p in enumerate(primitives)]) for seed in range(3)]
    rows=[]
    for cfg in configs:
        stats=[]
        for values,(q,b,d,mask) in zip(allvalues,datasets):
            v=aggregate(values,cfg);s=base.statistics(v,b,d,region=mask)
            pp,cc=old.independent(v,b,d,mask)
            np.testing.assert_array_equal(pp,s['p1p2_per_path']);np.testing.assert_array_equal(cc,s['certificate_per_path'])
            stats.append(s)
        cvs=[('constant',np.ones_like(v)),('mode_constant',np.where(d,0.,1.))]
        for seed in range(3):
            rng=np.random.RandomState(159300000+seed);sv=v.copy()
            for mode in np.unique(q):sv[q==mode]=rng.permutation(v[q==mode])
            cvs.append(('shuffle%d'%seed,sv));cvs.append(('random%d'%seed,aggregate(randoms[seed],cfg)))
        controls={name:base.statistics(cv,b,d) for name,cv in cvs}
        ok=exp.qualification(stats,controls)
        rows.append(dict(config=cfg,train=stats[0],internal=stats[1],controls=controls,status='KEEP_DEV' if ok else 'BORDERLINE'))
    eligible=[r for r in rows if r['train']['p1p2_paths']==80]
    best=max(eligible,key=lambda r:(r['status']=='KEEP_DEV',*exp.score(r['internal']),r['internal']['certificate_preview']))
    base.write(out/'report.json',dict(at=base.stamp(),rows=rows,selected=best,fit_hashes=[r['sha256'] for r in records],
        internal_hashes=hashes,independent_AND_replays=len(rows)*100,calibration_count=0,test_count=0))
    base.write(out/'selected_manifest.json',dict(at=base.stamp(),primitives=primitives,aggregation=best['config'],spec=spec,
        eta=.01,H=300,scope='Development only; V(x) uses the stored aggregation of scaled primitive predictions'))
    np.savez_compressed(out/'selected_values.npz',train=aggregate(allvalues[0],best['config']),internal=aggregate(allvalues[1],best['config']))
    lines=['# 固定低速→高速：函数组合检查','','固定8个已训练函数、40种预先列出的组合；80fit/20复用internal。',
        '所有基本函数在训练集P1/P2全过；正加权平均、逐点min/max/median不引入时间输入或改变detector。',
        '没有在internal上拟合权重；从固定网格选组合仍属于调参，不能当独立验证。','',
        '| 组合 | train整路径 | internal整路径 | internal P1违反 | internal P2违反 | 状态 |','|---|---|---|---|---|---|']
    for r in rows:
        a,b=r['train'],r['internal'];lines.append('| %s | %d/80 | %d/20 | %d | %d | %s |'%(r['config']['id'],a['p1p2_paths'],b['p1p2_paths'],b['p1_violations'],b['p2_violations'],r['status']))
    lines+=['','选中：'+json.dumps(best['config']),
        '对照整路径通过：'+str({k:v['p1p2_paths'] for k,v in best['controls'].items()}),
        '未启动正式校准或Test1。完整AND独立重算%d条候选路径。'%(len(rows)*100)]
    (out/'RESULT_CN.md').write_text('\n'.join(lines)+'\n')
    print('SELECTED',best['config'],'train',best['train']['p1p2_paths'],'dev',best['internal']['p1p2_paths'],
          'P1/P2 bad',best['internal']['p1_violations'],best['internal']['p2_violations'],best['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output.resolve())
