"""Independent replay and concise report for fixed motion-burst V refinement."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import l2_motion_v_refine as m
base=m.base;exp=m.exp


def audit(out):
    torch.set_num_threads(2);plan=json.loads((out/'plan.json').read_text());report=json.loads((out/'report.json').read_text())
    assert base.digest(m.__file__)==plan['code_sha256']
    assert base.digest(m.SOURCE/'plan.json')==plan['source_plan_sha256']
    fit,records=base.load_train();internal,hashes=exp.load_internal()
    assert records==plan['fit_records'] and hashes==report['internal_hashes']
    assert not set(hashes)&{r['sha256'] for r in records}
    region=np.load(exp.SOURCE/'region.npz');masks=[]
    for paths in (fit,internal):
        masks.append(np.linalg.norm((np.stack([base.physical(a) for a in paths])-region['mean'])/region['scale'],axis=-1)
                     <=float(region['radius'])+1e-10)
    frozen=json.loads((out/'frozen.json').read_text())['records'];count=0
    for rec in frozen:
        assert base.digest(out/rec['weights'])==rec['sha256']
        model=torch.load(out/rec['weights'],map_location='cpu');rows=[r for r in report['rows'] if r['id']==rec['id']]
        for label,paths,mask in zip(('train','internal'),(fit,internal),masks):
            x,q,b,d=m.products(paths,plan['spec'],model['config']['features']);v=m.predict(model,x,d)
            for path,dd in zip(paths,d):
                t=m.manual_completion(m.screen.signals(path),plan['spec']);ix=np.flatnonzero(dd)
                assert t==(int(ix[0]) if len(ix) else None)
            with np.load(out/(rec['id']+'_values.npz')) as a:np.testing.assert_array_equal(a[label],v)
            for row in rows:
                vv=v*row['gain'];stat=base.statistics(vv,b,d,region=mask)
                assert stat==row[label]
                pp,cc=m.screen.independent(vv,b,d,mask)
                np.testing.assert_array_equal(pp,stat['p1p2_per_path']);np.testing.assert_array_equal(cc,stat['certificate_per_path'])
                count+=len(paths)
        cvs=[('constant',np.ones_like(v)),('mode_constant',np.where(d,0.,1.))]
        for seed in range(3):
            rng=np.random.RandomState(159200000+seed);sv=v.copy()
            for mode in np.unique(q):sv[q==mode]=rng.permutation(v[q==mode])
            cvs.append(('shuffle%d'%seed,sv));torch.manual_seed(159201000+seed)
            rm=copy.deepcopy(model);rm['weights']=m.Value(x.shape[-1],model['config']).state_dict()
            cvs.append(('random%d'%seed,m.predict(rm,x,d)))
        for row in rows:
            for name,cv in cvs:assert base.statistics(row['gain']*cv,b,d)==row['controls'][name]
    best=report['selected'];baseline=report['baseline']
    lines=['# 固定低速→高速规范：V优化结果','',
        '固定detector：先speed<=0.2245011670，随后speed>=0.7390221069；H=300、eta=0.01、零容差。',
        '同一SafeDreamer checkpoint + CCEPlanner.policy；仅80fit+20复用internal，未使用正式calibration/Test1。',
        'V训练加入fit-only长等待加权、pending状态归一化、完成剩余步数辅助标签；未来标签不作为模型输入。',
        '8个函数、每个400步辅助预训练+4000步优化；原18/20候选保留。','',
        '| 候选 | 最佳输出倍数 | train整路径P1/P2 | internal整路径P1/P2 | internal P1违反 | internal P2违反 | 状态 |',
        '|---|---|---|---|---|---|---|']
    for ident in [c['id'] for c in plan['configs']]+[baseline['id']]:
        options=[r for r in report['rows']+[baseline] if r['id']==ident]
        if not options:continue
        r=max(options,key=lambda r:(r['status']=='KEEP_DEV',*exp.score(r['internal']),*exp.score(r['train']),-r['gain']))
        a,b=r['train'],r['internal'];lines.append('| %s | %.1f | %d/80 | %d/20 | %d/6000 | %d/%d | %s |'%(
            ident,r['gain'],a['p1p2_paths'],b['p1p2_paths'],b['p1_violations'],b['p2_violations'],b['bad_transitions'],r['status']))
    b=best['internal'];a=best['train']
    lines+=['','选中：%s，输出倍数%.1f，%s。'%(best['id'],best['gain'],best['status']),
        '选中函数：train P1 %d/%d，P2 %d/%d；internal P1 %d/%d，P2 %d/%d。'%(
            a['transitions']-a['p1_violations'],a['transitions'],a['bad_transitions']-a['p2_violations'],a['bad_transitions'],
            b['transitions']-b['p1_violations'],b['transitions'],b['bad_transitions']-b['p2_violations'],b['bad_transitions']),
        'internal完整AND通过%d/20，任务完成%d/20；区域失败%d、closure失败%d、坏模式进入Zfree状态%d。'%(
            b['certificate_preview'],b['finite_completion_paths'],b['region_failures'],b['closure_failure_paths'],b['bad_in_z_free'])]
    if 'controls' in best:lines+=['对照整路径通过：'+str({k:v['p1p2_paths'] for k,v in best['controls'].items()})]
    lines+=['','**即使达到KEEP_DEV，也只是反复用于调参的内部数据全部通过，不是新独立验证，更不是SAFE。**',
        '没有降低eta、放宽容差、改阈值、过滤失败路径或重设计时预算。接受态V=0来自因果完成状态；未完成态V>=eta。',
        '独立重算%d条候选路径与全部对照；权重/数据hash、完整AND和first-hit monitor检查通过。'%count]
    (out/'RESULT_CN.md').write_text('\n'.join(lines)+'\n')
    base.write(out/'audit.json',dict(at=base.stamp(),candidate_paths_replayed=count,controls_replayed=True,
        independent_completion_monitor=True,independent_gate_AND=True,fit_internal_disjoint=True,
        code_sha256=base.digest(m.__file__),audit_sha256=base.digest(__file__),calibration_count=0,test_count=0))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output.resolve())
