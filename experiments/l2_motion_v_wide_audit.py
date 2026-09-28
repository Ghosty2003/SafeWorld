"""Audit all wide-search candidates and controls, not just the best numeric fit."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import l2_motion_v_wide_search as run
base=run.base;exp=run.exp


def audit(out):
    torch.set_num_threads(2)
    plan=json.loads((out/'plan.json').read_text());report=json.loads((out/'numeric_report.json').read_text())
    assert base.digest(run.__file__)==plan['code_sha256']
    fit,records=base.load_train();internal,hashes=exp.load_internal()
    assert records==plan['fit_records'] and hashes==report['internal_hashes']
    assert not set(hashes)&{r['sha256'] for r in records}
    spec=plan['spec'];manifest=json.loads((out/'baseline_manifest.json').read_text())
    region=np.load(exp.SOURCE/'region.npz');masks=[];baseline=[];data_cache={}
    for paths in (fit,internal):
        masks.append(np.linalg.norm((np.stack([base.physical(a) for a in paths])-region['mean'])/region['scale'],axis=-1)<=float(region['radius'])+1e-10)
        for prim in manifest['primitives']:assert base.digest(prim['path'])==prim['sha256']
        vals=np.stack([run.ens.primitive_values(p,paths,spec) for p in manifest['primitives']])
        baseline.append(run.ens.aggregate(vals,manifest['aggregation']))
    random_baseline=[]
    for seed in range(3):
        vals=np.stack([run.ens.primitive_values(p,internal,spec,160301000+seed*100+i) for i,p in enumerate(manifest['primitives'])])
        random_baseline.append(run.ens.aggregate(vals,manifest['aggregation']))
    frozen=json.loads((out/'frozen.json').read_text())['records']
    evaluations=[];selected_values={};count=0;model_cache={}
    groups=[None]+frozen
    for rec in groups:
        if rec is None:
            group=[r for r in report['rows'] if r['network'] is None]
            kind='full';m=None;vs=[np.zeros_like(v) for v in baseline]
        else:
            assert base.digest(out/rec['file'])==rec['sha256'];m=torch.load(out/rec['file'],map_location='cpu')
            group=[r for r in report['rows'] if r['network']==rec['file']];kind=m['config']['features']
        if kind not in data_cache:data_cache[kind]=[run.products(paths,spec,kind) for paths in (fit,internal)]
        datasets=data_cache[kind]
        if m is not None:
            vs=[run.predict(m,x,q) for x,q,b,d in datasets]
            with np.load(out/(rec['id']+'_values.npz')) as saved:
                for key,v in zip(('train','internal'),vs):np.testing.assert_array_equal(saved[key],v)
        random_vs=[]
        x,q,b,d=datasets[1]
        for seed in range(3):
            if m is None:random_vs.append(np.zeros_like(baseline[1]));continue
            torch.manual_seed(160401000+seed);rm=copy.deepcopy(m)
            rm['weights']=run.Value(x.shape[-1],m['config']).state_dict();random_vs.append(run.predict(rm,x,q))
        for original in group:
            row=copy.deepcopy(original);values=[]
            for label,(x,q,b,d),v,old,mask in zip(('train','internal'),datasets,vs,baseline,masks):
                vv=run.combine(v,old,row['config']);stat=base.statistics(vv,b,d,region=mask)
                assert stat==row[label]
                pp,cc=run.screen.independent(vv,b,d,mask)
                np.testing.assert_array_equal(pp,stat['p1p2_per_path']);np.testing.assert_array_equal(cc,stat['certificate_per_path'])
                values.append(vv);count+=len(vv)
            v=values[1];controls={}
            cvs=[('constant',np.ones_like(v)),('mode_constant',np.where(d,0.,1.))]
            for seed in range(3):
                rng=np.random.RandomState(160501000+seed);sv=v.copy()
                for mode in np.unique(q):sv[q==mode]=rng.permutation(v[q==mode])
                cvs.append(('shuffle%d'%seed,sv));cvs.append(('random%d'%seed,run.combine(random_vs[seed],random_baseline[seed],row['config'])))
            for name,cv in cvs:controls[name]=base.statistics(cv,b,d)
            qualified=exp.qualification([row['train'],row['internal']],controls)
            row.update(controls=controls,status='KEEP_DEV' if qualified else 'BORDERLINE')
            evaluations.append(row)
            selected_values[row['config']['id']]=values
    best=max(evaluations,key=lambda r:(r['status']=='KEEP_DEV',*run.key(r)))
    cfg=best['config'];vtrain,v=selected_values[cfg['id']]
    _,q,b,d=run.products(internal,spec,'full');failures=[]
    for i in np.flatnonzero(~np.asarray(best['internal']['p1p2_per_path'])):
        delta=np.diff(v[i]);failures.append(dict(path=int(i),p1_times=np.flatnonzero(delta>0).tolist(),
            p2_times=np.flatnonzero(b[i,:-1]&(delta>-.01)).tolist(),first_completion=int(np.flatnonzero(d[i])[0]) if d[i].any() else None))
    base.write(out/'report.json',dict(at=base.stamp(),rows=evaluations,selected=best,selected_failures=failures,
        qualified=sum(r['status']=='KEEP_DEV' for r in evaluations),internal_hashes=hashes,calibration_count=0,test_count=0))
    bundle=dict(config=cfg,spec=spec,eta=.01,H=300,new_model=None if best['network'] is None else torch.load(out/best['network'],map_location='cpu'),
        baseline=manifest,baseline_models=[torch.load(p['path'],map_location='cpu') for p in manifest['primitives']],
        evaluator='l2_motion_v_wide_search.combine(new_prediction, baseline_prediction, config)',scope='Repeated internal-development selection; no formal confirmation')
    torch.save(bundle,out/'selected_bundle.pt');np.savez_compressed(out/'selected_values.npz',train=vtrain,internal=v)
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['candidate','train_paths','internal_paths','internal_P1_bad','internal_P2_bad','certificate','status'])
        for r in evaluations:
            a,c=r['train'],r['internal'];w.writerow([r['config']['id'],a['p1p2_paths'],c['p1p2_paths'],c['p1_violations'],c['p2_violations'],c['certificate_preview'],r['status']])
    lines=['# 固定低速→高速：48函数扩大搜索','',
        '48个新网络：12种架构/输入模板，各4个初始化与训练设置；每个4000步优化，部分另有400步辅助预训练。',
        '固定阈值0.2245011670→0.7390221069、H=300、eta=.01、checkpoint、CCEPlanner及gate。',
        '80条fit用于梯度与checkpoint选择，20条复用internal用于家族/尺度/组合选择。无时间输入、无未来标签输入。',
        '每个网络4个输出尺度、3种固定混合权重及min/max；连同原19/20基线共433个候选。',
        '没有读取正式calibration/Test1；结果只属开发阶段。','',
        '| 模板 | 该模板最佳候选 | train整路径 | internal整路径 | internal P1违反 | internal P2违反 | 状态 |','|---|---|---|---|---|---|---|']
    for template in list(dict.fromkeys(c['id'].rsplit('_s',1)[0] for c in plan['configs'])):
        options=[r for r in evaluations if r['network'] and r['network'].rsplit('_s',1)[0]==template]
        r=max(options,key=lambda r:(r['status']=='KEEP_DEV',*run.key(r)));a,c=r['train'],r['internal']
        lines.append('| %s | %s | %d/80 | %d/20 | %d/6000 | %d/%d | %s |'%(template,r['config']['id'],a['p1p2_paths'],c['p1p2_paths'],c['p1_violations'],c['p2_violations'],c['bad_transitions'],r['status']))
    a,c=best['train'],best['internal']
    lines+=['','选中：'+json.dumps(cfg),
        'train：P1 %d/%d，P2 %d/%d；internal：P1 %d/%d，P2 %d/%d。'%(a['transitions']-a['p1_violations'],a['transitions'],a['bad_transitions']-a['p2_violations'],a['bad_transitions'],c['transitions']-c['p1_violations'],c['transitions'],c['bad_transitions']-c['p2_violations'],c['bad_transitions']),
        '完整AND：%d/20；任务完成%d/20；区域失败%d、closure失败%d。'%(c['certificate_preview'],c['finite_completion_paths'],c['region_failures'],c['closure_failure_paths']),
        '对照整路径：'+str({k:v['p1p2_paths'] for k,v in best['controls'].items()}),
        '剩余失败：'+json.dumps(failures),
        '**内部集已反复用于选择，不能把其100%（若出现）当作新独立泛化保证。没有发SAFE，没有正式CP。**',
        '全部433个候选含失败项保存在comparison.csv/report.json；selected_bundle.pt保留选中权重和组合规则。']
    (out/'RESULT_CN.md').write_text('\n'.join(lines)+'\n')
    base.write(out/'audit.json',dict(at=base.stamp(),candidate_paths_replayed=count,controls_evaluated=len(evaluations)*8,
        independent_AND=True,fit_internal_hashes_disjoint=True,weights_verified=True,selected_bundle_sha256=base.digest(out/'selected_bundle.pt'),
        code_sha256=base.digest(run.__file__),audit_sha256=base.digest(__file__),calibration_count=0,test_count=0))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output.resolve())
