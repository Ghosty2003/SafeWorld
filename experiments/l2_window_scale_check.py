"""Explicit internal-development value-scale sweep, never change eta/gates."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from experiments import l2_window_progress_refine as prog
from experiments import l2_spec_continuation as exp
from experiments.l2_spec_continuation_audit import independent


def run():
    torch.set_num_threads(2)
    source=exp.ROOT/'artifacts/safedreamer_l2_window_progress_v1'
    out=exp.ROOT/'artifacts/safedreamer_l2_window_scale_v1';out.mkdir(exist_ok=False)
    file=source/'safe_window24_coefficient.pt'
    model=torch.load(file,map_location='cpu')
    plan=dict(created=exp.base.stamp(),source=str(file),source_sha256=exp.base.digest(file),
        spec=model['spec'],scales=[1.,1.5,2.],eta=.01,tolerance=0.,H=300,
        reason='Internal-development refinement: two failures had negative descent but insufficient fixed eta margin.',
        selection='Smallest tested positive scale giving strict train/internal P1/P2 and negative controls',
        caveat='Scale selection uses reused internal validation; NOT fresh confirmation or a warrant',
        fixed='No altered automaton, eta, region, endpoint or closure gate; Zfree remains V_scaled<eta',
        code_sha256=exp.base.digest(__file__),calibration_count=0,test_count=0)
    exp.base.write(out/'plan.json',plan)
    fit,records=exp.base.load_train();internal,hashes=exp.load_internal()
    region=np.load(exp.SOURCE/'region.npz');datasets=[]
    for paths in (fit,internal):
        x,q,b,d=prog.data(paths,model['spec']);v=prog.predict(model,x,q)
        mask=np.linalg.norm((np.stack([exp.base.physical(a) for a in paths])-region['mean'])/region['scale'],axis=-1)<=float(region['radius'])+1e-10
        datasets.append((x,q,b,d,v,mask))
    control_values=[('constant',np.ones_like(v)),('mode_constant',np.where(d,0.,1.)),('monitor_rank',.02*(24-q))]
    for seed in range(3):
        rng=np.random.RandomState(156400000+seed);sv=v.copy()
        for mode in np.unique(q):sv[q==mode]=rng.permutation(v[q==mode])
        control_values.append(('shuffle%d'%seed,sv));torch.manual_seed(156401000+seed)
        rm=copy.deepcopy(model);rm['weights']=prog.Value(x.shape[-1],rm['family']).state_dict()
        control_values.append(('random%d'%seed,prog.predict(rm,x,q)))
    rows=[];selected=None
    for scale in plan['scales']:
        stats=[]
        for x,q,b,d,v,mask in datasets:
            vv=scale*v;s=exp.base.statistics(vv,b,d,region=mask)
            pp,cc=independent(vv,b,d,mask)
            np.testing.assert_array_equal(pp,s['p1p2_per_path']);np.testing.assert_array_equal(cc,s['certificate_per_path'])
            stats.append(s)
        controls={name:exp.base.statistics(scale*cv,b,d) for name,cv in control_values}
        ok=exp.qualification(stats,{k:v for k,v in controls.items() if k!='monitor_rank'})
        rows.append(dict(scale=scale,train=stats[0],internal=stats[1],controls=controls,status='KEEP_DEV' if ok else 'BORDERLINE'))
        if ok and selected is None:selected=scale
        print('SCALE',scale,'train',stats[0]['p1p2_paths'],'internal',stats[1]['p1p2_paths'],'status',rows[-1]['status'],flush=True)
    if selected is not None:
        # This is a composite frozen candidate: evaluator MUST apply output_scale.
        torch.save(dict(base_model=model,output_scale=selected,predictor='V(x)=output_scale*prog.predict(base_model,x,q)',
                        scope='Development-selected, requires fresh confirmation'),out/'selected_scaled.pt')
    exp.base.write(out/'report.json',dict(rows=rows,selected_scale=selected,independent_gate_replay=True,
        fit_hashes=[r['sha256'] for r in records],internal_hashes=hashes,calibration_count=0,test_count=0))
    lines=['# L2 窗口候选的尺度检查','','规范：F[0,277] G[0,23](clearance>=0.2)，H=300，eta=0.01。',
        '当前输入为完整latent/planner、解码信号、因果safe-run计数，不含未来时间。',
        '基函数来自80条fit训练；20条旧internal用于选择家族和输出尺度，未用于梯度训练。',
        '不会改变旧48-state SAFE结果，也未使用其校准/Test1。','',
        '| V输出倍数 | train P1/P2整路径 | internal P1 | internal P2 | internal整路径 | 状态 |','|---|---|---|---|---|---|']
    for r in rows:
        a,b=r['train'],r['internal'];lines.append('| %.1f | %d/80 | %d/%d | %d/%d | %d/20 | %s |'%(r['scale'],a['p1p2_paths'],b['transitions']-b['p1_violations'],b['transitions'],b['bad_transitions']-b['p2_violations'],b['bad_transitions'],b['p1p2_paths'],r['status']))
    if selected is not None:
        best=next(r for r in rows if r['scale']==selected)
        lines+=['','开发集选中输出倍数%.1f。没有放松eta或容差，而是整体放大同一个非恒定V的下降幅度。'%selected,
            '正尺度不能修复V增加的转移；本候选原本已P1全过，两个失败仅是下降不足eta。',
            'internal值范围：[%g, %g]。'%(best['internal']['v_min'],best['internal']['v_max']),
            '常数/按mode常数/打乱/随机对照的整路径通过数：'+str({k:v['p1p2_paths'] for k,v in best['controls'].items()}),
            '20/20完成规范、无初始已完成；完整AND与逐路径独立实现精确一致。',
            '**KEEP_DEV不是SAFE：同一内部集经过反复选择，需要新独立确认；没有正式CP下界。**']
    (out/'RESULT_CN.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':run()
