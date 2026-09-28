"""Freeze and independently replay the selected motion value combination."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import l2_motion_v_ensemble as en
base=en.base;exp=en.exp


def audit(out):
    torch.set_num_threads(2)
    plan=json.loads((out/'plan.json').read_text());report=json.loads((out/'report.json').read_text())
    manifest=json.loads((out/'selected_manifest.json').read_text());best=report['selected'];cfg=best['config']
    assert base.digest(en.__file__)==plan['code_sha256']
    assert base.digest(en.new.__file__)==plan['new_predictor_sha256']
    assert base.digest(en.old.__file__)==plan['old_predictor_sha256']
    assert manifest['aggregation']==cfg and manifest['primitives']==plan['primitives']
    for prim in plan['primitives']:assert base.digest(prim['path'])==prim['sha256']
    fit,records=base.load_train();internal,hashes=exp.load_internal()
    assert [r['sha256'] for r in records]==report['fit_hashes'] and hashes==report['internal_hashes']
    assert not set(hashes)&set(report['fit_hashes'])
    region=np.load(exp.SOURCE/'region.npz');values=[]
    for name,paths in [('train',fit),('internal',internal)]:
        predictions=np.stack([en.primitive_values(p,paths,plan['spec']) for p in plan['primitives']])
        v=en.aggregate(predictions,cfg);values.append(v)
        x,q,b,d,_=en.old.products(paths,plan['spec'])
        mask=np.linalg.norm((np.stack([base.physical(a) for a in paths])-region['mean'])/region['scale'],axis=-1)<=float(region['radius'])+1e-10
        stat=base.statistics(v,b,d,region=mask);assert stat==best[name]
        with np.load(out/'selected_values.npz') as saved:np.testing.assert_array_equal(saved[name],v)
        pp,cc=en.old.independent(v,b,d,mask)
        np.testing.assert_array_equal(pp,stat['p1p2_per_path']);np.testing.assert_array_equal(cc,stat['certificate_per_path'])
    assert cfg['op']=='weighted'
    direct=sum(w*en.primitive_values(plan['primitives'][i],internal,plan['spec']) for i,w in zip(cfg['members'],cfg['weights']))
    np.testing.assert_array_equal(direct,values[1])
    failed=[]
    for i in np.flatnonzero(~np.asarray(best['internal']['p1p2_per_path'])):
        dv=np.diff(values[1][i]);bad=b[i,:-1]
        failed.append(dict(path=int(i),completion_t=int(np.flatnonzero(d[i])[0]),
            p1_failure_times=np.flatnonzero(dv>0).tolist(),p2_failure_times=np.flatnonzero(bad&(dv>-.01)).tolist(),
            max_delta=float(dv.max())))
    bundle=dict(spec=plan['spec'],eta=.01,H=300,aggregation=cfg,
        primitives=[dict(**prim,model=torch.load(prim['path'],map_location='cpu')) for prim in plan['primitives']],
        scope='Development-selected; no confidence guarantee; no calibration/Test1')
    # The manifest remains the executable recipe; bundle is a self-contained weight backup.
    torch.save(bundle,out/'selected_bundle.pt')
    base.write(out/'audit.json',dict(at=base.stamp(),selected_replayed=True,independent_gate_AND=True,
        independent_weighted_formula=True,all_primitive_hashes_checked=True,failed_internal_paths=failed,
        selected_bundle_sha256=base.digest(out/'selected_bundle.pt'),calibration_count=0,test_count=0,
        code_sha256=base.digest(en.__file__),audit_sha256=base.digest(__file__)))
    a,c=best['train'],best['internal']
    text=['# 低速→高速：本轮更好的V','',
        '固定阈值speed<=0.2245011670后再speed>=0.7390221069；H=300，eta=0.01。',
        '当前结果：BORDERLINE，未达到双100%，无正式SAFE结论。','',
        '| 指标 | 原V | 新组合V |','|---|---|---|',
        '| train整路径P1/P2 | 80/80 | %d/80 |'%a['p1p2_paths'],
        '| internal整路径P1/P2 | 18/20 | %d/20 |'%c['p1p2_paths'],
        '| internal P1 | 5987/6000 | %d/6000 |'%(6000-c['p1_violations']),
        '| internal P2 | 134/148 | %d/148 |'%(148-c['p2_violations']),
        '| internal任务完成 | 20/20 | 20/20 |',
        '| internal完整certificate AND | 18/20 | %d/20 |'%c['certificate_preview'],'',
        '新V=0.75×(4×原ELU128函数)+0.25×(4×新full_residual函数)。两个输入处理分别随模型冻结；不是重训policy。',
        'selected_manifest.json给出精确组合及路径/hash；selected_bundle.pt保留所有所需权重备份。',
        '常数/打乱/随机对照整路径通过：'+str({k:v['p1p2_paths'] for k,v in best['controls'].items()}),
        '剩余失败路径：'+json.dumps(failed,ensure_ascii=False),'',
        '训练仅80fit；20internal用于选择家族/尺度/组合，未加入梯度训练，但已反复调参，因此19/20不是独立统计证据。',
        '完整AND、权重hash、选中值数组及独立加权公式已重算一致。旧48-state SAFE和24-state KEEP_DEV结果不变。',
        '本轮没有正式calibration、CP下界或Test1。']
    (out/'SELECTED_CN.md').write_text('\n'.join(text)+'\n')
    print('\n'.join(text),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output.resolve())
