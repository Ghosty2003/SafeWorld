"""Replay fit/internal screening, summarize without calibration/test access."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import l2_spec_continuation as exp


def independent(v,bad,done,region):
    p1=[];p2=[];cert=[]
    for a,b,d,r in zip(v,bad,done,region):
        first=all(y<=x for x,y in zip(a[:-1],a[1:]))
        second=all(not z or y-x<=-.01 for x,y,z in zip(a[:-1],a[1:],b[:-1]))
        sound=all(x>=.01 or not z for x,z in zip(a,b))
        closure=all(x>=.01 or (y<.01 and not z) for x,y,z in zip(a[:-1],a[1:],b[1:]))
        p1.append(first);p2.append(second)
        cert.append(first and second and bool(d[-1]) and a[-1]<.01 and sound and closure
                    and bool(r.all()) and bool(np.isfinite(a).all()) and bool((a>=0).all()))
    return np.array(p1)&np.array(p2),np.array(cert)


def audit(out):
    torch.set_num_threads(2)
    plan=json.loads((out/'plan.json').read_text())
    assert exp.base.digest(exp.__file__)==plan['code_sha256']
    assert exp.base.digest(exp.SOURCE/'plan.json')==plan['source_plan_sha256']
    report=json.loads((out/'report.json').read_text())
    fit,records=exp.base.load_train();internal,hashes=exp.load_internal()
    assert records==plan['fit_records'] and hashes==report['internal_hashes']
    assert not set(hashes)&{r['sha256'] for r in records}
    region=np.load(exp.SOURCE/'region.npz')
    masks=[np.linalg.norm((np.stack([exp.base.physical(a) for a in p])-region['mean'])/region['scale'],axis=-1)
           <=float(region['radius'])+1e-10 for p in (fit,internal)]
    frozen=json.loads((out/'frozen.json').read_text())
    count=0
    for rec,row in zip(frozen['records'],report['rows']):
        assert rec['id']==row['id'] and exp.base.digest(out/rec['weights'])==rec['sha256']
        model=torch.load(out/rec['weights'],map_location='cpu')
        for label,paths,mask in zip(('train','internal'),(fit,internal),masks):
            x,q,b,d=exp.product(paths,row['spec']);v=exp.predict(model,x,b,d)
            s=exp.base.statistics(v,b,d,row['spec']['persistence'],mask)
            assert s==row[label]
            pp,cc=independent(v,b,d,mask)
            np.testing.assert_array_equal(pp,s['p1p2_per_path'])
            if not row['spec']['persistence']:
                np.testing.assert_array_equal(cc,s['certificate_per_path'])
            count+=len(paths)
            if row['spec']['kind']=='window':
                L=row['spec']['length']
                for a,dd in zip(paths,d):
                    c,_=exp.base.signals(a)
                    manual=any((c[i:i+L]>=.2).all() for i in range(len(c)-L+1))
                    assert bool(dd[-1])==manual
    old=json.loads((exp.SOURCE/'report.json').read_text())
    oldbest={r['spec']['id']:r for r in old['rows']}
    lines=['# L2 继续筛选：fit/internal-only','',
        '固定当前checkpoint、CCEPlanner、H=300、eta=0.01；80条fit、20条复用内部调参路径。',
        '8个规范×2类函数（线性softplus、128×128 ELU），各2000次更新；只用fit挑训练checkpoint。',
        '未读取正式校准或Test1，未改变已保存的48-state SAFE结果。本轮内部集已反复用于选择，不是新独立确认集。','',
        '| spec | 原轮内部整路径P1/P2 | 本轮train | 本轮internal | internal P1违反 | internal P2违反 | 完整事件preview | 状态 |',
        '|---|---|---|---|---|---|---|---|']
    bests=[]
    for spec in plan['candidates']:
        options=[r for r in report['rows'] if r['spec']['id']==spec['id']]
        if not options:
            continue
        best=max(options,key=lambda r:(r['status']=='KEEP',*exp.score(r['internal']),*exp.score(r['train'])))
        bests.append(best);a,b=best['train'],best['internal']
        previous=oldbest.get(spec['id'])
        prev='%d/20'%previous['internal_validation']['p1p2_paths'] if previous else '新增'
        lines.append('| %s | %s | %d/80 | %d/20 | %d/%d | %d/%d | %d/20 | %s |'%(
            spec['specification'],prev,a['p1p2_paths'],b['p1p2_paths'],b['p1_violations'],b['transitions'],
            b['p2_violations'],b['bad_transitions'],b['certificate_preview'],best['status']))
    lines+=['','## 非退化及语义检查','',
        '| spec | 初始已完成 | 初始未完成路径P1/P2 | mode内打乱最高通过 | 随机函数最高通过 | 仅monitor rank通过 |',
        '|---|---|---|---|---|---|']
    for r in bests:
        c=r['controls'];b=r['internal']
        shuffle=max(v['p1p2_paths'] for k,v in c.items() if k.startswith('within_mode'))
        random=max(v['p1p2_paths'] for k,v in c.items() if k.startswith('random'))
        baseline='%d/20'%c['monitor_rank_only']['p1p2_paths'] if 'monitor_rank_only' in c else '不适用'
        lines.append('| %s | %s | %d/%d | %d/20 | %d/20 | %s |'%(r['spec']['id'],b['already_complete_at_t0'],
            r['internal_pending_path_pass'],r['internal_initially_pending_count'],shuffle,random,baseline))
    lines+=['','KEEP要求：train/internal所有P1/P2严格通过、P2非空，常数/按mode常数/打乱/随机对照不能全通过。',
        '线性与ELU结果均保存在screening_summary.csv；不能把单步接近100%当作整路径100%。',
        '新增窗口从t=0读取，允许在H=300内任意完整窗口完成；不同于原F[1,17]G[0,47]的64步规范。',
        'Persistence仍是无限公式，有限样本不提供永久完成或全支持集closure证明；未发任何新warrant。',
        '有限图可行仅表示已采样转移约束有解，不表示神经函数必能泛化，也不表示无限性质已证。','',
        '本轮KEEP函数数：%d。'%report['qualified'],
        '精确重算%d条候选路径；monitor独立窗口枚举、权重hash、数据来源及严格AND已核验。'%count]
    (out/'RESULT_CN.md').write_text('\n'.join(lines)+'\n')
    exp.base.write(out/'audit.json',dict(completed=exp.base.stamp(),candidate_paths_replayed=count,
        exact_replay=True,manual_window_replay=True,fit_internal_hashes_disjoint=True,
        code_sha256=exp.base.digest(exp.__file__),audit_code_sha256=exp.base.digest(__file__),
        formal_calibration_accessed=False,formal_test_accessed=False))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    audit(p.parse_args().output.resolve())
