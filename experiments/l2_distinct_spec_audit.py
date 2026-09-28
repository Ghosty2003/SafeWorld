"""Replay non-window L2 screening and retain the unsuccessful candidates."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import l2_distinct_spec_screen as screen
base=screen.base;exp=screen.exp


def manual_completion(sig,spec):
    # Independent ordered first-hit scan (not the state-update implementation).
    previous=-1
    for pred in spec['stages']:
        values=sig[pred['signal']]
        hit=values<=pred['threshold'] if pred['op']=='le' else values>=pred['threshold']
        indices=np.flatnonzero(hit & (np.arange(len(hit))>previous))
        if not len(indices):return None
        previous=int(indices[0])
    if spec['safety_until_completion'] and (sig['clearance'][:previous+1]<.2).any():return None
    return previous


def audit(out):
    torch.set_num_threads(2)
    plan=json.loads((out/'plan.json').read_text());report=json.loads((out/'report.json').read_text())
    assert base.digest(screen.__file__)==plan['code_sha256']
    assert base.digest(exp.SOURCE/'plan.json')==plan['source_plan_sha256']
    fit,records=base.load_train();internal,hashes=exp.load_internal()
    assert records==plan['fit_records'] and hashes==report['internal_hashes']
    assert not set(hashes)&{r['sha256'] for r in records}
    specs,thresholds=screen.candidates(fit);assert specs==plan['specs'] and thresholds==plan['thresholds']
    region=np.load(exp.SOURCE/'region.npz')
    masks=[np.linalg.norm((np.stack([base.physical(a) for a in paths])-region['mean'])/region['scale'],axis=-1)
           <=float(region['radius'])+1e-10 for paths in (fit,internal)]
    frozen=json.loads((out/'frozen.json').read_text())['records'];replayed=0
    for rec in frozen:
        assert base.digest(out/rec['weights'])==rec['sha256']
        m=torch.load(out/rec['weights'],map_location='cpu')
        candidates=[r for r in report['rows'] if r['id']==rec['id']]
        for label,paths,mask in zip(('train','internal'),(fit,internal),masks):
            x,q,b,d,p=screen.products(paths,rec['spec']);v=screen.predict(m,x,d,p)
            with np.load(out/(rec['id']+'_values.npz')) as saved:np.testing.assert_array_equal(saved[label],v)
            for path,done in zip(paths,d):
                first=manual_completion(screen.signals(path),rec['spec'])
                indices=np.flatnonzero(done)
                assert first==(int(indices[0]) if len(indices) else None)
            for row in candidates:
                vv=row['gain']*v;s=base.statistics(vv,b,d,region=mask);assert s==row[label]
                pp,cc=screen.independent(vv,b,d,mask)
                np.testing.assert_array_equal(pp,s['p1p2_per_path']);np.testing.assert_array_equal(cc,s['certificate_per_path'])
                replayed+=len(paths)
        # Reproduce all scaled negative controls, not only main candidates.
        cvs=[('constant',np.ones_like(v)),('mode_constant',np.where(d,0.,1.)),('signal_prior_only',.02*p)]
        for seed in range(3):
            rng=np.random.RandomState(158100000+seed);sv=v.copy()
            for mode in np.unique(q):sv[q==mode]=rng.permutation(v[q==mode])
            cvs.append(('shuffle%d'%seed,sv));torch.manual_seed(158101000+seed)
            rm=dict(m);rm['weights']=screen.Value(x.shape[-1],m['family']).state_dict()
            cvs.append(('random%d'%seed,screen.predict(rm,x,d,p)))
        for row in candidates:
            for name,cv in cvs:assert base.statistics(row['gain']*cv,b,d)==row['controls'][name]
    lines=['# SafeDreamer：非连续安全窗口的 L2 specification screening','',
        '本轮仅80条train-fit、20条复用internal-development；固定已审计checkpoint和CCEPlanner，H=300、eta=0.01、零数值容差。',
        '8个不同事件任务×3函数家族×3预设尺度，共24个训练网络、72个尺度候选；每网络3000次优化更新。',
        '未读取正式校准/Test1。内部集已多次参与开发选择，不能据此声明独立泛化置信保证。',
        '没有重复“连续N步无hazard”窗口，也没有更改先前结果。','',
        '所有谓词顺序必须在0..300内完成，阶段之间至少间隔一个采样状态。q是因果阶段计数；接受态吸收；B是所有未接受模式。',
        'safe_recover在完成前发生hazard即永久拒绝；完成之后不再要求无hazard。其余规范本身不保证避障。',
        'Zfree={V<0.01}，坏模式V>=0.01，接受模式V=0。有限图可行只覆盖已有转移，不是无限时域证明。','',
        '## 阈值（全部来自fit）','',json.dumps(plan['thresholds'],ensure_ascii=False,indent=2),'',
        '净空/目标距离/速度/位移都是decoder输出派生量。Goal-relative distance不是环境goal_met；位移相对decoded t0位置，不是实际里程。','',
        '## 每个任务最佳候选，包括没有成功的任务','',
        '| 任务 | fit任务完成 | internal任务完成 | fit整路径P1/P2 | internal整路径P1/P2 | internal P1违反 | internal P2违反 | 完整证书preview | 状态 |',
        '|---|---|---|---|---|---|---|---|---|']
    failure_rows=[]
    for r in report['best_per_spec']:
        a,b=r['train'],r['internal'];spec=r['spec']
        lines.append('| %s | %d/80 | %d/20 | %d/80 | %d/20 | %d/6000 | %d/%d | %d/20 | %s |'%(
            spec['name'],a['finite_completion_paths'],b['finite_completion_paths'],a['p1p2_paths'],b['p1p2_paths'],
            b['p1_violations'],b['p2_violations'],b['bad_transitions'],b['certificate_preview'],r['status']))
        failure_rows.append(dict(spec=spec['id'],function=r['id'],gain=r['gain'],
            path_P1P2_failures=20-b['p1p2_paths'],endpoint_failures=b['endpoint_failures'],
            region_failures=b['region_failures'],closure_failures=b['closure_failure_paths'],bad_in_zfree=b['bad_in_z_free'],
            safety_rejected=r['internal_events']['safety_rejected']))
    lines+=['','P1检查全部300条转移；P2只在源模式属于B时要求下降eta。整路径比例不能由单步比例替代。',
        'KEEP_DEV必须train/internal严格100%且常数、打乱、随机对照不全通过；否则BORDERLINE。没有内部完成的任务单独REJECT。','',
        '## 具体函数和失败解释','']
    for r in report['best_per_spec']:
        b=r['internal'];c=r['controls'];events=r['internal_events']
        chosen=[v for v in frozen if v['id']==r['id']][0]
        m=torch.load(out/chosen['weights'],map_location='cpu')
        x,q,bad,done,p=screen.products(internal,r['spec']);v=r['gain']*screen.predict(m,x,done,p)
        delta=np.diff(v,axis=1);ix=np.argwhere(bad[:,:-1] & (delta>-.01))[:5]
        examples=[dict(path=int(i),source_t=int(t),mode_from=int(q[i,t]),mode_to=int(q[i,t+1]),delta_V=float(delta[i,t])) for i,t in ix]
        lines+=['### '+r['spec']['name'],'',
            '- 定义：'+json.dumps(r['spec']['stages'],ensure_ascii=False),
            '- 函数：%s，输出乘%.1f；初始已完成%d/20。'%(r['id'],r['gain'],b['already_complete_at_t0']),
            '- 失败项（可能重叠）：P1/P2失败%d条；终点失败%d条；区域gate失败%d条；closure失败%d条；安全拒绝%d条。'%(
                20-b['p1p2_paths'],b['endpoint_failures'],b['region_failures'],b['closure_failure_paths'],events['safety_rejected']),
            '- 对照整路径通过：'+str({k:v['p1p2_paths'] for k,v in c.items()}),
            '- 前5个P2失败转移：'+json.dumps(examples,ensure_ascii=False),
            '- 完成时刻（未完成记null）：'+json.dumps(events['first_completion']),
            '- 解读：'+('20/20已完成行为，当前失败来自V/证书检查而不是任务未完成；放大V不能修复正的delta_V。' if b['finite_completion_paths']==20 else
                       '任务未完成与证书函数失败均需单独检查；P1/P2不通过本身不是规范为假的证明。'),'']
    old=json.loads((exp.SOURCE/'report.json').read_text())
    lines+=['## 此前还试过但未达到双100%的非窗口规范','',
        '| 规范 | 当轮最佳内部整路径P1/P2 | 说明 |','|---|---|---|']
    for r in old['rows']:
        lines.append('| %s | %d/20 | 当前函数未达到双100%%，不等于数学上无解 |'%(r['spec']['specification'],r['internal_validation']['p1p2_paths']))
    lines+=['','上表是早期6规范筛选；高净空任务后来单独换函数曾提高到13/20，仍未全通过。',
        '## 结论','',
        '本轮KEEP_DEV尺度候选数：%d。'%report['qualified'],
        '低速后进入高速最接近双100%，但多数路径在2–10步完成（另有45步）；它衡量启动/运动能力，不代表避障、导航或长期recurrence。',
        '本轮只筛选，不发SAFE，不计算CP下界，不启动正式calibration/Test1。',
        '所有未成功项保留在all_candidates.csv；failure_breakdown.csv按gate给出失败原因，report.json保留每条路径结果。',
        '复核：%d条尺度候选路径、全部随机/打乱对照、独立first-hit事件检测、完整AND、数据/权重哈希重算通过。'%replayed]
    (out/'RESULT_CN.md').write_text('\n'.join(lines)+'\n')
    with (out/'failure_breakdown.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(failure_rows[0]));w.writeheader();w.writerows(failure_rows)
    audit_path=out/'audit.json'
    if audit_path.exists():audit_path=out/('audit_replay_'+base.digest(__file__)[:12]+'.json')
    base.write(audit_path,dict(at=base.stamp(),candidate_path_replays=replayed,all_controls_replayed=True,
        independent_first_hit=True,independent_gate_AND=True,fit_internal_disjoint=True,
        thresholds_fit_only=True,calibration_accessed=False,test_accessed=False,code_sha256=base.digest(screen.__file__),audit_sha256=base.digest(__file__)))
    print('AUDIT PASS',replayed,'candidate paths; report',out/'RESULT_CN.md',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);audit(p.parse_args().output.resolve())
