"""Replay sampled constraints and controls; write a qualified-or-abstain audit."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import safedreamer_l2_spec_search as s


def run(out):
    torch.set_num_threads(2)
    plan=json.loads((out/'plan.json').read_text());report=json.loads((out/'report.json').read_text())
    assert s.digest(Path(s.__file__))==plan['code_sha256']
    assert s.digest(plan['checkpoint'])==plan['checkpoint_sha256']
    assert s.digest(ROOT/'wrappers/safedreamer_wrapper.py')==plan['wrapper_sha256']
    train,records=s.load_train();assert records==plan['train_records']
    collection=json.loads((out/'internal_collection.json').read_text());val=[]
    assert collection['calibration']==collection['test']==0
    train_seeds={r[k] for r in records for k in ('reset_seed','imagination_seed')}
    val_seeds=[r[k] for r in collection['records'] for k in ('reset_seed','imagination_seed')]
    assert len(val_seeds)==len(set(val_seeds)) and not train_seeds.intersection(val_seeds)
    train_fps={r['fingerprint'] for r in records};val_fps=[r['fingerprint'] for r in collection['records']]
    assert not train_fps.intersection(val_fps) and len(val_fps)==len(set(val_fps))
    frozen=json.loads((out/'candidates_frozen.json').read_text())
    assert all(r['time']>frozen['at'] for r in collection['records'])
    for r in collection['records']:
        file=out/'internal_raw'/r['file'];assert s.digest(file)==r['sha256']
        if r['eligible']:
            with np.load(file,allow_pickle=False) as a:val.append({k:a[k].copy() for k in a.files})
    assert len(train)==80 and len(val)==20
    thresholds=np.quantile(np.concatenate([s.signals(a)[0] for a in train]),[.25,.75])
    assert all(np.array_equal(thresholds,[c['low'],c['high']]) for c in plan['candidates'])
    region=np.load(out/'region.npz');cases=0;control_cases=0;graph_rows=[]
    for spec in plan['candidates']:
        x,q,b,d=s.product_data(train,spec);g,w=s.graph_feasibility(x,q,b)
        assert g==json.loads((out/(spec['id']+'_feasibility.json')).read_text());graph_rows.append(dict(id=spec['id'],**g))
        if w is not None:
            with np.load(out/(spec['id']+'_graph_witness.npz')) as stored:np.testing.assert_array_equal(w,stored['values'])
    for r in report['all_candidates']:
        file=out/r['weights'];assert s.digest(file)==r['sha256'];model=torch.load(file,map_location='cpu',weights_only=False)
        for paths,key in ((train,'train'),(val,'internal_validation')):
            x,q,b,d=s.product_data(paths,r['spec'])
            core=np.stack([s.physical(a) for a in paths]);mask=np.linalg.norm((core-region['mean'])/region['scale'],axis=-1)<=float(region['radius'])+1e-10
            v=s.predict(model,x,b,d);actual=s.statistics(v,b,d,r['spec']['persistence'],mask)
            assert actual==r[key],(r['spec']['id'],r['width'],key);cases+=len(paths)
        assert s.controls(model,x,b,d,r['spec']['persistence'])==r['controls'];control_cases+=len(r['controls'])*len(val)
        joint=all(r[k]['p1_violations']==0 and r[k]['p2_violations']==0 and r[k]['bad_transitions']>0 for k in ('train','internal_validation'))
        assert joint==r['strict_sampled_100']
    qualified=[r for r in report['rows'] if r['status']=='KEEP']
    assert len(qualified)==report['qualified_count']
    usable=[r for r in report['rows'] if 'train' in r]
    closest=max(usable,key=lambda r:(r['internal_validation']['p1p2_paths']/r['internal_validation']['n'],
        min(r['internal_validation']['p1_pass'],r['internal_validation']['p2_pass'] or 0),
        r['train']['p1p2_paths']/r['train']['n'])) if usable else None
    result=dict(completed=s.stamp(),constraint_path_replays=cases,control_path_replays=control_cases,
        checkpoint_hash_verified=True,wrapper_hash_verified=True,code_hash_verified=True,
        thresholds_fit_only=True,validation_new_after_freeze=True,seeds_disjoint=True,latent_fingerprints_disjoint=True,
        exact_replay=True,graph_feasibility=graph_rows,calibration_count=0,test_count=0,
        qualified_count=len(qualified),closest_borderline=None if closest is None else dict(
            spec=closest['spec']['id'],width=closest['width'],train=closest['train'],validation=closest['internal_validation']),
        audit_code_sha256=s.digest(__file__))
    s.write(out/'audit.json',result)
    header='Layer | Specification | MP class | Backend / carrier | H | Verdict | p_hat_gamma (CP lower, 95%) | Test 1 (N=1000)'
    if not qualified:
        table=header+'\n'+' | '.join(['---']*8)+'\nL2 | no-qualified-spec | — | SafeDreamer / SafetyPointGoal1-v0 | 300 | ABSTAIN | —（未校准） | —（未运行）\n'
        (out/'final_table.md').write_text(table)
    lines=['# SafeDreamer L2 specification screening 审计结果','',
        '结论：'+report['conclusion']+'。这不是规范为假、模型必然不安全或有效V不存在的证明。','',
        '本轮固定已审计checkpoint和CCEPlanner.policy()，H=300。80条既有fit路径，仅使用其train_fit分区；',
        '另采20条新internal-validation，seed从141000000起，与以往internal-check及正式cal/test分离。',
        '18个权重均在新internal-validation采样前冻结。候选/阈值/架构/训练规则事先写入plan.json。',
        '这20条用于本轮选择，之后不能再当最终校准或独立测试。未读取旧calibration/test样本或结果。','',
        '## 固定定义','',
        'LOW阈值=%.10f，HIGH阈值=%.10f，来自80条fit路径的clearance状态分布25%%/75%%分位数。'%tuple(thresholds),
        'clearance=min(norm(decoded hazard vectors))，无hazard定义为clearance>=0.2，goal半径为严格<0.3。',
        'q_t消费到t的观测；初始标签在t=0消费。P2使用源状态q_t。每条301个状态恰好300条真实转移，无人工末尾自环。',
        'eta=0.01。所有评价用float64、P1/P2容差均为0。P1检查全部转移，P2检查q_t∈B。',
        'V输入为完整RSSM、planner carry、当前解码clearance/goal距离和因果自动机mode；没有剩余时间、路径ID或未来信息。',
        '训练辅助标签使用有限图rank；未完成路径仅施加下界，不把H截断当作成功。',
        'Z_free={V<eta}；bad-mode输出结构性>=eta。Guarantee接受态V=0且自动机吸收；这不证明物理安全区域不变。',
        'Persistence的good状态不是吸收态，有限安全后缀不等于已经永久完成。','',
        '## 候选与结果','',
        '| Spec | MP class | B / accepting | train P1违反 | train P2违反 | internal P1违反 | internal P2违反 | 整路径P1P2 train/internal | preview完整事件 | 状态 |',
        '|---|---|---|---|---|---|---|---|---|---|']
    for r in report['rows']:
        spec=r['spec']
        if 'train' not in r:continue
        t=r['train'];v=r['internal_validation']
        lines.append('| %s | %s | %s / %s | %d/%d | %d/%d | %d/%d | %d/%d | %d/80；%d/20 | %d/20 | %s |'%(
            spec['specification'],spec['mp_class'],spec['bad_set'],spec['accepting'],t['p1_violations'],t['transitions'],
            t['p2_violations'],t['bad_transitions'],v['p1_violations'],v['transitions'],v['p2_violations'],v['bad_transitions'],
            t['p1p2_paths'],v['p1p2_paths'],v['certificate_preview'],r['status']))
    lines+=['','以上每个spec选一个最佳候选；全部18个函数及其常数/按mode打乱/随机函数对照见report.json。',
        'P1/P2比例分母不同；不能用大量接受态零差值掩盖pending状态的P2违反。','',
        '## 有限图可行性的限制','',
        '以完整product state的精确字节身份建图，正权边表示要求下降eta。SCC内有正权边即不可行。',
        '没有正权环时用凝聚DAG最长加权路径构造非负可行见证。没有合并近似latent，也没有虚构末尾转移。',
        '本轮图由有限路径构成；可行仅说明采到的约束有解，不说明某个神经函数可泛化，亦不说明无限支持集可证。','',
        '## 完整certificate AND和非退化检查','',
        'preview = 全路径P1 AND P2 AND endpoint AND Zfree/bad-exclusion AND sampled closure AND region AND finite/nonnegative。',
        'region在fit的标准化完整物理状态半径上固定；不利用validation扩大区域。',
        '真正常数V、仅按mode的常数、5次mode内打乱V、5个随机网络均检查；不以仅测全局常数替代其余对照。',
        'Persistence另缺无限closure依据，不把sampled predicate持续为真写成无限warrant。','',
        '## 模型选择与停止','',
        '每个spec尝试线性softplus、128×128 ELU、256×256 ELU，各1200更新；train每200更新选checkpoint。',
        '在新internal-validation上优先整路径P1/P2通过、再平衡P1/P2单步通过率；完全没有使用CP下界选模型。',
        '未达到要求时只记BORDERLINE；不能发SAFE。也不能由P1/P2失败推出spec violation。',
        '正式calibration和Test1均未运行，p_hat_gamma不填0而标未计算。',
        'Persistence候选既没有有限完成证明，也没有无限closure证明；与有限终止Guarantee区别见plan.json。','',
        '## 复核','',
        '%d条候选路径和%d条对照路径独立重算完全一致；数据、checkpoint、代码、权重哈希已复核。'%(cases,control_cases),
        '审计文件：audit.json；逐模型细节：report.json；对比表：screening_summary.csv；总表：final_table.md。']
    lines+=['','## 起点已完成与最接近候选','',
        '| Spec | train 起点已完成 | internal 起点已完成 | internal 单纯完成 | internal 全路径P1/P2 |',
        '|---|---|---|---|---|']
    for r in usable:
        t=r['train'];v=r['internal_validation']
        lines.append('| %s | %s | %s | %s | %d/20 |'%(r['spec']['id'],t['already_complete_at_t0'],
            v['already_complete_at_t0'],v['finite_completion_paths'],v['p1p2_paths']))
    if closest is not None:
        t=closest['train'];v=closest['internal_validation']
        lines+=['','按预先规定的全路径P1/P2优先规则，最接近BORDERLINE：%s，width=%d。'%(closest['spec']['specification'],closest['width']),
            'train整路径通过%d/80；internal通过%d/20。internal P1违反%d/%d，P2违反%d/%d。'%(t['p1p2_paths'],v['p1p2_paths'],
                v['p1_violations'],v['transitions'],v['p2_violations'],v['bad_transitions']),
            '这只是本轮开发候选排序，不是KEEP，也不是统计置信声明。',
            '所有候选已经在用户要求的数值100%条件处失败；Persistence的额外语义保留条件没有改变本轮结论。']
    (out/'RESULT_CN.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output.resolve())
