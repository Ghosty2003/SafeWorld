"""Frozen original 48-state certificate: new calibration500 then Test1=1000.

Explicit independent simulator-reset and imagination seeds. No training,
selection, optional stopping, future filtering, or changes to policy/planner.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import l2_clear_window_holdout as old
from experiments import l2_clear_window_v as fitted
from experiments.safedreamer_finite_recurrence_pilot import SeededResetRecorder,extract
from experiments.l3_safedreamer_clear_pipeline import digest,write,stamp

SOURCE=ROOT/'artifacts/safedreamer_l2_clear_window_holdout'
SEEDS={'calibration':151000000,'test':152000000}
COUNTS={'calibration':500,'test':1000}


def cp(k,n):
    return float(beta.ppf(.05,k,n-k+1)) if k else 0.


def full_events(v,done):
    delta=v[:,:-1]-v[:,1:];z=v<.01
    p1=(delta>=0).all(1);p2=(done[:,:-1]|(delta>=.01)).all(1)
    closure=(~z[:,:-1]|z[:,1:]).all(1)
    exclusion=(~z|done).all(1)
    return p1&p2&done[:,-1]&z[:,-1]&closure&exclusion&np.isfinite(v).all(1)&(v>=0).all(1)


def evaluate_arrays(model,a):
    # Preserve original model, monitor and input processing. No refit/clipping.
    row=old.audit(model,a)
    run,done=fitted.monitor(a['aps'])
    features=np.concatenate([old.base.value_features(a,'policy'),run[...,None]/48.],axis=-1)
    v=old.predict_family(model,features,done)
    events=full_events(v,done)
    legacy=np.asarray(row['per_path_candidate_event'])&done[:,-1]
    np.testing.assert_array_equal(events,legacy)
    assert row[old.EVENT]['successes']==int(events.sum())
    row['complete_gate_AND_events']=events.tolist()
    row['explicit_gate_AND_equals_original_event']=True
    row['certificate_without_completion']=int((events&~done[:,-1]).sum())
    return row,v,done


def prepare(out):
    p=json.loads((SOURCE/'plan.json').read_text());frozen=json.loads((SOURCE/'frozen.json').read_text())
    assert digest(SOURCE/'selected.pt')==p['model_sha256']==frozen['model_sha256']
    assert digest(p['checkpoint'])==p['checkpoint_sha256']
    assert digest(SOURCE/'policy_config.json')==p['policy_config_sha256']
    assert p['spec']==fitted.SPEC and p['horizon']==64 and p['eta']==.01
    for file,sha in p['code_sha256'].items():assert digest(file)==sha,('Source changed',file)
    out.mkdir(parents=True,exist_ok=False)
    for name in ['selected.pt','policy_config.json']:shutil.copyfile(SOURCE/name,out/name)
    # Identity metadata only. No previous outcomes are used in this rerun.
    seen=set(json.loads((SOURCE/'prior_identities.json').read_text())['fingerprints'])
    for file in (ROOT/'artifacts').glob('safedreamer*/*_provenance.json'):
        data=json.loads(file.read_text());seen.update(data.get('fingerprints',[]))
        for batch in data.get('batches',[]):seen.update(batch.get('fingerprints',[]))
    for file in (ROOT/'artifacts').glob('safedreamer*/*collection.json'):
        data=json.loads(file.read_text())
        for r in data.get('records',[]):
            if 'fingerprint' in r:seen.add(r['fingerprint'])
            for key in ('reset_seed','imagination_seed'):
                seed=r.get(key,-1)
                assert not any(b<=seed<b+100000 for b in SEEDS.values()),('Seed reuse',file,seed)
    write(out/'prior_identities.json',dict(fingerprints=sorted(seen)))
    code={**p['code_sha256'],str(Path(__file__).resolve()):digest(__file__),
        str(ROOT/'experiments/safedreamer_finite_recurrence_pilot.py'):digest(ROOT/'experiments/safedreamer_finite_recurrence_pilot.py')}
    plan=dict(created=stamp(),source=str(SOURCE),source_plan_sha256=digest(SOURCE/'plan.json'),
        spec=p['spec'],H=64,eta=.01,model_sha256=p['model_sha256'],original_freeze=frozen,
        checkpoint=p['checkpoint'],checkpoint_sha256=p['checkpoint_sha256'],
        policy_sha256=p['policy_config_sha256'],policy=json.loads((out/'policy_config.json').read_text()),
        initial_condition=p['initial_condition'],counts=COUNTS,seeds=SEEDS,max_raw_draws={k:5*n for k,n in COUNTS.items()},
        seed_rule='reset=base+2*draw; imagination=reset+1; actual gym reset explicitly seeded',
        sampling='First N eligible in draw order; only initial decoded goal_dist margin>=0.70; no future filtering',
        inference='Fixed-N iid same-distribution conditional-draw assumption; individual one-sided95% CP lower; no pooled or simultaneous claim',
        action_law='Unmodified CCEPlanner.policy()',warrant_threshold=.95,confidence=.95,
        event='All64 P1/P2 AND completion AND endpoint V<eta AND sampled sublevel closure/exclusion AND finite/nonnegative',
        decision='Calibration CP lower>=.95 AND no observed sublevel-region counterevidence; SAFE finite-model-scope else ABSTAIN',
        scope='64-step finite model-only certificate AND completion; not infinite safety or support-wide L2 proof',
        evaluate_test='Same frozen event, 1000 independent paths; own CP bound; never tune on test',
        freeze_before_sampling=True,code_sha256=code,prior_identities_sha256=digest(out/'prior_identities.json'))
    write(out/'plan.json',plan)
    print('FROZEN',plan['spec'],'cal=500 test=1000; threshold=.95',flush=True)


def check(out):
    p=json.loads((out/'plan.json').read_text())
    for file,sha in p['code_sha256'].items():assert digest(file)==sha,('Frozen code changed',file)
    assert digest(out/'selected.pt')==p['model_sha256']
    assert digest(out/'policy_config.json')==p['policy_sha256']
    assert digest(out/'prior_identities.json')==p['prior_identities_sha256']
    return p


def records_for(out,split):
    return [json.loads(f.read_text()) for f in sorted((out/(split+'_raw')).glob('draw_*.json'))]


def retained_arrays(out,split,records):
    arrays=[]
    for r in records:
        if not r['eligible']:continue
        path=out/(split+'_raw')/r['file'];assert digest(path)==r['sha256']
        with np.load(path,allow_pickle=False) as a:arrays.append({k:a[k].copy() for k in a.files})
    return {k:np.stack([a[k] for a in arrays]) for k in arrays[0]}


def summarize_split(out,split,model):
    plan=check(out);records=records_for(out,split);assert sum(r['eligible'] for r in records)==plan['counts'][split]
    a=retained_arrays(out,split,records);row,v,done=evaluate_arrays(model,a)
    np.savez_compressed(out/(split+'_evaluated.npz'),values=v,done=done,events=row['complete_gate_AND_events'])
    write(out/(split+'_result.json'),row)
    write(out/(split+'_provenance.json'),dict(completed=stamp(),retained=len(a['latent']),draws=len(records),records=records,
        model_sha256=plan['model_sha256'],plan_sha256=digest(out/'plan.json')))
    if split=='calibration':
        decision=old.verdict(row)
        write(out/'prediction.json',dict(saved_at=stamp(),plan_sha256=digest(out/'plan.json'),model_sha256=plan['model_sha256'],
            result_sha256=digest(out/'calibration_result.json'),decision=decision,
            verdict='SAFE' if decision['result']=='FINITE_EVENT_THRESHOLD_MET' else 'ABSTAIN',
            certificate=row[old.EVENT],completion=row['completion_reached'],next_test_count=1000,scope=plan['scope']))
        print('CALIBRATION FROZEN',json.dumps(decision),flush=True)
    else:print('TEST COMPLETE',json.dumps(row[old.EVENT]),flush=True)


def collect(out):
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    torch.set_num_threads(2);plan=check(out);assert digest(plan['checkpoint'])==plan['checkpoint_sha256']
    seen=set(json.loads((out/'prior_identities.json').read_text())['fingerprints'])
    for split in COUNTS:
        for r in records_for(out,split):
            assert r['fingerprint'] not in seen;seen.add(r['fingerprint'])
    model=torch.load(out/'selected.pt',map_location='cpu',weights_only=False)
    extra=build_extra(argparse.Namespace(repo_root=None,checkpoint=plan['checkpoint']))
    extra['action_source']='policy';extra['config_overrides']['jax'].update(platform='gpu',logical_gpus=0)
    cfg=RolloutConfig(horizon=64,n_rollouts=1,seed=151000001,action_source='policy',extra=extra)
    started=time.monotonic()
    with SafeDreamerWrapper(cfg) as wrapper:
        wrapper.load()
        import jax
        assert jax.default_backend()=='gpu'
        actual=dict(action_source='policy',expl_behavior=wrapper._config.expl_behavior,
            planner={k:getattr(wrapper._config.planner,k) for k in plan['policy']['planner']},cost_limit=wrapper._config.cost_limit,
            planner_source_sha256=digest(Path(extra['repo_root'])/'SafeDreamer/behaviors.py'),note=plan['policy']['note'])
        assert actual==plan['policy']
        recorder=SeededResetRecorder(wrapper._env);wrapper._env=recorder
        try:
            for split,n in plan['counts'].items():
                check(out)
                if (out/(split+'_provenance.json')).exists():continue
                if split=='test':
                    pred=json.loads((out/'prediction.json').read_text())
                    assert pred['plan_sha256']==digest(out/'plan.json') and pred['model_sha256']==plan['model_sha256']
                    if not (out/'test_started.json').exists():write(out/'test_started.json',dict(at=stamp(),prediction_sha256=digest(out/'prediction.json')))
                raw=out/(split+'_raw');raw.mkdir(exist_ok=True)
                existing=records_for(out,split);kept=sum(r['eligible'] for r in existing)
                assert len(list(raw.glob('draw_*.npz')))==len(existing),'Uncommitted raw draw; audit before resuming'
                for draw in range(len(existing),plan['max_raw_draws'][split]):
                    if kept==n:break
                    reset=plan['seeds'][split]+2*draw;recorder.pending=reset
                    data=wrapper.sample_latent_rollouts(RolloutConfig(horizon=64,n_rollouts=1,seed=reset+1,action_source='policy',extra=extra))
                    assert data['action_source']=='policy' and recorder.pending is None
                    a=extract(data,recorder.observation)
                    a['aps']=np.asarray([[s[k] for k in ('goal_dist','hazard_dist','velocity')] for s in data['aps'][0]],dtype=np.float32)
                    assert a['latent'].shape==(65,512) and a['aps'].shape==(65,3)
                    # Exact same initial predicate as the original experiment.
                    eligible=bool(a['aps'][0,0]>=.70)
                    fp=hashlib.sha256(a['latent'].tobytes()).hexdigest()
                    assert fp not in seen,'Repeated raw trajectory; abort, never silently replace';seen.add(fp)
                    file=raw/('draw_%05d.npz'%draw);assert not file.exists();np.savez_compressed(file,**a)
                    r=dict(split=split,draw=draw,file=file.name,sha256=digest(file),fingerprint=fp,eligible=eligible,
                        retained_index=kept if eligible else None,initial_goal_margin=float(a['aps'][0,0]),reset_seed=reset,
                        imagination_seed=reset+1,time=stamp())
                    write(file.with_suffix('.json'),r);kept+=int(eligible)
                    if draw%10==0 or kept==n:print('PROGRESS',split,kept,'/',n,'draws',draw+1,'elapsed_s',round(time.monotonic()-started,1),flush=True)
                assert kept==n,'Draw budget exhausted; no partial-N warrant'
                summarize_split(out,split,model)
        finally:recorder.restore();wrapper._env=recorder.outer


def audit(out):
    torch.set_num_threads(2);plan=check(out)
    model=torch.load(out/'selected.pt',map_location='cpu',weights_only=False)
    seen=set(json.loads((out/'prior_identities.json').read_text())['fingerprints']);used_seeds=set();rows={};raw_count=0
    prediction=json.loads((out/'prediction.json').read_text());test_start=json.loads((out/'test_started.json').read_text())
    assert prediction['plan_sha256']==digest(out/'plan.json')
    assert test_start['prediction_sha256']==digest(out/'prediction.json')
    assert prediction['saved_at']<=test_start['at']
    for split,n in plan['counts'].items():
        records=records_for(out,split)
        for draw,r in enumerate(records):
            assert draw==r['draw'] and r['reset_seed']==plan['seeds'][split]+2*draw and r['imagination_seed']==r['reset_seed']+1
            assert r['fingerprint'] not in seen;seen.add(r['fingerprint'])
            for key in ['reset_seed','imagination_seed']:assert r[key] not in used_seeds;used_seeds.add(r[key])
            file=out/(split+'_raw')/r['file'];assert digest(file)==r['sha256']
            with np.load(file,allow_pickle=False) as a:
                assert hashlib.sha256(a['latent'].tobytes()).hexdigest()==r['fingerprint']
                assert bool(a['aps'][0,0]>=.70)==r['eligible']
                recompute=np.array([bool((a['aps'][s:s+48,1]>=0).all()) for s in range(1,18)]).any()
                assert fitted.monitor(a['aps'][None])[1][0,-1]==recompute
            if split=='test':assert r['time']>=test_start['at']
        assert sum(r['eligible'] for r in records)==n
        assert records[-1]['eligible'];raw_count+=len(records)
        a=retained_arrays(out,split,records);row,v,d=evaluate_arrays(model,a)
        assert row==json.loads((out/(split+'_result.json')).read_text())
        with np.load(out/(split+'_evaluated.npz')) as saved:np.testing.assert_array_equal(saved['values'],v)
        rows[split]=row
    assert prediction['result_sha256']==digest(out/'calibration_result.json')
    assert prediction['decision']==old.verdict(rows['calibration'])
    result=dict(completed=stamp(),raw_draws=raw_count,retained=plan['counts'],all_seed_ranges_disjoint=True,
        raw_hashes_verified=True,old_identities_excluded=True,exact_replay=True,detector_independent_replay=True,
        prediction_before_test=True,calibration=rows['calibration'],test=rows['test'],prediction=prediction,
        test_defects=dict(certificate_without_completion=rows['test']['certificate_without_completion'],
            candidate_region_status=rows['test']['sampled_region_status']),scope=plan['scope'])
    write(out/'audit.json',result)
    c=rows['calibration'][old.EVENT];t=rows['test'][old.EVENT]
    header='Layer | Specification | MP class | Backend / carrier | H | Verdict | p_hat_gamma (CP lower, 95%) | Test 1 (N=1000)'
    table=header+'\n'+' | '.join(['---']*8)+'\nL2 | F[1,17] G[0,47](¬hazard) | Guarantee (bounded) | SafeDreamer / SafetyPointGoal1-v0 | 64 | %s | %.4f | %d/1000, %.4f\n'%(prediction['verdict'],c['cp_lower'],t['successes'],t['cp_lower'])
    (out/'final_table.md').write_text(table)
    text=['# 冻结48-state规范：新校准与Test1','',table,'',
        '固定原linear_max4（step4800）V、eta=.01、48-state detector、H=64、checkpoint和CCEPlanner。',
        '500条新校准先固定结论，再生成1000条独立Test1。无训练、调参、根据结果追加样本或未来成功筛选。',
        '与旧实验一致，仅按初始解码goal距离>=1.0筛起点；现在显式记录实际gym reset及imagination独立seed。',
        '这里SAFE仅指固定有限时域model-scope完整证书事件概率下界过门槛，不是无限时域安全或支持集全局证明。','',
        '| 指标 | calibration500 | Test1 1000 |','|---|---|---|',
        '| 完整证书 | %d/500 | %d/1000 |'%(c['successes'],t['successes']),
        '| 完成48-state窗口 | %d/500 | %d/1000 |'%(rows['calibration']['completion_reached']['successes'],rows['test']['completion_reached']['successes']),
        '| 单侧95%%证书CP下界 | %.6f | %.6f |'%(c['cp_lower'],t['cp_lower']),
        '| P1违反转移 | %d | %d |'%(rows['calibration']['p1']['violations'],rows['test']['p1']['violations']),
        '| P2违反转移 | %d | %d |'%(rows['calibration']['p2']['violations'],rows['test']['p2']['violations']),
        '| 未完成却通过完整证书 | %d | %d |'%(rows['calibration']['certificate_without_completion'],rows['test']['certificate_without_completion']),
        '', '校准判定原因：'+json.dumps(prediction['decision'],ensure_ascii=False),
        'Test1 sublevel检查：'+rows['test']['sampled_region_status'],
        'Test1自身下界独立报告，不替代校准下界、不用于重选V；这些95%置信界是各自的界，不是联合95%声明。',
        '没有观察到错误发证也不能证明错误概率为零。有限样本预测成立依赖固定模型、策略、起点分布与独立同分布抽样假设。',
        '审计：所有raw（含起点被排除者）均留存；seed/哈希/时序/完整AND/独立窗口检测及指标重算核验通过。']
    (out/'RESULT_CN.md').write_text('\n'.join(text)+'\n')
    print(table,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','collect','audit']);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a.output.resolve())
