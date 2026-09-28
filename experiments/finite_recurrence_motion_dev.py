"""Finite recurrence development on cached fit data; NEVER opens cal/test.

Reset-budget descent is constructed accounting, not a learned W drift theorem.
All observed edges, including arrivals before reset and censored tails, count.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import safedreamer_b_fit100_train as old
from experiments import safedreamer_high_clearance_pilot as detector
from experiments import safedreamer_recurrence_screening as screen
from experiments.l3_safedreamer_clear_pipeline import digest,write,stamp
from core.finite_recurrence import segments,path_score,combine_gates

SOURCE=ROOT/'artifacts/safedreamer_b_fit100_v1'
CONFIGS=[dict(name='uniform64',width=64,weighted=False,under=1.,penalty=.01,features='full'),
         dict(name='stratified128',width=128,weighted=True,under=3.,penalty=.01,features='full'),
         dict(name='stratified256',width=256,weighted=True,under=3.,penalty=.03,features='full'),
         dict(name='deter128',width=128,weighted=True,under=3.,penalty=.01,features='deter')]


def load(role):
    collection=json.loads((SOURCE/'collection.json').read_text())
    assert collection['formal_calibration_count']==0 and collection['test_count']==0
    rows=[]
    for r in collection['records']:
        if r['role']!=role or not r['eligible']:continue
        f=SOURCE/'raw'/r['file'];assert digest(f)==r['sha256']
        with np.load(f) as a:rows.append(dict(record=r,arrays={k:a[k].copy() for k in a.files}))
    assert len(rows)==(80 if role=='train_fit' else 20)
    return rows


def candidates(paths):
    speed=np.concatenate([screen.signals(p['arrays'])['speed'] for p in paths])
    original=json.loads((SOURCE/'plan.json').read_text())['detector']
    out=[]
    for name,quantiles in [('motion_middle',[.25,.75]),('motion_wide',[.1,.9])]:
        lo,hi=map(float,np.quantile(speed,quantiles))
        out.append(dict(id=name,signal='speed',low=lo,high=hi,threshold_source='80 train pooled quantiles '+str(quantiles)))
    out.append(dict(id='clearance_B',signal='clearance',low=original['d_low'],high=original['d_high'],
                    threshold_source='Previously frozen B detector; unchanged'))
    return out


def products(paths,spec,kind):
    result=[]
    for p in paths:
        sig=screen.signals(p['arrays'])[spec['signal']]
        mon=detector.monitor(sig,spec['low'],spec['high'])
        assert detector.source_consistent(sig,mon,spec['low'],spec['high'])
        full=screen.product_features(p['arrays'],mon,sig,spec['signal'])
        x=full if kind=='full' else np.concatenate([full[:,:256],full[:,512:]],axis=1)
        result.append(dict(record=p['record'],x=x,full=full,signal=sig,monitor=mon))
    return result


def predict(model,paths):
    st=model['state'];net=old.BoundedReturnNet(st['mean'].numpy(),st['scale'].numpy(),model['config']['width'],2.4)
    net.load_state_dict(st);net.eval()
    with torch.no_grad():return [net(torch.tensor(p['x'])).numpy().astype(float) for p in paths]


def evaluate(paths,preds,spec,region,delta):
    rows=[];ys=[];ps=[];allocations=[]
    for path,pred in zip(paths,preds):
        events=path['monitor']['event_times'];total=np.minimum(2.4,np.asarray(pred)+delta)
        mask=detector.inside_region(path['full'],region)
        flags,_=detector.evaluate(path['signal'],path['monitor'],spec['low'],spec['high'],total,0.,mask)
        gate=combine_gates(flags,len(events),2)
        seg=segments(events,300)
        # Independent pathwise allocation check, not the debit-loop implementation.
        manual=all(total[s]+1e-7>=.01*(e-s) for s,e,_ in seg)
        assert manual==flags['delta_margin_pass']
        assert gate['fit_preview_C_rec']==bool(manual and mask.all() and len(events)>=2 and
            flags['event_source_pass'] and flags['transition_soundness_pass'] and flags['drift_pass'])
        for s,e,complete in seg:
            allocations.append(total[s])
            if complete:ys.append(.01*(e-s));ps.append(pred[s])
        pre=np.array(flags['budget_pre_reset']);bad=np.flatnonzero(pre< -1e-7)
        rows.append(dict(fit_index=path['record']['fit_index'],events=events,count=len(events),
            pass_certificate=gate['fit_preview_C_rec'],failure_reasons=gate['failure_reasons'],
            first_budget_failure=int(bad[0]+1) if len(bad) else None,
            margin_min=float(pre.min()),budget_pass=flags['delta_margin_pass'],
            geometric_region_pass=bool(mask.all()),drift_pass=flags['drift_pass'],
            source_pass=flags['event_source_pass'],soundness_pass=flags['transition_soundness_pass']))
    counts=np.array([r['count'] for r in rows]);err=old.metrics_error(ys,ps)
    summary=dict(n=len(paths),certificate=sum(r['pass_certificate'] for r in rows),
        count_pass=int((counts>=2).sum()),event_distribution={str(k):v for k,v in sorted(Counter(counts.tolist()).items())},
        count_min=int(counts.min()),count_median=float(np.median(counts)),count_max=int(counts.max()),
        zero_fraction=float((counts==0).mean()),count_variance=float(counts.var()),
        raw_completed_reset_error=err,allocation_mean=float(np.mean(allocations)),
        cap_fraction=float(np.mean(np.array(allocations)>=2.4-1e-6)),
        failure_breakdown=dict(Counter(reason for r in rows for reason in r['failure_reasons'])))
    return summary,rows


def controls(paths,spec,region,baseline,model_predictions,delta):
    result={}
    for name,value in [('fit_median',baseline),('cap_constant',2.4),('zero',0.)]:
        result[name]=evaluate(paths,[np.full(301,value) for _ in paths],spec,region,0.)[0]
    for seed in range(3):
        predictions=[a.copy() for a in model_predictions]
        sites=[(i,s) for i,p in enumerate(paths) for s,_,_ in segments(p['monitor']['event_times'],300)]
        vals=np.random.RandomState(18203000+seed).permutation([predictions[i][s] for i,s in sites])
        for (i,s),value in zip(sites,vals):predictions[i][s]=value
        result['shuffled%d'%seed]=evaluate(paths,predictions,spec,region,delta)[0]
    return result


def qualifies(train,dev,ctrl):
    # A declared developmental filter, NOT a calibrated theorem or significance test.
    e=dev['raw_completed_reset_error'];baseline=ctrl['fit_median']['raw_completed_reset_error']
    if e['MAE'] is None or baseline['MAE'] is None:return False
    return (train['certificate']/train['n']>=.95 and dev['certificate']/dev['n']>=.95
        and dev['count_variance']>0 and dev['cap_fraction']<.5
        and e['MAE']<baseline['MAE']
        and e['mean_underestimation']<=baseline['mean_underestimation']
        and e['MAE']<np.mean([ctrl['shuffled%d'%i]['raw_completed_reset_error']['MAE'] for i in range(3)]))


def run(out,updates):
    torch.set_num_threads(2);fit=load('train_fit');specs=candidates(fit)
    sourceplan=json.loads((SOURCE/'plan.json').read_text())
    out.mkdir(parents=True,exist_ok=False)
    plan=dict(at=stamp(),specs=specs,configs=CONFIGS,updates=updates,H=300,M_MIN=2,eta=.01,budget_cap=2.4,
        checkpoint=sourceplan['checkpoint'],checkpoint_sha256=sourceplan['checkpoint_sha256'],policy=sourceplan['policy'],
        source=str(SOURCE),source_collection_sha256=digest(SOURCE/'collection.json'),code_sha256=digest(__file__),
        scope='FIT_ONLY finite event-count AND pathwise reset-budget; NOT infinite L3 or W/U theorem',
        counts=dict(train_fit=80,internal_development=20,cal_delta=0,cal_CP=0,test=0),
        detector='causal LOW then later HIGH; must encounter a fresh LOW to emit again; t0 may arm; no dwell/refractory',
        delta='max train whole-path residual only, biased development preview; total allocation capped at2.4',
        region='full RSSM+planner+signal+armed empirical train envelope, no invariant-region claim',
        selection='KEEP_DEV: >=95% train/dev certificates, variable counts, <50% capped allocations, raw MAE beats fit-median and shuffled mean, underestimation no worse than fit-median; see qualifies()',
        controls='matched fit median, constant cap, zero, 3 within-development reset-site permutations; diagnostic only',
        stopping='No automatic calibration even if KEEP_DEV; review feasibility and freeze first',
        future_delta='Fresh disjoint cal_delta, whole-path residual, ceil((n+1)*.95) order; then separate calCP/Test1; no cal-max+arbitrary offset',
        caveat='Debit drift is constructed, so drift_pass alone is NOT evidence a neural ranking function was learned')
    write(out/'plan.json',plan);frozen=[];started=time.monotonic()
    for spec in specs:
        paths=products(fit,spec,'full');core=np.concatenate([p['full'] for p in paths])
        mean=core.mean(0);scale=np.maximum(core.std(0),.05);mean[-2:]=0.;scale[-2:]=1.
        region=dict(mean=mean,scale=scale,radius=float(np.linalg.norm((core-mean)/scale,axis=1).max()))
        np.savez_compressed(out/(spec['id']+'_region.npz'),**region)
        for ci,cfg in enumerate(CONFIGS):
            paths=products(fit,spec,cfg['features']);labels=[];xs=[]
            for path in paths:
                lab=old.labels(path);labels.extend(lab);xs.extend(path['x'][r['t']] for r in lab)
            xx=np.stack(xs);mu=xx.mean(0);sd=np.maximum(xx.std(0),.05);mu[-2:]=0.;sd[-2:]=1.
            x=torch.tensor(xx);y=torch.tensor([r['target'] for r in labels],dtype=torch.float32)
            censor=torch.tensor([r['censored'] for r in labels]);weights=torch.tensor(old.sampling_weights(labels))
            torch.manual_seed(18201000+ci)
            net=old.BoundedReturnNet(mu,sd,cfg['width'],2.4);opt=torch.optim.Adam(net.parameters(),lr=.001)
            for _ in range(updates):
                ix=torch.multinomial(weights,256,replacement=True) if cfg['weighted'] else torch.randint(len(x),(256,))
                pred=net(x[ix]);res=pred-y[ix]
                loss=torch.where(censor[ix],torch.relu(-res).square(),torch.where(res<0,cfg['under']*res.square(),res.square())).mean()
                loss=loss+cfg['penalty']*(pred/2.4).square().mean();opt.zero_grad();loss.backward();opt.step()
            model=dict(state=net.state_dict(),config=cfg,spec=spec)
            preds=predict(model,paths);delta=max(path_score(p,a['monitor']['event_times']) for p,a in zip(preds,paths))
            ys=[.01*(e-s) for a in paths for s,e,completed in segments(a['monitor']['event_times'],300) if completed]
            baseline=float(np.median(ys)) if ys else 0.
            ident=spec['id']+'_'+cfg['name'];torch.save(model,out/(ident+'.pt'))
            tr,_=evaluate(paths,preds,spec,region,delta)
            frozen.append(dict(id=ident,sha256=digest(out/(ident+'.pt')),spec=spec,config=cfg,delta_fit=delta,baseline=baseline,train=tr))
            print('FIT',ident,'certificate',tr['certificate'],'/80','delta',round(delta,4),'elapsed',round(time.monotonic()-started,1),flush=True)
    write(out/'frozen_candidates.json',dict(at=stamp(),records=frozen))
    replay(out)


def replay(out):
    torch.set_num_threads(2);plan=json.loads((out/'plan.json').read_text())
    assert plan['code_sha256']==digest(__file__) and plan['source_collection_sha256']==digest(SOURCE/'collection.json')
    fit=load('train_fit');dev=load('internal_validation')
    assert not {p['record']['fingerprint'] for p in fit}&{p['record']['fingerprint'] for p in dev}
    data=[];pathrows=[]
    for rec in json.loads((out/'frozen_candidates.json').read_text())['records']:
        file=out/(rec['id']+'.pt');assert digest(file)==rec['sha256'];model=torch.load(file,map_location='cpu')
        region=dict(np.load(out/(rec['spec']['id']+'_region.npz')))
        ps=products(fit,rec['spec'],rec['config']['features']);pred=predict(model,ps)
        assert max(path_score(p,a['monitor']['event_times']) for p,a in zip(pred,ps))==rec['delta_fit']
        tr,rows=evaluate(ps,pred,rec['spec'],region,rec['delta_fit']);assert tr==rec['train']
        pathrows.extend([dict(candidate=rec['id'],role='train',**r) for r in rows])
        ds=products(dev,rec['spec'],rec['config']['features']);pr=predict(model,ds)
        dv,rows=evaluate(ds,pr,rec['spec'],region,rec['delta_fit'])
        pathrows.extend([dict(candidate=rec['id'],role='internal',**r) for r in rows])
        ctrl=controls(ds,rec['spec'],region,rec['baseline'],pr,rec['delta_fit'])
        status='KEEP_DEV' if qualifies(tr,dv,ctrl) else 'BORDERLINE'
        data.append(dict(**rec,internal=dv,controls=ctrl,status=status))
    write(out/'report.json',dict(at=stamp(),candidates=data,cal_delta=0,cal_CP=0,test=0))
    detector.csv_write(out/'path_results.csv',pathrows)
    lines=['# 有限 recurrence：80fit/20internal 开发对照','',
        '固定checkpoint、CCEPlanner、H=300、至少2次因果LOW→HIGH。不是无限GF证明。',
        '12个预算网络：3个detector ×4个训练设置，每个'+str(plan['updates'])+'次更新。',
        '只读取已有fit数据；20条internal已被重复用于开发。没有新采样、正式cal_delta、CP校准或Test1。','',
        '| detector/预算模型 | fit完整 | internal次数>=2 | internal完整 | internal原始MAE(步) | 中位常数MAE(步) | 顶到预算上限比例 | 状态 |',
        '|---|---|---|---|---|---|---|---|']
    csvrows=[]
    for r in data:
        t=r['train'];d=r['internal'];c=r['controls']['fit_median']
        lines.append('| %s | %d/80 | %d/20 | %d/20 | %.2f | %.2f | %.1f%% | %s |'%(r['id'],t['certificate'],d['count_pass'],d['certificate'],d['raw_completed_reset_error']['latency_MAE_steps'],c['raw_completed_reset_error']['latency_MAE_steps'],100*d['cap_fraction'],r['status']))
        csvrows.append(dict(candidate=r['id'],train=t['certificate'],internal_count=d['count_pass'],internal_certificate=d['certificate'],status=r['status']))
    for spec in plan['specs']:
        options=[r for r in data if r['spec']['id']==spec['id']]
        best=max(options,key=lambda r:(r['status']=='KEEP_DEV',r['internal']['certificate'],-r['internal']['raw_completed_reset_error']['MAE']))
        d=best['internal'];ctrl=best['controls']
        lines+=['','## '+spec['id'],
            '阈值：'+str(spec['low'])+' → '+str(spec['high'])+'；'+spec['threshold_source'],
            '事件分布fit/internal：'+json.dumps(best['train']['event_distribution'])+' / '+json.dumps(d['event_distribution']),
            '开发最佳：'+best['id']+'；failure breakdown：'+json.dumps(d['failure_breakdown']),
            '内部对照完整通过数：'+json.dumps({k:v['certificate'] for k,v in ctrl.items()})]
    qualified=[r['id'] for r in data if r['status']=='KEEP_DEV']
    lines+=['','## 范围和停止点',
        'KEEP_DEV候选：'+str(qualified),
        '每步下降由预算扣减构造，不能把drift=100%当作学到了原论文W。gate只检查有限观察边，不替代U/Ville或support-wide证明。',
        'Delta是fit最大残差，不是正式conformal结果；预算上限作用于g+Delta整体，不能用Delta绕过。',
        '常数cap可能达到相同证书通过率；因此同时报告原始预测误差、打乱对照和cap饱和，而不把高通过率当非退化证明。',
        '本轮到此停止。若没有KEEP_DEV，不宣称已找到可正式发证的候选；若有，仍须复核后冻结并另采cal_delta/cal_CP/test。']
    (out/'RESULT_CN.md').write_text('\n'.join(lines)+'\n')
    detector.csv_write(out/'comparison.csv',csvrows)
    write(out/'audit.json',dict(at=stamp(),weights_verified=True,fit_replay_exact=True,
        disjoint_fit_dev_fingerprints=True,independent_segment_budget_check=True,independent_event_sources=True,
        candidates=len(data),candidate_paths=1200,control_paths=1440,code_sha256=digest(__file__),
        report_sha256=digest(out/'report.json'),cal_delta=0,cal_CP=0,test=0))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=2000)
    args=p.parse_args();run(args.output.resolve(),args.updates)
