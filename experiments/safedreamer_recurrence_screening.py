"""Bounded fit-only spec screening. No calibration/test data or CP computation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.safedreamer_high_clearance_pilot import (
    load_fit, monitor, product, inside_region, evaluate, csv_write, distribution)
from experiments.safedreamer_finite_recurrence_pilot import train_budget
from experiments.safedreamer_b_budget_diagnosis import prediction_metrics
from experiments.l3_safedreamer_clear_pipeline import digest, write, stamp
from core.finite_recurrence import segments, path_score, combine_gates

FAMILIES=[
    ('clearance','clearance',False,'LOW clearance -> HIGH clearance; relative clearance recovery',0),
    ('motion','speed',False,'LOW decoded speed -> HIGH decoded speed; movement resumes, not goal success',2),
    ('approach','approach',False,'FAR from decoded goal -> NEAR decoded goal; not original goal attainment',1),
    ('progress','progress',False,'Receding from decoded goal -> approaching it; may reflect decoded jitter',3),
    ('clearance_safe','clearance',True,'Clearance returns AND no decoded hazard throughout states0..300',0),
    ('motion_safe','speed',True,'Motion resumes AND no decoded hazard throughout states0..300',2),
]
QUANTILES=[('q25_75',.25,.75),('q10_90',.10,.90)]


def signals(a):
    o=np.asarray(a['decoded'],float)
    if o.shape!=(301,29) or not np.isfinite(o).all(): raise ValueError('Invalid decoded observation')
    goal=np.linalg.norm(o[:,7:9],axis=1)
    return dict(clearance=np.linalg.norm(o[:,9:25].reshape(-1,8,2),axis=2).min(1),
        speed=np.linalg.norm(o[:,:2],axis=1),approach=1/(1+goal),
        progress=(1+np.tanh(np.r_[0.,goal[:-1]-goal[1:]]/.1))/2)


def product_features(a,mon,signal,kind):
    x=product(a,mon['armed'])
    # Progress uses previous goal distance: include its current causal signal
    # to make the product-state budget aware of detector-relevant memory.
    return np.column_stack([x[:,:-1],signal,x[:,-1]]).astype(np.float32)


def augmented_gate(flags,events,safety_required,safety_ok):
    result=combine_gates(flags,len(events),2)
    result['gates']['safety_pass']=bool(safety_ok) if safety_required else None
    if safety_required and not safety_ok:
        result['failure_reasons'].append('SAFETY_CONJUNCT_FAIL')
        result['fit_preview_C_rec']=False
    return result


def characterize(counts):
    a=np.array(counts)
    return dict(counts=list(map(int,a)),n=len(a),min=int(a.min()),median=float(np.median(a)),max=int(a.max()),
        variance=float(np.var(a)),zero_fraction=float(np.mean(a==0)),
        count_pass_fraction=float(np.mean(a>=2)),all_paths_count_pass=bool((a>=2).all()),
        modal_fraction=float(max(Counter(a).values())/len(a)))


def status_for(train,check,mae,baseline_mae,gap):
    # Predeclared multi-criterion rules; no CP, no sorting by success rate.
    if train['median']<1 or check['median']<1 or train['variance']==0 or train['zero_fraction']>.5:
        return 'REJECT','No useful repeated-event distribution in the fit split'
    if train['median']>6 or check['median']>6:
        return 'BORDERLINE','Events too frequent for intended interpretable 1-6 return regime'
    if train['median']<2 or check['median']<2 or train['zero_fraction']>.25 or check['zero_fraction']>.25:
        return 'BORDERLINE','Recurrence sparse or weak internal-check coverage'
    if mae is None or baseline_mae is None or mae>=baseline_mae or gap>.35:
        return 'BORDERLINE','Budget generalization lacks sufficient descriptive advantage'
    return 'KEEP','Nonconstant repeated events plus provisional budget advantage; not certification-ready'


def choose(details):
    # Favor passing all descriptive checks, then predeclared semantic simplicity.
    # Prefer middle-gap thresholds; no maximization of certificate success.
    order={'KEEP':0,'BORDERLINE':1,'REJECT':2}
    ranked=sorted(details,key=lambda d:(order[d['status']],d['semantic_priority'],
        d['safety_required'],d['threshold_label']!='q25_75',d['id']))
    eligible=[d for d in ranked if d['status']!='REJECT']
    if not eligible: return None,None
    primary=eligible[0]
    backup=next((d for d in eligible if d['signal']!=primary['signal']),None)
    return primary,backup


def run(args):
    torch.set_num_threads(2)
    source=args.source.resolve(); original,paths=load_fit(source)
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    ss=[signals(p['arrays']) for p in paths]
    configs=[]
    for family,signal,safety,semantic,priority in FAMILIES:
        pool=np.concatenate([s[signal] for s in ss[:12]])
        for label,ql,qh in QUANTILES:
            low,high=map(float,np.quantile(pool,[ql,qh]))
            if not low<high: raise ValueError('Degenerate thresholds: '+family)
            if signal=='progress' and not low<.5<high: raise ValueError('Progress thresholds must straddle zero improvement')
            configs.append(dict(id=family+'_'+label,family=family,signal=signal,safety_required=safety,
                semantic=semantic,semantic_priority=priority,threshold_label=label,d_low=low,d_high=high,
                quantiles=[ql,qh],dwell=1,refractory=0,M_MIN=2))
    plan=dict(created=stamp(),stage='FIT_ONLY_SCREENING',source=str(source),new_rollouts=0,
        source_plan_sha256=digest(source/'plan.json'),horizon=300,policy=original['policy'],
        checkpoint=original['checkpoint'],checkpoint_sha256=original['checkpoint_sha256'],
        start_distribution=original['start_distribution'],split_counts=dict(fit=16,cal_delta=0,cal_CP=0,test=0),
        thresholds_from='12 train-fit paths; pooled states0..300; not iid observations',candidates=configs,
        M_MIN=2,updates=400,eta=.01,shuffle_seeds=[31000+i for i in range(10)],
        ranking='Event eligibility, budget MAE vs train-mean constant, gap <=.35, then semantic priority; never CP',
        semantic_priority=['clearance','approach','motion','progress'],
        KEEP_does_not_mean='No guarantee of nondegenerate certificate or infinite recurrence',
        formal_future_delta_rule=dict(method='split conformal whole-path upper residual',alpha=.05,
            score='max(0,max_segments(eta*(end-start)-g(x_start))); includes observed tail',
            order='ceil((n_cal_delta+1)*(1-alpha)); k>n -> INVALID_NO_WARRANT, never finite replacement',
            prohibit_retraining=True),
        candidate_code_hashes={str(p):digest(p) for p in [Path(__file__),ROOT/'experiments/safedreamer_high_clearance_pilot.py',
            ROOT/'experiments/safedreamer_finite_recurrence_pilot.py',ROOT/'core/finite_recurrence.py']})
    # Written before any fitting/evaluation: candidate and ranking rules immutable for this run.
    write(out/'plan.json',plan)
    details=[]; csvrows=[]; failures=[]; saved_rows=[]
    for config in configs:
        signal,low,high=config['signal'],config['d_low'],config['d_high']
        fit=[]
        for p,s in zip(paths,ss):
            mon=monitor(s[signal],low,high)
            fit.append(dict(monitor=mon,x=product_features(p['arrays'],mon,s[signal],signal)))
        net,region,training=train_budget(fit,12,400)
        torch.save(dict(state=net.state_dict(),region_radius=region['radius']),out/(config['id']+'.pt'))
        with torch.no_grad(): predictions=[net(torch.tensor(f['x'])).numpy().astype(float) for f in fit]
        delta=max(path_score(p,f['monitor']['event_times']) for p,f in zip(predictions[:12],fit[:12]))
        seg=[]
        for i,(f,pred) in enumerate(zip(fit,predictions)):
            for s,e,done in segments(f['monitor']['event_times'],300):
                seg.append(dict(fit_index=i,partition=paths[i]['partition'],start=s,end=e,completed=done,
                    required_budget=.01*(e-s),raw_g=float(pred[s]),budget=float(pred[s]+delta)))
        tr=[r for r in seg if r['partition']=='train']; completed=[r['required_budget'] for r in tr if r['completed']]
        regression_constant=float(np.mean(completed)) if completed else 0.
        matched=float(np.mean([r['budget'] for r in tr]))
        rows=[]
        def evaluate_arm(arm,preds,correction,shuffle_seed=None):
            for i,(f,p,sig,pr) in enumerate(zip(fit,paths,ss,preds)):
                mon=f['monitor']; inside=inside_region(f['x'],region)
                flags,resets=evaluate(sig[signal],mon,low,high,pr,correction,inside)
                safety_ok=bool((sig['clearance']>=.2).all())
                gate=augmented_gate(flags,mon['event_times'],config['safety_required'],safety_ok)
                rows.append(dict(candidate=config['id'],fit_index=i,partition=p['partition'],arm=arm,
                    shuffle_seed=shuffle_seed,count=mon['event_count'],events=mon['event_times'],
                    low_entries=mon['low_entry_timestamps'],event_pairs=mon['pairs'],
                    safety_pass=safety_ok,drift_pass=flags['drift_pass'],budget_pass=flags['delta_margin_pass'],
                    region_pass=flags['region_gate_pass'],soundness_pass=flags['transition_soundness_pass'],
                    event_source_pass=flags['event_source_pass'],full_pass=gate['fit_preview_C_rec'],
                    failure_reasons=gate['failure_reasons'],resets=resets,
                    geometric_outside=flags['n_region_outside_states']))
        evaluate_arm('learned',predictions,delta)
        evaluate_arm('constant_matched',[np.full(301,matched) for _ in fit],0.)
        evaluate_arm('constant_H_eta',[np.full(301,3.) for _ in fit],0.)
        for seed in plan['shuffle_seeds']:
            rng=np.random.default_rng(seed); permuted=[p.copy()+delta for p in predictions]
            # Diagnostic only: permute allocations among existing reset sites within
            # each split. Future event locations are not used to generate learned inputs.
            for part in ('train','internal_check'):
                sites=[(r['fit_index'],r['start']) for r in seg if r['partition']==part]
                values=np.array([permuted[i][s] for i,s in sites]); values=rng.permutation(values)
                for (i,s),v in zip(sites,values): permuted[i][s]=v
            evaluate_arm('shuffled',permuted,0.,seed)
        stats={}
        for part in ('train','internal_check'):
            idx=[i for i,p in enumerate(paths) if p['partition']==part]
            selected=[r for r in rows if r['partition']==part]
            learned=[r for r in selected if r['arm']=='learned']
            count_stats=characterize([fit[i]['monitor']['event_count'] for i in idx])
            predmetrics=prediction_metrics([r for r in seg if r['partition']==part],regression_constant)
            trivial=[int((ss[i][signal][1:]>=high).sum()) for i in idx]
            # A hold-high pulse is deliberately an invalid detector: compare full
            # timestamps to show it cannot pass source consistency for chosen spec.
            mismatches=sum(np.flatnonzero(ss[i][signal][1:]>=high).tolist()!=
                [t-1 for t in fit[i]['monitor']['event_times']] for i in idx)
            stats[part]=dict(events=count_stats,prediction=predmetrics,
                learned_pass=sum(r['full_pass'] for r in learned),n=len(idx),
                arm_pass_rates={arm:float(np.mean([r['full_pass'] for r in selected if r['arm']==arm]))
                    for arm in ('learned','constant_matched','constant_H_eta','shuffled')},
                shuffled_passes_per_seed=[sum(r['full_pass'] for r in selected if r['arm']=='shuffled'
                    and r['shuffle_seed']==seed) for seed in plan['shuffle_seeds']],
                trivial_high_only_count_stats=characterize(trivial),trivial_timestamp_mismatch_paths=mismatches,
                mechanical_48_count=6,mechanical_count_pass_fraction=1.,
                failure_breakdown=dict(Counter(v for r in learned for v in r['failure_reasons'])))
            for arm in ('learned','constant_matched','constant_H_eta','shuffled'):
                rs=[r for r in selected if r['arm']==arm]
                failures.append(dict(candidate=config['id'],partition=part,arm=arm,n_evaluations=len(rs),
                    drift_fail=sum(not r['drift_pass'] for r in rs),budget_fail=sum(not r['budget_pass'] for r in rs),
                    region_fail=sum(not r['region_pass'] for r in rs),soundness_fail=sum(not r['soundness_pass'] for r in rs),
                    source_fail=sum(not r['event_source_pass'] for r in rs),
                    safety_fail=sum(config['safety_required'] and not r['safety_pass'] for r in rs),
                    count_fail=sum(r['count']<2 for r in rs),full_pass=sum(r['full_pass'] for r in rs)))
        gap=stats['train']['arm_pass_rates']['learned']-stats['internal_check']['arm_pass_rates']['learned']
        checkpred=stats['internal_check']['prediction']
        status,reason=status_for(stats['train']['events'],stats['internal_check']['events'],
            checkpred.get('learned_mae'),checkpred.get('constant_mae'),gap)
        detail=dict(**config,status=status,status_reason=reason,stats=stats,train_check_gap=gap,
            delta_fit=delta,matched_budget=matched,training=training,segments=seg,
            model_sha256=digest(out/(config['id']+'.pt')),region_radius=region['radius'],
            degeneracy_warning='Large constants remove budget difficulty; shuffled and trivial controls do not certify anything',
            semantic_warning='Decoded model-space signals only, no physical-ground-truth or infinite recurrence claim')
        details.append(detail); saved_rows.extend(rows)
        csvrows.append(dict(candidate=config['id'],event_definition=config['semantic'],d_low=low,d_high=high,
            train_counts=stats['train']['events'],internal_counts=stats['internal_check']['events'],
            train_certificate=f"{stats['train']['learned_pass']}/12",internal_certificate=f"{stats['internal_check']['learned_pass']}/4",
            train_check_gap=gap,internal_budget_MAE=checkpred.get('learned_mae'),
            internal_latency_MAE_steps=None if checkpred.get('learned_mae') is None else checkpred['learned_mae']/.01,
            internal_constant_MAE=checkpred.get('constant_mae'),
            internal_controls=stats['internal_check']['arm_pass_rates'],
            failure_breakdown=stats['internal_check']['failure_breakdown'],
            degeneracy_warning=detail['degeneracy_warning'],status=status,reason=reason))
        print(config['id'],status,'counts',stats['train']['events']['counts'],stats['internal_check']['events']['counts'],
            'passes',stats['train']['learned_pass'],stats['internal_check']['learned_pass'],flush=True)
    primary,backup=choose(details)
    write(out/'spec_candidate_details.json',dict(details=details,primary=None if primary is None else primary['id'],
        backup=None if backup is None else backup['id'],confidence=None,CP_lower=None,stage='FIT_ONLY'))
    write(out/'path_diagnostics.json',saved_rows)
    csv_write(out/'spec_screening_summary.csv',csvrows); csv_write(out/'spec_failure_breakdown.csv',failures)
    if primary is not None:
        write(out/'frozen_primary.json',dict(frozen=stamp(),selection_status=primary['status'],
            scope='FROZEN_FINITE_EXPERIMENT_DESIGN_NOT_A_VALIDATED_CERTIFICATE',
            definition={k:primary[k] for k in ['id','family','signal','semantic','d_low','d_high','M_MIN','dwell','refractory','safety_required']},
            hysteresis='Initial state may arm. Only subsequent LOW->HIGH emits once then unarms. Inclusive <=low, >=high.',
            horizon=300,eta=.01,model=str(out/(primary['id']+'.pt')),model_sha256=primary['model_sha256'],
            checkpoint=original['checkpoint'],checkpoint_sha256=original['checkpoint_sha256'],policy=original['policy'],
            start_distribution=original['start_distribution'],feature_rule='RSSM + planner carry + current causal scalar signal + ARMED',
            budget_rule='At t0 and legal returns g(x)+Delta; subtract .01 each edge before optional reset; no other resets',
            gate_rule='All300 edges: drift AND margin AND empirical-region/low-budget-source AND soundness AND event-source AND count>=2 AND safety if selected; CV/sliding disabled',
            region_rule='Frozen train feature normalization and max normalized L2 norm envelope, not invariant proof',
            region_radius=primary['region_radius'],delta_rule=plan['formal_future_delta_rule'],
            delta_fit_preview_not_for_formal_use=primary['delta_fit'],formal_delta_value=None,
            code_sha256=plan['candidate_code_hashes'],calibration_started=False,test_started=False,
            heldout_use='Future roles require disjoint path seeds from all existing fit/dev/cal/test; no tuning after release',
            readiness='Budget degeneracy remains; freeze does not establish predictive or infinite-time guarantee'))
    write(out/'audit.json',dict(time=stamp(),code_sha256=digest(__file__),source_hashes_checked=True,
        fit_paths=16,cal_delta=0,cal_CP=0,test=0,candidates=len(details),models=len(details),
        cases=len(saved_rows),output_hashes={p.name:digest(p) for p in out.iterdir() if p.is_file()}))
    print('PRIMARY',None if primary is None else primary['id'],'BACKUP',None if backup is None else backup['id'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'artifacts/safedreamer_finite_recurrence_pilot_v2')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
