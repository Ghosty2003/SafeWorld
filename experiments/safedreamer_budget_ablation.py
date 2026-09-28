"""Read-only source-data budget scaling; all outcomes remain fit diagnostics."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.finite_recurrence import budget_diagnostics,combine_gates,safe_window_monitor,reference_events,segments
from experiments.safedreamer_finite_recurrence_pilot import BudgetNet,features,hazard_mask,region_membership
from experiments.l3_safedreamer_clear_pipeline import digest,write,stamp

SCALES=[.5,.75,1.,1.25,1.5,2.]


def dist(values):
    if not len(values): return None
    a=np.asarray(values,float)
    return dict(n=len(a),mean=float(a.mean()),**dict(zip(('min','q10','median','q90','max'),
        map(float,np.quantile(a,[0,.1,.5,.9,1])))))


def minimal_scale(pred,events,delta,arm,eta=.01):
    required=[]
    for s,e,_ in segments(events,len(pred)-1):
        denominator=float(pred[s])+(delta if arm=='whole_budget' else 0.)
        numerator=eta*(e-s)-(delta if arm=='g_only' else 0.)
        required.append(max(0.,numerator)/denominator if denominator>0 else (0. if numerator<=0 else float('inf')))
    return float(max(required))


def scaled_diagnostics(pred,events,ref,delta,region,scale,arm):
    if arm not in ('g_only','whole_budget'): raise ValueError('Unknown scale arm')
    flags=budget_diagnostics(np.asarray(pred,float)*scale,events,ref,
                            delta if arm=='g_only' else delta*scale,region)
    decision=combine_gates(flags,len(events),6)
    pre=np.array(flags['budget_pre_reset']); post=np.array(flags['budget_post_reset'])
    candidates=[]
    for t in np.flatnonzero((pre< -1e-7)|~np.isfinite(pre)): candidates.append(int(t+1))
    for t in np.flatnonzero((post< -1e-7)|~np.isfinite(post)): candidates.append(int(t))
    for t in np.flatnonzero(~np.asarray(region,bool)): candidates.append(int(t))
    for t in np.flatnonzero(pre-post[:-1]>-.01+1e-7): candidates.append(int(t+1))
    if events!=ref: candidates.extend(sorted(set(events)^set(ref)))
    if len(events)<6: candidates.append(len(pred)-1)
    return dict(scale=scale,arm=arm,fit_preview_C_rec=decision['fit_preview_C_rec'],
        failure_reasons=decision['failure_reasons'],first_failure_time=min(candidates) if candidates else None,
        minimum_budget_margin=flags['min_pre_reset_budget'],
        minimum_drift_slack=float((-.01-(pre-post[:-1])).min()),
        n_unsound_transitions=flags['n_unsound_transitions'],
        gates=decision['gates'])


def run(args):
    torch.set_num_threads(2); source=args.source.resolve(); out=args.output.resolve()
    plan=json.loads((source/'plan.json').read_text()); old=json.loads((source/'report.json').read_text())
    meta=json.loads((source/'fit_model.json').read_text()); audit=json.loads((source/'audit.json').read_text())
    assert audit['exact_replay'] and digest(source/'report.json')==audit['report_sha256']
    for name,sha in plan['code_sha256'].items(): assert digest(name)==sha,name
    assert digest(source/'g_init.pt')==meta['sha256']
    checkpoint=torch.load(source/'g_init.pt',map_location='cpu',weights_only=True)
    state=checkpoint['state']; net=BudgetNet(state['mean'],state['scale']); net.load_state_dict(state); net.eval()
    region=dict(mean=state['mean'].numpy(),scale=state['scale'].numpy(),radius=checkpoint['region_radius'])
    delta=meta['delta_fit_preview']
    out.mkdir(parents=True,exist_ok=False)
    write(out/'plan.json',dict(created=stamp(),source=str(source),source_plan_sha256=digest(source/'plan.json'),
        source_report_sha256=digest(source/'report.json'),source_model_sha256=meta['sha256'],
        code_sha256=digest(__file__),M_MIN=6,M_MIN_scope='USER_SELECTED_FOR_SUBSEQUENT_PILOT_ONLY',
        final_experiment_frozen=False,scales=SCALES,delta_fit_only=delta,
        primary_arm='g_only: B_s=scale*g_init(x_s)+fixed Delta_fit',
        sensitivity_arm='whole_budget: B_s=scale*(g_init(x_s)+Delta_fit)',
        fixed=['16 paths','g_init weights','eta=.01','events','empirical region','CV/sliding disabled'],
        scope='FIT_ONLY_ABLATION_NOT_CALIBRATION',new_rollouts=0,cal_delta_count=0,cal_CP_count=0,test_count=0))
    paths=[]; rows=[]; segment_rows=[]; phase_values={p:{c:[] for c in range(48)} for p in ('train','internal_check')}
    for oldrow in old['rows']:
        file=source/'fit'/oldrow['file']; r=json.loads(file.with_suffix('.json').read_text())
        assert digest(file)==r['sha256'] and r['eligible']
        with np.load(file,allow_pickle=False) as f: a={k:f[k].copy() for k in f.files}
        monitor=safe_window_monitor(hazard_mask(a['decoded'])); x=features(a,monitor['counter'])
        events=monitor['event_times']; ref=reference_events(hazard_mask(a['decoded']))
        with torch.no_grad(): pred=net(torch.tensor(x)).numpy().astype(float)
        inside=region_membership(x,region)
        base=budget_diagnostics(pred,events,ref,delta,inside)
        assert base==oldrow['previews']['delta_fit_only']['flags']
        identity=dict(fit_index=oldrow['fit_index'],partition=oldrow['fit_partition'])
        thresholds={arm:minimal_scale(pred,events,delta,arm) for arm in ('g_only','whole_budget')}
        # Analytic boundary checked independently through the full gate evaluator.
        permanently_ok=bool(inside.all() and events==ref and len(events)>=6)
        for arm,threshold in thresholds.items():
            assert scaled_diagnostics(pred,events,ref,delta,inside,threshold+1e-5,arm)['fit_preview_C_rec']==permanently_ok
            if threshold>1e-3:
                assert not scaled_diagnostics(pred,events,ref,delta,inside,threshold-1e-3,arm)['fit_preview_C_rec']
        paths.append(dict(**identity,event_count=len(events),event_times=events,
            minimum_required_scale=thresholds,nonbudget_gates_pass=permanently_ok))
        for arm in ('g_only','whole_budget'):
            for scale in SCALES:
                rows.append(dict(**identity,**scaled_diagnostics(pred,events,ref,delta,inside,scale,arm)))
        for j,(s,e,completed) in enumerate(segments(events,300)):
            segment_rows.append(dict(**identity,segment=j,start=s,end=e,completed=completed,
                observed_length=e-s,required_budget=.01*(e-s),g_prediction=float(pred[s]),
                corrected_budget=float(pred[s]+delta),residual=float(.01*(e-s)-pred[s]),
                margin=float(pred[s]+delta-.01*(e-s))))
        for c in range(48):
            # Compare neural g on states with the SAME safe-counter phase; not
            # the constructed budget's linear-in-time shape, which is tautological.
            phase_values[identity['partition']][c].extend(pred[monitor['counter']==c].tolist())
    summary=[]
    for arm in ('g_only','whole_budget'):
        for scale in SCALES:
            r=[x for x in rows if x['arm']==arm and x['scale']==scale]
            summary.append(dict(arm=arm,scale=scale,n=16,successes=sum(x['fit_preview_C_rec'] for x in r),
                train_successes=sum(x['fit_preview_C_rec'] for x in r if x['partition']=='train'),
                check_successes=sum(x['fit_preview_C_rec'] for x in r if x['partition']=='internal_check'),
                first_failure_time=dist([x['first_failure_time'] for x in r if x['first_failure_time'] is not None]),
                minimum_budget_margin=min(x['minimum_budget_margin'] for x in r),
                failure_breakdown=dict(Counter(reason for x in r for reason in x['failure_reasons']))))
    comparisons={}
    for partition in ('train','internal_check'):
        selected=[x for x in segment_rows if x['partition']==partition]
        complete=[x for x in selected if x['completed']]
        comparisons[partition]=dict(n_paths=12 if partition=='train' else 4,
            completed_length_counts=dict(Counter(x['observed_length'] for x in complete)),
            completed_segment_distributions={key:dist([x[key] for x in complete]) for key in
                ('observed_length','required_budget','g_prediction','corrected_budget','residual','margin')},
            all_segment_distributions={key:dist([x[key] for x in selected]) for key in
                ('g_prediction','corrected_budget','residual','margin')},
            per_completed_segment_position={str(j):dict(g=dist([x['g_prediction'] for x in complete if x['segment']==j]),
                required=dist([x['required_budget'] for x in complete if x['segment']==j])) for j in range(6)},
            phase_profile={str(c):dist(v) for c,v in phase_values[partition].items()})
    tr=np.array([np.mean(phase_values['train'][c]) for c in range(48)])
    ck=np.array([np.mean(phase_values['internal_check'][c]) for c in range(48)])
    shift=float((tr-ck).mean())
    shape=dict(matched_counter_phase_mean_correlation=float(np.corrcoef(tr,ck)[0,1]),
        mean_fit_minus_check=shift,raw_rmse=float(np.sqrt(np.mean((tr-ck)**2))),
        offset_corrected_rmse=float(np.sqrt(np.mean((tr-(ck+shift))**2))),
        note='Descriptive, pooled correlated states from12/4 paths. No causal/existence or independent generalization conclusion.')
    write(out/'report.json',dict(completed=stamp(),summary=summary,per_path=paths,rows=rows,segment_rows=segment_rows,
        comparisons=comparisons,shape_comparison=shape,
        minimum_scale_all16={arm:max(p['minimum_required_scale'][arm] for p in paths) for arm in ('g_only','whole_budget')},
        scope='FIT_ONLY_NO_CONFIDENCE_NO_WARRANT',M_MIN=6,M_MIN_scope='PILOT_ONLY',
        calibration_started=False,test_started=False,confidence=None))
    for filename,data in [('scale_summary.csv',summary),('path_scale_rows.csv',rows),('segment_budgets.csv',segment_rows),('path_thresholds.csv',paths)]:
        with (out/filename).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    write(out/'audit.json',dict(time=stamp(),source_baseline_exact_replay=True,all16_analytic_boundaries_checked=True,
        source_model_and_path_hashes_checked=True,report_sha256=digest(out/'report.json'),
        code_sha256=digest(__file__),new_rollouts=0,source_files_modified=False))
    print(json.dumps(dict(summary=summary,shape=shape,minimum_scale_all16={arm:max(p['minimum_required_scale'][arm] for p in paths) for arm in ('g_only','whole_budget')}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'artifacts/safedreamer_finite_recurrence_pilot_v2')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
