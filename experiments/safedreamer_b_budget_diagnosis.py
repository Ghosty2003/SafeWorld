"""Candidate B frozen-weight budget diagnosis; cached fit only, no calibration."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.safedreamer_high_clearance_pilot import (
    load_fit, monitor, product, inside_region, evaluate, csv_write, distribution)
from experiments.safedreamer_finite_recurrence_pilot import BudgetNet
from experiments.l3_safedreamer_clear_pipeline import digest, write, stamp
from core.finite_recurrence import segments, combine_gates


def prediction_metrics(rows, constant):
    # A censored tail is not a measured return time. Do not include it in MAE.
    complete = [r for r in rows if r['completed']]
    if not complete: return dict(n=0)
    y = np.array([r['required_budget'] for r in complete])
    pred = np.array([r['raw_g'] for r in complete])
    corr = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred)>0 and np.std(y)>0 else None
    return dict(n=len(y), learned_mae=float(np.abs(pred-y).mean()),
        constant_mae=float(np.abs(constant-y).mean()), correlation=corr,
        note='Completed segments only; segments within paths are correlated, descriptive not confidence estimates.')


def failure_detail(flags, events, m):
    pre = np.asarray(flags['budget_pre_reset'])
    hits = np.flatnonzero(pre < -1e-7)
    first = int(hits[0]+1) if len(hits) else None
    reached = events[m-1] if len(events)>=m else None
    return dict(first_budget_failure=first, Mth_event_time=reached,
        budget_failure_after_Mth_event=bool(first is not None and reached is not None and first>reached),
        minimum_margin=float(pre.min()))


def run(args):
    torch.set_num_threads(2)
    source = args.source.resolve()
    old = json.loads((source/'report.json').read_text())
    oldplan = json.loads((source/'plan.json').read_text())
    audit = json.loads((source/'audit.json').read_text())
    for name, sha in audit['outputs'].items(): assert digest(source/name)==sha, name
    for name, sha in oldplan['code_sha256'].items(): assert digest(name)==sha, name
    _, paths = load_fit(Path(oldplan['source']))
    b = next(s for s in old['summaries'] if s['candidate']=='B')
    low, high, delta = b['d_low'], b['d_high'], b['delta_fit_preview']
    weights = torch.load(source/'g_init_B.pt', map_location='cpu', weights_only=True)
    assert digest(source/'g_init_B.pt')==b['model_sha256']
    state = weights['state']; net = BudgetNet(state['mean'], state['scale'])
    net.load_state_dict(state); net.eval()
    region = dict(mean=state['mean'].numpy(), scale=state['scale'].numpy(), radius=weights['region_radius'])
    original = {(r['fit_index'],r['preview']):r for r in json.loads((source/'traces.json').read_text())
        if r['candidate']=='B'}
    segment_rows = []; prepared = []
    for p in paths:
        mon = monitor(p['clearance'], low, high); x = product(p['arrays'], mon['armed'])
        with torch.no_grad(): pred = net(torch.tensor(x)).numpy().astype(float)
        inside = inside_region(x, region)
        flags, resets = evaluate(p['clearance'], mon, low, high, pred, delta, inside)
        assert flags == original[p['fit_index'],'delta_fit']['flags']
        assert resets == original[p['fit_index'],'delta_fit']['resets']
        prepared.append(dict(path=p, monitor=mon, pred=pred, inside=inside))
        for j,(s,e,completed) in enumerate(segments(mon['event_times'],300)):
            pair = mon['pairs'][j] if completed else None
            armed_t = pair['low_entry_timestep'] if completed else mon['pending_low_entry']
            segment_rows.append(dict(fit_index=p['fit_index'], partition=p['partition'], segment=j,
                start=s, end=e, completed=completed, armed_at_start=mon['armed'][s],
                low_entry=armed_t, wait_to_low=None if armed_t is None else armed_t-s,
                low_to_return=e-armed_t if completed else None,
                observed_length=e-s, raw_g=float(pred[s]), corrected_budget=float(pred[s]+delta),
                required_budget=.01*(e-s), margin=float(pred[s]+delta-.01*(e-s)),
                segment_first_budget_failure=next((t for t in range(s+1,e+1)
                    if pred[s]+delta-.01*(t-s)<-1e-7),None)))
    train = [r for r in segment_rows if r['partition']=='train']
    matched = float(np.mean([r['corrected_budget'] for r in train]))
    train_max = max(r['required_budget'] for r in train)
    mean_target = float(np.mean([r['required_budget'] for r in train if r['completed']]))
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    write(out/'plan.json', dict(created=stamp(), candidate='B', d_low=low, d_high=high,
        threshold_scope='FROZEN_FOR_THIS_DIAGNOSIS_NOT_FORMAL_CALIBRATION', M_MIN_main=2, M_MIN_control=1,
        source=str(source), model_sha256=b['model_sha256'], code_sha256=digest(__file__),
        source_report_sha256=digest(source/'report.json'), delta_fit_only=delta,
        fixed=['16 cached fit paths','weights','events','eta=.01','H=300','original gate definitions'],
        scales=[.5,.75,1.,1.25,1.5,2.],
        matched_constant_train_mean_budget=matched, constant_train_max_observed_segment=train_max,
        constant_H_eta=3., constant_regression_train_completed_mean=mean_target,
        split_counts=dict(fit=16, cal_delta=0, cal_CP=0, test=0),
        retraining=False, new_sampling=False, confidence=None, formal_M_MIN=None))
    rows=[]
    for item in prepared:
        p,mon,pred,inside = [item[k] for k in ('path','monitor','pred','inside')]
        arms=[('learned',1.,pred+delta),('constant_matched',1.,np.full(301,matched)),
            ('constant_train_max',1.,np.full(301,train_max)),('constant_H_eta',1.,np.full(301,3.))]
        for scale in [.5,.75,1.25,1.5,2.]:
            arms.extend([('learned',scale,(pred+delta)*scale),
                ('constant_matched',scale,np.full(301,matched*scale))])
        for arm,scale,budget in arms:
            flags,_=evaluate(p['clearance'],mon,low,high,budget,0.,inside)
            starts=[s for s,_,_ in segments(mon['event_times'],300)]
            for m in (1,2):
                result=combine_gates(flags,mon['event_count'],m)
                rows.append(dict(fit_index=p['fit_index'],partition=p['partition'],
                    seed=p['meta']['imagination_seed'], arm=arm,scale=scale,M_MIN=m,
                    event_count=mon['event_count'],event_times=mon['event_times'],
                    budget_pass=flags['delta_margin_pass'],fit_preview_pass=result['fit_preview_C_rec'],
                    failure_reasons=result['failure_reasons'],
                    mean_reset_budget=float(np.mean(budget[starts])),
                    **failure_detail(flags,mon['event_times'],m)))
    summary=[]
    for arm,scale in sorted({(r['arm'],r['scale']) for r in rows}):
        for m in (1,2):
            for partition in ('all','train','internal_check'):
                selected=[r for r in rows if r['arm']==arm and r['scale']==scale and r['M_MIN']==m
                    and (partition=='all' or r['partition']==partition)]
                n=len(selected)
                summary.append(dict(arm=arm,scale=scale,M_MIN=m,partition=partition,n=n,
                    count_pass=sum(r['event_count']>=m for r in selected),
                    budget_pass=sum(r['budget_pass'] for r in selected),
                    full_pass=sum(r['fit_preview_pass'] for r in selected),
                    full_pass_rate=sum(r['fit_preview_pass'] for r in selected)/n,
                    event_sufficient_but_budget_fail=sum(r['event_count']>=m and not r['budget_pass'] for r in selected),
                    budget_failure_after_Mth_event=sum(r['budget_failure_after_Mth_event'] for r in selected),
                    failure_breakdown=dict(Counter(v for r in selected for v in r['failure_reasons']))))
    metrics={}
    for partition in ('train','internal_check'):
        selected=[r for r in segment_rows if r['partition']==partition]
        metrics[partition]=dict(prediction=prediction_metrics(selected,mean_target),
            completed_lengths=distribution([r['observed_length'] for r in selected if r['completed']]),
            tail_lengths=distribution([r['observed_length'] for r in selected if not r['completed']]),
            mean_budget=distribution([r['corrected_budget'] for r in selected]),
            completed_margin=distribution([r['margin'] for r in selected if r['completed']]),
            tail_margin=distribution([r['margin'] for r in selected if not r['completed']]))
    # Full-scale baseline must agree with original results; no altered gate.
    for part in ('all','train','internal_check'):
        for m in (1,2):
            previous=next(r for r in old['breakdown'] if r['candidate']=='B' and r['partition']==part
                and r['preview']=='delta_fit' and r['M_MIN_candidate']==m)
            current=next(r for r in summary if r['arm']=='learned' and r['scale']==1
                and r['partition']==part and r['M_MIN']==m)
            assert current['full_pass']==previous['certificate_successes']
    csv_write(out/'segments.csv',segment_rows); csv_write(out/'path_diagnostics.csv',rows)
    csv_write(out/'summary.csv',summary)
    write(out/'report.json',dict(summary=summary,metrics=metrics,
        constant_matched=matched,constant_train_max=train_max,constant_regression=mean_target,
        status='FIT_ONLY_DIAGNOSIS_NO_WARRANT_NO_CALIBRATION',
        limitations=['No independent predictive accuracy estimate: all paths are development data.',
            'Matched budget averages reset allocations on TRAIN observed segment starts, not unconditional state distribution.',
            'Large constant dominance in pass rate is mathematical, not evidence of superior prediction.',
            'Full-horizon gate may fail after the Mth event; event completion and certificate are distinct.',
            'Finite pathwise constructed drift is not an infinite recurrence theorem.']))
    write(out/'audit.json',dict(time=stamp(),source_hashes_verified=True,baseline_flags_exact_replay=True,
        baseline_decisions_match=True,models_and_sources_unchanged=True,
        report_sha256=digest(out/'report.json'),rows=len(rows),segments=len(segment_rows),
        split_counts=dict(fit=16,cal_delta=0,cal_CP=0,test=0)))
    for r in summary:
        if r['scale']==1 and r['partition']=='all': print(r,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'artifacts/safedreamer_high_clearance_pilot_v1')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
