"""Bounded remaining-return-time fit on80 paths; select on20 fresh internal paths."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.safedreamer_recurrence_screening import signals,product_features
from experiments.safedreamer_high_clearance_pilot import monitor,inside_region,evaluate,csv_write
from experiments.l3_safedreamer_clear_pipeline import digest,write,stamp
from core.finite_recurrence import segments,path_score,combine_gates


class BoundedReturnNet(nn.Module):
    def __init__(self,mean,scale,width,cap):
        super().__init__();self.cap=cap
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(len(mean),width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,1))
    def forward(self,x):return self.cap*torch.sigmoid(self.net((x-self.mean)/self.scale).squeeze(-1))


def labels(path):
    """Completed next-return times; censored rows only bound observed remaining time."""
    events=path['monitor']['event_times']; rows=[]
    for j,(s,e,complete) in enumerate(segments(events,300)):
        pair=path['monitor']['pairs'][j] if complete else None
        for t in range(s,e):
            remaining=e-t
            # Sample by the completed LOW->HIGH episode's latency, while the
            # regression target is time remaining from this state's timestamp.
            latency=None if pair is None else e-pair['low_entry_timestep']
            bucket=int(np.searchsorted([48,120,240],latency,side='left')) if complete else 4
            rows.append(dict(t=t,target=.01*remaining,censored=not complete,bucket=bucket,
                segment_start=s,segment_end=e,
                low_to_high_latency=latency))
    return rows


def sampling_weights(rows):
    """Balance LOW->HIGH latency strata, oversampling long completed episodes."""
    counts=Counter(r['bucket'] for r in rows); priorities={0:1.,1:1.,2:3.,3:5.,4:1.}
    return np.array([priorities[r['bucket']]/counts[r['bucket']] for r in rows],float)


def load_paths(source,records,role,plan):
    result=[];lo=plan['detector']['d_low'];hi=plan['detector']['d_high']
    for r in records:
        if not r['eligible'] or r['role']!=role:continue
        file=source/'raw'/r['file'];assert digest(file)==r['sha256']
        with np.load(file,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
        sig=signals(a)['clearance'];mon=monitor(sig,lo,hi)
        x=product_features(a,mon,sig,'clearance')
        result.append(dict(record=r,x=x,signal=sig,monitor=mon))
    assert len(result)==plan['roles'][role]['n']
    return result


def metrics_error(y,p):
    if not len(y):return dict(n=0,MAE=None,mean_underestimation=None)
    y=np.asarray(y);p=np.asarray(p);under=np.maximum(y-p,0)
    return dict(n=len(y),MAE=float(np.abs(y-p).mean()),mean_underestimation=float(under.mean()),
        underestimate_fraction=float((p<y-1e-7).mean()),underestimation_p90=float(np.quantile(under,.9)),
        latency_MAE_steps=float(np.abs(y-p).mean()/.01),underestimation_steps=float(under.mean()/.01))


def eval_paths(paths,predictions,delta,region,plan,arm):
    y=[];p=[];alloc=[];all_y=[];all_p=[];rows=[];segment_rows=[]
    cap=plan['budget_cap']
    for path,pred in zip(paths,predictions):
        mon=path['monitor'];total=np.minimum(cap,np.asarray(pred)+delta)
        mask=inside_region(path['x'],region)
        flags,resets=evaluate(path['signal'],mon,plan['detector']['d_low'],plan['detector']['d_high'],total,0.,mask)
        result=combine_gates(flags,mon['event_count'],2)
        seg=segments(mon['event_times'],300)
        for j,(s,e,completed) in enumerate(seg):
            required=.01*(e-s)
            if completed:y.append(required);p.append(pred[s]);alloc.append(total[s])
            segment_rows.append(dict(arm=arm,role=path['record']['role'],fit_index=path['record']['fit_index'],
                start=s,end=e,completed=completed,required=required,predicted=float(pred[s]),
                allocated=float(total[s]),margin=float(total[s]-required),cap_impossible=required>cap+1e-7,
                low_to_high_latency=None if not completed else e-mon['pairs'][j]['low_entry_timestep']))
        for r in labels(path):
            if not r['censored']:all_y.append(r['target']);all_p.append(pred[r['t']])
        pre=np.asarray(flags['budget_pre_reset']);bad=np.flatnonzero(pre < -1e-7)
        rows.append(dict(arm=arm,role=path['record']['role'],fit_index=path['record']['fit_index'],
            seed=path['record']['imagination_seed'],events=mon['event_times'],count=mon['event_count'],
            full_pass=result['fit_preview_C_rec'],failure_reasons=result['failure_reasons'],
            budget_pass=flags['delta_margin_pass'],drift_pass=flags['drift_pass'],region_pass=flags['region_gate_pass'],
            soundness_pass=flags['transition_soundness_pass'],source_pass=flags['event_source_pass'],
            cap_impossible=any(.01*(e-s)>cap+1e-7 for s,e,_ in seg),
            first_budget_failure=int(bad[0]+1) if len(bad) else None,
            min_margin=flags['min_pre_reset_budget'],geometric_outside=flags['n_region_outside_states'],
            resets=resets))
    return dict(n_paths=len(paths),count_pass=sum(r['count']>=2 for r in rows),
        full_pass=sum(r['full_pass'] for r in rows),full_pass_rate=float(np.mean([r['full_pass'] for r in rows])),
        budget_pass=sum(r['budget_pass'] for r in rows),
        event_counts=dict(Counter(r['count'] for r in rows)),cap_impossible_paths=sum(r['cap_impossible'] for r in rows),
        cap_impossible_and_count_sufficient=sum(r['cap_impossible'] and r['count']>=2 for r in rows),
        reset_completed_raw_error=metrics_error(y,p),reset_completed_allocated_error=metrics_error(y,alloc),
        all_completed_state_raw_error=metrics_error(all_y,all_p),
        mean_allocated_budget=float(np.mean([r['allocated'] for r in segment_rows])),
        capped_allocation_fraction=float(np.mean([r['allocated']>=cap-1e-6 for r in segment_rows])),
        failure_breakdown=dict(Counter(v for r in rows for v in r['failure_reasons']))),rows,segment_rows


def select_candidate(summaries):
    vals=[]
    for s in summaries:
        v=s['internal_validation'];e=v['reset_completed_raw_error']
        vals.append([e['mean_underestimation'],e['MAE'],1-v['full_pass_rate']])
    a=np.asarray(vals,float)
    if not np.isfinite(a).all():raise ValueError('No complete validation return labels; selection undefined')
    nondominated=[i for i in range(len(a)) if not any(np.all(a[j]<=a[i]) and np.any(a[j]<a[i])
        for j in range(len(a)) if j!=i)]
    norm=(a-a.min(0))/np.maximum(a.max(0)-a.min(0),1e-12)
    best=min(nondominated,key=lambda i:(float(norm[i].sum()),i))
    return best,dict(criteria=['validation_mean_underestimation','validation_MAE','1-validation_certificate_pass'],
        values=a.tolist(),pareto_candidates=[summaries[i]['name'] for i in nondominated],
        normalized_sums=norm.sum(1).tolist(),chosen=summaries[best]['name'])


def run(args):
    torch.set_num_threads(2);source=args.source.resolve()
    plan=json.loads((source/'plan.json').read_text());collection=json.loads((source/'collection.json').read_text())
    assert collection['retained']==dict(train_fit=80,internal_validation=20)
    assert collection['formal_calibration_count']==0 and collection['test_count']==0
    records=collection['records'];seedsets=[]
    for role in ('train_fit','internal_validation'):
        seedsets.append({r[k] for r in records if r['role']==role for k in ('reset_seed','imagination_seed')})
    assert not seedsets[0]&seedsets[1]
    assert digest(ROOT/'experiments/safedreamer_b_fit100_collect.py')==plan['collection_code_sha256']
    train=load_paths(source,records,'train_fit',plan)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    write(out/'training_plan.json',dict(started=stamp(),source=str(source),source_plan_sha256=digest(source/'plan.json'),
        source_collection_sha256=digest(source/'collection.json'),code_sha256=digest(__file__),
        validation_not_loaded=True,candidates=plan['candidates'],updates=plan['updates'],
        stratification='LOW->HIGH completed episode latency bins <=48,49..120,121..240,>240, plus separate censored bin',
        sampling_refinement='Collection plan sketched remaining-latency bins; before training use actual LOW->HIGH latency as requested, target remains e-t',
        cap=plan['budget_cap'],split_counts=dict(train_fit=80,internal_validation=20,cal_delta=0,cal_CP=0,test=0)))
    core=np.concatenate([p['x'] for p in train]);mean=core.mean(0);scale=np.maximum(core.std(0),.05)
    mean[-2:]=0;scale[-2:]=1
    region=dict(mean=mean,scale=scale,radius=float(np.linalg.norm((core-mean)/scale,axis=1).max()))
    flat=[];xs=[]
    for path in train:
        rs=labels(path);flat.extend(rs);xs.extend(path['x'][r['t']] for r in rs)
    x=torch.tensor(np.stack(xs));y=torch.tensor([r['target'] for r in flat],dtype=torch.float32)
    censor=torch.tensor([r['censored'] for r in flat]);weights=torch.tensor(sampling_weights(flat),dtype=torch.float64)
    frozen=[];start=time.monotonic()
    for cfg in plan['candidates']:
        torch.manual_seed(plan['training_seed'])
        net=BoundedReturnNet(mean,scale,cfg['width'],plan['budget_cap']);optimizer=torch.optim.Adam(net.parameters(),lr=.001)
        history=[];samplecounts=Counter()
        for step in range(plan['updates']):
            idx=torch.multinomial(weights,256,replacement=True) if cfg['weighted'] else torch.randint(len(x),(256,))
            pred=net(x[idx]);residual=pred-y[idx]
            completed_loss=torch.where(residual<0,cfg['under_weight']*residual.square(),residual.square())
            target_loss=torch.where(censor[idx],torch.relu(-residual).square(),completed_loss).mean()
            magnitude=(pred/plan['budget_cap']).square().mean()
            loss=target_loss+cfg['magnitude']*magnitude
            optimizer.zero_grad();loss.backward();optimizer.step()
            samplecounts.update(flat[i]['bucket'] for i in idx.tolist())
            if step==0 or (step+1)%200==0:
                history.append(dict(step=step+1,loss=float(loss),target_loss=float(target_loss),magnitude=float(magnitude)))
        net.eval()
        with torch.no_grad():preds=[net(torch.tensor(p['x'])).numpy().astype(float) for p in train]
        delta=max(path_score(pred,p['monitor']['event_times']) for p,pred in zip(train,preds))
        file=out/(cfg['name']+'.pt');torch.save(dict(state=net.state_dict(),width=cfg['width'],cap=plan['budget_cap'],region_radius=region['radius']),file)
        frozen.append(dict(name=cfg['name'],file=file.name,sha256=digest(file),delta_fit=delta,
            training=cfg,history=history,sampled_bucket_counts=dict(samplecounts),frozen=stamp()))
        print('FROZEN',cfg['name'],'delta_fit',delta,'elapsed',round(time.monotonic()-start,1),flush=True)
    write(out/'all_candidates_frozen.json',dict(frozen=stamp(),validation_not_loaded=True,candidates=frozen,
        original_bucket_counts=dict(Counter(r['bucket'] for r in flat)),censored_count=int(censor.sum()),
        labels_above_cap=int((y>plan['budget_cap']+1e-7).sum()),region_radius=region['radius']))
    # First access to validation observations occurs AFTER all candidate weights freeze.
    validation=load_paths(source,records,'internal_validation',plan)
    write(out/'validation_opened.json',dict(time=stamp(),weights_already_frozen=True,n=20))
    summaries=[];allrows=[];allsegments=[]
    for f in frozen:
        w=torch.load(out/f['file'],map_location='cpu',weights_only=True)
        net=BoundedReturnNet(mean,scale,w['width'],w['cap']);net.load_state_dict(w['state']);net.eval()
        summary=dict(name=f['name'],delta_fit=f['delta_fit'],weights_sha256=f['sha256'])
        predictions={}
        for role,paths in [('train_fit',train),('internal_validation',validation)]:
            with torch.no_grad():preds=[net(torch.tensor(p['x'])).numpy().astype(float) for p in paths]
            predictions[role]=preds
            m,rows,segs=eval_paths(paths,preds,f['delta_fit'],region,plan,f['name'])
            summary[role]=m;allrows.extend(rows);allsegments.extend(segs)
        matched=summary['train_fit']['mean_allocated_budget']
        controls={}
        for label,value in [('matched_constant',matched),('cap_constant',plan['budget_cap'])]:
            controls[label],rows,segs=eval_paths(validation,[np.full(301,value) for _ in validation],0.,region,plan,f['name']+'_'+label)
            allrows.extend(rows);allsegments.extend(segs)
        summary['validation_controls']=controls;summaries.append(summary)
        print('EVAL',f['name'],summary['internal_validation'],flush=True)
    best,selection=select_candidate(summaries)
    chosen=frozen[best]
    write(out/'selected_candidate.json',dict(selected=stamp(),**chosen,selection=selection,
        detector=plan['detector'],M_MIN=2,budget_cap=plan['budget_cap'],
        total_budget_rule='min(2.4,g(x)+Delta_fit); realized debit .01 per step, reset only on legal event',
        formal_delta=None,confidence=None,status='FIT_INTERNAL_SELECTION_ONLY_NOT_WARRANT'))
    csv_write(out/'path_results.csv',allrows);csv_write(out/'segment_results.csv',allsegments)
    write(out/'report.json',dict(completed=stamp(),summaries=summaries,selection=selection,
        scope='FIT_ONLY',cal_delta=0,cal_CP=0,test=0,elapsed_training_and_eval=time.monotonic()-start,
        limitations=['Validation used for model selection, not an independent final accuracy estimate.',
            'Observed gaps >240 steps are incompatible with the predeclared total budget cap.',
            'Magnitude penalty and cap limit allocation but do not prove certificate nondegeneracy.',
            'Censored labels are lower bounds on unobserved return time, not completed returns.',
            'Finite pathwise budget not an infinite recurrence proof.']))
    print('SELECTED',chosen['name'],selection,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
