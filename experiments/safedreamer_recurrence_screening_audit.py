"""Replay all screening arms from saved weights, including seeded shuffles."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.safedreamer_recurrence_screening import signals,product_features,augmented_gate,choose
from experiments.safedreamer_high_clearance_pilot import load_fit,monitor,inside_region,evaluate
from experiments.safedreamer_finite_recurrence_pilot import BudgetNet
from experiments.l3_safedreamer_clear_pipeline import digest,write,stamp
from core.finite_recurrence import path_score


def run(out):
    torch.set_num_threads(2)
    audit=json.loads((out/'audit.json').read_text()); plan=json.loads((out/'plan.json').read_text())
    for name,sha in audit['output_hashes'].items(): assert digest(out/name)==sha,name
    for name,sha in plan['candidate_code_hashes'].items(): assert digest(name)==sha,name
    _,paths=load_fit(Path(plan['source'])); ss=[signals(p['arrays']) for p in paths]
    saved=json.loads((out/'path_diagnostics.json').read_text())
    indexed={(r['candidate'],r['fit_index'],r['arm'],r['shuffle_seed']):r for r in saved}
    report=json.loads((out/'spec_candidate_details.json').read_text()); n=0
    for d in report['details']:
        file=out/(d['id']+'.pt'); assert digest(file)==d['model_sha256']
        w=torch.load(file,map_location='cpu',weights_only=True); st=w['state']
        net=BudgetNet(st['mean'],st['scale']); net.load_state_dict(st); net.eval()
        region=dict(mean=st['mean'].numpy(),scale=st['scale'].numpy(),radius=w['region_radius'])
        fit=[]; predictions=[]
        for p,s in zip(paths,ss):
            mon=monitor(s[d['signal']],d['d_low'],d['d_high'])
            x=product_features(p['arrays'],mon,s[d['signal']],d['signal'])
            fit.append((mon,x))
            with torch.no_grad(): predictions.append(net(torch.tensor(x)).numpy().astype(float))
        delta=max(path_score(p,mon['event_times']) for p,(mon,x) in zip(predictions[:12],fit[:12]))
        assert delta==d['delta_fit']
        arms=[('learned',None,predictions,delta),
            ('constant_matched',None,[np.full(301,d['matched_budget']) for _ in fit],0.),
            ('constant_H_eta',None,[np.full(301,3.) for _ in fit],0.)]
        for seed in plan['shuffle_seeds']:
            rng=np.random.default_rng(seed); permuted=[p.copy()+delta for p in predictions]
            for part in ('train','internal_check'):
                sites=[(r['fit_index'],r['start']) for r in d['segments'] if r['partition']==part]
                values=rng.permutation(np.array([permuted[i][s] for i,s in sites]))
                for (i,s),v in zip(sites,values):permuted[i][s]=v
            arms.append(('shuffled',seed,permuted,0.))
        for arm,seed,preds,corr in arms:
            for i,((mon,x),s,pred) in enumerate(zip(fit,ss,preds)):
                flags,resets=evaluate(s[d['signal']],mon,d['d_low'],d['d_high'],pred,corr,inside_region(x,region))
                gate=augmented_gate(flags,mon['event_times'],d['safety_required'],bool((s['clearance']>=.2).all()))
                old=indexed[d['id'],i,arm,seed]
                assert resets==old['resets'] and mon['pairs']==old['event_pairs']
                assert mon['event_times']==old['events'] and gate['fit_preview_C_rec']==old['full_pass']
                assert gate['failure_reasons']==old['failure_reasons']
                for key,original in [('drift_pass','drift_pass'),('delta_margin_pass','budget_pass'),
                        ('region_gate_pass','region_pass'),('transition_soundness_pass','soundness_pass'),
                        ('event_source_pass','event_source_pass')]:assert flags[key]==old[original]
                n+=1
    primary,backup=choose(report['details'])
    assert primary['id']==report['primary'] and backup['id']==report['backup']
    frozen=json.loads((out/'frozen_primary.json').read_text())
    assert frozen['definition']['id']==primary['id'] and frozen['model_sha256']==primary['model_sha256']
    write(out/'audit_replay.json',dict(time=stamp(),exact_cases=n,seeded_shuffle_replayed=True,
        gate_and_reset_replayed=True,delta_fit_recomputed=True,selection_reproduced=True,
        calibration_count=0,test_count=0))
    print('Exact replay',n,'cases; all weights, resets, gates, shuffles and selection match.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    run(p.parse_args().output.resolve())
