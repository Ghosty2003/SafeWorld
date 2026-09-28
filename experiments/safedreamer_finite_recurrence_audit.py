"""Read saved FIT pilot data and replay all diagnostics without new rollouts."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.finite_recurrence import safe_window_monitor,reference_events,path_score,budget_diagnostics,combine_gates
from experiments.safedreamer_finite_recurrence_pilot import BudgetNet,features,hazard_mask,region_membership,seed_pair
from experiments.l3_safedreamer_clear_pipeline import digest,write,stamp


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('output',type=Path)
    out=p.parse_args().output.resolve(); torch.set_num_threads(2)
    plan=json.loads((out/'plan.json').read_text()); report=json.loads((out/'report.json').read_text())
    metadata=json.loads((out/'fit_model.json').read_text())
    for file,sha in plan['code_sha256'].items(): assert digest(file)==sha, file
    assert digest(plan['checkpoint'])==plan['checkpoint_sha256']
    assert digest(out/'g_init.pt')==metadata['sha256']
    checkpoint=torch.load(out/'g_init.pt',map_location='cpu',weights_only=True)
    state=checkpoint['state']; net=BudgetNet(state['mean'],state['scale']); net.load_state_dict(state); net.eval()
    region=dict(mean=state['mean'].numpy(),scale=state['scale'].numpy(),radius=checkpoint['region_radius'])
    accepted=[]; seen=set(); reset_seeds=set(); imagination_seeds=set()
    for file in sorted((out/'fit').glob('path_*.json')):
        r=json.loads(file.read_text()); assert r['role']=='fit'
        assert (r['reset_seed'],r['imagination_seed'])==seed_pair('fit',r['draw'])
        assert r['reset_seed'] not in reset_seeds and r['imagination_seed'] not in imagination_seeds
        reset_seeds.add(r['reset_seed']); imagination_seeds.add(r['imagination_seed'])
        raw=file.with_suffix('.npz'); assert digest(raw)==r['sha256']
        with np.load(raw,allow_pickle=False) as f: a={k:f[k].copy() for k in f.files}
        fp=hashlib.sha256(a['latent'].tobytes()).hexdigest()
        assert fp==r['fingerprint'] and fp not in seen; seen.add(fp)
        assert a['latent'].shape==(301,512) and a['actions'].shape==(300,2)
        assert r['eligible']==bool(np.linalg.norm(a['decoded'][0,7:9])>=1.)
        if r['eligible']: accepted.append((r,a))
    assert len(accepted)==plan['fit_count']==report['split_counts']['fit']
    for role in ('cal_delta','cal_CP','test'):
        assert report['split_counts'][role]==0 and not (out/role).exists()
        assert plan['roles'][role]['status']=='LOCKED_NOT_COLLECTED'
    assert plan['M_MIN'] is None and report['M_MIN'] is None
    assert report['calibrated_delta'] is None and report['confidence'] is None
    assert report['seed_replay']['all_exact']
    scores=[]
    for row,(record,a) in zip(report['rows'],accepted):
        m=safe_window_monitor(hazard_mask(a['decoded'])); x=features(a,m['counter'])
        assert record['fit_index']==row['fit_index']
        for k,v in m.items():
            if k!='counter': assert row[k]==v,(row['fit_index'],k)
        with torch.no_grad(): pred=net(torch.tensor(x)).numpy()
        if row['fit_partition']=='train': scores.append(path_score(pred,m['event_times']))
        for label in ('delta_zero','delta_fit_only'):
            delta=row['previews'][label]['delta']
            replay=budget_diagnostics(pred,m['event_times'],reference_events(hazard_mask(a['decoded'])),delta,region_membership(x,region))
            assert replay==row['previews'][label]['flags'],(row['fit_index'],label)
            for threshold in (3,4,5,6):
                assert combine_gates(replay,m['event_count'],threshold)==row['previews'][label]['by_M_MIN'][str(threshold)]
        constant=budget_diagnostics(np.full(301,3.),m['event_times'],reference_events(hazard_mask(a['decoded'])),0.,region_membership(x,region))
        for threshold in (3,4,5,6):
            assert combine_gates(constant,m['event_count'],threshold)==row['constant_H_eta_baseline'][str(threshold)]
    assert max(scores)==metadata['delta_fit_preview']
    for partition,saved in report['summary'].items():
        rows=[r for r in report['rows'] if partition=='all' or r['fit_partition']==partition]
        assert saved['n']==len(rows)
        assert saved['event_count_distribution']=={str(k):v for k,v in Counter(r['event_count'] for r in rows).items()}
        for label in ('delta_zero','delta_fit_only'):
            for threshold in ('3','4','5','6'):
                view=saved['previews'][label][threshold]
                assert view['successes']==sum(r['previews'][label]['by_M_MIN'][threshold]['fit_preview_C_rec'] for r in rows)
                assert view['failure_breakdown']==dict(Counter(reason for r in rows for reason in r['previews'][label]['by_M_MIN'][threshold]['failure_reasons']))
    with (out/'fit_rows.csv').open('x',newline='') as f:
        w=csv.writer(f); w.writerow(['fit_index','partition','event_count','hazard_interruption_count','hazard_state_count',
            'preview','M_MIN','drift_pass','delta_margin_pass','region_gate_pass','transition_soundness_pass','event_source_pass',
            'CV_pass','sliding_pass','fit_preview_C_rec','failure_reasons'])
        for row in report['rows']:
            for label in ('delta_zero','delta_fit_only'):
                v=row['previews'][label]
                for threshold,r in v['by_M_MIN'].items():
                    w.writerow([row['fit_index'],row['fit_partition'],row['event_count'],row['hazard_interruption_count'],
                        row['hazard_state_count'],label,threshold,*[v['flags'][k] for k in ('drift_pass','delta_margin_pass',
                        'region_gate_pass','transition_soundness_pass','event_source_pass','CV_pass','sliding_pass')],
                        r['fit_preview_C_rec'],'|'.join(r['failure_reasons'])])
    result=dict(time=stamp(),exact_replay=True,fit_paths=len(accepted),all_raw_draws=len(seen),
        transitions_checked_per_path=300,event_timestamp_reference_match=True,
        fit_only_delta_recomputed=True,source_model_data_hashes_checked=True,
        cal_delta_count=0,cal_CP_count=0,test_count=0,M_MIN=None,
        seed_replay_exact=True,report_sha256=digest(out/'report.json'),
        scope='FIT_ONLY_NO_CONFIDENCE_NO_WARRANT')
    write(out/'audit.json',result); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
