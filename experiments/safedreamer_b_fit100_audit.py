"""Read-only raw/model replay for the bounded B fit100 study."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.safedreamer_b_fit100_train import load_paths,BoundedReturnNet,eval_paths,select_candidate
from experiments.l3_safedreamer_clear_pipeline import digest,write,stamp
from core.finite_recurrence import path_score


def run(source,out):
    torch.set_num_threads(2)
    plan=json.loads((source/'plan.json').read_text());collection=json.loads((source/'collection.json').read_text())
    trainplan=json.loads((out/'training_plan.json').read_text());report=json.loads((out/'report.json').read_text())
    frozen=json.loads((out/'all_candidates_frozen.json').read_text())
    opened=json.loads((out/'validation_opened.json').read_text())
    assert frozen['frozen']<opened['time']
    assert digest(source/'plan.json')==trainplan['source_plan_sha256']
    assert digest(source/'collection.json')==trainplan['source_collection_sha256']
    assert digest(ROOT/'experiments/safedreamer_b_fit100_train.py')==trainplan['code_sha256']
    seeds=[];fingerprints=[]
    for r in collection['records']:
        assert digest(source/'raw'/r['file'])==r['sha256']
        seeds.extend([r['reset_seed'],r['imagination_seed']]);fingerprints.append(r['fingerprint'])
    assert len(seeds)==len(set(seeds)) and len(fingerprints)==len(set(fingerprints))
    assert all(json.loads((source/'seed_replay.json').read_text())['exact'].values())
    paths={role:load_paths(source,collection['records'],role,plan) for role in ('train_fit','internal_validation')}
    checks=0
    for f,saved in zip(frozen['candidates'],report['summaries']):
        file=out/f['file'];assert digest(file)==f['sha256']
        w=torch.load(file,map_location='cpu',weights_only=True);st=w['state']
        net=BoundedReturnNet(st['mean'].numpy(),st['scale'].numpy(),w['width'],w['cap'])
        net.load_state_dict(st);net.eval()
        region=dict(mean=st['mean'].numpy(),scale=st['scale'].numpy(),radius=w['region_radius'])
        for role,ps in paths.items():
            with torch.no_grad():pred=[net(torch.tensor(p['x'])).numpy().astype(float) for p in ps]
            # Float32 sigmoid output has rounding; total allocation below is
            # clipped in float64 and is checked against the exact configured cap.
            assert all(((p>=0)&(p<=2.4+1e-6)).all() for p in pred)
            if role=='train_fit':
                delta=max(path_score(p,x['monitor']['event_times']) for p,x in zip(pred,ps))
                assert delta==f['delta_fit']
            metrics,rows,segs=eval_paths(ps,pred,f['delta_fit'],region,plan,f['name'])
            # JSON object keys (event counts) are strings on disk.
            assert json.loads(json.dumps(metrics))==saved[role]
            assert all(r['allocated']<=2.4 for r in segs)
            assert all(not r['budget_pass'] for r in rows if r['cap_impossible'])
            checks+=len(rows)
        for arm,value in [('matched_constant',saved['train_fit']['mean_allocated_budget']),('cap_constant',2.4)]:
            metrics,_,_=eval_paths(paths['internal_validation'],[np.full(301,value) for _ in range(20)],0.,region,plan,'audit')
            assert json.loads(json.dumps(metrics))==saved['validation_controls'][arm];checks+=20
    _,selection=select_candidate(report['summaries']);assert selection==report['selection']
    write(out/'audit.json',dict(time=stamp(),source_raw_hashes_verified=True,seed_replay_exact=True,
        seeds_disjoint=True,latents_not_duplicated=True,validation_opened_after_all_freezes=True,
        exact_metric_replay=True,cases_replayed=checks,total_budget_cap_verified=True,
        impossible_gaps_fail_closed=True,selection_replayed=True,cal_delta=0,cal_CP=0,test=0,
        hashes={p.name:digest(p) for p in out.iterdir() if p.is_file()}))
    print('AUDIT OK',checks,'cases, frozen-before-validation, cap and selection verified.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source.resolve(),a.output.resolve())
