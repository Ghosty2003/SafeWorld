"""Replay every saved W/U diagnostic; no conversion to formal H2 evidence."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.l3_expected_drift_ablation import load_source,evaluate,core_check
from experiments.l3_spec_ablation import SPECS,relabel
from experiments.l3_spec_ablation_audit import load_queries
from experiments.l3_wu_data_ablation import restore
from experiments.l3_safedreamer_clear_pipeline import digest,write,stamp


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('output',type=Path)
    out=p.parse_args().output.resolve(); torch.set_num_threads(2)
    plan=json.loads((out/'plan.json').read_text()); frozen=json.loads((out/'frozen.json').read_text())
    report=json.loads((out/'report.json').read_text()); source=Path(plan['source'])
    assert digest(out/'plan.json')==frozen['plan_sha256']
    assert digest(source/'frozen.json')==plan['source_frozen_sha256']
    for name,sha in plan['code_sha256'].items(): assert digest(ROOT/name)==sha
    _,_,train,dev,tq,dq,ts,ds=load_source(source)
    extra,seeds=load_queries(out,'fit',train,'')
    assert not set(seeds)&(set(ts)|set(ds))
    assert len(extra)==len(tq)==plan['fit_anchors']
    combined=[]
    for a,b in zip(tq,extra):
        assert (a['path_index'],a['t'])==(b['path_index'],b['t'])
        assert len(a['successors'])==32 and len(b['successors'])==96
        combined.append(dict(path_index=a['path_index'],t=a['t'],
            successors=np.concatenate([a['successors'],b['successors']]),
            decoded=np.concatenate([a['decoded'],b['decoded']])))
    spec=next(s for s in SPECS if s['id']=='B_S24')
    tp,tb32=relabel(train,tq,spec); _,tb128=relabel(train,combined,spec); dp,db=relabel(dev,dq,spec)
    for a,b in zip(tb32,tb128):
        np.testing.assert_array_equal(a.anchor,b.anchor)
        np.testing.assert_array_equal(a.successors,b.successors[:32])
    slack=json.loads((source/'report.json').read_text())['rows'][0]['sampling_radius']
    for row in report['rows']:
        file=out/f'{row["name"]}.pt'
        assert digest(file)==row['sha256']==frozen['models'][row['name']]
        net=restore(file)
        for role,paths,batches in [('train',tp,tb32 if row['kappa']==32 else tb128),('development',dp,db)]:
            replay=evaluate(net,row['kind'],row['retention'],batches,paths,row['margin'],slack)
            assert replay==row[role],(row['name'],role)
    for role,batches in [('train32',tb32),('train128',tb128),('development',db)]:
        assert core_check(batches)==report['core_checks'][role]
    assert report['confidence'] is None and report['global_L3']=='ABSTAIN'
    assert report['final_validation']=='NOT_RUN'
    result=dict(time=stamp(),exact_replay=True,models=len(report['rows']),
        new_fit_chunks=len(seeds),new_fit_successors=32*len(seeds),nested_kappa_prefix_exact=True,
        source_code_and_weight_hashes_checked=True,cross_role_query_seeds_disjoint=True,
        final_validation='NOT_RUN',H2_invariance='NOT_ESTABLISHED',
        note='Development outcomes deliberately reused; not independent of adaptive model selection.')
    write(out/'audit.json',result); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
