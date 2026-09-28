"""Replay saved spec comparison; add descriptive common-nonaccepting W checks."""
import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.l3_spec_ablation import SPECS,load_role,relabel
from experiments.l3_wu_data_ablation import checked_array,restore,scan
from experiments.l3_safedreamer_clear_pipeline import digest,write


def load_queries(out,role,paths,frozen_time):
    groups={}; seeds=[]
    for file in sorted((out/(role+'_queries')).glob('*.npz')):
        pi,t,chunk=map(int,re.fullmatch(r'p(\d+)_t(\d+)_c(\d+)\.npz',file.name).groups())
        m=json.loads(file.with_suffix('.json').read_text())
        a=checked_array(file,m['sha256']); seeds.append(m['seed'])
        if m['path_sha256']!=paths[pi]['record']['sha256'] or m['t']!=t:
            raise ValueError('Wrong path/query association')
        np.testing.assert_array_equal(a['anchor'],paths[pi]['base'][t])
        if role=='development' and m['time']<=frozen_time: raise ValueError('Query precedes model freeze')
        groups.setdefault((pi,t),[]).append((chunk,a))
    if len(seeds)!=len(set(seeds)): raise ValueError('Repeated query seed')
    result=[]
    for (pi,t),parts in sorted(groups.items()):
        parts.sort(key=lambda p:p[0])
        if [p[0] for p in parts]!=list(range(len(parts))): raise ValueError('Missing chunk')
        result.append(dict(path_index=pi,t=t,successors=np.concatenate([a['successors'] for _,a in parts]),
                           decoded=np.concatenate([a['decoded'] for _,a in parts])))
    if len(result)!=sum(len(p['times']) for p in paths): raise ValueError('Missing anchors')
    return result,seeds


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('output',type=Path)
    args=p.parse_args(); out=args.output.resolve(); torch.set_num_threads(2)
    plan=json.loads((out/'plan.json').read_text()); frozen=json.loads((out/'frozen.json').read_text())
    report=json.loads((out/'report.json').read_text()); source=Path(plan['source'])
    if digest(out/'plan.json')!=frozen['plan_sha256']: raise ValueError('Changed plan')
    for name,sha in plan['source_sha256'].items():
        if digest(ROOT/name)!=sha: raise ValueError(f'Changed source: {name}')
    paths=load_role(source,'development'); fitpaths=load_role(source,'fit')
    queries,seeds=load_queries(out,'development',paths,frozen['time'])
    _,fitseeds=load_queries(out,'fit',fitpaths,frozen['time'])
    if set(seeds)&set(fitseeds): raise ValueError('Cross-role RNG seed reuse')
    masks=[]; scores={}
    for spec,saved in zip(SPECS,report['rows']):
        sid=spec['id']; nets=[]
        for k in ('W','U'):
            file=out/f'{sid}_{k}.pt'
            if digest(file)!=frozen['models'][sid][k]: raise ValueError('Changed frozen weights')
            nets.append(restore(file))
        _,batches=relabel(paths,queries,spec)
        row=scan(*nets,batches,[plan['dev_kappa']],len(SPECS))[0]
        for k,v in row.items():
            if v!=saved[k]: raise ValueError(f'Replay differs: {sid}/{k}')
        if spec['window']>1:
            x=torch.tensor(np.stack([b.anchor for b in batches]))
            with torch.no_grad(): mask=(x[:,-1].numpy()<1)&(nets[1](x).numpy()<=.8)
            masks.append(mask); scores[sid]=row
    common=np.all(masks,axis=0)
    matched={sid:dict(n=int(common.sum()),raw_pass=int((np.array(r['W']['drifts'])[common]<=-.01).sum()),
                     sampling_corrected_pass=int((np.array(r['W']['drifts'])[common]+r['sampling_radius']<=-.01).sum()))
             for sid,r in scores.items()}
    audit=dict(exact_replay=True,dev_anchors=len(queries),development_chunks=len(seeds),
               fit_chunks=len(fitseeds),all_dev_queries_after_freeze=True,
               matched_nonaccepting_W=matched,
               note='Descriptive common physical-anchor subset for S16/S24/S32/S48 only; no significance or recurrence probability claim.')
    write(out/'audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__': main()
