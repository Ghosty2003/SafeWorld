"""Fixed original goal radius, three PointGoal1 checkpoints, fresh 64-step data.

Each checkpoint gets independent V fitting/selection, calibration and test.
No calibration or test statistic selects V, the radius, or another checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.l2_latent_prediction_check import digest, now

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/safedreamer_l2_goal03_policy64'
ACTION_SOURCE = 'policy'
CKPTS = ROOT.parent/'SafeDreamer/checkpoint'
PREFIX = '20240307-010600_osrp_vector_safetygymcoor_SafetyPointGoal1-v0_'
IDS = (0,5,10)


def write(path, obj):
    with path.open('x') as f:
        json.dump(obj,f,indent=2,allow_nan=False)


def compare():
    rows=[]
    comparison_plan=json.loads((OUT/'plan.json').read_text())
    for seed in IDS:
        directory=OUT/f'seed{seed}'
        plan=json.loads((directory/'plan.json').read_text())
        assert plan['radii']==[.3] and plan['horizon']==64
        assert plan['action_source']==comparison_plan['action_source']
        expected=comparison_plan['checkpoints'][str(seed)]
        assert plan['checkpoint']==expected['path']
        assert plan['checkpoint_sha256']==expected['sha256']
        assert digest(Path(plan['checkpoint']))==plan['checkpoint_sha256']
        r=json.loads((directory/'report.json').read_text())
        frozen=json.loads((directory/'frozen.json').read_text())
        simultaneous={}
        for event in ('goal_reached','finite_goal_and_certificate'):
            k=r['calibration'][event]['successes']; n=r['calibration'][event]['trials']
            simultaneous[event]=float(beta.ppf(.05/6,k,n-k+1)) if k else 0.
        rows.append(dict(checkpoint_seed=seed,checkpoint=plan['checkpoint'],
            checkpoint_sha256=plan['checkpoint_sha256'],variant=r['variant'],update=r['step'],
            train=frozen['best']['train'],validation=frozen['best']['validation'],
            calibration=r['calibration'],test=r['test'],comparisons=r['comparisons'],
            simultaneous_95pct_calibration_lowers=simultaneous,region_status=r['region_status']))
    result=dict(scope=f"Model-only {comparison_plan['action_source']}-action F_[0,64](decoded distance < 0.30)",
        rows=rows,support_wide_validity='NOT_ESTABLISHED',
        notes=[
            'Strict <0.30 matches the requested predicate; environment uses <=0.30.',
            'Each world model has a separately fitted V; old latent coordinates/V are not transferred.',
            'All checkpoints are reported; none is selected or retuned using holdout outcomes.',
            'Individual 95% intervals are not simultaneous; extra lowers use alpha=.05/6 for six primary claims.',
            'CP bounds concern population event probabilities, not guaranteed next-batch fractions.',
            'Same collection rules and imagination seeds; environment reset RNG is not fully controlled, '
            'so this is not an exactly matched initial-state or action-sequence comparison.',
            'Original paper architecture/loss is compared with a learned accepting-scalar/anchor extension.',
            'No threshold is used to assert global or infinite-horizon warrant from sampled checks.',
        ])
    path=OUT/'comparison.json'
    if path.exists(): assert json.loads(path.read_text())==result
    else: write(path,result)
    for r in rows:
        print('SUMMARY',r['checkpoint_seed'],r['variant'],r['update'],
              {s:{k:r[s][k] for k in ('goal_reached','zfree_entered','finite_goal_and_certificate','p1','p2')}
               for s in ('calibration','test')},flush=True)


def run():
    checkpoints={seed:CKPTS/f'{PREFIX}{seed}.ckpt' for seed in IDS}
    for p in checkpoints.values():
        if not p.is_file(): raise FileNotFoundError(p)
    OUT.mkdir(exist_ok=False)
    cpus=sorted(os.sched_getaffinity(0))
    # Bound each independent process to disjoint CPU subsets; no GPU use.
    width=min(8,len(cpus)//len(IDS))
    if width<1: raise RuntimeError('At least three available CPUs required')
    write(OUT/'plan.json',dict(created_at=now(),checkpoints={str(s):dict(path=str(p),sha256=digest(p))
        for s,p in checkpoints.items()},horizon=64,radius=.3,action_source=ACTION_SOURCE,
        counts=dict(train=120,validation=60,calibration=100,test=100),seed_base=18500,
        updates_per_variant=1500,variants=['paper','anchored_accepting_scalar'],
        cpus_per_worker=width,selection='Select V on each development set; report all three checkpoints.',
        confidence='Individual 95% CP plus six-claim Bonferroni 95% lower bounds; no test-based selection'))
    env=dict(os.environ,MUJOCO_GL='egl',MPLCONFIGDIR='/tmp/safeworld-mpl',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
    jobs=[]
    try:
        for i,seed in enumerate(IDS):
            logpath=OUT/f'seed{seed}.log'
            writer=logpath.open('x'); reader=logpath.open()
            subset=','.join(str(c) for c in cpus[i*width:(i+1)*width])
            command=['taskset','-c',subset,sys.executable,'-u',str(ROOT/'experiments/l2_achievement64.py'),
                '--checkpoint',str(checkpoints[seed]),'--output',str(OUT/f'seed{seed}'),
                '--radii','0.3','--seed-base','18500','--action-source',ACTION_SOURCE]
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=writer,stderr=subprocess.STDOUT)
            jobs.append((seed,process,writer,reader))
        while True:
            for seed,p,w,reader in jobs:
                for line in reader.readlines():
                    if line.startswith(('FIT ','COLLECTED ','FORECAST ','SafeDreamerWrapper: imagined','Traceback')):
                        print(f'[{seed}] {line.rstrip()}',flush=True)
            if all(p.poll() is not None for _,p,_,_ in jobs): break
            time.sleep(5)
        codes={str(s):p.returncode for s,p,_,_ in jobs}
        write(OUT/'completion.json',dict(completed_at=now(),exit_codes=codes))
        if any(codes.values()): raise RuntimeError(f'Failed workers: {codes}; inspect saved logs')
    finally:
        for _,p,w,r in jobs:
            if p.poll() is None: p.terminate(); p.wait()
            w.close(); r.close()
    compare()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare-only',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--action-source',choices=['policy','random'],default='policy')
    args=parser.parse_args()
    if args.output: OUT=args.output.resolve()
    ACTION_SOURCE=args.action_source
    if args.compare_only: compare()
    else: run()
