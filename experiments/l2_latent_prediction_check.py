"""Prospective probability prediction versus fresh imagination outcomes.

Freeze the latest V, estimate events on new calibration trajectories, write
the predictions to disk, then collect independent test imagination paths.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.stats import beta, fisher_exact

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.l2_latent_v_search import OLD,read,predictor
from core.lppm.latent_reachability import audit_values

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/'artifacts/safedreamer_l2_v_refine'
OUT=ROOT/'artifacts/safedreamer_l2_prediction_check'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()


def interval(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,
            float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]


def compare(cal,test,n_test):
    k,n=cal['successes'],cal['trials'];j,m=test['successes'],test['trials']
    diff=j/m-k/n
    return dict(predicted_rate=k/n,expected_successes=n_test*k/n,
                observed_successes=j,observed_trials=m,observed_rate=j/m,
                signed_error_percentage_points=100*diff,
                absolute_error_percentage_points=100*abs(diff),
                calibration_95pct_lower=cal['cp_lower'],
                calibration_two_sided_95pct_interval=interval(k,n),
                test_two_sided_95pct_interval=interval(j,m),
                two_sample_fisher_pvalue=float(fisher_exact([[k,n-k],[j,m-j]])[1]))


def existing_fingerprints():
    paths=[OLD/f'{s}.npz' for s in ('train','validation','calibration','test')]
    paths += [OLD.parent/'safedreamer_l2_v_search/fresh_test.npz']
    paths += [MODEL_DIR/f'{s}.npz' for s in ('extra_train','extra_validation','fresh_test')]
    seen=set()
    for path in paths:
        with np.load(path) as d:
            seen.update(hashlib.sha256(z.tobytes()).hexdigest() for z in d['latent'])
    return seen


def evaluate():
    frozen=json.loads((OUT/'prediction.json').read_text())
    assert digest(MODEL_DIR/'selected.pt')==frozen['model_sha256']
    pred=predictor(torch.load(MODEL_DIR/'selected.pt',weights_only=False))
    x,d=read(OUT/'calibration.npz')
    cal=audit_values(pred(x),d)
    assert cal==frozen['calibration']
    x,d=read(OUT/'test.npz')
    test=audit_values(pred(x),d)
    report=dict(model_sha256=frozen['model_sha256'],prediction_saved_at=frozen['saved_at'],
                comparison_generated_at=now(),spec='F(goal)',horizon=50,eta=.01,
                scope='model imagination only; random actions; no environment trajectory comparison',
                calibration=cal,test=test,comparisons={},
                confidence_note='95% CP bound concerns an event probability, not a guaranteed count in the next 100 paths')
    for metric in ('certificate_event','goal_reached','p1p2_paths'):
        report['comparisons'][metric]=compare(cal[metric],test[metric],100)
        print(metric,json.dumps(report['comparisons'][metric]),flush=True)
    report['transition_p2']=dict(
        calibration_violation_rate=cal['p2']['violations']/cal['p2']['checked'],
        test_violation_rate=test['p2']['violations']/test['p2']['checked'],
        note='descriptive transition rates; within-path transitions are correlated')
    report['result']='NO_WARRANT' if cal['certificate_event']['cp_lower']<.8 else 'SAMPLED_EVENT_THRESHOLD_MET'
    (OUT/'report.json').write_text(json.dumps(report,indent=2))


def run():
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    OUT.mkdir(parents=True,exist_ok=True)
    if any((OUT/name).exists() for name in ('calibration.npz','test.npz','prediction.json')):
        raise FileExistsError('Do not overwrite a prospective prediction; use --evaluate-only')
    log=logging.getLogger('wrappers.safedreamer_wrapper')
    log.setLevel(logging.INFO);log.addHandler(logging.StreamHandler(sys.stdout))
    log.propagate=False
    weight_hash=digest(MODEL_DIR/'selected.pt')
    pred=predictor(torch.load(MODEL_DIR/'selected.pt',weights_only=False))
    plan=dict(model_sha256=weight_hash,model=str(MODEL_DIR/'selected.pt'),spec='ltl_goal_reach',
              n_calibration=100,n_test=100,seed_calibration=11301,seed_test=11302,
              horizon=50,eta=.01,action_source='random',created_at=now(),
              design='write forecast before collecting test; no training or selection')
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    seen=existing_fingerprints()
    fingerprints={}
    extra=build_extra(argparse.Namespace(repo_root=None,checkpoint=None))
    cfg=RolloutConfig(horizon=50,n_rollouts=100,seed=11301,action_source='random',extra=extra)
    with SafeDreamerWrapper(cfg) as w:
        w.load()
        for split,seed in (('calibration',11301),('test',11302)):
            assert digest(MODEL_DIR/'selected.pt')==weight_hash
            if split=='test':
                assert (OUT/'prediction.json').exists()
                print('Prediction frozen; collecting test now.',flush=True)
            data=w.sample_latent_rollouts(RolloutConfig(horizon=50,n_rollouts=100,seed=seed,action_source='random',extra=extra))
            hashes=[hashlib.sha256(z.tobytes()).hexdigest() for z in data['latent']]
            assert len(set(hashes))==100 and not set(hashes)&seen
            seen.update(hashes);fingerprints[split]=hashes
            aps=np.array([[[z[k] for k in ('goal_dist','hazard_dist','velocity')] for z in tr] for tr in data['aps']],dtype=np.float32)
            np.savez_compressed(OUT/f'{split}.npz',latent=data['latent'],aps=aps,decoded=data['decoded'],
                                **{f'rssm_{k}':v for k,v in data['rssm'].items()})
            if split=='calibration':
                x,d=read(OUT/'calibration.npz');summary=audit_values(pred(x),d)
                frozen=dict(saved_at=now(),model_sha256=weight_hash,calibration=summary,
                            next_n_paths=100,expected_certificate_successes=100*summary['certificate_event']['rate'],
                            expected_goal_successes=100*summary['goal_reached']['rate'])
                (OUT/'prediction.json').write_text(json.dumps(frozen,indent=2))
                print('FROZEN FORECAST:',frozen['expected_certificate_successes'],'certificate events;',
                      frozen['expected_goal_successes'],'goal events per 100 paths',flush=True)
    (OUT/'fingerprints.json').write_text(json.dumps(fingerprints,indent=2))
    evaluate()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluate-only',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    if a.evaluate_only:evaluate()
    else:run()
