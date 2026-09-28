"""Freeze the latest selected V; forecast before collecting fresh test paths.

No fitting or checkpoint selection. Statistics concern fixed 50-step events
under random-action model imagination, not a support-wide proof. The optional
80% acceptance check is reported separately from probability-bound validity.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.lppm.latent_reachability import audit_values
from experiments.l2_latent_prediction_check import compare, digest, now
from experiments.l2_latent_v_search import read
from experiments.l2_latent_v_validation_search import OUT as SOURCE, predictor

OUT = SOURCE.parent / 'safedreamer_l2_latest_prediction_check'
N, HORIZON, ETA, THRESHOLD = 100, 50, .01, .80
SEEDS = dict(calibration=14301, test=14302)


def write_json(name, data):
    with (OUT/name).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def path_hashes(latents):
    import hashlib
    return [hashlib.sha256(z.tobytes()).hexdigest() for z in latents]


def prior_hashes():
    """Read fingerprints only; never evaluate past test data to select V."""
    seen, sources = set(), []
    for path in sorted(SOURCE.parent.glob('safedreamer_l2_*/*.npz')):
        if path.parent == OUT:
            continue
        with np.load(path) as data:
            if 'latent' not in data.files:
                continue
            hashes = path_hashes(data['latent'])
            seen.update(hashes)
            sources.append(dict(path=str(path), paths=len(hashes)))
    return seen, sources


def evaluate():
    plan = json.loads((OUT/'plan.json').read_text())
    forecast = json.loads((OUT/'prediction.json').read_text())
    collection = json.loads((OUT/'collection.json').read_text())
    assert digest(OUT/'frozen_v.pt') == plan['model_sha256'] == forecast['model_sha256']
    assert digest(OUT/'prediction.json') == collection['prediction_sha256_before_test']
    assert forecast['saved_at'] <= collection['test_started_at']
    fn = predictor(torch.load(OUT/'frozen_v.pt', weights_only=False))
    summaries = {}
    for split in SEEDS:
        assert digest(OUT/f'{split}.npz') == collection['dataset_sha256'][split]
        x, done = read(OUT/f'{split}.npz')
        assert x.shape[:2] == (N, HORIZON+1)
        summaries[split] = audit_values(fn(x), done, eta=ETA)
    cal, test = summaries['calibration'], summaries['test']
    assert cal == forecast['calibration']
    comparisons = {key: compare(cal[key], test[key], N)
                   for key in ('certificate_event', 'goal_reached', 'p1p2_paths')}
    lower = cal['certificate_event']['cp_lower']
    report = dict(
        model_sha256=plan['model_sha256'], candidate_name=plan['candidate_name'],
        generated_at=now(), prediction_saved_at=forecast['saved_at'],
        test_started_at=collection['test_started_at'],
        scope=plan['scope'], spec='F(goal)', observed_horizon=HORIZON, eta=ETA,
        calibration=cal, test=test, comparisons=comparisons,
        acceptance_check=dict(threshold=THRESHOLD, lower=lower,
                              met=lower >= THRESHOLD,
                              result='THRESHOLD_MET' if lower >= THRESHOLD else 'BELOW_ACCEPTANCE_THRESHOLD'),
        deductive_status='NOT_ESTABLISHED',
        full_warrant_audit='NOT_PERFORMED; acceptance check alone is not a full warrant audit',
        confidence_note='Individual one-sided 95% CP bounds, conditional on fixed V and iid same-distribution paths; not a confidence score for V, simultaneous bounds, or guaranteed counts in the next batch.',
        zfree_semantics='Accepting done branch, forced V=0 after the first decoded goal visit. Closure here is structural, not a learned physical invariant region.',
        violation_note='P2 failures concern this candidate. Failure to reach by step 50 does not falsify unbounded F(goal).',
        fitting_performed=False,
    )
    if (OUT/'report.json').exists():
        previous = json.loads((OUT/'report.json').read_text())
        assert previous['comparisons'] == comparisons
        assert previous['calibration'] == cal and previous['test'] == test
        print('Saved report reproduced without changes.', flush=True)
    else:
        write_json('report.json', report)
    for key, values in comparisons.items():
        print(key, json.dumps(values), flush=True)
    print('ACCEPTANCE CHECK', json.dumps(report['acceptance_check']), flush=True)


def run():
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper

    OUT.mkdir(exist_ok=False)
    source_report = json.loads((SOURCE/'report.json').read_text())
    weight_hash = digest(SOURCE/'selected.pt')
    assert weight_hash == source_report['model_sha256']
    shutil.copyfile(SOURCE/'selected.pt', OUT/'frozen_v.pt')
    assert digest(OUT/'frozen_v.pt') == weight_hash
    fn = predictor(torch.load(OUT/'frozen_v.pt', weights_only=False))
    extra = build_extra(argparse.Namespace(repo_root=None, checkpoint=None))
    # Separate simulator-reset RNG stream; model checkpoint and policy unchanged.
    extra['config_overrides']['seed'] = 14300
    seen, prior_sources = prior_hashes()
    write_json('plan.json', dict(
        created_at=now(), source_model=str(SOURCE/'selected.pt'),
        model_sha256=weight_hash, candidate_name=source_report['best']['name'],
        world_model_checkpoint=extra['checkpoint_path'],
        world_model_sha256=digest(Path(extra['checkpoint_path'])),
        n_calibration=N, n_test=N, horizon=HORIZON, eta=ETA,
        seeds=SEEDS, environment_config_seed=14300, action_source='random',
        scope='Same frozen SafeDreamer checkpoint; reset-encoded starts, random-action imagination only; no environment trajectory comparison.',
        goal='decoded goal_dist < 0, radius 0.30; includes initial state',
        acceptance_threshold=THRESHOLD, prior_fingerprint_sources=prior_sources,
        protocol='Fixed sizes; calibration prediction saved before any test collection; no refitting or selection.',
    ))
    logger = logging.getLogger('wrappers.safedreamer_wrapper')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.propagate = False
    fingerprints, dataset_hashes = {}, {}
    cfg = RolloutConfig(horizon=HORIZON, n_rollouts=N, seed=SEEDS['calibration'],
                        action_source='random', extra=extra)
    with SafeDreamerWrapper(cfg) as wrapper:
        wrapper.load()
        for split, seed in SEEDS.items():
            assert digest(OUT/'frozen_v.pt') == weight_hash
            if split == 'test':
                assert (OUT/'prediction.json').exists()
                forecast_hash = digest(OUT/'prediction.json')
                test_started_at = now()
                print('Forecast saved and frozen; collecting fresh test paths.', flush=True)
            data = wrapper.sample_latent_rollouts(RolloutConfig(
                horizon=HORIZON, n_rollouts=N, seed=seed, action_source='random', extra=extra))
            hashes = path_hashes(data['latent'])
            assert len(hashes) == len(set(hashes)) == N and not set(hashes) & seen
            seen.update(hashes)
            fingerprints[split] = hashes
            aps = np.array([[[s[k] for k in ('goal_dist', 'hazard_dist', 'velocity')]
                             for s in trajectory] for trajectory in data['aps']], dtype=np.float32)
            np.savez_compressed(OUT/f'{split}.npz', latent=data['latent'], aps=aps,
                                decoded=data['decoded'],
                                **{f'rssm_{k}':v for k,v in data['rssm'].items()})
            dataset_hashes[split] = digest(OUT/f'{split}.npz')
            if split == 'calibration':
                x, done = read(OUT/'calibration.npz')
                cal = audit_values(fn(x), done, eta=ETA)
                write_json('prediction.json', dict(
                    saved_at=now(), model_sha256=weight_hash, calibration=cal,
                    next_n_paths=N,
                    expected_certificate_successes=N*cal['certificate_event']['rate'],
                    expected_goal_successes=N*cal['goal_reached']['rate']))
                print('FROZEN FORECAST', 'certificate', cal['certificate_event'],
                      'goal', cal['goal_reached'], flush=True)
    write_json('fingerprints.json', fingerprints)
    write_json('collection.json', dict(test_started_at=test_started_at,
                                      completed_at=now(),
                                      prediction_sha256_before_test=forecast_hash,
                                      dataset_sha256=dataset_hashes,
                                      duplicate_paths_found=False))
    evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    evaluate() if args.evaluate_only else run()
