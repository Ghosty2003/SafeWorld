"""Freeze the clear-window V; collect fresh calibration before independent test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import l2_achievement64 as base
from experiments import l2_approach1_holdout as shared
from experiments import l2_clear_window_v as fitted
from experiments.l2_approach1_outside_v import write
from experiments.l2_latent_prediction_check import digest, now
from core.lppm.value_families import predict_family

SEEDS = dict(calibration=28503000, test=28504000)
EVENT = 'finite_completion_and_certificate'


def audit(model, data):
    if model['spec'] != fitted.SPEC or model['eta'] != .01:
        raise ValueError('Frozen value specification differs')
    if (data['aps'][:, 0, 0] < .70).any():
        raise ValueError('Expected decoded initial goal distance >=1.0')
    run, done = fitted.monitor(data['aps'])
    z = np.concatenate((base.value_features(data, 'policy'), run[..., None]/48.), axis=-1)
    values = predict_family(model, z, done)
    row = base.summary(values, done, inferential=True)
    row = {k.replace('goal', 'completion'): v for k,v in row.items()}
    passed = np.asarray(row['per_path_candidate_event']) & done[:, -1]
    delta = values[:, :-1]-values[:, 1:]
    pp = ~((delta < 0) | (~done[:, :-1] & (delta < .01))).any(1)
    row['p1p2_given_completion'] = base.rate(
        pp[done[:, -1]], True)
    row['confusion'] = dict(completed_and_certificate=int((passed & done[:, -1]).sum()),
                           completed_without_certificate=int((~passed & done[:, -1]).sum()),
                           not_completed=int((~done[:, -1]).sum()),
                           candidate_certificate_without_completion=row['candidate_event_without_completion'])
    return row


def verdict(row):
    lower = row[EVENT]['cp_lower']
    clean = row['sampled_region_status'] == 'NO_VIOLATION_ON_OBSERVED_SOURCES'
    sufficient = lower >= .95
    return dict(required_probability=.95, confidence=.95, lower_bound=lower,
                probability_threshold_met=sufficient, no_observed_region_counterevidence=clean,
                result='FINITE_EVENT_THRESHOLD_MET' if sufficient and clean else 'NO_WARRANT',
                reasons=([] if sufficient else ['Certificate-event lower bound below 0.95'])
                        + ([] if clean else ['Candidate region lacks clean observed-source evidence']),
                scope='Finite model-only certificate AND completion probability under fixed initial '
                      'distribution and policy. Not infinite-horizon safety or support-wide proof.')


def prepare(source, out):
    fitted.evaluate(source)
    p = json.loads((source/'plan.json').read_text())
    frozen = json.loads((source/'frozen.json').read_text())
    if (p['spec'], p['horizon'], p['eta'], p['warrant_threshold'], p['future_confidence']) != (
            fitted.SPEC, 64, .01, .95, .95):
        raise ValueError('Source scope/threshold changed')
    if p['initial_condition'] != 'norm(decoded initial goal vector)>=1.0':
        raise ValueError('Wrong initial population')
    policy = Path(p['source'])/'policy_config.json'
    if digest(policy) != p['policy_config_sha256'] or digest(Path(p['checkpoint'])) != p['checkpoint_sha256']:
        raise ValueError('World model or policy changed')
    # Identity metadata only, including rejected/surplus old raw draws.
    seen = set()
    for path in (base.ROOT/'artifacts').glob('safedreamer*/*_provenance.json'):
        prov = json.loads(path.read_text())
        seen.update(prov.get('fingerprints', []))
        for batch in prov.get('batches', []):
            seen.update(batch.get('fingerprints', []))
    for path in (base.ROOT/'artifacts').glob('safedreamer*/plan.json'):
        other = json.loads(path.read_text())
        seeds = other.get('seeds', {})
        if isinstance(seeds, dict) and set(SEEDS.values()).intersection(
                v for v in seeds.values() if isinstance(v, int)):
            raise ValueError('Fresh seed range already recorded by another experiment')
    out.mkdir(exist_ok=False, parents=True)
    shutil.copyfile(source/'selected.pt', out/'selected.pt')
    shutil.copyfile(policy, out/'policy_config.json')
    write(out/'prior_identities.json', dict(fingerprints=sorted(seen)))
    code_paths = [Path(__file__).resolve(), Path(shared.__file__).resolve(),
                  Path(fitted.__file__).resolve(), base.ROOT/'core/lppm/value_families.py',
                  base.ROOT/'core/lppm/learned_sublevel.py', Path(base.__file__).resolve(),
                  base.ROOT/'wrappers/safedreamer_wrapper.py']
    write(out/'frozen.json', dict(frozen_at=frozen['frozen_at'], confirmed_at=now(),
          source_run=str(source), source_plan_sha256=frozen['plan_sha256'],
          model_sha256=frozen['model_sha256'], selected_candidate=frozen['best']['candidate'],
          step=frozen['best']['step']))
    write(out/'plan.json', dict(created_at=now(), source_run=str(source),
          model_sha256=frozen['model_sha256'], spec=fitted.SPEC, horizon=64, eta=.01,
          checkpoint=p['checkpoint'], checkpoint_sha256=p['checkpoint_sha256'],
          policy_config_sha256=p['policy_config_sha256'], initial_condition=p['initial_condition'],
          action_source='policy', device='gpu', initialization_seed=28502, seeds=SEEDS,
          counts=dict(calibration=100, test=100), batch_size=20, max_batches_per_split=50,
          prior_identities_sha256=digest(out/'prior_identities.json'),
          code_sha256={str(path):digest(path) for path in code_paths},
          certificate_key=EVENT, task_key='completion_reached',
          primary_event='All 64 model transitions satisfy P1/P2; endpoint V<eta; '
                        'the selected 48-clear-state window specification completed by t64',
          warrant_threshold=.95, confidence=.95,
          decision='Primary one-sided 95% Clopper-Pearson lower bound >=0.95 and no '
                   'observed candidate-region counterevidence on calibration paths',
          sampling='First 100 eligible paths in draw order; only decoded t0 goal distance>=1.0 '
                   'filters paths. Keep all future successes/failures. No repeated testing to pass.',
          inference='Assumes iid same-distribution draws conditional on initial eligibility. '
                    'Bounds are individual, not simultaneous. Fixed finite model-only specification; '
                    'no real-world or infinite-horizon guarantee. Test rates are not lower bounds.',
          reset_rng_note='Imagination RNG seeds explicit; simulator reset RNG not fully controlled by them.',
          sequencing='Weights/spec frozen; calibration forecast saved before test generation; no fitting'))


def check_frozen(out):
    p = json.loads((out/'plan.json').read_text())
    for filename, sha in p['code_sha256'].items():
        if digest(Path(filename)) != sha:
            raise ValueError('Frozen collection/monitor/predictor code changed: '+filename)
    if digest(out/'prior_identities.json') != p['prior_identities_sha256']:
        raise ValueError('Prior identity list changed')
    if digest(out/'selected.pt') != p['model_sha256']:
        raise ValueError('Frozen V changed')
    return json.loads((out/'prior_identities.json').read_text())['fingerprints']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=base.ROOT/'artifacts/safedreamer_l2_clear_window_v_linear6000')
    parser.add_argument('--output', type=Path, default=base.ROOT/'artifacts/safedreamer_l2_clear_window_holdout')
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(2)
    source, out = args.source.resolve(), args.output.resolve()
    if not args.evaluate_only:
        prepare(source, out)
        shared.collect(source, out, audit_fn=audit, verdict_fn=verdict, extra_seen=check_frozen(out))
    shared.evaluate(out, audit_fn=audit, verdict_fn=verdict, extra_seen=check_frozen(out))


if __name__ == '__main__': main()
