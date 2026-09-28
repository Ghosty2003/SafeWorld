"""Descriptive 100-step goal-arrival screen; no V, calibration, or warrant.

Compare the first 64 and all 100 transitions of the SAME fresh paths.
Eligibility depends only on decoded initial goal distance >=1.0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
N, HORIZON, BATCH, MAX_BATCHES, SEED_BASE = 100, 100, 20, 50, 29505000


def digest(path):
    with Path(path).open('rb') as f:
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, obj):
    with Path(path).open('x') as f: json.dump(obj, f, indent=2, allow_nan=False)


def fingerprints(z):
    return [hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for a in z]


def select(aps, remaining):
    if aps.ndim != 3 or aps.shape[1:] != (101, 3) or not np.isfinite(aps).all():
        raise ValueError('Expected N x 101 x 3 finite APs')
    if remaining < 1: raise ValueError('Positive remaining count required')
    eligible = aps[:, 0, 0] >= .70  # stored goal AP is raw distance minus .30
    return eligible, np.flatnonzero(eligible)[:remaining]


def summarize(aps):
    select(aps, len(aps))  # schema/finite validation
    reached = aps[..., 0] < 0  # strict decoded goal distance <0.30
    first = np.where(reached.any(1), reached.argmax(1), -1)
    by64 = reached[:, :65].any(1)
    by100 = reached.any(1)
    return dict(n_paths=len(aps), initially_satisfied=int(reached[:, 0].sum()),
                success_by64=int(by64.sum()), success_by100=int(by100.sum()),
                rate_by64=float(by64.mean()), rate_by100=float(by100.mean()),
                newly_successful_65_to100=int((by100 & ~by64).sum()),
                not_reached_by100=int((~by100).sum()),
                success_at_final_state_only=int(reached[:, -1].sum()),
                first_arrival_steps=first.tolist(),
                median_first_arrival_among_successes=float(np.median(first[by100])) if by100.any() else None,
                per_path_min_decoded_goal_distance=(aps[..., 0].min(1)+.30).tolist())


def prepare(source, out):
    p = json.loads((source/'plan.json').read_text())
    if digest(Path(p['checkpoint'])) != p['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    if digest(source/'policy_config.json') != p['policy_config_sha256']:
        raise ValueError('Policy changed')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source/'policy_config.json', out/'policy_config.json')
    write(out/'plan.json', dict(created_at=now(), source_config=str(source),
          checkpoint=p['checkpoint'], checkpoint_sha256=p['checkpoint_sha256'],
          policy_config_sha256=p['policy_config_sha256'], horizon=HORIZON, count=N,
          batch_size=BATCH, max_batches=MAX_BATCHES, seed_base=SEED_BASE,
          spec='F[0,100](norm(decoded goal vector)<0.30)',
          initial_condition='norm(decoded initial goal vector)>=1.0',
          action_source='policy', planner='unchanged local CCEPlanner',
          comparison='First 64 versus full 100 transitions of the same accepted trajectories',
          selection='First eligible paths in draw order; t0-only condition; no future-success filtering',
          scope='EXPLORATORY_MODEL_ONLY_GOAL_RATE; no V fitting/evaluation, no certificate/warrant',
          seed_note='Explicit imagination seeds; simulator reset randomness not completely controlled by them'))


def collect(out):
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    plan = json.loads((out/'plan.json').read_text())
    extra = build_extra(argparse.Namespace(repo_root=None, checkpoint=plan['checkpoint']))
    extra['action_source'] = 'policy'
    extra['config_overrides']['jax'].update(platform='gpu', logical_gpus=0)
    logger = logging.getLogger('wrappers.safedreamer_wrapper'); logger.setLevel(logging.INFO)
    if not logger.handlers: logger.addHandler(logging.StreamHandler(sys.stdout))
    cfg = RolloutConfig(horizon=HORIZON, n_rollouts=BATCH, seed=29505, action_source='policy', extra=extra)
    seen, pieces, records, kept = set(), [], [], 0
    (out/'raw').mkdir()
    with SafeDreamerWrapper(cfg) as wrapper:
        wrapper.load()
        import jax
        if jax.default_backend() != 'gpu': raise RuntimeError('GPU required')
        print('IMAGINATION DEVICE', jax.devices(), flush=True)
        actual = dict(action_source='policy', expl_behavior=wrapper._config.expl_behavior,
            planner={k:getattr(wrapper._config.planner,k) for k in
                     ('horizon','num_samples','num_elites','iterations','mixture_coef','momentum','init_std')},
            cost_limit=wrapper._config.cost_limit,
            planner_source_sha256=digest(Path(extra['repo_root'])/'SafeDreamer/behaviors.py'),
            note='Local reconstructed evaluation planner; no claim of exact historical training config.')
        if actual != json.loads((out/'policy_config.json').read_text()): raise ValueError('Policy mismatch')
        for batch in range(MAX_BATCHES):
            seed = SEED_BASE+batch
            print('COLLECTING', batch, 'seed', seed, 'retained', kept, '/', N, flush=True)
            d = wrapper.sample_latent_rollouts(RolloutConfig(
                horizon=HORIZON, n_rollouts=BATCH, seed=seed, action_source='policy', extra=extra))
            if d['action_source'] != 'policy' or d['latent'].shape != (BATCH,101,512):
                raise ValueError('Wrong rollout shape/scope')
            aps = np.asarray([[[s[k] for k in ('goal_dist','hazard_dist','velocity')] for s in path]
                              for path in d['aps']], dtype=np.float32)
            arrays = dict(latent=d['latent'], aps=aps, decoded=d['decoded'], actions=d['actions'],
                          **d['policy_state'], **{f'rssm_{k}':v for k,v in d['rssm'].items()})
            fp = fingerprints(arrays['latent'])
            if len(set(fp)) != BATCH or seen.intersection(fp): raise ValueError('Duplicate path')
            seen.update(fp)
            eligible, ids = select(aps, N-kept)
            path = out/'raw'/f'batch_{batch:03d}.npz'; np.savez_compressed(path, **arrays)
            record = dict(seed=seed, raw_file=str(path.relative_to(out)), sha256=digest(path),
                          fingerprints=fp, eligible_indices=np.flatnonzero(eligible).tolist(),
                          selected_indices=ids.tolist(), initial_rejected=int((~eligible).sum()),
                          surplus_eligible=int(eligible.sum()-len(ids)))
            write(path.with_suffix('.json'), record); records.append(record)
            if len(ids): pieces.append({k:v[ids] for k,v in arrays.items()})
            kept += len(ids)
            print('PROGRESS retained', kept, '/', N, 'drawn', (batch+1)*BATCH, flush=True)
            if kept == N: break
    if kept != N: raise RuntimeError('Draw budget exhausted')
    accepted = {k:np.concatenate([a[k] for a in pieces], axis=0) for k in pieces[0]}
    np.savez_compressed(out/'rollouts.npz', **accepted)
    write(out/'provenance.json', dict(ended_at=now(), sha256=digest(out/'rollouts.npz'),
          plan_sha256=digest(out/'plan.json'), accepted=N, drawn=len(records)*BATCH,
          initial_rejected=sum(r['initial_rejected'] for r in records),
          surplus_eligible=sum(r['surplus_eligible'] for r in records), batches=records))


def evaluate(out):
    p = json.loads((out/'plan.json').read_text()); prov = json.loads((out/'provenance.json').read_text())
    if digest(out/'plan.json') != prov['plan_sha256'] or digest(out/'rollouts.npz') != prov['sha256']:
        raise ValueError('Saved inputs changed')
    if digest(out/'policy_config.json') != p['policy_config_sha256']: raise ValueError('Policy changed')
    pieces, remaining, seen = [], N, set()
    for batch, rec in enumerate(prov['batches']):
        path = out/rec['raw_file']
        if digest(path) != rec['sha256'] or rec['seed'] != p['seed_base']+batch:
            raise ValueError('Raw data/seed changed')
        with np.load(path) as a:
            fp = fingerprints(a['latent'])
            if fp != rec['fingerprints'] or len(set(fp)) != len(fp) or seen.intersection(fp):
                raise ValueError('Repeated/changed raw paths')
            seen.update(fp)
            eligible, ids = select(a['aps'], remaining)
            if ids.tolist() != rec['selected_indices'] or np.flatnonzero(eligible).tolist() != rec['eligible_indices']:
                raise ValueError('Initial-only sampling rule changed')
            if len(ids): pieces.append({k:a[k][ids] for k in a.files})
            remaining -= len(ids)
    if remaining: raise ValueError('Wrong retained count')
    with np.load(out/'rollouts.npz') as a:
        for k in a.files:
            np.testing.assert_array_equal(a[k], np.concatenate([x[k] for x in pieces], axis=0))
        np.testing.assert_allclose(a['aps'][...,0], np.linalg.norm(a['decoded'][...,7:9],axis=-1)-.30, atol=1e-6)
        if not (a['aps'][:,0,0] >= .70).all(): raise ValueError('Initial distribution differs')
        report = dict(scope=p['scope'], spec=p['spec'], initial_condition=p['initial_condition'], **summarize(a['aps']))
    if (out/'report.json').exists():
        if json.loads((out/'report.json').read_text()) != report: raise ValueError('Report changed')
    else: write(out/'report.json', report)
    print('RESULT', json.dumps({k:v for k,v in report.items() if not isinstance(v,list)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'artifacts/safedreamer_l2_clear_window_holdout')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/safedreamer_goal03_policy100')
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args(); source, out = args.source.resolve(), args.output.resolve()
    if not args.evaluate_only: prepare(source,out); collect(out)
    evaluate(out)


if __name__ == '__main__': main()
