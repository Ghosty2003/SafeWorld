"""Finite screening of GF G[0,47](decoded hazard margin >= 0).

A causal counter accepts after 48 consecutive safe states; hazard resets it.
Acceptance is NOT absorbing. This is not an infinite recurrence verifier.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def monitor(safe, window=48):
    p = np.asarray(safe)
    if p.ndim != 1 or p.dtype != np.bool_ or not len(p) or window < 1:
        raise ValueError('Expected a nonempty boolean path and positive window')
    count = 0
    longest = 0
    runs = []
    ends = []
    counters = []
    for t, hit in enumerate(p):
        if hit:
            count += 1
            longest = max(longest, count)
        else:
            if count:
                runs.append([t-count, t-1])
            count = 0
        counters.append(min(count, window))
        if count >= window:
            ends.append(t)
    if count:
        runs.append([len(p)-count, len(p)-1])
    halfway = (len(p)-1)//2
    return dict(completion_steps=ends, counter=counters,
                overlapping_windows=len(ends),
                nonoverlapping_windows=sum((b-a+1)//window for a,b in runs),
                qualifying_safe_runs=sum(b-a+1 >= window for a,b in runs),
                longest_safe_run=longest, entirely_safe=bool(p.all()),
                completion_after_halfway=any(t > halfway for t in ends),
                window_wholly_after_halfway=any(t-window+1 > halfway for t in ends),
                safe_runs=runs)


def summarize(safe):
    p = np.asarray(safe)
    if p.ndim != 2 or p.shape[0] == 0:
        raise ValueError('Expected N x (T+1) boolean paths')
    rows = [monitor(row) for row in p]
    return dict(n_paths=len(rows), horizon=p.shape[1]-1, includes_initial_state=True,
                paths_with_one_window=sum(r['overlapping_windows'] >= 1 for r in rows),
                paths_with_two_overlapping_windows=sum(r['overlapping_windows'] >= 2 for r in rows),
                paths_with_two_nonoverlapping_windows=sum(r['nonoverlapping_windows'] >= 2 for r in rows),
                paths_with_two_hazard_separated_safe_runs=sum(r['qualifying_safe_runs'] >= 2 for r in rows),
                paths_with_completion_after_halfway=sum(r['completion_after_halfway'] for r in rows),
                paths_with_window_wholly_after_halfway=sum(r['window_wholly_after_halfway'] for r in rows),
                entirely_safe_paths=sum(r['entirely_safe'] for r in rows),
                total_overlapping_windows=sum(r['overlapping_windows'] for r in rows),
                min_longest_safe_run=min(r['longest_safe_run'] for r in rows),
                median_longest_safe_run=float(np.median([r['longest_safe_run'] for r in rows])),
                per_path=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    provenance = json.loads((source.parent/'provenance.json').read_text())
    if digest != provenance['sha256']:
        raise ValueError('Source data does not match provenance')
    with np.load(source, allow_pickle=False) as data:
        d = data['decoded']
        if d.ndim != 3 or d.shape[-1] != 29 or not np.isfinite(d).all():
            raise ValueError('Invalid decoder observations')
        distances = np.linalg.norm(d[...,9:25].reshape(*d.shape[:2],8,2), axis=-1).min(-1)
        margin = distances.astype(np.float64)-.20
        np.testing.assert_allclose(margin, data['aps'][...,1], atol=1e-6)
        safe = margin >= 0
    report = dict(spec='GF G[0,47](decoded_hazard_margin>=0)',
                  scope='EXPLORATORY_FINITE_PREFIX_ONLY', source=str(source), source_sha256=digest,
                  warrant='NOT_EVALUATED', monitor='nonabsorbing consecutive-safe counter saturated at 48',
                  notes=['Finite observations neither prove nor refute infinite GF.',
                         'No new rollouts, training, calibration, or confidence bounds.',
                         'Overlapping windows are not independent observations.',
                         'Always-safe behavior is allowed; hazard-separated windows are not required.',
                         'Source initial distribution: decoded goal distance >= 1.0.',
                         'Late uncompleted windows are censored, not GF failures.'],
                  full_horizon=summarize(safe))
    if safe.shape[1] >= 65:
        report['first64_same_paths'] = summarize(safe[:,:65])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    for name in ('full_horizon','first64_same_paths'):
        if name in report:
            print(name, {k:v for k,v in report[name].items() if k != 'per_path'})


if __name__ == '__main__': main()
