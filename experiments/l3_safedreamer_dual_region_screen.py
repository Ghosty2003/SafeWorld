"""Finite exploratory diagnostics for GF(d<0.5) & GF(d>1.0).

No training, confidence bound, or infinite-horizon verdict. Distance is to
the decoded CURRENT goal, not a verified fixed physical landmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def cycles(distance):
    """Causal A -> B -> A monitor; consecutive cycles can share closing A.

    Initial B is ignored until A has been reached. Neutral states do not
    reset progress. Strict boundaries .5 and 1.0 belong to neither region.
    """
    d = np.asarray(distance)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('Expected finite nonnegative one-dimensional distances')
    start = middle = None
    result = []
    for t, value in enumerate(d):
        if start is None:
            if value < .5:
                start = t
        elif middle is None:
            if value > 1.:
                middle = t
        elif value < .5:
            result.append([start, middle, t])
            start, middle = t, None
    return result


def summarize(distance):
    d = np.asarray(distance)
    if d.ndim != 2 or len(d) == 0 or d.shape[1] < 3:
        raise ValueError('Expected N paths with at least three states')
    triples = [cycles(row) for row in d]
    counts = np.array([len(x) for x in triples])
    halfway = (d.shape[1]-1)//2
    return dict(n_paths=len(d), horizon=d.shape[1]-1,
                ever_A=int((d < .5).any(1).sum()),
                ever_B=int((d > 1.).any(1).sum()),
                both_regions_any_order=int(((d < .5).any(1) & (d > 1.).any(1)).sum()),
                at_least_one_ABA=int((counts >= 1).sum()),
                at_least_two_ABA=int((counts >= 2).sum()),
                at_least_three_ABA=int((counts >= 3).sum()),
                total_ABA=int(counts.sum()),
                paths_with_ABA_finishing_after_halfway=sum(any(x[2] > halfway for x in ts) for ts in triples),
                paths_with_ABA_wholly_after_halfway=sum(any(x[0] > halfway for x in ts) for ts in triples),
                mean_cycles=float(counts.mean()), median_cycles=float(np.median(counts)),
                cycle_counts=counts.tolist(), cycle_indices=triples)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source = args.source.resolve()
    with np.load(source, allow_pickle=False) as a:
        decoded = a['decoded']
        if decoded.ndim != 3 or decoded.shape[-1] != 29 or not np.isfinite(decoded).all():
            raise ValueError('Expected finite decoded observations of width 29')
        distance = np.linalg.norm(decoded[..., 7:9], axis=-1)
    report = dict(spec='GF(decoded_current_goal_distance<0.5) & GF(decoded_current_goal_distance>1.0)',
                  scope='EXPLORATORY_FINITE_PREFIX_ONLY', source=str(source),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  verdict='NOT_EVALUATED', fixed_physical_goal_identity='UNVERIFIED',
                  notes=['Saved paths only; no new calibration or test data.',
                         'Finite cycles do not establish infinite recurrence.',
                         'Goal reset and model error can mimic physical departure.',
                         'No success-based path filtering; cycles may share boundary A.'],
                  full_horizon=summarize(distance))
    if distance.shape[1] >= 65:
        report['first64_same_paths'] = summarize(distance[:, :65])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    for name in ('full_horizon', 'first64_same_paths'):
        if name in report:
            print(name, {k:v for k,v in report[name].items() if not isinstance(v, list)})


if __name__ == '__main__':
    main()
