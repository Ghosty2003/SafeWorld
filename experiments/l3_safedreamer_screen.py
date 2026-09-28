"""Exploratory finite-prefix screen for L3 candidates, never a GF proof.

Old imagination paths are development data only. No calibration or warrant.
Repeated accepting states and separate re-entries are reported separately:
staying in p satisfies GF(p) on an infinite continuation, without re-entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(labels):
    p = np.asarray(labels)
    if p.dtype != np.bool_ or p.ndim != 2 or not p.shape[0] or p.shape[1] < 2:
        raise ValueError('Expected nonempty boolean paths including initial state')
    entries = ((~p[:, :-1]) & p[:, 1:]).sum(1)
    accepting_runs = entries + p[:, 0]
    # Count suffix lengths, not future failures: an unfinished wait is censored.
    tail = []
    longest = []
    for row in p:
        run = peak = 0
        for hit in row:
            run = 0 if hit else run + 1
            peak = max(peak, run)
        tail.append(run)
        longest.append(peak)
    return dict(n_paths=len(p), horizon=p.shape[1]-1,
                initially_accepting=int(p[:, 0].sum()),
                ever_accepting=int(p.any(1).sum()),
                always_accepting=int(p.all(1).sum()),
                two_or_more_accepting_runs=int((accepting_runs >= 2).sum()),
                accepts_in_each_half=int((p[:, 1:1+(p.shape[1]-1)//2].any(1)
                                          & p[:, 1+(p.shape[1]-1)//2:].any(1)).sum()),
                accepting_state_counts=p.sum(1).tolist(),
                false_to_true_entry_counts=entries.tolist(),
                unfinished_nonaccepting_suffix_lengths=tail,
                longest_nonaccepting_runs=longest)


def screen(decoded):
    d = np.asarray(decoded)
    if d.ndim != 3 or d.shape[-1] != 29 or not np.isfinite(d).all():
        raise ValueError('Expected finite N x (T+1) x 29 decoded observations')
    goal = np.linalg.norm(d[..., 7:9], axis=-1)
    hazard = np.linalg.norm(d[..., 9:25].reshape(*d.shape[:2], 8, 2), axis=-1).min(-1)
    return {name: summarize(labels) for name, labels in (
        ('GF(decoded_goal_distance<0.3)', goal < .3),
        ('GF(decoded_goal_distance<1.0)', goal < 1.),
        ('GF(decoded_hazard_margin>=0)', hazard >= .2))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source, allow_pickle=False) as data:
        result = screen(data['decoded'])
    report = dict(scope='EXPLORATORY_FINITE_PREFIX_ONLY', source=str(source),
                  source_sha256=digest, candidates=result,
                  warrant='NOT_EVALUATED',
                  notes=['No finite prefix establishes GF.',
                         'No infinite GF violation inferred from an unfinished suffix.',
                         'Re-entry is diagnostic, not required by GF when p stays true.',
                         'Old paths used for screening, not new calibration/test.',
                         'Decoded model predicates; no real-world validity claim.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    for name, row in result.items():
        print(name, {k: v for k, v in row.items() if not isinstance(v, list)})


if __name__ == '__main__':
    main()
