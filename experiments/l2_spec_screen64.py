"""Exploratory finite-spec screening on development data, never a warrant.

Keep every eligible failure. Initial-state filters use time zero only.
No V fitting, calibration, test reuse, or change to the rollout controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def window_event(predicate, start, stop, length):
    """Exists s in [start, stop]: predicate holds at s..s+length-1."""
    if length < 1 or start < 0 or stop < start or stop + length > predicate.shape[1]:
        raise ValueError('Window exceeds saved horizon')
    windows = np.lib.stride_tricks.sliding_window_view(predicate, length, axis=1)
    return windows[:, start:stop + 1].all(-1).any(1)


def counts(event, eligible, initially=None):
    n = int(eligible.sum())
    k = int((event & eligible).sum())
    return dict(successes=k, total=n, rate=k / n if n else None,
                initial_target_members=int((initially & eligible).sum()) if initially is not None else None)


def screen(aps):
    if aps.ndim != 3 or aps.shape[1:] != (65, 3) or not np.isfinite(aps).all():
        raise ValueError('Expected finite N x 65 x 3 APs')
    distance = aps[..., 0] + .30
    initial = distance[:, 0]
    clear = aps[..., 1] >= 0
    all_paths = np.ones(len(aps), dtype=bool)
    rows = {}
    for radius in (.3, .5, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5):
        goal = distance < radius
        for scope, eligible in (('all', all_paths), ('initial_outside', initial >= radius)):
            rows[f'F[0,64](d<{radius:g}); {scope}'] = counts(goal.any(1), eligible, goal[:, 0])
    for lo, hi in ((1., 1.25), (1., 1.5), (1., 1.75), (1.25, 1.75), (1.5, 2.), (1.75, 2.5)):
        eligible = (initial >= lo) & (initial < hi)
        for radius in (.3, .5, 1.):
            rows[f'F[1,64](d<{radius:g}); {lo:g}<=d0<{hi:g}'] = counts(
                (distance[:, 1:] < radius).any(1), eligible, initial < radius)
    for progress in (.01, .05, .1, .2, .3, .5):
        rows[f'F[1,64](d<d0-{progress:g})'] = counts(
            (distance[:, 1:] < initial[:, None] - progress).any(1), all_paths,
            np.zeros(len(aps), dtype=bool))
    rows['G[0,64](hazard_margin>=0)'] = counts(clear.all(1), all_paths)
    for length in (5, 10, 16, 24, 32, 48, 64):
        event = window_event(clear, 1, 65 - length, length)
        name = f'F[1,{65-length}](G[0,{length-1}](hazard_margin>=0))'
        rows[name] = counts(event, all_paths)
        rows[name + '; initial_hazard'] = counts(event, ~clear[:, 0])
    for start in (4, 8, 12, 16, 17):
        rows[f'G[{start},64](hazard_margin>=0)'] = counts(clear[:, start:].all(1), all_paths)
    for radius in (2., 2.5, 3., 3.5, 4.):
        rows[f'G[0,64](d<{radius:g})'] = counts((distance < radius).all(1), all_paths)
    return rows


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root / 'artifacts/safedreamer_l2_approach1_more_data')
    parser.add_argument('--output', type=Path, default=root / 'artifacts/safedreamer_l2_spec_screen64')
    args = parser.parse_args()
    results, inputs, diagnostics = {}, {}, {}
    for split in ('train', 'validation'):
        path = args.source / f'{split}.npz'
        with np.load(path) as a:
            aps, decoded = a['aps'], a['decoded']
        # Cross-check stored predicates against the saved decoded observations.
        np.testing.assert_allclose(aps[..., 0], np.linalg.norm(decoded[..., 7:9], axis=-1) - .30, atol=1e-6)
        hazard = np.linalg.norm(decoded[..., 9:25].reshape(*decoded.shape[:2], 8, 2), axis=-1).min(-1) - .20
        np.testing.assert_allclose(aps[..., 1], hazard, atol=1e-6)
        if (aps[:, 0, 0] + .30 < 1.).any():
            raise ValueError('Unexpected source initial distribution')
        results[split] = screen(aps)
        inputs[split] = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        diagnostics[split] = dict(hazard_counts_by_time=(aps[..., 1] < 0).sum(0).tolist(),
                                  minimum_hazard_margin_from_t17=float(aps[:, 17:, 1].min()))
    plan = json.loads((args.source / 'plan.json').read_text())
    report = dict(status='EXPLORATORY_SPEC_SCREENING_NOT_CALIBRATED', horizon=64,
                  checkpoint=plan['checkpoint'], policy='unchanged local CCEPlanner',
                  source_initial_condition='decoded d0 >= 1.0', inputs=inputs,
                  results=results, diagnostics=diagnostics,
                  notes=['Training and repeatedly used development data only; no fresh test.',
                         'Adaptive specification selection: no confidence bounds or warrant issued.',
                         'All quantities are decoded imagination APs, not real-environment cost.',
                         'Window indices are inclusive state indices; 64 transitions give 65 states.',
                         'Post-hoc narrowed initial distributions require fresh conditional sampling.',
                         'A long clear window permits earlier or later hazards; it is not full-horizon safety.',
                         'No V was trained for the newly screened formulas.'])
    args.output.mkdir(exist_ok=False, parents=True)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# 64-step exploratory specification screening', '',
             'Development data only. No new V, confidence bound, or warrant. All distances are decoded.', '',
             '| Specification / initial scope | Training | Development | Initially in target (train/dev) |',
             '|---|---:|---:|---:|']
    for name, tr in results['train'].items():
        va = results['validation'][name]
        lines.append(f"| `{name}` | {tr['successes']}/{tr['total']} | {va['successes']}/{va['total']} | "
                     f"{tr['initial_target_members']}/{va['initial_target_members']} |")
    lines += ['', '## Limitations', ''] + [f'- {note}' for note in report['notes']]
    (args.output / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    print('Saved', args.output, 'candidates:', len(results['train']))


if __name__ == '__main__':
    main()
