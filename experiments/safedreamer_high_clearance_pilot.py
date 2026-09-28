"""Fit-only LOW -> HIGH recurrence on audited cached CCE imagination paths.

No formal thresholds, M_MIN, calibration, CP bounds, or infinite-time claim.
Old experiments and raw data are read-only. State 0 may arm, but never emits.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.finite_recurrence import budget_diagnostics, combine_gates, path_score
from experiments.safedreamer_finite_recurrence_pilot import train_budget, require_fit_stage
from experiments.l3_safedreamer_clear_pipeline import digest, write, stamp

QUANTILES = [('A', .25, .50), ('B', .25, .75), ('C', .10, .90), ('D', .50, .75)]


def validate(d, low, high):
    d = np.asarray(d, dtype=float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError('Clearance must be a finite trace including state 0')
    if not np.isfinite([low, high]).all() or not 0 <= low < high:
        raise ValueError('Require finite 0 <= d_low < d_high')
    return d


def monitor(d, low, high):
    d = validate(d, low, high)
    armed = False
    start = None
    states, entries, pairs = [], [], []
    for t, value in enumerate(d):
        if not armed:
            if value <= low:
                armed = True
                start = t
                entries.append(t)
        elif value >= high:
            pairs.append(dict(event_id=len(pairs)+1, low_entry_timestep=start,
                return_timestep=t, event_timestep=t, reset_timestep=t,
                low_entry_clearance=float(d[start]), return_clearance=float(value)))
            armed = False
            start = None
        states.append(int(armed))
    events = [p['event_timestep'] for p in pairs]
    below = d <= low
    excursions = np.flatnonzero(below & np.r_[True, ~below[:-1]]).tolist()
    return dict(config=dict(d_low=float(low), d_high=float(high), initial_state_can_arm=True),
        event_times=events, return_timestamps=events.copy(), low_entry_timestamps=entries,
        low_excursion_timestamps=excursions, event_count=len(events), pairs=pairs,
        armed=states, pending_low_entry=start, inter_event_intervals=np.diff(events).tolist())


def reference(d, low, high):
    """Independent interval-search implementation, used only for auditing."""
    d = validate(d, low, high)
    cursor, pairs, entries = 0, [], []
    while cursor < len(d):
        hits = np.flatnonzero(d[cursor:] <= low)
        if not len(hits): break
        s = cursor + int(hits[0])
        entries.append(s)
        hits = np.flatnonzero(d[s+1:] >= high)
        if not len(hits): break
        e = s + 1 + int(hits[0])
        pairs.append((s, e))
        cursor = e + 1
    return entries, pairs


def source_consistent(d, mon, low, high, resets=None):
    entries, pairs = reference(d, low, high)
    expected = monitor(d, low, high)
    # Reconcile full timestamps, detector state, values, config, and reset sites.
    if mon != expected or entries != mon['low_entry_timestamps']:
        return False
    if pairs != [(p['low_entry_timestep'], p['return_timestep']) for p in mon['pairs']]:
        return False
    if resets is not None:
        if len(resets) != len(mon['pairs']): return False
        for row, p in zip(resets, mon['pairs']):
            if any(row.get(k) != v for k, v in p.items()): return False
    return True


def product(a, armed):
    n = len(armed)
    return np.concatenate([a['latent'], a['planner_action_mean'].reshape(n, -1),
        a['planner_action_std'].reshape(n, -1), a['planner_initialized'].reshape(n, 1),
        np.asarray(armed)[:, None]], axis=1).astype(np.float32)


def inside_region(x, region):
    # ARMED is binary and 1 is legitimate (old window counter required <1).
    score = np.linalg.norm((x-region['mean'])/region['scale'], axis=1)
    return np.isfinite(x).all(1) & (score <= region['radius']+1e-5) & np.isin(x[:, -1], [0, 1])


def evaluate(d, mon, low, high, pred, delta, inside):
    events = [e for _, e in reference(d, low, high)[1]]
    flags = budget_diagnostics(pred, mon['event_times'], events, delta, inside)
    resets = [dict(**p, pre_reset_budget=flags['budget_pre_reset'][p['event_timestep']-1],
        post_reset_budget=flags['budget_post_reset'][p['event_timestep']]) for p in mon['pairs']]
    source_ok = source_consistent(d, mon, low, high, resets)
    flags['event_source_pass'] = source_ok
    flags['transition_soundness_pass'] = flags['transition_soundness_pass'] and source_ok
    return flags, resets


def distribution(values):
    a = np.asarray(values, float)
    return dict(n=int(a.size), mean=float(a.mean()), **dict(zip(
        ['min', 'q10', 'q25', 'median', 'q75', 'q90', 'max'],
        map(float, np.quantile(a, [0, .1, .25, .5, .75, .9, 1])))))


def csv_write(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def load_fit(source):
    plan = json.loads((source/'plan.json').read_text())
    audit = json.loads((source/'audit.json').read_text())
    report = json.loads((source/'report.json').read_text())
    assert audit['exact_replay'] and digest(source/'report.json') == audit['report_sha256']
    assert plan['stage'] == 'FIT_ONLY' and plan['horizon'] == 300
    assert plan['policy']['expl_behavior'] == 'CCEPlanner' and plan['policy']['action_source'] == 'policy'
    assert digest(plan['checkpoint']) == plan['checkpoint_sha256']
    for file, sha in plan['code_sha256'].items(): assert digest(file) == sha, file
    assert all(report['split_counts'][r] == 0 for r in ('cal_delta', 'cal_CP', 'test'))
    paths = []
    for row in report['rows']:
        file = source/'fit'/row['file']
        meta = json.loads(file.with_suffix('.json').read_text())
        assert meta['role'] == 'fit' and meta['eligible'] and digest(file) == meta['sha256']
        with np.load(file, allow_pickle=False) as f: a = {k: f[k].copy() for k in f.files}
        assert a['latent'].shape == (301, 512) and a['actions'].shape == (300, 2)
        d = np.linalg.norm(a['decoded'][:, 9:25].reshape(301, 8, 2), axis=-1).min(-1).astype(float)
        assert np.isfinite(d).all()
        paths.append(dict(arrays=a, clearance=d, meta=meta, file=str(file),
            fit_index=row['fit_index'], partition=row['fit_partition']))
    assert len(paths) == 16 and [p['partition'] for p in paths] == ['train']*12+['internal_check']*4
    assert len({p['meta']['fingerprint'] for p in paths}) == 16
    return plan, paths


def run(args):
    require_fit_stage(args.role)
    torch.set_num_threads(2)
    source = args.source.resolve()
    original, paths = load_fit(source)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    pooled = np.concatenate([p['clearance'] for p in paths[:12]])
    candidates = [dict(candidate=name, low_quantile=ql, high_quantile=qh,
        d_low=float(np.quantile(pooled, ql)), d_high=float(np.quantile(pooled, qh))) for name, ql, qh in QUANTILES]
    write(out/'plan.json', dict(created=stamp(), stage='FIT_ONLY', source=str(source),
        source_plan_sha256=digest(source/'plan.json'), reused_paths=16, new_rollouts=0,
        backend='SafeDreamer', checkpoint=original['checkpoint'], checkpoint_sha256=original['checkpoint_sha256'],
        policy=original['policy'], start_distribution=original['start_distribution'], horizon=300,
        specification='GF(return-to-high-clearance); only finite count proxy evaluated',
        formal_thresholds=None, M_MIN=None, candidates=candidates,
        threshold_source='12 train-fit paths, states 0..300, pooled descriptive quantiles',
        initial_state='State0 may arm; only later LOW->HIGH returns emit events',
        features='Full latent + planner carry + current ARMED bit; no future/timestep input',
        roles=original['roles'], training_updates=args.updates,
        delta='Maximum train-fit whole-path residual, not calibrated',
        eta=.01, CV='DISABLED_FOR_PILOT', sliding='DISABLED_FOR_PILOT',
        code_sha256={str(p):digest(p) for p in [Path(__file__), ROOT/'core/finite_recurrence.py',
            ROOT/'experiments/safedreamer_finite_recurrence_pilot.py']},
        scope='FINITE_MODEL_FIT_PREVIEW_NOT_THEOREM_5_6'))
    clearance_rows = [dict(fit_index=p['fit_index'], partition=p['partition'],
        reset_seed=p['meta']['reset_seed'], seed=p['meta']['imagination_seed'],
        **distribution(p['clearance'])) for p in paths]
    write(out/'clearance_distribution.json', dict(threshold_source='train only',
        train=distribution(pooled), internal_check=distribution(np.concatenate([p['clearance'] for p in paths[12:]])),
        all_fit=distribution(np.concatenate([p['clearance'] for p in paths])), per_rollout=clearance_rows))
    event_rows, certificate_rows, failure_rows, summaries, traces = [], [], [], [], []
    for candidate in candidates:
        name, low, high = candidate['candidate'], candidate['d_low'], candidate['d_high']
        fit_paths = []
        for p in paths:
            mon = monitor(p['clearance'], low, high)
            assert source_consistent(p['clearance'], mon, low, high)
            fit_paths.append(dict(monitor=mon, x=product(p['arrays'], mon['armed'])))
        net, region, training = train_budget(fit_paths, 12, args.updates)
        torch.save(dict(state=net.state_dict(), region_radius=region['radius']), out/f'g_init_{name}.pt')
        with torch.no_grad(): predictions = [net(torch.tensor(p['x'])).numpy() for p in fit_paths]
        delta = max(path_score(pred, p['monitor']['event_times']) for pred, p in zip(predictions[:12], fit_paths[:12]))
        max_count = max(p['monitor']['event_count'] for p in fit_paths)
        # Include one threshold with zero count successes; do not select M_MIN.
        m_candidates = list(range(1, max(4, max_count+1)+1))
        for i, (p, f, pred) in enumerate(zip(paths, fit_paths, predictions)):
            mon = f['monitor']
            identity = dict(candidate=name, fit_index=i, partition=p['partition'],
                seed=p['meta']['imagination_seed'], reset_seed=p['meta']['reset_seed'])
            event_rows.append(dict(**identity, d_low=low, d_high=high,
                min_clearance=float(p['clearance'].min()), mean_clearance=float(p['clearance'].mean()),
                N_low_entries=len(mon['low_entry_timestamps']), N_low_excursions=len(mon['low_excursion_timestamps']),
                N_returns=mon['event_count'], low_entry_timestamps=mon['low_entry_timestamps'],
                return_timestamps=mon['return_timestamps'], event_timestamps=mon['event_times'],
                inter_event_intervals=mon['inter_event_intervals'], pending_low_entry=mon['pending_low_entry']))
            inside = inside_region(f['x'], region)
            for label, prediction, correction in [('delta_zero', pred, 0.), ('delta_fit', pred, delta),
                    ('constant_H_eta', np.full(301, 3.), 0.)]:
                flags, resets = evaluate(p['clearance'], mon, low, high, prediction, correction, inside)
                traces.append(dict(**identity, preview=label, flags=flags, resets=resets, monitor=mon))
                for m in m_candidates:
                    result = combine_gates(flags, mon['event_count'], m)
                    certificate_rows.append(dict(**identity, preview=label, M_MIN_candidate=m,
                        N_returns=mon['event_count'],
                        **{k:flags[k] for k in ('drift_pass','delta_margin_pass','region_gate_pass',
                            'transition_soundness_pass','event_source_pass')},
                        count_pass=mon['event_count']>=m, certificate_pass=result['fit_preview_C_rec'],
                        failure_reason=result['failure_reasons'], min_budget_margin=flags['min_pre_reset_budget'],
                        n_region_outside_states=flags['n_region_outside_states'], formal_certificate=None))
        selected = [r for r in certificate_rows if r['candidate']==name]
        for partition in ('all', 'train', 'internal_check'):
            for label in ('delta_zero', 'delta_fit', 'constant_H_eta'):
                for m in m_candidates:
                    rows = [r for r in selected if r['preview']==label and r['M_MIN_candidate']==m
                        and (partition=='all' or r['partition']==partition)]
                    reasons = Counter(reason for r in rows for reason in r['failure_reason'])
                    failure_rows.append(dict(candidate=name, partition=partition, preview=label, M_MIN_candidate=m,
                        n=len(rows), count_successes=sum(r['count_pass'] for r in rows),
                        certificate_successes=sum(r['certificate_pass'] for r in rows),
                        failure_breakdown=dict(reasons)))
        counts = [f['monitor']['event_count'] for f in fit_paths]
        summaries.append(dict(**candidate, delta_fit_preview=delta, region_radius=region['radius'],
            event_counts=counts, count_distribution=dict(Counter(counts)), count_stats=distribution(counts),
            M_MIN_candidates=m_candidates, training=training,
            model_sha256=digest(out/f'g_init_{name}.pt')))
        print(name, 'thresholds', low, high, 'counts', counts, 'delta_fit', delta, flush=True)
    csv_write(out/'threshold_candidates.csv', [{k:v for k,v in s.items() if k!='training'} for s in summaries])
    csv_write(out/'rollout_event_counts.csv', event_rows)
    csv_write(out/'rollout_certificate_results.csv', certificate_rows)
    csv_write(out/'failure_breakdown.csv', failure_rows)
    write(out/'traces.json', traces)
    write(out/'report.json', dict(completed=stamp(), summaries=summaries, breakdown=failure_rows,
        M_MIN=None, formal_thresholds=None, confidence=None, CP_lower=None,
        split_counts=dict(fit=16, cal_delta=0, cal_CP=0, test=0),
        status='FIT_ONLY_PENDING_USER_THRESHOLD_AND_M_CHOICE',
        limitations=['Threshold comparison and all previews reuse D_fit; not independent calibration.',
            'Constructed realized budget decreases by eta; not learned expected drift.',
            'Constant H*eta makes budget checks vacuous on a finite horizon, but cannot create events or fix region failures.',
            'Region is an empirical training envelope, not a proved invariant or Z_free.',
            'No infinite recurrence proof, no warrant, no CP bound.',
            'Low/high are relative decoded-clearance thresholds, not physical safety labels.']))
    write(out/'audit.json', dict(completed=stamp(), original_raw_hashes_verified=True,
        audited_source_report_sha256=digest(source/'report.json'),
        raw_paths=[dict(file=p['file'], sha256=p['meta']['sha256'], seed=p['meta']['imagination_seed']) for p in paths],
        independent_timestamp_reference_matches=True, states_per_path=301, transitions_per_path=300,
        report_sha256=digest(out/'report.json'), split_counts=dict(fit=16, cal_delta=0, cal_CP=0, test=0),
        outputs={p.name:digest(p) for p in out.iterdir() if p.is_file()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'artifacts/safedreamer_finite_recurrence_pilot_v2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--role', default='fit', choices=['fit', 'cal_delta', 'cal_CP', 'test'])
    parser.add_argument('--updates', type=int, default=400)
    args = parser.parse_args()
    if args.updates < 1: parser.error('Positive updates required')
    run(args)
