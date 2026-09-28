"""Replay saved fit pilot with frozen weights; no sampling/training/calibration."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.safedreamer_high_clearance_pilot import load_fit, monitor, product, inside_region, evaluate
from experiments.safedreamer_finite_recurrence_pilot import BudgetNet
from experiments.l3_safedreamer_clear_pipeline import digest, write, stamp
from core.finite_recurrence import combine_gates, path_score


def run(out):
    torch.set_num_threads(2)
    plan = json.loads((out/'plan.json').read_text())
    report = json.loads((out/'report.json').read_text())
    audit = json.loads((out/'audit.json').read_text())
    for name, sha in audit['outputs'].items(): assert digest(out/name) == sha, name
    for file, sha in plan['code_sha256'].items(): assert digest(file) == sha, file
    _, paths = load_fit(Path(plan['source']))
    saved = {(r['candidate'], r['fit_index'], r['preview']): r
        for r in json.loads((out/'traces.json').read_text())}
    with (out/'rollout_certificate_results.csv').open() as f: decisions = list(csv.DictReader(f))
    recomputed = {}
    for summary in report['summaries']:
        name, low, high = summary['candidate'], summary['d_low'], summary['d_high']
        assert digest(out/f'g_init_{name}.pt') == summary['model_sha256']
        weights = torch.load(out/f'g_init_{name}.pt', map_location='cpu', weights_only=True)
        state = weights['state']
        net = BudgetNet(state['mean'], state['scale']); net.load_state_dict(state); net.eval()
        region = dict(mean=state['mean'].numpy(), scale=state['scale'].numpy(), radius=weights['region_radius'])
        scores = []
        for i, p in enumerate(paths):
            mon = monitor(p['clearance'], low, high); x = product(p['arrays'], mon['armed'])
            with torch.no_grad(): pred = net(torch.tensor(x)).numpy()
            scores.append(path_score(pred, mon['event_times']))
            for label, prediction, delta in [('delta_zero', pred, 0.),
                    ('delta_fit', pred, summary['delta_fit_preview']), ('constant_H_eta', np.full(301, 3.), 0.)]:
                flags, resets = evaluate(p['clearance'], mon, low, high, prediction, delta, inside_region(x, region))
                original = saved[name, i, label]
                assert flags == original['flags'] and resets == original['resets'] and mon == original['monitor']
                for m in summary['M_MIN_candidates']:
                    recomputed[name, i, label, m] = combine_gates(flags, mon['event_count'], m)
        assert max(scores[:12]) == summary['delta_fit_preview']
    for row in decisions:
        r = recomputed[row['candidate'], int(row['fit_index']), row['preview'], int(row['M_MIN_candidate'])]
        assert str(r['fit_preview_C_rec']) == row['certificate_pass']
        assert r['failure_reasons'] == json.loads(row['failure_reason'])
    write(out/'audit_replay.json', dict(time=stamp(), exact_saved_weight_replay=True,
        traces_checked=len(saved), decisions_checked=len(decisions),
        full_source_and_reset_metadata_checked=True, source_hashes_checked=True,
        fit_only_delta_recomputed=True, calibration_and_test_not_run=True))
    print('Exact replay:', len(saved), 'traces;', len(decisions), 'gate-AND decisions.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    run(p.parse_args().output.resolve())
