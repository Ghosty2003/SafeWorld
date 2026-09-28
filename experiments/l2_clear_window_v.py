"""Fit finite-horizon V for F[1,17] G[0,47] !hazard, development only.

The causal monitor counts consecutive clear states from t=1 (not t=0).
Acceptance is absorbing after 48 clear observations by t=64. The value
function sees latent/planner state and the monitor's run length, not time,
path identity, or future completion labels. This is NOT an infinite proof.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.lppm.latent_reachability import fit_scaler
from core.lppm.value_families import ValueFamily, predict_family
from experiments import l2_achievement64 as base
from experiments.l2_approach1_outside_v import write
from experiments.l2_latent_prediction_check import digest, now
from experiments.l2_policy_v_refine import objective
from experiments.l2_spec_screen64 import window_event

SPEC = 'F[1,17] G[0,47](decoded hazard_margin>=0)'


def monitor(aps):
    if aps.ndim != 3 or aps.shape[1:] != (65, 3) or not np.isfinite(aps).all():
        raise ValueError('Expected finite N x 65 x 3 APs')
    clear = aps[..., 1] >= 0
    run = np.zeros(clear.shape, dtype=np.int64)
    done = np.zeros(clear.shape, dtype=bool)
    for t in range(1, 65):
        run[:, t] = np.where(done[:, t-1], 48,
                            np.where(clear[:, t], np.minimum(run[:, t-1]+1, 48), 0))
        done[:, t] = done[:, t-1] | (run[:, t] == 48)
    np.testing.assert_array_equal(done[:, -1], window_event(clear, 1, 17, 48))
    return run, done


def load_data(source, expected=None):
    data, metadata, seen = {}, {}, set()
    for split in ('train', 'validation'):
        path = source / f'{split}.npz'
        sha = digest(path)
        prov = json.loads((source / f'{split}_provenance.json').read_text())
        if sha != prov['sha256'] or (expected and sha != expected[split]['sha256']):
            raise ValueError('Data hash changed')
        ids = prov['fingerprints']
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError('Repeated trajectory identity')
        seen.update(ids)
        with np.load(path) as arrays:
            run, done = monitor(arrays['aps'])
            z = base.value_features(arrays, 'policy')
        # Run length is past/present monitor memory, not remaining horizon.
        features = np.concatenate((z, run[..., None] / 48.), axis=-1)
        data[split] = (features, done)
        metadata[split] = dict(path=str(path), sha256=sha, n=len(done),
                               completed=int(done[:, -1].sum()))
    return data, metadata


def summarize(values, done):
    row = base.summary(values, done, inferential=False)
    # The shared audit calls generic accepting labels 'goal'; rename them.
    return {key.replace('goal', 'completion'): value for key, value in row.items()}


def rank(row):
    return (row['p1p2_paths']['successes'],
            row['finite_completion_and_certificate']['successes'],
            -row['pending_sublevel_states'], -row['p2']['violations'],
            -row['p1']['violations'], -row['mean_paper_transition_loss'])


def evaluate(out):
    plan = json.loads((out / 'plan.json').read_text())
    frozen = json.loads((out / 'frozen.json').read_text())
    report = json.loads((out / 'report.json').read_text())
    if digest(out / 'selected.pt') != frozen['model_sha256'] or digest(out / 'plan.json') != frozen['plan_sha256']:
        raise ValueError('Frozen model or plan changed')
    data, _ = load_data(Path(plan['source']), plan['data'])
    model = torch.load(out / 'selected.pt', weights_only=False)
    for split, (z, done) in data.items():
        actual = summarize(predict_family(model, z, done), done)
        if actual != report[split]:
            raise ValueError('Metrics do not reproduce')
        print('VERIFIED', split, actual['p1p2_paths'], flush=True)


def train(source, out, steps, families=('linear', 'elu', 'residual', 'tanh')):
    data, metadata = load_data(source)
    source_plan = json.loads((source / 'plan.json').read_text())
    if digest(Path(source_plan['checkpoint'])) != source_plan['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    if digest(source / 'policy_config.json') != source_plan['policy_config_sha256']:
        raise ValueError('Policy changed')
    out.mkdir(parents=True, exist_ok=False)
    candidates = [dict(name=f'{family}_max{weight}', family=family, width=128,
                       seed=28500+2*i+j, path_weight=weight)
                  for i, family in enumerate(families)
                  for j, weight in enumerate((0, 4))]
    plan = dict(created_at=now(), source=str(source), data=metadata, spec=SPEC,
                checkpoint=source_plan['checkpoint'], checkpoint_sha256=source_plan['checkpoint_sha256'],
                policy_config_sha256=source_plan['policy_config_sha256'],
                initial_condition=source_plan['initial_condition'], horizon=64, eta=.01,
                value_inputs='577 latent/planner features plus causal clear-run counter/48; '
                             'absorbing accepting branch. No elapsed/remaining time or future input.',
                monitor='Ignore t0, increment clear run at t1..64, reset on hazard, '
                        'accept after 48 clear states; completion absorbing. Unaccepted at64 fails.',
                bad_states='All not-yet-accepting monitor states, including failed finite prefixes',
                objective='Mean/worst-path hinge, training margin 2*eta, learned accepting scalar '
                          'anchor, waiting sublevel penalty 10; evaluation exact eta=.01.',
                selection='Whole-path P1P2 then complete certificate then fewer pending sublevel states',
                candidates=candidates, updates=steps, training_device='cpu',
                warrant_threshold=.95, future_confidence=.95,
                scope='DEVELOPMENT_ONLY; adaptively chosen finite specification, no deductive claim')
    write(out / 'plan.json', plan)
    z, done = data['train']
    mean, scale = fit_scaler(z)
    x = torch.tensor((z-mean)/scale, dtype=torch.float32)
    labels = torch.tensor(done, dtype=torch.float32)
    records, best = [], None
    for cfg in candidates:
        torch.manual_seed(cfg['seed'])
        net = ValueFamily(cfg['family'], z.shape[-1], cfg['width'])
        optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=1e-4)
        candidate_best = None
        for step in range(1, steps+1):
            ix = torch.randperm(len(x))[:32]
            values = net(x[ix], labels[ix])
            loss = objective(values, labels[ix], margin=2., path_weight=cfg['path_weight'])
            loss = loss + 9*torch.relu(.02-values[~labels[ix].bool()]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
            optimizer.step()
            if step % 200 and step != steps:
                continue
            model = dict(family=cfg['family'], width=cfg['width'], mean=mean, scale=scale,
                         weights={k:v.detach().clone() for k,v in net.state_dict().items()},
                         eta=.01, spec=SPEC, step=step, config=cfg, feature_schema=plan['value_inputs'])
            row = dict(candidate=cfg['name'], step=step,
                       **{s:summarize(predict_family(model, zz, dd), dd) for s,(zz,dd) in data.items()})
            key = rank(row['validation'])
            records.append(row)
            if candidate_best is None or key > candidate_best:
                candidate_best = key
                torch.save(model, out / f"{cfg['name']}.pt")
            if best is None or key > rank(best['validation']):
                best = row
                torch.save(model, out / 'selected.pt')
            (out / 'progress.json').write_text(json.dumps(dict(best=best, records=records), indent=2))
            print('FIT', cfg['name'], step, 'train P1P2', row['train']['p1p2_paths']['successes'],
                  'dev P1P2', row['validation']['p1p2_paths']['successes'],
                  'dev C', row['validation']['finite_completion_and_certificate']['successes'],
                  'dev P1/P2', row['validation']['p1']['violations'], row['validation']['p2']['violations'],
                  'pending low', row['validation']['pending_sublevel_states'], flush=True)
    write(out / 'frozen.json', dict(frozen_at=now(), model_sha256=digest(out/'selected.pt'),
                                    plan_sha256=digest(out/'plan.json'), best=best))
    write(out / 'report.json', dict(result='TRAINED_NOT_CALIBRATED', warrant_status='NOT_ASSESSED',
                                    confidence_lower_bound=None, spec=SPEC, **best))
    print('SELECTED', best['candidate'], best['step'], flush=True)
    evaluate(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=base.ROOT/'artifacts/safedreamer_l2_approach1_more_data')
    parser.add_argument('--output', type=Path, default=base.ROOT/'artifacts/safedreamer_l2_clear_window_v')
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--families', nargs='+', choices=('linear', 'elu', 'residual', 'tanh'),
                        default=['linear', 'elu', 'residual', 'tanh'])
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    if args.steps < 1: parser.error('Positive steps required')
    torch.set_num_threads(2)
    if args.evaluate_only: evaluate(args.output.resolve())
    else: train(args.source.resolve(), args.output.resolve(), args.steps, args.families)


if __name__ == '__main__':
    main()
