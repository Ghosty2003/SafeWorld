"""Fresh full-RSSM reachability experiment with disjoint selection/calibration.

H actual model transitions use H+1 states. No terminal self-loop is added.
This opt-in experiment does not use the legacy AP-only synthetic-edge path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'artifacts/safedreamer_l2_latent_trial'
AP_KEYS = ['goal_dist', 'hazard_dist', 'velocity']
SPLITS = {'train': (60, 7301), 'validation': (40, 7302),
          'calibration': (100, 7303), 'test': (100, 7304)}


def collect(out):
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper

    paths = [out / f'{name}.npz' for name in SPLITS]
    if any(p.exists() for p in paths):
        raise FileExistsError('Collection would overwrite a split; use --phase fit for existing data')
    extra = build_extra(argparse.Namespace(repo_root=None, checkpoint=None))
    prereg = dict(spec='ltl_goal_reach', formula='F(goal_dist < 0)', horizon_transitions=50,
                  states_per_path=51, eta=0.01, gamma=0.05, threshold=0.8,
                  splits={k: dict(n=n, seed=s) for k, (n, s) in SPLITS.items()},
                  action_source='random', initial_states='encoded simulator resets',
                  environment_comparison=False, checkpoint=extra['checkpoint_path'],
                  candidate_families=['normalized_mlp', 'gaussian_kernel_ridge'],
                  selection='validation full-path event, then P2 pass rate; train fit gate',
                  semantics='q_t has consumed label(z_t); edges are (z_t,q_t)->(z_t+1,q_t+1)',
                  limitation='finite observed model transitions; no support-wide deductive proof')
    (out / 'preregistration.json').write_text(json.dumps(prereg, indent=2))
    cfg = RolloutConfig(horizon=50, n_rollouts=1, seed=7301, action_source='random', extra=extra)
    with SafeDreamerWrapper(cfg) as wrapper:
        wrapper.load()
        for name, (n, seed) in SPLITS.items():
            print(f'Collecting {name}: {n} paths, 50 transitions + actual successor', flush=True)
            cfg = RolloutConfig(horizon=50, n_rollouts=n, seed=seed, action_source='random', extra=extra)
            sampled = wrapper.sample_latent_rollouts(cfg)
            aps = np.array([[[z[k] for k in AP_KEYS] for z in tr] for tr in sampled['aps']], dtype=np.float32)
            arrays = dict(latent=sampled['latent'], decoded=sampled['decoded'], aps=aps)
            arrays.update({f'rssm_{k}': v for k, v in sampled['rssm'].items()})
            assert arrays['latent'].shape[:2] == (n, 51)
            assert all(np.isfinite(v).all() for v in arrays.values())
            np.savez_compressed(out / f'{name}.npz', **arrays)
            hits = (aps[:, :, 0] < 0).any(1).sum()
            print(f'Saved {name}: latent {arrays["latent"].shape}, goal paths {hits}/{n}', flush=True)
    hashes = {}
    for name in SPLITS:
        with np.load(out / f'{name}.npz') as data:
            hashes[name] = [hashlib.sha256(x.tobytes()).hexdigest() for x in data['latent']]
    all_hashes = sum(hashes.values(), [])
    if len(all_hashes) != len(set(all_hashes)):
        raise ValueError('Duplicate full latent paths across/within splits')
    (out / 'split_fingerprints.json').write_text(json.dumps(hashes, indent=2))


def fit_and_evaluate(out):
    import torch
    from scipy.linalg import solve
    from scipy.spatial.distance import cdist
    from core.lppm.latent_reachability import (
        make_features, training_targets, fit_scaler, normalize,
        kernel_predict, audit_values,
    )

    torch.set_num_threads(2)
    eta, floor = 0.01, 0.011

    def load_split(name):
        with np.load(out / f'{name}.npz') as data:
            return make_features(data['latent'], data['aps'])

    x, done = load_split('train')
    xv, dv = load_split('validation')
    mean, scale = fit_scaler(x)
    normalized = normalize(x, mean, scale)
    target = training_targets(done)
    records, models = [], {}

    def register(name, predict, artifact):
        train = audit_values(predict(x), done, eta)
        validation = audit_values(predict(xv), dv, eta)
        fitted = train['p1']['violations'] == train['p2']['violations'] == 0
        record = dict(name=name, train=train, validation=validation,
                      train_fit_pass=fitted, artifact=str(artifact))
        records.append(record)
        models[name] = predict
        print(f'{name}: train P1={train["p1"]} P2={train["p2"]}; '
              f'validation P2={validation["p2"]}, event={validation["certificate_event"]["successes"]}/40', flush=True)

    # Train two nonnegative neural candidates on full latent features.
    # One uses auxiliary finite-graph targets; neither uses future labels
    # or trajectory time as an inference feature.
    flat = torch.tensor(normalized, dtype=torch.float32)
    mask = torch.tensor(~done, dtype=torch.float32)
    targets = torch.tensor(target, dtype=torch.float32)
    for seed, target_weight in ((0, 0.0), (1, 0.2)):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(x.shape[-1], 128), torch.nn.Tanh(),
                                  torch.nn.Linear(128, 128), torch.nn.Tanh(),
                                  torch.nn.Linear(128, 1))
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        for epoch in range(600):
            v = (floor + torch.nn.functional.softplus(net(flat).squeeze(-1))) * mask
            gaps = torch.relu(v[:, 1:]-v[:, :-1]+0.015*mask[:, :-1])
            loss = gaps.mean() + torch.topk(gaps.flatten(), 150).values.mean()
            loss = loss + target_weight*(v-targets).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 200 == 199:
                print(f'MLP seed={seed} epoch={epoch+1} loss={loss.item():.6g}', flush=True)
        def predict(states, net=net):
            with torch.no_grad():
                inp = torch.tensor(normalize(states, mean, scale), dtype=torch.float32)
                return floor + torch.nn.functional.softplus(net(inp).squeeze(-1)).numpy()
        name = f'mlp_seed{seed}_target{target_weight}'
        path = out / f'{name}.pt'
        torch.save(dict(weights=net.state_dict(), mean=mean, scale=scale,
                        floor=floor, feature_schema='deter+flatten(stoch)+goal_dist,hazard_dist,velocity',
                        input_dim=x.shape[-1], hidden=128), path)
        register(name, predict, path)

    # Gaussian kernels can fit the genuine finite training graph exactly;
    # varying bandwidth tests whether smooth interpolation transfers better
    # than the earlier nearest-neighbor lookup. All choices use validation.
    centers = normalized[~done]
    y = target[~done]
    rng = np.random.default_rng(4001)
    ia, ib = rng.integers(len(centers), size=(2, 3000))
    median_distance = float(np.median(np.linalg.norm(centers[ia]-centers[ib], axis=1)))
    distance2 = cdist(centers, centers, 'sqeuclidean')
    for factor in (0.25, 0.5, 1.0, 2.0):
        bandwidth = median_distance*factor
        kernel = np.exp(-distance2/(2*bandwidth**2))
        kernel.flat[::len(kernel)+1] += 1e-7
        alpha = solve(kernel, y, assume_a='pos')
        model = dict(centers=centers, alpha=alpha, mean=mean, scale=scale,
                     bandwidth=bandwidth, floor=floor)
        name = f'gaussian_bandwidth{factor}'
        path = out / f'{name}.npz'
        np.savez_compressed(path, **model)
        register(name, lambda states, model=model: kernel_predict(states, model), path)

    # Prefer actual zero-violation train fits; choose using only validation.
    def selection_key(record):
        val = record['validation']
        return (record['train_fit_pass'], val['certificate_event']['successes'],
                val['p1p2_paths']['successes'], -val['p2']['violations'])
    winner = max(records, key=selection_key)
    selection = dict(selected=winner['name'], candidate_records=records,
                     rule='train zero-violation gate, validation certificate count, validation path pass count, validation P2 violations',
                     horizon=50, eta=eta, waiting_floor=floor,
                     no_time_input=True, no_test_selection=True,
                     terminal_semantics='actual post-label product states; 51 states, 50 edges')
    # This file is committed to disk BEFORE opening calibration/test arrays.
    (out / 'frozen_selection.json').write_text(json.dumps(selection, indent=2))
    print('FROZEN SELECTION:', winner['name'], flush=True)
    evaluate_frozen(out)


def evaluate_frozen(out):
    """Resume evaluation without refitting or changing the frozen selection."""
    import torch
    from core.lppm.latent_reachability import make_features, normalize, kernel_predict, audit_values
    selection = json.loads((out / 'frozen_selection.json').read_text())
    winner = next(r for r in selection['candidate_records'] if r['name'] == selection['selected'])
    eta = selection['eta']
    def load_split(name):
        with np.load(out / f'{name}.npz') as data:
            return make_features(data['latent'], data['aps'])
    selected_path = Path(winner['artifact'])
    if selected_path.suffix == '.npz':
        with np.load(selected_path) as saved:
            model = {k: saved[k] for k in saved.files}
        predict = lambda states: kernel_predict(states, model)
    else:
        saved = torch.load(selected_path, map_location='cpu', weights_only=False)
        net = torch.nn.Sequential(torch.nn.Linear(saved['input_dim'], 128), torch.nn.Tanh(),
                                  torch.nn.Linear(128, 128), torch.nn.Tanh(), torch.nn.Linear(128, 1))
        net.load_state_dict(saved['weights'])
        net.eval()
        def predict(states):
            with torch.no_grad():
                inp = torch.tensor(normalize(states, saved['mean'], saved['scale']), dtype=torch.float32)
                return saved['floor']+torch.nn.functional.softplus(net(inp).squeeze(-1)).numpy()
    # Check persisted inference exactly reproduces the selected train audit.
    x, done = load_split('train')
    train_audit = audit_values(predict(x), done, eta)
    assert all(train_audit[k] == v for k, v in winner['train'].items())
    xv, dv = load_split('validation')
    val_audit = audit_values(predict(xv), dv, eta)
    assert all(val_audit[k] == v for k, v in winner['validation'].items())
    results = dict(train=train_audit, validation=val_audit)
    for split in ('calibration', 'test'):
        xs, ds = load_split(split)
        results[split] = audit_values(predict(xs), ds, eta)
        print(split, json.dumps({k: v for k, v in results[split].items() if k != 'per_path_event'}), flush=True)
    cal = results['calibration']['certificate_event']
    test = results['test']['certificate_event']
    result = 'SAMPLED_EVENT_THRESHOLD_MET' if cal['cp_lower'] >= 0.8 else 'NO_WARRANT'
    # Training/selection samples are adaptive; no inferential CP claim for them.
    for split in ('train', 'validation'):
        for value in results[split].values():
            if isinstance(value, dict):
                value.pop('cp_lower', None)
                value.pop('confidence', None)
    report = dict(selected=winner['name'], splits=results, eta=eta, confidence=0.95,
                  result=result, calibration_lower_threshold=0.8,
                  certificate_event='all 50 observed P1/P2 checks and final V<eta in accepting state',
                  zfree='accepting q; waiting branch is >=0.011 > eta=0.01',
                  closure='accepting state is absorbing for F(goal); V=0 there by construction',
                  deductive_status='NOT_ESTABLISHED',
                  probability_comparison=dict(calibration_rate=cal['rate'], test_rate=test['rate'],
                                              absolute_rate_error=abs(cal['rate']-test['rate'])),
                  scope='finite 50-step model-imagination event; random actions; reset-encoded initial states',
                  preprocessing='train-only mean/std with scale floor 0.05, reused unchanged at inference',
                  valid_probability_bounds_for=['calibration', 'test'],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print('RESULT:', result, 'report:', out / 'report.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--phase', choices=['collect', 'fit', 'evaluate', 'all'], default='all')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase in ('collect', 'all'):
        collect(args.out)
    if args.phase in ('fit', 'all'):
        fit_and_evaluate(args.out)
    if args.phase == 'evaluate':
        evaluate_frozen(args.out)


if __name__ == '__main__':
    main()
