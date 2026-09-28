"""Frozen motion V: fresh calibration500, then independent Test1=1000.

No fitting or selection. Development 19/20 is disclosed, not upgraded to 100%.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import l2_motion_v_wide_search as w
from experiments.safedreamer_finite_recurrence_pilot import SeededResetRecorder, extract
base = w.base
SOURCE = ROOT / 'artifacts/safedreamer_l2_motion_v_wide_v1'
COUNTS = dict(calibration=500, test=1000)
SEEDS = dict(calibration=171000000, test=172000000)


def cp(k, n):
    return float(beta.ppf(.05, k, n-k+1)) if k else 0.


def values(bundle, paths):
    """Evaluate embedded frozen weights only; no dependency on source weights."""
    spec = bundle['spec']; manifest = bundle['baseline']
    primitives = []
    for p, m in zip(manifest['primitives'], bundle['baseline_models']):
        if p['kind'] == 'old':
            x, q, b, d, prior = w.screen.products(paths, spec)
            v = w.screen.predict(m, x, d, prior)
        else:
            x, q, b, d = w.prev.products(paths, spec, m['config']['features'])
            v = w.prev.predict(m, x, d)
        primitives.append(4*v)
    old = w.ens.aggregate(np.stack(primitives), manifest['aggregation'])
    m = bundle['new_model']
    x, q, b, d = w.products(paths, spec, m['config']['features'])
    return w.combine(w.predict(m, x, q), old, bundle['config']), q, b, d


def evaluate(bundle, region, paths):
    v, q, b, d = values(bundle, paths)
    mask = np.linalg.norm((np.stack([base.physical(a) for a in paths])-region['mean'])/
                          region['scale'], axis=-1) <= float(region['radius'])+1e-10
    stat = base.statistics(v, b, d, region=mask)
    pp, cert = w.screen.independent(v, b, d, mask)
    np.testing.assert_array_equal(pp, stat['p1p2_per_path'])
    np.testing.assert_array_equal(cert, stat['certificate_per_path'])
    # Independent two-stage implementation, including t=0 and strict ordering.
    for a, done in zip(paths, d):
        speed = np.linalg.norm(a['decoded'][:, :2].astype(float), axis=-1)
        low = np.flatnonzero(speed <= bundle['spec']['stages'][0]['threshold'])
        manual = np.zeros(len(speed), bool)
        if len(low):
            high = np.flatnonzero((np.arange(len(speed)) > low[0]) &
                                 (speed >= bundle['spec']['stages'][1]['threshold']))
            if len(high): manual[high[0]:] = True
        np.testing.assert_array_equal(done, manual)
    rows = []
    for i in range(len(paths)):
        dv = np.diff(v[i]); z = v[i] < .01
        fail1 = np.flatnonzero(dv > 0).tolist()
        fail2 = np.flatnonzero(b[i, :-1] & (dv > -.01)).tolist()
        rows.append(dict(task=bool(d[i, -1]), p1=not fail1, p2=not fail2,
            p1p2=bool(pp[i]), certificate=bool(cert[i]), p1_failure_times=fail1,
            p2_failure_times=fail2, region=bool(mask[i].all()),
            endpoint=bool(d[i, -1] and z[-1]), exclusion=bool((~z | ~b[i]).all()),
            closure=bool((~z[:-1] | (z[1:] & ~b[i, 1:])).all()),
            finite_nonnegative=bool(np.isfinite(v[i]).all() and (v[i] >= 0).all()),
            first_completion=int(np.flatnonzero(d[i])[0]) if d[i].any() else None,
            bad_transitions=int(b[i, :-1].sum())))
    return rows, v, stat


def prepare(out):
    assert not out.exists()
    source_plan = json.loads((SOURCE/'plan.json').read_text())
    audit = json.loads((SOURCE/'audit.json').read_text())
    assert base.digest(SOURCE/'selected_bundle.pt') == audit['selected_bundle_sha256']
    bundle = torch.load(SOURCE/'selected_bundle.pt', map_location='cpu')
    assert bundle['spec']['id'] == 'motion_burst' and bundle['H'] == 300 and bundle['eta'] == .01
    prior = json.loads((ROOT/'artifacts/safedreamer_b_fit100_v1/plan.json').read_text())
    assert base.digest(prior['checkpoint']) == prior['checkpoint_sha256']
    assert base.digest(ROOT/'wrappers/safedreamer_wrapper.py') == prior['wrapper_sha256']
    # Identity metadata, not outcomes, from prior collections and raw sidecars.
    fingerprints = set(); seeds = set()
    for f in (ROOT/'artifacts').glob('safedreamer*/**/*.json'):
        if not ('collection' in f.name or f.name.startswith('draw_') or 'provenance' in f.name
                or f.name == 'prior_identities.json'):
            continue
        obj = json.loads(f.read_text())
        records = obj if isinstance(obj, list) else obj.get('records', [obj])
        if isinstance(obj, dict): fingerprints.update(obj.get('fingerprints', []))
        for r in records:
            if not isinstance(r, dict): continue
            if 'fingerprint' in r: fingerprints.add(r['fingerprint'])
            for k in ('reset_seed', 'imagination_seed'):
                if k in r: seeds.add(r[k])
    assert not any(start <= s < start+10000 for start in SEEDS.values() for s in seeds)
    out.mkdir(parents=True); (out/'raw').mkdir()
    shutil.copyfile(SOURCE/'selected_bundle.pt', out/'selected_bundle.pt')
    shutil.copyfile(w.exp.SOURCE/'region.npz', out/'region.npz')
    base.write(out/'prior_identities.json', dict(fingerprints=sorted(fingerprints), seeds=sorted(seeds)))
    modules = [Path(__file__), Path(w.__file__), Path(w.prev.__file__), Path(w.ens.__file__),
               Path(w.screen.__file__), Path(base.__file__), Path(w.exp.__file__),
               ROOT/'experiments/l2_spec_continuation_audit.py',
               ROOT/'experiments/safedreamer_finite_recurrence_pilot.py',
               ROOT/'experiments/l2_safedreamer.py', ROOT/'wrappers/safedreamer_wrapper.py']
    plan = dict(at=base.stamp(), checkpoint=prior['checkpoint'], checkpoint_sha256=prior['checkpoint_sha256'],
        policy=prior['policy'], spec=bundle['spec'], H=300, eta=.01, tolerance=0.,
        counts=COUNTS, seeds=SEEDS, max_draws={k:5*n for k,n in COUNTS.items()},
        model_sha256=base.digest(out/'selected_bundle.pt'), region_sha256=base.digest(out/'region.npz'),
        prior_identities_sha256=base.digest(out/'prior_identities.json'),
        code_sha256={str(p.resolve()):base.digest(p) for p in modules},
        source_plan_sha256=base.digest(SOURCE/'plan.json'), candidate=bundle['config'],
        development='80/80 fit, 19/20 reused internal; double100 qualification NOT met; user-authorized frozen evaluation',
        start_distribution=prior['start_distribution'], eligibility='decoded initial goal distance >=1.0 only; first N eligible; no future filtering',
        gates=['P1 all300','P2 all source-bad transitions','completion and endpoint V<eta',
               'Zfree bad exclusion','sampled Zfree closure','original fit-only region','finite nonnegative V'],
        z_free='V(x)<.01; absorbing completed q=2; not a physical forever-safe region',
        warrant_threshold=.95, confidence=.95,
        decision='Calibration certificate CP>=.95 and no exclusion/closure counterevidence -> SAFE (statistical finite-model only); else ABSTAIN',
        test='1000 fresh seeds after immutable calibration prediction, separate CP; no pooling or selection',
        inference='Fixed N, iid conditional reset/imagination draws assumed. Each reported CP is marginal, not simultaneous.',
        limitation='Not sampled double100 qualified, not global P1/P2 proof, not real-environment or infinite-horizon safety')
    base.write(out/'plan.json', plan)
    print('FROZEN calibration500 then test1000; H300; original V/eta/region/detector', flush=True)


def check(out):
    p = json.loads((out/'plan.json').read_text())
    for file, sha in p['code_sha256'].items(): assert base.digest(file) == sha, file
    for file, key in [('selected_bundle.pt','model_sha256'), ('region.npz','region_sha256'),
                      ('prior_identities.json','prior_identities_sha256')]:
        assert base.digest(out/file) == p[key], file
    return p


def records(out, split):
    return [json.loads(f.read_text()) for f in sorted((out/'raw'/split).glob('draw_*.json'))]


def summarize(out, split, bundle, region):
    p = check(out); rs = records(out, split); rows = []; all_v = []
    for r in rs:
        if not r['eligible']: continue
        f = out/'raw'/split/r['file']; assert base.digest(f) == r['sha256']
        with np.load(f) as a: paths = [{k:a[k] for k in a.files}]
        row, v, _ = evaluate(bundle, region, paths)
        row[0].update(draw=r['draw'], reset_seed=r['reset_seed'], imagination_seed=r['imagination_seed'])
        rows.extend(row); all_v.append(v[0])
    n = len(rows); assert n == p['counts'][split]
    metrics = {key:dict(k=sum(r[key] for r in rows), n=n,
                       cp_lower95=cp(sum(r[key] for r in rows), n))
               for key in ('task','p1','p2','p1p2','certificate')}
    failures = {k:sum(not r[k] for r in rows) for k in
                ('p1','p2','endpoint','exclusion','closure','region','finite_nonnegative')}
    result = dict(at=base.stamp(), split=split, metrics=metrics, failure_paths=failures,
        p1_violations=sum(len(r['p1_failure_times']) for r in rows),
        p2_violations=sum(len(r['p2_failure_times']) for r in rows),
        bad_transitions=sum(r['bad_transitions'] for r in rows), rows=rows,
        plan_sha256=base.digest(out/'plan.json'), model_sha256=p['model_sha256'])
    base.write(out/(split+'_report.json'), result)
    np.savez_compressed(out/(split+'_values.npz'), values=np.stack(all_v))
    if split == 'calibration':
        ok = metrics['certificate']['cp_lower95'] >= .95 and failures['exclusion'] == failures['closure'] == 0
        base.write(out/'prediction.json', dict(at=base.stamp(), verdict='SAFE' if ok else 'ABSTAIN',
            cp_lower95=metrics['certificate']['cp_lower95'], threshold=.95,
            calibration_report_sha256=base.digest(out/'calibration_report.json'),
            model_sha256=p['model_sha256'], plan_sha256=base.digest(out/'plan.json'),
            scope='Statistical finite-model certificate event ONLY; internal double100 NOT met'))
    print('FINISHED', split, metrics, flush=True)


def collect(out):
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    torch.set_num_threads(2); p = check(out)
    assert base.digest(p['checkpoint']) == p['checkpoint_sha256']
    bundle = torch.load(out/'selected_bundle.pt', map_location='cpu'); region = dict(np.load(out/'region.npz'))
    seen = set(json.loads((out/'prior_identities.json').read_text())['fingerprints'])
    for split in p['counts']:
        for r in records(out, split):
            assert r['fingerprint'] not in seen; seen.add(r['fingerprint'])
    extra = build_extra(argparse.Namespace(repo_root=None, checkpoint=p['checkpoint']))
    extra['action_source'] = 'policy'; extra['config_overrides']['jax'].update(platform='gpu', logical_gpus=0)
    start = time.monotonic()
    with SafeDreamerWrapper(RolloutConfig(horizon=300,n_rollouts=1,seed=SEEDS['calibration']+1,action_source='policy',extra=extra)) as wrapper:
        wrapper.load()
        import jax
        assert jax.default_backend() == 'gpu'
        actual = dict(action_source='policy', expl_behavior=wrapper._config.expl_behavior,
            planner={k:getattr(wrapper._config.planner,k) for k in p['policy']['planner']},
            cost_limit=wrapper._config.cost_limit,
            planner_source_sha256=base.digest(Path(extra['repo_root'])/'SafeDreamer/behaviors.py'), note=p['policy']['note'])
        assert actual == p['policy']
        recorder = SeededResetRecorder(wrapper._env); wrapper._env = recorder
        try:
            for split, n in p['counts'].items():
                if (out/(split+'_report.json')).exists(): continue
                check(out)
                if split == 'test':
                    pred=json.loads((out/'prediction.json').read_text())
                    assert pred['plan_sha256']==base.digest(out/'plan.json') and pred['model_sha256']==p['model_sha256']
                marker=out/(split+'_started.json')
                if not marker.exists(): base.write(marker,dict(at=base.stamp(),plan_sha256=base.digest(out/'plan.json'),
                    prediction_sha256=base.digest(out/'prediction.json') if split=='test' else None))
                folder=out/'raw'/split; folder.mkdir(exist_ok=True)
                rs=records(out,split); kept=sum(r['eligible'] for r in rs)
                assert len(list(folder.glob('*.npz')))==len(rs), 'Uncommitted raw; audit before resume'
                for draw in range(len(rs),p['max_draws'][split]):
                    if kept==n: break
                    reset=p['seeds'][split]+2*draw; recorder.pending=reset
                    data=wrapper.sample_latent_rollouts(RolloutConfig(horizon=300,n_rollouts=1,seed=reset+1,action_source='policy',extra=extra))
                    assert recorder.pending is None and data['action_source']=='policy'
                    a=extract(data,recorder.observation); assert a['latent'].shape==(301,512)
                    fp=hashlib.sha256(a['latent'].tobytes()).hexdigest()
                    assert fp not in seen, 'Duplicate trajectory, abort without replacement'; seen.add(fp)
                    initial=float(np.linalg.norm(a['decoded'][0,7:9])); eligible=initial>=1.
                    f=folder/('draw_%05d.npz'%draw); assert not f.exists(); np.savez_compressed(f,**a)
                    base.write(f.with_suffix('.json'),dict(draw=draw,file=f.name,sha256=base.digest(f),fingerprint=fp,
                        eligible=eligible,initial_goal=initial,reset_seed=reset,imagination_seed=reset+1,at=base.stamp()))
                    kept+=int(eligible)
                    if draw%10==0 or kept==n:
                        print('PROGRESS',split,kept,'/',n,'raw',draw+1,'elapsed_s',round(time.monotonic()-start,1),flush=True)
                assert kept==n, 'Fixed-N sampling incomplete; no warrant'
                summarize(out,split,bundle,region)
        finally:
            recorder.restore(); wrapper._env=recorder.outer
    audit(out)


def audit(out):
    torch.set_num_threads(2); p=check(out)
    bundle=torch.load(out/'selected_bundle.pt',map_location='cpu'); region=dict(np.load(out/'region.npz'))
    seen=set(json.loads((out/'prior_identities.json').read_text())['fingerprints']); used=set(); reports={}
    for split in p['counts']:
        report=json.loads((out/(split+'_report.json')).read_text()); reports[split]=report
        saved=np.load(out/(split+'_values.npz'))['values']; kept=0
        rs=records(out,split)
        for i,r in enumerate(rs):
            assert r['draw']==i and r['reset_seed']==p['seeds'][split]+2*i and r['imagination_seed']==r['reset_seed']+1
            for key in ('reset_seed','imagination_seed'):
                assert r[key] not in used; used.add(r[key])
            f=out/'raw'/split/r['file']; assert base.digest(f)==r['sha256']
            with np.load(f) as a: path={k:a[k] for k in a.files}
            fp=hashlib.sha256(path['latent'].tobytes()).hexdigest()
            assert fp==r['fingerprint'] and fp not in seen; seen.add(fp)
            assert r['eligible']==bool(np.linalg.norm(path['decoded'][0,7:9])>=1.)
            if not r['eligible']: continue
            rows,v,_=evaluate(bundle,region,[path]); row=rows[0]
            row.update(draw=i,reset_seed=r['reset_seed'],imagination_seed=r['imagination_seed'])
            assert row==report['rows'][kept]; np.testing.assert_array_equal(v[0],saved[kept]); kept+=1
        assert kept==p['counts'][split] and rs[-1]['eligible']
        for key,metric in report['metrics'].items():
            k=sum(r[key] for r in report['rows'])
            assert metric==dict(k=k,n=kept,cp_lower95=cp(k,kept))
    pred=json.loads((out/'prediction.json').read_text()); marker=json.loads((out/'test_started.json').read_text())
    assert marker['prediction_sha256']==base.digest(out/'prediction.json')
    assert pred['calibration_report_sha256']==base.digest(out/'calibration_report.json')
    assert pred['at']<=marker['at']<=records(out,'test')[0]['at']
    base.write(out/'audit.json',dict(at=base.stamp(),weights_code_verified=True,complete_AND_replayed=1500,
        manual_detector_agrees=True,raw_fingerprints_unique=True,seeds_disjoint=True,
        calibration_prediction_before_test=True,counts=p['counts'],model_sha256=p['model_sha256']))
    header=['Layer','Specification','MP class','Backend / carrier','H','Verdict','p_hat_gamma (CP lower, 95%)','Test 1 (N=1000)']
    t=reports['test']['metrics']['certificate']
    row=['L2','F[0,300](LOW speed then strictly later HIGH speed)','Guarantee (bounded)',
         'SafeDreamer / SafetyPointGoal1-v0',300,pred['verdict'],'%.4f'%pred['cp_lower95'],
         '%d/1000, %.4f'%(t['k'],t['cp_lower95'])]
    with (out/'final_table.csv').open('x',newline='') as f:
        writer=csv.writer(f); writer.writerow(header); writer.writerow(row)
    lines=['# 低速后进入高速：冻结校准与 Test1','',
           '| '+' | '.join(header)+' |','| '+' | '.join(['---']*8)+' |','| '+' | '.join(map(str,row))+' |','',
           'LOW <=0.2245011670；严格之后 HIGH >=0.7390221069；H=300，eta=.01。',
           '内部开发仍为19/20，不是双100%合格；SAFE若出现，仅指预注册阈值下的统计有限模型结论。',
           '无真实环境安全或无限时域保证；各下界为分别95%，不是联合95%。','',
           '| 指标 | 新校准500 | 校准CP下界 | 独立Test1000 | Test自身CP下界 |','|---|---|---|---|---|']
    for key in ('task','p1','p2','p1p2','certificate'):
        a=reports['calibration']['metrics'][key]; b=reports['test']['metrics'][key]
        lines.append('| %s | %d/500 | %.4f | %d/1000 | %.4f |'%(key,a['k'],a['cp_lower95'],b['k'],b['cp_lower95']))
    for split in p['counts']: lines+=['',split+' failure breakdown: '+json.dumps(reports[split]['failure_paths'])]
    lines+=['','两批未混用；全部raw、逐路径失败时间和V值保留；无测试后调参。']
    (out/'RESULT_CN.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','collect','audit'])
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    globals()[args.stage](args.output.resolve())
