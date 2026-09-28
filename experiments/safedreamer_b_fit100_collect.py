"""Collect new 80 train / 20 internal-validation paths; FIT ONLY."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.safedreamer_finite_recurrence_pilot import SeededResetRecorder,extract
from experiments.l3_safedreamer_clear_pipeline import digest,write,stamp


def run(out):
    frozen_file=ROOT/'artifacts/safedreamer_recurrence_screening_v1/frozen_primary.json'
    frozen=json.loads(frozen_file.read_text())
    assert digest(frozen['checkpoint'])==frozen['checkpoint_sha256']
    for file,sha in frozen['code_sha256'].items(): assert digest(ROOT/file if not Path(file).is_absolute() else file)==sha
    out.mkdir(parents=True,exist_ok=False)
    plan=dict(created=stamp(),scope='NEW_FIT_ONLY_B_LATENCY_STUDY',source_freeze_sha256=digest(frozen_file),
        checkpoint=frozen['checkpoint'],checkpoint_sha256=frozen['checkpoint_sha256'],policy=frozen['policy'],
        detector=frozen['definition'],H=300,eta=.01,M_MIN=2,budget_cap=2.4,
        budget_cap_note='Total post-residual allocation <=2.4; observed gaps>240 necessarily fail',
        delta_rule='Train-only maximum whole-path residual preview; clip total budget to cap after adding; not calibration',
        start_distribution=frozen['start_distribution'],
        roles=dict(train_fit=dict(n=80,seed_base=131000000),internal_validation=dict(n=20,seed_base=132000000),
            cal_delta=dict(n=0,status='LOCKED'),cal_CP=dict(n=0,status='LOCKED'),test=dict(n=0,status='LOCKED')),
        sampling='No future filtering. Train batching stratified by completed remaining latency <=48,49..120,121..240,>240 plus censored; long bins oversampled',
        candidates=[dict(name='bounded_uniform',width=64,weighted=False,under_weight=1.,magnitude=.01),
                    dict(name='stratified',width=64,weighted=True,under_weight=3.,magnitude=.01),
                    dict(name='stratified_wide_penalty',width=128,weighted=True,under_weight=3.,magnitude=.05)],
        updates=1200,training_seed=918,
        model_selection='Pareto on validation underestimation mean, MAE (completed reset segments), full certificate pass; then normalized sum of only these three',
        selection_note='No performance claim from used-for-selection validation; no calibration/Test1',
        collection_code_sha256=digest(__file__),
        wrapper_sha256=digest(ROOT/'wrappers/safedreamer_wrapper.py'))
    write(out/'plan.json',plan)
    from configs.settings import RolloutConfig
    from experiments.l2_safedreamer import build_extra
    from wrappers.safedreamer_wrapper import SafeDreamerWrapper
    extra=build_extra(argparse.Namespace(repo_root=None,checkpoint=frozen['checkpoint']))
    extra['action_source']='policy'; extra['config_overrides']['jax'].update(platform='gpu',logical_gpus=0)
    raw=out/'raw'; raw.mkdir(); seen=set(); records=[]; started=time.monotonic()
    with SafeDreamerWrapper(RolloutConfig(horizon=300,n_rollouts=1,seed=131000001,action_source='policy',extra=extra)) as wrapper:
        wrapper.load()
        import jax
        assert jax.default_backend()=='gpu'
        actual=dict(action_source='policy',expl_behavior=wrapper._config.expl_behavior,
            planner={k:getattr(wrapper._config.planner,k) for k in frozen['policy']['planner']},
            cost_limit=wrapper._config.cost_limit,
            planner_source_sha256=digest(Path(extra['repo_root'])/'SafeDreamer/behaviors.py'),
            note=frozen['policy']['note'])
        assert actual==frozen['policy']
        recorder=SeededResetRecorder(wrapper._env);wrapper._env=recorder
        try:
            for role in ('train_fit','internal_validation'):
                count=0; base=plan['roles'][role]['seed_base']; target=plan['roles'][role]['n']
                for draw in range(target*5):
                    reset_seed=base+2*draw;imagination_seed=reset_seed+1
                    def sample():
                        recorder.pending=reset_seed
                        data=wrapper.sample_latent_rollouts(RolloutConfig(horizon=300,n_rollouts=1,
                            seed=imagination_seed,action_source='policy',extra=extra))
                        assert recorder.pending is None and data['action_source']=='policy'
                        return extract(data,recorder.observation)
                    a=sample()
                    if not records:
                        repeat=sample();exact={k:bool(np.array_equal(v,repeat[k])) for k,v in a.items()}
                        write(out/'seed_replay.json',dict(reset_seed=reset_seed,imagination_seed=imagination_seed,exact=exact))
                        assert all(exact.values())
                    assert a['latent'].shape==(301,512) and a['actions'].shape==(300,2)
                    fp=hashlib.sha256(a['latent'].tobytes()).hexdigest();assert fp not in seen;seen.add(fp)
                    initial_goal=float(np.linalg.norm(a['decoded'][0,7:9]));eligible=bool(initial_goal>=1.)
                    file=raw/f'{role}_{draw:04d}.npz';np.savez_compressed(file,**a)
                    record=dict(role=role,draw=draw,fit_index=count if eligible else None,eligible=eligible,
                        initial_goal=initial_goal,reset_seed=reset_seed,imagination_seed=imagination_seed,
                        file=file.name,sha256=digest(file),fingerprint=fp,time=stamp())
                    write(file.with_suffix('.json'),record);records.append(record)
                    if eligible:count+=1
                    print(role,'draw',draw,'eligible',count,'/',target,'elapsed_s',round(time.monotonic()-started,1),flush=True)
                    if count==target:break
                if count!=target:raise RuntimeError('Draw budget exhausted; do not train incomplete data')
        finally:
            recorder.restore();wrapper._env=recorder.outer
    write(out/'collection.json',dict(completed=stamp(),elapsed_seconds=time.monotonic()-started,records=records,
        retained=dict(train_fit=80,internal_validation=20),formal_calibration_count=0,test_count=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output.resolve())
