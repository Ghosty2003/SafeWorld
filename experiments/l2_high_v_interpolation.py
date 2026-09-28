"""Fit-only compact radial interpolation diagnostic: memorization != proof.

Fixed centers are training product states. Each bump radius is one quarter
of that center's nearest other training-center distance; balls are disjoint.
The nearest-center bump is continuous, nonnegative, and state-only. This is
a radial basis function, NOT a learned dynamics model or a global certificate.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments import l2_high_v_refinement as ref
base=ref.base


def radial_predict(model,x,done):
    z=(x-model['mean'])/model['scale'];pending=~done
    v=np.zeros(done.shape);inside=np.zeros(done.shape,bool)
    tree=cKDTree(model['centers'])
    if pending.any():
        distance,index=tree.query(z[pending],k=1)
        activation=np.maximum(0.,1.-distance/model['radii'][index])
        v[pending]=model['default']+(model['targets'][index]-model['default'])*activation
        inside[pending]=activation>0
    return v,inside


def run(out):
    # Fail rather than touch an incomplete live set of neural candidates.
    assert (out/'all_candidates_frozen.json').exists()
    plan=json.loads((out/'plan.json').read_text())
    train,_=base.load_train();x,q,b,d=base.product_data(train,plan['spec'])
    mean=x.mean((0,1));scale=np.maximum(x.std((0,1)),.05)
    graph,witness=base.graph_feasibility(x,q,b);assert graph['feasible']
    centers=((x-mean)/scale)[b];targets=1.5*witness[b]+2*base.ETA
    tree=cKDTree(centers);distance,index=tree.query(centers,k=2)
    radii=.25*distance[:,1];assert (radii>0).all()
    model=dict(centers=centers,targets=targets,radii=radii,mean=mean,scale=scale,default=float(targets.max()))
    file=out/'radial_interpolation.npz';assert not file.exists();np.savez_compressed(file,**model)
    base.write(out/'radial_frozen.json',dict(frozen=base.stamp(),sha256=base.digest(file),code_sha256=base.digest(__file__),
        uses='TRAIN ONLY; diagnostic, fixed compact radial basis centers',
        centers=len(centers),radius_fraction=.25,target_rule='1.5*sampled graph rank+2eta',default=model['default'],
        validation_role='After fitting only; not calibration/test',warning='May memorize train; outside all training balls V is constant'))
    val=[]
    for r in json.loads((ref.SOURCE/'internal_collection.json').read_text())['records']:
        if not r['eligible']:continue
        p=ref.SOURCE/'internal_raw'/r['file'];assert base.digest(p)==r['sha256']
        with np.load(p,allow_pickle=False) as a:val.append({k:a[k].copy() for k in a.files})
    results={};region=np.load(ref.SOURCE/'region.npz')
    for name,paths in [('train',train),('dev',val)]:
        x,q,b,d=base.product_data(paths,plan['spec']);v,inside=radial_predict(model,x,d)
        mask=np.linalg.norm((np.stack([base.physical(a) for a in paths])-region['mean'])/region['scale'],axis=-1)<=float(region['radius'])+1e-10
        stats=base.statistics(v,b,d,region=mask)
        stats.update(pending_states=int(b.sum()),pending_states_inside_training_balls=int((inside&b).sum()))
        results[name]=stats
    base.write(out/'radial_diagnostic.json',results)
    print('RADIAL',json.dumps({k:{f:v[f] for f in ['p1p2_paths','p1_violations','p2_violations','pending_states','pending_states_inside_training_balls']} for k,v in results.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);run(p.parse_args().output.resolve())
