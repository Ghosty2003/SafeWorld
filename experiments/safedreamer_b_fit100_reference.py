"""Evaluate prior small-fit B on the new internal split; never select/retrain it."""
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.safedreamer_b_fit100_train import load_paths,eval_paths
from experiments.safedreamer_finite_recurrence_pilot import BudgetNet
from experiments.l3_safedreamer_clear_pipeline import digest,write,stamp


def run(source,out):
    torch.set_num_threads(2)
    # Reference may run only after the new model choice has already been saved.
    selected=json.loads((out/'selected_candidate.json').read_text())
    plan=json.loads((source/'plan.json').read_text());coll=json.loads((source/'collection.json').read_text())
    olddir=ROOT/'artifacts/safedreamer_recurrence_screening_v1'
    old=json.loads((olddir/'frozen_primary.json').read_text())
    assert old['definition']==plan['detector']
    assert digest(old['model'])==old['model_sha256']
    weight=torch.load(old['model'],map_location='cpu',weights_only=True);state=weight['state']
    net=BudgetNet(state['mean'],state['scale']);net.load_state_dict(state);net.eval()
    current=torch.load(out/selected['file'],map_location='cpu',weights_only=True)
    region=dict(mean=current['state']['mean'].numpy(),scale=current['state']['scale'].numpy(),radius=current['region_radius'])
    validation=load_paths(source,coll['records'],'internal_validation',plan)
    with torch.no_grad():predictions=[net(torch.tensor(p['x'])).numpy().astype(float) for p in validation]
    metrics,rows,segs=eval_paths(validation,predictions,old['delta_fit_preview_not_for_formal_use'],region,plan,'prior_small_fit')
    write(out/'prior_model_reference.json',dict(time=stamp(),old_model_sha256=old['model_sha256'],
        scope='Diagnostic after selection; not in candidate ranking; current shared region and2.4 cap',
        metrics=metrics,rows=rows,segments=segs,selected_model_unchanged=selected['sha256']))
    print('PRIOR SMALL-FIT REFERENCE',metrics)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.source.resolve(),a.output.resolve())
