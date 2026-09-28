import numpy as np
import torch
import unittest
from core.lbsm.operator_model import AnchorSuccessorBatch
from experiments.l3_expected_drift_ablation import expected_loss, required_mask, core_check, evaluate


def test_expected_loss_allows_one_upward_successor():
    values=torch.tensor([[.9,.1]],requires_grad=True)
    loss=expected_loss(values,torch.tensor([.6]),torch.tensor([True]),.05)
    assert loss.item()==0
    assert torch.relu(values-.6+.05).mean().item()>0
    loss.backward()
    assert torch.isfinite(values.grad).all()


def test_h2_candidate_mask_is_not_h1_mask():
    x=torch.tensor([[0.,0.],[0.,1.]])
    assert required_mask(x,'U','H1').tolist()==[True,True]
    assert required_mask(x,'U','H2_candidate').tolist()==[True,False]
    assert required_mask(x,'W','H2_candidate').tolist()==[True,False]


def test_sampled_no_exit_never_proves_invariance():
    b=AnchorSuccessorBatch(np.array([0.,1.],np.float32),
                           np.array([[0.,1.],[0.,1.]],np.float32),'a','test')
    r=core_check([b])
    assert r['sample_status']=='SAMPLED_NO_EXIT'
    assert r['support_wide_invariance']=='NOT_ESTABLISHED'
    b.successors[1,-1]=0
    r=core_check([b])
    assert r['exit_successors']==1
    assert r['sample_status']=='COUNTEREXAMPLE'


def test_empty_core_is_unobserved_not_pass():
    r=core_check([])
    assert r['sample_status']=='UNOBSERVED'
    assert r['support_wide_invariance']=='NOT_ESTABLISHED'


def test_empty_training_domain_zero_loss_has_gradient():
    values=torch.tensor([[.9,.1]],requires_grad=True)
    loss=expected_loss(values,torch.tensor([.6]),torch.tensor([False]),.05)
    assert loss.item()==0
    loss.backward()
    assert values.grad is not None


def test_h2_report_keeps_h1_denominator_and_fixed_thresholds():
    class Coordinate(torch.nn.Module):
        def forward(self,x): return x[...,0]
    batches=[AnchorSuccessorBatch(np.array([.5,q],np.float32),
             np.array([[v,q],[v,q]],np.float32),str(q),'test')
             for q,v in [(0.,.4),(1.,.6)]]
    paths=np.array([[[.5,0.],[.5,1.]]],np.float32)
    r=evaluate(Coordinate(),'U','H2_candidate',batches,paths,.03,.137)
    assert r['candidate_domain']['required']==1
    assert r['candidate_domain']['raw_pass']==1
    assert r['H1_all_anchors']==dict(required=2,raw_pass=1)
    assert r['verification_threshold']==0.
    assert r['global_I_subset_C']=='NOT_ESTABLISHED'
    assert r['confidence'] is None


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(f) for name,f in globals().items()
                              if name.startswith('test_') and callable(f))


if __name__=='__main__':
    unittest.main()
