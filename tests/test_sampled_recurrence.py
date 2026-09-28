import unittest
import numpy as np
import torch
from core.lbsm.operator_model import AnchorSuccessorBatch
from core.lbsm.sampled_recurrence import (NormalizedBoundedCertificate,fit_h1,
                                        drift_diagnostics,formal_status)


class FirstCoordinate(torch.nn.Module):
    B=1.
    def forward(self,x): return x[:,0]


def batch(x,y,k=10000):
    return AnchorSuccessorBatch(np.array(x,dtype=np.float32),
                               np.tile(np.array(y,dtype=np.float32),(k,1)))


class H1Tests(unittest.TestCase):
    def test_accepting_sources_still_require_U(self):
        stats=drift_diagnostics(FirstCoordinate(),FirstCoordinate(),
              [batch([.2,1],[.7,1])],fit_batches=[batch([.1,0],[0,0])])
        self.assertEqual(stats['n_W_required'],0)
        self.assertEqual(stats['n_U_required'],1)
        self.assertEqual(stats['n_U_empirical_pass'],0)

    def test_in_sample_is_not_concentration_evidence(self):
        b=[batch([.5,0],[.1,0])]
        stats=drift_diagnostics(FirstCoordinate(),FirstCoordinate(),b,fit_batches=b,independent=False)
        self.assertIsNone(stats['pointwise_confidence'])
        self.assertIsNone(stats['n_W_pointwise_ucb_pass'])

    def test_leakage_rejected(self):
        b=[batch([.5,0],[.1,0])]
        with self.assertRaises(ValueError):
            drift_diagnostics(FirstCoordinate(),FirstCoordinate(),b,fit_batches=b)

    def test_all_sampled_checks_pass_still_not_global_warrant(self):
        stats=drift_diagnostics(FirstCoordinate(),FirstCoordinate(),
              [batch([.5,0],[.1,0])],fit_batches=[batch([.2,0],[.1,0])])
        self.assertEqual(stats['n_W_pointwise_ucb_pass'],1)
        self.assertEqual(stats['n_U_pointwise_ucb_pass'],1)
        report=formal_status(stats)
        self.assertEqual(report['verdict'],'ABSTAIN')
        self.assertIsNone(report['probability_lower_bound'])

    def test_multiple_comparison_budget(self):
        stats=drift_diagnostics(FirstCoordinate(),FirstCoordinate(),
              [batch([.5,0],[.1,0]),batch([.6,0],[.1,0])],fit_batches=[])
        self.assertAlmostEqual(stats['per_function_per_anchor_delta'],.05/4)

    def test_structure_and_fit(self):
        bs=[batch([.5,0,0],[.4,1,1],k=4),batch([.6,0,1],[.4,1,1],k=4)]
        core=np.array([[.5,0,0],[.6,0,1],[.4,1,1]],dtype=np.float32)
        W,U,history=fit_h1(bs,core,epochs=3)
        for net in (W,U):
            v=net(torch.tensor([[1000.,-20.,1.],[-1000.,1.,0.]]))
            self.assertTrue(bool(((v>=0)&(v<=1)).all()))
        self.assertEqual(history[-1]['epoch'],3)


if __name__=='__main__': unittest.main()
