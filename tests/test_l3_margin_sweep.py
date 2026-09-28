import unittest
import numpy as np
import torch
from torch import nn
from core.lbsm.operator_model import AnchorSuccessorBatch
from experiments.l3_margin_sweep import distribution,diagnose,ranking


class Coordinate(nn.Module):
    def forward(self,x): return x[...,0]


class MarginTests(unittest.TestCase):
    def test_quantiles(self):
        q=distribution([-1.,0.,1.])
        self.assertEqual(q['median'],0.)
        self.assertEqual(q['min'],-1.)
        self.assertAlmostEqual(q['q10'],-.8)

    def test_verification_margin_stays_fixed(self):
        b=AnchorSuccessorBatch(np.array([.5,0.],np.float32),np.tile([.48,0.],(32,1)).astype(np.float32))
        paths=np.array([[[.5,0.],[.48,0.]]],np.float32)
        d=diagnose(Coordinate(),'W',[b],paths,.15,.137)
        self.assertEqual(d['verification_threshold'],-.01)
        self.assertEqual(d['raw_pass'],1)
        self.assertEqual(d['train_margin_pass'],0)
        self.assertEqual(d['fixed_slack_pass'],0)
        self.assertIsNone(d['confidence'])

    def test_u_includes_acceptance_and_outside_C(self):
        b=AnchorSuccessorBatch(np.array([.9,1.],np.float32),np.tile([.95,1.],(32,1)).astype(np.float32))
        paths=np.array([[[.9,1.],[.95,1.]]],np.float32)
        d=diagnose(Coordinate(),'U',[b],paths,.03,.137)
        self.assertEqual(d['required'],1)
        self.assertEqual(d['raw_pass'],0)
        self.assertEqual(d['below_ell_anchor_count'],0)
        self.assertFalse(ranking(dict(kind='U',development=d))[0])


if __name__=='__main__': unittest.main()
