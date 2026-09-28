import unittest
import numpy as np
import torch
from torch import nn
from experiments.l3_wu_data_ablation import anchor_times,scan
from core.lbsm.operator_model import AnchorSuccessorBatch


class Coordinate(nn.Module):
    def forward(self,x): return x[:,0]


class DataAblationTests(unittest.TestCase):
    def test_stratified_anchors(self):
        c=list(range(1,49))+[48]*145
        t=anchor_times(c,4)
        self.assertEqual(len(t),2)
        self.assertLess(c[t[0]],48)
        self.assertEqual(c[t[1]],48)
        self.assertTrue(all(1<=x<192 for x in t))
        self.assertEqual(t,anchor_times(c,4))
        self.assertEqual(len(anchor_times([0]*193,4)),1)

    def test_nested_sampling_correction(self):
        b=AnchorSuccessorBatch(np.array([.8,0],np.float32),np.tile([.5,0],(256,1)).astype(np.float32))
        rows=scan(Coordinate(),Coordinate(),[b],[8,256],1)
        self.assertEqual(rows[0]['W']['raw_pass'],1)
        self.assertEqual(rows[0]['W']['sampling_corrected_pass'],0)
        self.assertEqual(rows[1]['W']['sampling_corrected_pass'],1)
        self.assertLess(rows[1]['sampling_radius'],rows[0]['sampling_radius'])
        self.assertEqual(rows[0]['delta_per_check'],.05/4)
        self.assertIsNone(rows[0]['cell_correction'])

    def test_accepting_still_checks_u(self):
        b=AnchorSuccessorBatch(np.array([.5,1],np.float32),np.tile([.6,1],(8,1)).astype(np.float32))
        row=scan(Coordinate(),Coordinate(),[b],[8],4)[0]
        self.assertEqual(row['W']['required_in_C'],0)
        self.assertEqual(row['U']['required_in_C'],1)
        self.assertEqual(row['U']['raw_pass'],0)
        self.assertEqual(row['delta_per_check'],.05/8)

    def test_no_silent_short_query(self):
        b=AnchorSuccessorBatch(np.array([.5,0],np.float32),np.tile([.4,0],(8,1)).astype(np.float32))
        with self.assertRaises(ValueError): scan(Coordinate(),Coordinate(),[b],[16],1)


if __name__=='__main__': unittest.main()
