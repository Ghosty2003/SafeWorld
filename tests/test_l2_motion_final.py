import unittest
import numpy as np
import torch
from experiments import l2_motion_final as f


class Tests(unittest.TestCase):
    def test_cp(self):
        self.assertEqual(f.cp(0,1000),0.)
        self.assertAlmostEqual(f.cp(1000,1000),.05**.001,12)

    def test_splits_disjoint(self):
        a=set(range(f.SEEDS['calibration'],f.SEEDS['calibration']+10*f.COUNTS['calibration']))
        b=set(range(f.SEEDS['test'],f.SEEDS['test']+10*f.COUNTS['test']))
        self.assertFalse(a&b)

    def test_frozen_bundle_reproduces_internal(self):
        torch.set_num_threads(2)
        bundle=torch.load(f.SOURCE/'selected_bundle.pt',map_location='cpu')
        paths,_=f.w.exp.load_internal()
        region=dict(np.load(f.w.exp.SOURCE/'region.npz'))
        rows,v,stat=f.evaluate(bundle,region,paths)
        saved=np.load(f.SOURCE/'selected_values.npz')['internal']
        np.testing.assert_array_equal(v,saved)
        self.assertEqual(sum(r['certificate'] for r in rows),19)
        self.assertEqual(stat['p1_violations'],7)
        self.assertEqual(stat['p2_violations'],8)
        # Production evaluates one path at a time; verify identical gates.
        for a,expected in zip(paths,rows):
            got,_,_=f.evaluate(bundle,region,[a])
            self.assertEqual(got[0],expected)


if __name__=='__main__':unittest.main()
