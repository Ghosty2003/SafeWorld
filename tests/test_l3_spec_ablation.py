import unittest
import numpy as np
from experiments.l3_spec_ablation import counters,next_counter,finite_behavior,relabel


class SpecAblationTests(unittest.TestCase):
    def test_window_lengths_reset_without_absorption(self):
        for length in (1,16,24,32,48):
            c=counters([True]*length+[False,True],length)
            self.assertEqual(c[length-1],length)
            self.assertEqual(c[length],0)
            self.assertEqual(c[length+1],1)

    def test_successor_counter(self):
        np.testing.assert_array_equal(next_counter(15,np.array([True,False]),16),[1.,0.])
        np.testing.assert_array_equal(next_counter(16,np.array([True,False]),16),[1.,0.])

    def test_overlap_not_double_nonoverlap(self):
        self.assertFalse(finite_behavior([True]*60,48)['two_nonoverlapping_windows'])
        r=finite_behavior([True]*96,48)
        self.assertTrue(r['two_nonoverlapping_windows'])
        self.assertFalse(r['return_after_exit'])

    def test_simple_predicate_exit_return(self):
        r=finite_behavior([False,True,False,True],1)
        self.assertTrue(r['return_after_exit'])
        self.assertEqual(r['return_count'],1)

    def test_relabel_preserves_dynamics(self):
        base=np.zeros((2,578),np.float32)
        paths=[dict(base=base,monitors={'test':np.array([15,16])},record={'seed':123})]
        obs=np.ones((2,29),np.float32); obs[1,9:25]=0
        y=np.ones((2,578),np.float32)*.25
        q=[dict(path_index=0,t=0,successors=y,decoded=obs)]
        p,b=relabel(paths,q,dict(id='test',window=16,threshold=0.))
        np.testing.assert_array_equal(b[0].successors[:,:-1],y[:,:-1])
        np.testing.assert_array_equal(b[0].successors[:,-1],[1,0])
        self.assertAlmostEqual(p[0,0,-1],15/16)
        self.assertTrue((y[:,-1]==.25).all())


if __name__=='__main__': unittest.main()
