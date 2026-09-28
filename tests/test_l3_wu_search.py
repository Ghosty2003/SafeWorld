import unittest
import numpy as np
import torch
from experiments.l3_wu_search import Candidate, return_targets, pair_score


class WUSearchTests(unittest.TestCase):
    def test_return_targets_censoring(self):
        np.testing.assert_array_equal(return_targets([46,47,48,48,0])[:4],[2,1,0,0])
        self.assertTrue(np.isnan(return_targets([46,47,48,48,0])[-1]))
        self.assertTrue(np.isnan(return_targets([0,1,2])).all())

    def test_every_family_bounded(self):
        x=torch.randn(8,10); x[:,-1]=torch.arange(8)/48
        for family in ('plain','wide','mode','progress','sublevel','quadratic'):
            net=Candidate(np.zeros(10),np.ones(10),family)
            y=net(x)
            self.assertEqual(y.shape,(8,))
            self.assertTrue(((y>=0)&(y<=1)).all())
            y.sum().backward()

    def test_acceptance_does_not_bypass_u(self):
        w=dict(required_anchors=1,required=[True,False],anchor_pass=[True,True],anchor_rate=1.)
        u=dict(initial_in_C=2,n_initial=2,all_path_state_in_C_rate=1.,value_max=.4,value_min=.1,
               anchor_in_C=[True,True],anchor_pass=[True,False],anchor_rate=.5)
        self.assertEqual(pair_score(w,u),(True,.5,.5,.75))
        u['value_max']=u['value_min']
        self.assertFalse(pair_score(w,u)[0])


if __name__=='__main__': unittest.main()
