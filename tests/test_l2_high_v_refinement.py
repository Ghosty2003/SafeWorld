import unittest
import numpy as np
import torch
from experiments import l2_high_v_refinement as r


class TestHighRefinement(unittest.TestCase):
    def test_warm_start_preserves_pointwise_function(self):
        torch.manual_seed(17)
        old=r.base.Value(7,256).double();new=r.ProgressValue(7,r.CONFIGS[0]).double()
        new.load_state_dict(old.state_dict())
        x=torch.randn(3,4,7,dtype=torch.float64);done=torch.rand(3,4)>.5
        np.testing.assert_array_equal(old(x,(~done).double(),done).detach().numpy(),new(x,done).detach().numpy())

    def test_nonnegative_and_absorbing(self):
        for cfg in r.CONFIGS:
            net=r.ProgressValue(5,cfg)
            x=torch.randn(20,5);done=torch.arange(20)%2==0
            v=net(x,done)
            self.assertTrue(bool((v[done]==0).all()))
            self.assertTrue(bool((v[~done]>=r.base.ETA).all()))

    def test_feature_omission_only_stochastic_block(self):
        idx=r.feature_indices(581,r.CONFIGS[3])
        self.assertEqual(len(idx),325)
        self.assertFalse(bool(np.isin(np.arange(256,512),idx).any()))
        self.assertTrue(bool(np.isin(np.arange(512,581),idx).all()))

    def test_no_clock_or_history_dependence(self):
        net=r.ProgressValue(5,r.CONFIGS[1]).double()
        x=torch.randn(3,5,dtype=torch.float64);done=torch.zeros(3,dtype=torch.bool)
        a=net(x,done).detach().numpy();b=net(x[[2,0,1]],done).detach().numpy()
        np.testing.assert_allclose(a[[2,0,1]],b,atol=1e-12,rtol=1e-12)

    def test_unchanged_state_pending_cannot_cheat_p2(self):
        net=r.ProgressValue(5,r.CONFIGS[1]).double()
        x=torch.ones(1,2,5,dtype=torch.float64);done=torch.zeros(1,2,dtype=torch.bool)
        v=net(x,done).detach().numpy()
        s=r.base.statistics(v,np.ones((1,2),bool),np.zeros((1,2),bool))
        self.assertEqual(s['p2_violations'],1)


if __name__=='__main__':unittest.main()
