import unittest
import numpy as np
import torch
from experiments import l2_motion_v_refine as m


class TestMotionValue(unittest.TestCase):
    def test_bounds_all_families(self):
        for cfg in m.CONFIGS:
            net=m.Value(4,cfg)
            v=net(torch.zeros(3,4),torch.tensor([False,False,True]))
            self.assertTrue(bool((v[:2]>=.01).all()))
            self.assertEqual(float(v[-1]),0.)

    def test_prediction_float64_matches_direct(self):
        cfg=m.CONFIGS[0];net=m.Value(4,cfg).double()
        model=dict(config=cfg,mean=np.zeros(4),scale=np.ones(4),weights=net.state_dict())
        x=np.zeros((1,3,4));d=np.array([[False,False,True]])
        v=m.predict(model,x,d)
        self.assertEqual(v.dtype,np.float64)
        np.testing.assert_array_equal(v,net(torch.tensor(x),torch.tensor(d)).detach().numpy())

    def test_long_burst_requires_low_first(self):
        spec=dict(stages=[dict(signal='speed',op='le',threshold=.2),dict(signal='speed',op='ge',threshold=.7)],
                  accepting=2,reject=None,safety_until_completion=False)
        sig=dict(speed=np.array([.3,.8,.1,.4,.8]),clearance=np.ones(5))
        q,_,d=m.screen.monitor(sig,spec)
        np.testing.assert_array_equal(q,[0,0,1,1,2])
        self.assertEqual(m.manual_completion(sig,spec),4)


if __name__=='__main__':unittest.main()
