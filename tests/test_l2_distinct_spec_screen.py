import unittest
import numpy as np
import torch
from experiments import l2_distinct_spec_screen as s


def task(stages,safe=False):
    return dict(stages=stages,accepting=len(stages),reject=len(stages)+1 if safe else None,
                safety_until_completion=safe)


class Tests(unittest.TestCase):
    def test_order_and_no_simultaneous_stages(self):
        t=task([dict(signal='speed',op='le',threshold=.2),dict(signal='speed',op='ge',threshold=.7)])
        sig=dict(clearance=np.ones(5),speed=np.array([1.,.1,.3,.8,.1]))
        q,b,d=s.monitor(sig,t)
        np.testing.assert_array_equal(q,[0,1,1,2,2])
        self.assertTrue(d[-1]);self.assertFalse(d[2])

    def test_causality(self):
        t=task([dict(signal='clearance',op='le',threshold=.4),dict(signal='clearance',op='ge',threshold=.8)])
        sig=dict(clearance=np.array([.9,.3,.6,.8,.1]))
        q,_,_=s.monitor(sig,t)
        for n in range(1,6):np.testing.assert_array_equal(s.monitor({k:v[:n] for k,v in sig.items()},t)[0],q[:n])

    def test_safety_reject_absorbing(self):
        t=task([dict(signal='clearance',op='le',threshold=.4),dict(signal='clearance',op='ge',threshold=.8)],True)
        q,_,d=s.monitor(dict(clearance=np.array([.3,.1,.9])),t)
        np.testing.assert_array_equal(q,[1,3,3]);self.assertFalse(d.any())
        q,_,d=s.monitor(dict(clearance=np.array([.3,.9,.1])),t)
        np.testing.assert_array_equal(q,[1,2,2]);self.assertTrue(d[-1])

    def test_displacement_is_relative_and_causal(self):
        decoded=np.zeros((3,29));decoded[:,27]=[3.,4.,3.]
        a=dict(decoded=decoded)
        np.testing.assert_array_equal(s.signals(a)['displacement'],[0,1,0])

    def test_nonnegative_bad_and_zero_accepting(self):
        for family in ['linear','elu128','signal_progress']:
            net=s.Value(4,family);v=net(torch.zeros(3,4),torch.tensor([False,False,True]),torch.tensor([1.,2.,0.]))
            self.assertTrue(bool((v[:2]>=.01).all()));self.assertEqual(float(v[-1]),0.)

    def test_scale_does_not_repair_positive_drift(self):
        v=np.array([[.03,.04,0.]])
        b=np.array([[True,True,False]]);d=~b
        self.assertEqual(s.base.statistics(v,b,d)['p1_violations'],1)
        self.assertEqual(s.base.statistics(4*v,b,d)['p1_violations'],1)


if __name__=='__main__':unittest.main()
