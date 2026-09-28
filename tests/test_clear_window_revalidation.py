import unittest
import numpy as np
from experiments.l2_clear_window_revalidation import full_events,cp,COUNTS,SEEDS
from experiments.l2_clear_window_v import monitor


class RevalidationTests(unittest.TestCase):
    def test_full_gate_and(self):
        done=np.array([[False,False,True]])
        self.assertTrue(full_events(np.array([[.045,.025,.005]]),done)[0])
        self.assertFalse(full_events(np.array([[.035,.030,.005]]),done)[0])
        self.assertFalse(full_events(np.array([[.045,.025,.015]]),done)[0])
        self.assertFalse(full_events(np.array([[.045,.025,.005]]),np.zeros_like(done))[0])
        self.assertFalse(full_events(np.array([[.005,.025,.005]]),done)[0])

    def test_window_endpoint_inclusive_and_no_t0(self):
        aps=np.ones((3,65,3));aps[:,:,1]=-1
        aps[0,1:49,1]=1;aps[1,17:65,1]=1;aps[2,0:48,1]=1
        run,done=monitor(aps)
        np.testing.assert_array_equal(done[:,-1],[True,True,False])
        self.assertEqual(done[0].argmax(),48);self.assertEqual(done[1].argmax(),64)

    def test_fixed_counts_seeds_and_cp(self):
        self.assertEqual(COUNTS,{'calibration':500,'test':1000})
        self.assertGreater(SEEDS['test']-SEEDS['calibration'],2*5*1000)
        self.assertAlmostEqual(cp(1000,1000),.05**.001,places=12)
        self.assertEqual(cp(0,500),0.)
        self.assertLess(cp(490,500),490/500)


if __name__=='__main__':unittest.main()
