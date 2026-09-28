import unittest
import numpy as np
from experiments import finite_recurrence_motion_dev as e


class Tests(unittest.TestCase):
    def path(self, signal):
        mon=e.detector.monitor(signal,.2,.8)
        full=np.column_stack([signal,mon['armed']]).astype(np.float32)
        return dict(record={'fit_index':0},signal=signal,monitor=mon,full=full,x=full)

    def evaluate(self,signal,pred,delta=0):
        path=self.path(signal)
        region=dict(mean=np.zeros(2),scale=np.ones(2),radius=10.)
        return e.evaluate([path],[np.full(301,pred)],dict(low=.2,high=.8),region,delta)

    def test_high_forever_not_recurring(self):
        stat,rows=self.evaluate(np.ones(301),2.4,100.)
        self.assertEqual(stat['count_pass'],0)
        self.assertEqual(stat['certificate'],0)
        self.assertIn('DELTA_MARGIN_FAIL',rows[0]['failure_reasons'])

    def test_repeated_events_require_new_low(self):
        signal=np.tile([.1,1.],151)[:301]
        stat,rows=self.evaluate(signal,.03)
        self.assertEqual(rows[0]['count'],150)
        self.assertEqual(stat['certificate'],1)

    def test_arrival_checked_before_reset(self):
        signal=np.tile([.1,1.],151)[:301]
        stat,rows=self.evaluate(signal,.005)
        self.assertEqual(stat['certificate'],0)
        self.assertEqual(rows[0]['first_budget_failure'],1)

    def test_tail_is_not_masked(self):
        signal=np.ones(301);signal[[0,2]]=.1
        stat,rows=self.evaluate(signal,.1)
        self.assertEqual(stat['count_pass'],1)
        self.assertEqual(stat['certificate'],0)
        self.assertGreater(rows[0]['first_budget_failure'],3)


if __name__=='__main__':unittest.main()
