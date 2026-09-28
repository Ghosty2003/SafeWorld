import unittest
from unittest.mock import patch
import numpy as np
from core.lppm.learned_sublevel import rate
from experiments.l2_approach1_holdout import take_eligible,verdict,audit


class ConditionalHoldoutTests(unittest.TestCase):
    def test_initial_condition_only_and_chronological_quota(self):
        aps=np.zeros((5,65,3))
        aps[:,:,0]=1.
        aps[:,0,0]=[.5,.7,.9,.3,1.2]
        aps[2,1:,0]=0.
        eligible,ix=take_eligible(aps,2)
        np.testing.assert_array_equal(eligible,[False,True,True,False,True])
        np.testing.assert_array_equal(ix,[1,2])
        aps[:,1:,0]=100.
        np.testing.assert_array_equal(take_eligible(aps,2)[1],ix)

    def test_threshold_is_lower_bound_not_empirical_rate(self):
        for k,expected in [(95,'NO_WARRANT'),(98,'NO_WARRANT'),(99,'FINITE_EVENT_THRESHOLD_MET')]:
            row=dict(finite_goal_and_certificate=rate(np.arange(100)<k,True))
            self.assertEqual(verdict(row)['result'],expected)

    def test_invalid_quota_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError): take_eligible(np.zeros((2,65,3)),0)
        with self.assertRaises(ValueError): take_eligible(np.full((2,65,3),np.nan),1)

    def test_primary_event_requires_goal_not_only_progress_and_low_endpoint(self):
        aps=np.ones((1,3,3))
        with patch('experiments.l2_approach1_holdout.base.value_features',return_value=np.zeros((1,3,577))):
            with patch('experiments.l2_approach1_holdout.predict',return_value=np.array([[.04,.02,.005]])):
                row=audit({},dict(aps=aps))
        self.assertEqual(row['candidate_certificate_event']['successes'],1)
        self.assertEqual(row['finite_goal_and_certificate']['successes'],0)
        self.assertEqual(verdict(row)['result'],'NO_WARRANT')


if __name__=='__main__': unittest.main()
