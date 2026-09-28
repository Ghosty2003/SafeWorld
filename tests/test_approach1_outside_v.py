import unittest
import numpy as np
from experiments.l2_approach1_outside_v import conditional_labels, summarize, selection_key


class ApproachOutsideTests(unittest.TestCase):
    def test_filter_initial_only_and_keep_nonarrival(self):
        aps=np.zeros((3,3,3))
        aps[:,:,0]=[[.5,1.2,1.3],[.8,.6,.9],[.8,.9,1.0]]
        keep,d=conditional_labels(aps)
        np.testing.assert_array_equal(keep,[False,True,True])
        np.testing.assert_array_equal(d,[[False,True,True],[False,False,False]])

    def test_boundary_is_outside_and_final_arrival_counts(self):
        aps=np.zeros((1,3,3)); aps[0,:,0]=[1.0-.3,1.0-.3,.69]
        keep,d=conditional_labels(aps)
        self.assertTrue(keep[0])
        np.testing.assert_array_equal(d,[[False,False,True]])

    def test_goal_coverage_and_endpoint_are_distinct(self):
        d=np.array([[False,False,True],[False,False,False]])
        row=summarize(np.array([[.08,.05,.02],[.08,.05,.02]]),d)
        self.assertEqual(row['p1p2_paths']['successes'],2)
        self.assertEqual(row['p1p2_given_goal']['successes'],1)
        self.assertEqual(row['p1p2_given_goal']['trials'],1)
        self.assertEqual(row['finite_goal_and_certificate']['successes'],0)
        self.assertNotIn('cp_lower',row['p1p2_given_goal'])

    def test_selection_prefers_successful_path_coverage(self):
        d=np.array([[False,False,True],[False,False,False]])
        good=summarize(np.array([[.04,.02,.005],[.04,.04,.04]]),d)
        bad=summarize(np.array([[.04,.05,.005],[.08,.05,.02]]),d)
        self.assertGreater(selection_key(good),selection_key(bad))


if __name__=='__main__': unittest.main()
