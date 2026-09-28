import unittest
import numpy as np
from experiments.goal03_policy100 import select, summarize


class Goal100Tests(unittest.TestCase):
    def test_paired_horizons_and_strict_boundary(self):
        a = np.ones((4,101,3))
        a[0,64,0] = -.01
        a[1,65,0] = -.01
        a[2,100,0] = -.01
        a[3,100,0] = 0  # exactly .30 is not < .30
        r = summarize(a)
        self.assertEqual(r['success_by64'], 1)
        self.assertEqual(r['success_by100'], 3)
        self.assertEqual(r['newly_successful_65_to100'], 2)
        self.assertEqual(r['first_arrival_steps'], [64,65,100,-1])

    def test_selection_uses_initial_state_only(self):
        a = np.ones((3,101,3)); a[0,0,0] = .5
        np.testing.assert_array_equal(select(a,1)[1], [1])
        a[1,1:,0] = 100; a[2,1:,0] = -.1
        np.testing.assert_array_equal(select(a,1)[1], [1])


if __name__ == '__main__': unittest.main()
