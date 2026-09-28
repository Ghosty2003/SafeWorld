import unittest
import numpy as np
from experiments.l3_safedreamer_dual_region_screen import cycles, summarize


class DualRegionTests(unittest.TestCase):
    def test_shared_endpoint_and_neutral_states(self):
        self.assertEqual(cycles([1.2,.4,.8,1.2,.7,.4,1.2,.3]), [[1,3,5],[5,6,7]])

    def test_strict_boundaries_no_false_cycles(self):
        self.assertEqual(cycles([.5,1.,.5,1.]), [])
        self.assertEqual(cycles([.4,.4,.4,.4]), [])
        self.assertEqual(cycles([1.2,1.2,.4,.4]), [])

    def test_late_completion_not_wholly_late(self):
        r = summarize(np.array([[.4,.7,1.2,.8,.4]]))
        self.assertEqual(r['at_least_one_ABA'], 1)
        self.assertEqual(r['paths_with_ABA_finishing_after_halfway'], 1)
        self.assertEqual(r['paths_with_ABA_wholly_after_halfway'], 0)

    def test_invalid_distance_rejected(self):
        for row in ([1., float('nan')], [-.1,1.]):
            with self.assertRaises(ValueError): cycles(row)


if __name__ == '__main__': unittest.main()
