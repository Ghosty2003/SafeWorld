import unittest

import numpy as np

from experiments.l2_spec_screen64 import counts, screen, window_event


class SpecScreenTests(unittest.TestCase):
    def test_exact_window_endpoints(self):
        p = np.zeros((2, 65), dtype=bool)
        p[0, 17:65] = True
        p[1, 18:65] = True
        np.testing.assert_array_equal(window_event(p, 1, 17, 48), [True, False])
        with self.assertRaises(ValueError):
            window_event(p, 1, 18, 48)

    def test_initial_exclusion_is_not_future_filter(self):
        a = np.ones((3, 65, 3))
        a[..., 0] = 1.2 - .3
        a[0, 0, 0] = .5 - .3
        a[1, 64, 0] = .9 - .3
        rows = screen(a)
        self.assertEqual(rows['F[0,64](d<1); all']['successes'], 2)
        self.assertEqual(rows['F[0,64](d<1); initial_outside']['total'], 2)
        self.assertEqual(rows['F[0,64](d<1); initial_outside']['successes'], 1)

    def test_no_data_is_not_success(self):
        r = counts(np.array([True]), np.array([False]))
        self.assertEqual(r['total'], 0)
        self.assertIsNone(r['rate'])


if __name__ == '__main__':
    unittest.main()
