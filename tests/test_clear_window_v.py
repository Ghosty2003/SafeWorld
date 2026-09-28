import unittest
import numpy as np
from experiments.l2_clear_window_v import monitor


class ClearWindowTests(unittest.TestCase):
    def test_completion_timing_and_absorption(self):
        a = np.zeros((3, 65, 3))
        a[..., 1] = -1
        a[0, 1:49, 1] = 1
        a[1, 17:65, 1] = 1
        a[2, 18:65, 1] = 1
        run, done = monitor(a)
        self.assertFalse(done[0, 47])
        self.assertTrue(done[0, 48:].all())
        self.assertFalse(done[1, 63])
        self.assertTrue(done[1, 64])
        self.assertFalse(done[2].any())
        self.assertTrue((run[0, 48:] == 48).all())

    def test_t0_does_not_count_and_hazard_resets(self):
        a = np.zeros((1, 65, 3))
        a[..., 1] = 1
        a[0, 20, 1] = -1
        run, done = monitor(a)
        self.assertEqual(run[0, 0], 0)
        self.assertEqual(run[0, 19], 19)
        self.assertEqual(run[0, 20], 0)
        self.assertFalse(done.any())

    def test_causal_labels(self):
        a = np.zeros((1, 65, 3))
        b = a.copy(); b[:, 25:, 1] = -1
        ra, da = monitor(a); rb, db = monitor(b)
        np.testing.assert_array_equal(ra[:, :25], rb[:, :25])
        np.testing.assert_array_equal(da[:, :25], db[:, :25])


if __name__ == '__main__': unittest.main()
