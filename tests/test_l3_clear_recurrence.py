import unittest
import numpy as np
from experiments.l3_safedreamer_clear_recurrence import monitor


class ClearRecurrenceTests(unittest.TestCase):
    def test_exact_boundary(self):
        self.assertEqual(monitor(np.ones(47, dtype=bool))['completion_steps'], [])
        self.assertEqual(monitor(np.ones(48, dtype=bool))['completion_steps'], [47])

    def test_overlap_not_independent_repeat(self):
        r = monitor(np.ones(101, dtype=bool))
        self.assertEqual(r['overlapping_windows'], 54)
        self.assertEqual(r['nonoverlapping_windows'], 2)
        self.assertEqual(r['qualifying_safe_runs'], 1)
        self.assertTrue(r['window_wholly_after_halfway'])

    def test_hazard_resets_not_absorbing(self):
        p = np.ones(97, dtype=bool); p[48] = False
        r = monitor(p)
        self.assertEqual(r['completion_steps'], [47,96])
        self.assertEqual(r['counter'][48], 0)
        self.assertEqual(r['qualifying_safe_runs'], 2)

    def test_causal_future_does_not_change_prefix(self):
        p = np.ones(100, dtype=bool); p[53] = False
        self.assertEqual(monitor(p)['counter'][:51], monitor(p[:51])['counter'])

    def test_all_unsafe(self):
        r = monitor(np.zeros(101, dtype=bool))
        self.assertEqual(r['overlapping_windows'], 0)
        self.assertEqual(r['longest_safe_run'], 0)


if __name__ == '__main__': unittest.main()
