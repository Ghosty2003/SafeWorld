import unittest
import numpy as np
from experiments.l3_safedreamer_screen import summarize, screen


class RecurrenceScreenTests(unittest.TestCase):
    def test_runs_not_repeated_states(self):
        p = np.array([[1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=bool)
        r = summarize(p)
        self.assertEqual(r['ever_accepting'], 2)
        self.assertEqual(r['always_accepting'], 1)
        self.assertEqual(r['two_or_more_accepting_runs'], 1)
        self.assertEqual(r['false_to_true_entry_counts'], [0, 2, 0])
        self.assertEqual(r['unfinished_nonaccepting_suffix_lengths'], [0, 1, 5])

    def test_strict_goal_and_hazard_boundary(self):
        d = np.zeros((1, 3, 29), dtype=float)
        d[..., 7] = .3
        d[..., 9:25:2] = .2
        r = screen(d)
        self.assertEqual(r['GF(decoded_goal_distance<0.3)']['ever_accepting'], 0)
        self.assertEqual(r['GF(decoded_goal_distance<1.0)']['always_accepting'], 1)
        self.assertEqual(r['GF(decoded_hazard_margin>=0)']['always_accepting'], 1)


if __name__ == '__main__':
    unittest.main()
