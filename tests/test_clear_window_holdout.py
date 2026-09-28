import unittest
from unittest.mock import patch
import numpy as np

from core.lppm.learned_sublevel import rate
from experiments.l2_clear_window_holdout import audit, verdict, EVENT
from experiments.l2_clear_window_v import SPEC


class ClearWindowHoldoutTests(unittest.TestCase):
    def test_probability_threshold_and_region_gate(self):
        for k, expected in ((98, 'NO_WARRANT'), (99, 'FINITE_EVENT_THRESHOLD_MET')):
            row = {EVENT: rate(np.arange(100) < k, True),
                   'sampled_region_status': 'NO_VIOLATION_ON_OBSERVED_SOURCES'}
            self.assertEqual(verdict(row)['result'], expected)
            row['sampled_region_status'] = 'COUNTEREVIDENCE'
            self.assertEqual(verdict(row)['result'], 'NO_WARRANT')

    def test_completed_window_vs_goal_and_early_sublevel(self):
        aps = np.ones((2, 65, 3))
        aps[1, :, 1] = -1
        # Both V sequences satisfy finite P1/P2 and endpoint sublevel.
        # Only the first trajectory satisfies the requested window property.
        values = np.tile(.005 + .02*np.arange(64, -1, -1), (2, 1))
        with patch('experiments.l2_clear_window_holdout.base.value_features',
                   return_value=np.zeros((2, 65, 577))):
            with patch('experiments.l2_clear_window_holdout.predict_family', return_value=values):
                row = audit(dict(spec=SPEC, eta=.01), dict(aps=aps))
        self.assertEqual(row['completion_reached']['successes'], 1)
        self.assertEqual(row['candidate_certificate_event']['successes'], 2)
        self.assertEqual(row[EVENT]['successes'], 1)
        self.assertEqual(row['candidate_event_without_completion'], 1)
        self.assertEqual(row['sampled_region_status'], 'COUNTEREVIDENCE')

    def test_wrong_initial_scope_and_model_rejected(self):
        with self.assertRaises(ValueError):
            audit(dict(spec='different', eta=.01), {})
        with self.assertRaises(ValueError):
            audit(dict(spec=SPEC, eta=.01), dict(aps=np.zeros((1,65,3))))


if __name__ == '__main__': unittest.main()
