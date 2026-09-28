import copy
import unittest
import numpy as np
from experiments.safedreamer_high_clearance_pilot import monitor, reference, source_consistent, evaluate, product, inside_region
from core.finite_recurrence import combine_gates


class HighClearanceTests(unittest.TestCase):
    def test_always_high_never_emits(self):
        m = monitor(np.ones(301), .2, .5)
        self.assertEqual(m['event_count'], 0)
        self.assertEqual(m['low_entry_timestamps'], [])

    def test_initial_low_arms_and_inclusive_boundaries(self):
        m = monitor([.2, .3, .5, .6, .2, .5], .2, .5)
        self.assertEqual(m['event_times'], [2, 5])
        self.assertEqual(m['low_entry_timestamps'], [0, 4])
        self.assertEqual(m['armed'], [1, 1, 0, 0, 1, 0])

    def test_hysteresis_and_pending_tail(self):
        m = monitor([.6, .2, .21, .19, .49, .5, .49, .5, .2], .2, .5)
        self.assertEqual(m['event_times'], [5])
        self.assertEqual(m['low_entry_timestamps'], [1, 8])
        self.assertEqual(m['pending_low_entry'], 8)
        self.assertEqual(m['low_excursion_timestamps'], [1, 3, 8])

    def test_causal_prefix_and_independent_reference(self):
        rng = np.random.default_rng(310)
        for _ in range(20):
            d = rng.random(301)
            full = monitor(d, .2, .7)
            self.assertTrue(source_consistent(d, full, .2, .7))
            for stop in [2, 10, 100, 200]:
                part = monitor(d[:stop], .2, .7)
                self.assertEqual(part['event_times'], [t for t in full['event_times'] if t<stop])
                self.assertEqual(part['armed'], full['armed'][:stop])

    def test_reject_same_count_wrong_source_config_reset(self):
        d = [.1, .3, .6, .1, .6]
        mon = monitor(d, .2, .5)
        for key, value in [('low_entry_timestamps', [1, 3]), ('event_times', [1, 4]),
                ('config', dict(d_low=.3, d_high=.5, initial_state_can_arm=True))]:
            bad = copy.deepcopy(mon); bad[key] = value
            self.assertFalse(source_consistent(d, bad, .2, .5))
        resets = copy.deepcopy(mon['pairs']); resets[0]['reset_timestep'] = 1
        self.assertFalse(source_consistent(d, mon, .2, .5, resets))

    def test_budget_only_resets_at_return_not_low_or_high(self):
        d = [.6, .6, .1, .3, .6, .6]
        mon = monitor(d, .2, .5)
        f, resets = evaluate(d, mon, .2, .5, np.ones(6), 0., np.ones(6, bool))
        np.testing.assert_allclose(f['budget_post_reset'], [1., .99, .98, .97, 1., .99])
        self.assertEqual([r['reset_timestep'] for r in resets], [4])
        self.assertAlmostEqual(resets[0]['pre_reset_budget'], .96)

    def test_pre_reset_insufficient_not_hidden(self):
        d = [.1, .3, .6]
        f, _ = evaluate(d, monitor(d, .2, .5), .2, .5, np.full(3, .01), 0., np.ones(3, bool))
        self.assertFalse(f['transition_soundness_pass'])
        self.assertFalse(combine_gates(f, 1, 1)['fit_preview_C_rec'])

    def test_large_budget_does_not_manufacture_returns(self):
        d = np.ones(301)
        f, _ = evaluate(d, monitor(d, .2, .5), .2, .5, np.full(301, 3.), 0., np.ones(301, bool))
        self.assertTrue(f['drift_pass'] and f['delta_margin_pass'])
        r = combine_gates(f, 0, 1)
        self.assertEqual(r['failure_reasons'], ['INSUFFICIENT_EVENTS'])

    def test_armed_is_legal_region_state(self):
        x = np.array([[0., 1.], [0., 0.]])
        region = dict(mean=np.zeros(2), scale=np.ones(2), radius=2.)
        self.assertTrue(inside_region(x, region).all())

    def test_invalid_data_and_thresholds(self):
        for low, high in [(.5, .5), (.6, .5), (-.1, .5)]:
            with self.assertRaises(ValueError): monitor([.1, .6], low, high)
        with self.assertRaises(ValueError): monitor([.1, float('nan')], .2, .5)


if __name__ == '__main__': unittest.main()
