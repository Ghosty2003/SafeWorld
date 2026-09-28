"""Observed terminal, real self-loop, and frozen preprocessing regressions."""
import unittest
import numpy as np

from core.lppm.latent_reachability import (
    make_features, fit_scaler, normalize, training_targets,
    audit_values, kernel_predict,
)


class LatentReachabilityTests(unittest.TestCase):
    def test_last_observation_goal_is_counted_without_extra_transition(self):
        aps = np.array([[[1., 1., 0.], [0.2, 1., 0.], [-0.1, 1., 0.]]])
        _, done = make_features(np.zeros((1, 3, 2)), aps)
        result = audit_values(np.array([[0.04, 0.02, 0.011]]), done)
        self.assertEqual(result['transitions_per_path'], 2)
        self.assertEqual(result['p2']['checked'], 2)
        self.assertEqual(result['p2']['violations'], 0)
        self.assertEqual(result['certificate_event']['successes'], 1)

    def test_censored_waiting_endpoint_is_not_forced_to_zero(self):
        done = np.zeros((1, 3), dtype=bool)
        target = training_targets(done)
        self.assertGreater(target[0, -1], 0.01)
        result = audit_values(target, done)
        self.assertEqual(result['p2']['checked'], 2)
        self.assertEqual(result['p2']['violations'], 0)
        self.assertEqual(result['certificate_event']['successes'], 0)

    def test_real_waiting_selfloop_still_fails_p2(self):
        result = audit_values(np.array([[0.1, 0.1]]), np.array([[False, False]]))
        self.assertEqual(result['p2']['violations'], 1)

    def test_goal_then_departure_stays_accepting(self):
        aps = np.array([[[-0.1, 1., 0.], [2., 1., 0.], [3., 1., 0.]]])
        _, done = make_features(np.zeros((1, 3, 2)), aps)
        self.assertTrue(done.all())
        result = audit_values(np.full((1, 3), 9.), done)
        self.assertEqual(result['p1']['checked'], 2)
        self.assertEqual(result['p1']['violations'], 0)
        self.assertEqual(result['certificate_event']['successes'], 1)

    def test_scaler_is_frozen_and_kernel_reuses_it(self):
        train = np.array([[[0., 2.], [2., 4.]]])
        mean, scale = fit_scaler(train)
        reference = mean.copy(), scale.copy()
        heldout = np.array([[[100., 102.], [200., 202.]]])
        transformed = normalize(heldout, mean, scale)
        self.assertTrue(np.array_equal(mean, reference[0]))
        self.assertTrue(np.array_equal(scale, reference[1]))
        self.assertTrue(np.array_equal(transformed, heldout-mean))
        model = dict(centers=normalize(train, mean, scale).reshape(2, 2),
                     alpha=np.array([0.1, 0.2]), mean=mean, scale=scale,
                     bandwidth=1., floor=0.011)
        result = kernel_predict(heldout, model)
        self.assertTrue(np.all(result >= 0.011))


if __name__ == '__main__':
    unittest.main()
