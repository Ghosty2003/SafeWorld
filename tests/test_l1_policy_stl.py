import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from experiments.l1_policy_stl import (
    ALL_GROUNDED, collect, evaluate, formula_requirements, select_specs, write_new,
)
from specs.stl_specs import G, F, atom


class PolicySTLTests(unittest.TestCase):
    def test_nested_horizon(self):
        self.assertEqual(formula_requirements(G(0, 40, F(0, 14, atom('hazard_dist', 0)))),
                         (54, {'hazard_dist'}))

    def test_grounding(self):
        self.assertEqual(len(select_specs(ALL_GROUNDED)), 7)
        with self.assertRaises(ValueError):
            select_specs(['stl_safe_dual_patrol'])
        with self.assertRaises(ValueError):
            select_specs(['ltl_goal_reach'])

    def test_no_truncation(self):
        spec = select_specs(['stl_hazard_avoidance'])[0]
        with self.assertRaisesRegex(ValueError, 'truncated'):
            evaluate(spec, [[{'hazard_dist': 1.}] * 10], .05)
        row = evaluate(spec, [[{'hazard_dist': 1.}] * 50] * 100, .05)
        self.assertEqual(row['successes'], 100)
        self.assertAlmostEqual(row['cp_lower'], .05 ** (1/100))

    def test_missing_or_nonfinite_not_satisfied(self):
        spec = select_specs(['stl_hazard_avoidance'])[0]
        for state in ({}, {'hazard_dist': float('nan')}, {'hazard_dist': float('inf')}):
            with self.assertRaises(ValueError):
                evaluate(spec, [[state] * 50], .05)

    def test_observed_failure_separate_from_probability(self):
        spec = select_specs(['stl_hazard_avoidance'])[0]
        row = evaluate(spec, [[{'hazard_dist': -1.}] * 50], .05)
        self.assertEqual(row['cp_lower'], 0)
        self.assertEqual(row['observed_nonpositive_margins'], 1)
        # A single success still gives only a 5% lower confidence bound.
        row = evaluate(spec, [[{'hazard_dist': 1.}] * 50], .05)
        self.assertAlmostEqual(row['cp_lower'], .05)

    def test_policy_collection_no_environment_pairs(self):
        wrapper = Mock()
        wrapper.sample_latent_rollouts.return_value = dict(
            action_source='policy', aps=[[{'hazard_dist': 1.}] * 3],
            actions=np.zeros((1, 2, 2)))
        result = collect(wrapper, 2, 1, 123, {'action_source': 'policy'})
        cfg = wrapper.sample_latent_rollouts.call_args[0][0]
        self.assertEqual(cfg.action_source, 'policy')
        self.assertEqual(cfg.extra['action_source'], 'policy')
        self.assertEqual(result['horizon'], 2)
        wrapper.sample_paired_rollouts.assert_not_called()
        wrapper.sample_latent_rollouts.return_value['action_source'] = 'random'
        with self.assertRaises(RuntimeError):
            collect(wrapper, 2, 1, 123, {'action_source': 'policy'})

    def test_terminal_state_required(self):
        wrapper = Mock()
        wrapper.sample_latent_rollouts.return_value = dict(
            action_source='policy', aps=[[{'hazard_dist': 1.}] * 2])
        with self.assertRaises(RuntimeError):
            collect(wrapper, 2, 1, 123, {'action_source': 'policy'})

    def test_never_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'prediction.json'
            write_new(path, {'original': True})
            with self.assertRaises(FileExistsError):
                write_new(path, {'original': False})


if __name__ == '__main__':
    unittest.main()
