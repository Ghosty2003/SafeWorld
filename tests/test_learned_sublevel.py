import unittest
import numpy as np
import torch
from core.lppm.learned_sublevel import audit, ProductValue, paper_loss


class LearnedSublevelTests(unittest.TestCase):
    def test_acceptance_does_not_force_membership(self):
        r = audit(np.array([[.5, .4]]), np.array([[True, True]]))
        self.assertEqual(r['zfree_entered']['successes'], 0)
        self.assertEqual(r['goal_reached']['successes'], 1)

    def test_pending_low_values_are_not_hidden_by_done_gate(self):
        r = audit(np.array([[.005, .004]]), np.array([[False, False]]))
        self.assertEqual(r['zfree_entered']['successes'], 1)
        self.assertEqual(r['pending_sublevel_states'], 2)
        self.assertEqual(r['zfree_sources']['p2_violations'], 1)
        self.assertEqual(r['sampled_region_status'], 'COUNTEREVIDENCE')

    def test_terminal_only_false_positive_is_reported(self):
        r = audit(np.array([[.02, .005]]), np.array([[False, False]]))
        self.assertEqual(r['candidate_certificate_event']['successes'], 1)
        self.assertEqual(r['finite_goal_and_certificate']['successes'], 0)
        self.assertEqual(r['terminal_only_entry_paths'], 1)
        self.assertIsNone(r['sampled_closure']['rate'])
        self.assertEqual(r['p1']['checked'], 1)

    def test_accepting_exit_and_increase_are_checked(self):
        r = audit(np.array([[.005, .02]]), np.array([[True, True]]))
        self.assertEqual(r['zfree_sources']['exits'], 1)
        self.assertEqual(r['p1']['violations'], 1)
        self.assertEqual(r['sampled_closure']['successes'], 0)

    def test_valid_observed_absorption(self):
        r = audit(np.array([[.04, .02, .005, .004]]), np.array([[False, False, True, True]]))
        self.assertEqual(r['finite_goal_and_certificate']['successes'], 1)
        self.assertEqual(r['sampled_closure']['successes'], 1)
        self.assertEqual(r['p1']['checked'], 3)
        self.assertEqual(r['p2']['checked'], 2)

    def test_p1_also_applies_to_bad_sources(self):
        r = audit(np.array([[.02, .03]]), np.array([[False, False]]))
        self.assertEqual(r['p1']['violations'], 1)
        self.assertEqual(r['p2']['violations'], 1)

    def test_positive_offset_preserves_constraints_but_changes_sublevel(self):
        values = np.array([[.04, .02, .005, .004]])
        done = np.array([[False, False, True, True]])
        a, b = audit(values, done), audit(values+1., done)
        self.assertEqual(a['p1'], b['p1'])
        self.assertEqual(a['p2'], b['p2'])
        self.assertEqual(a['zfree_entered']['successes'], 1)
        self.assertEqual(b['zfree_entered']['successes'], 0)

    def test_trainable_accepting_branch_and_gradient_loss(self):
        torch.manual_seed(4)
        net = ProductValue(latent_dim=2, width=8)
        x = torch.randn(2, 3, 2, requires_grad=True)
        done = torch.tensor([[0., 0., 1.], [1., 1., 1.]])
        v = net(x, done)
        self.assertTrue(torch.all(v > 0))
        loss, _ = paper_loss(v, done, x, torch.ones(2), .01, .001)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()))


if __name__ == '__main__':
    unittest.main()
