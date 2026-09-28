import unittest
import numpy as np
import torch
from experiments.l2_policy_v_refine import objective, rank
from experiments.l2_achievement64 import summary


class RefinementTests(unittest.TestCase):
    def test_training_margin_does_not_change_audit(self):
        v=np.array([[.04,.025,.005]])
        d=np.array([[False,False,True]])
        self.assertEqual(summary(v,d)['p1p2_paths']['successes'],1)
        self.assertGreater(objective(torch.tensor(v),torch.tensor(d)).item(),0)

    def test_rank_prioritizes_whole_paths_not_edges(self):
        good=summary(np.array([[.04,.02,.005],[.02,.03,.005]]),
                     np.array([[0,0,1],[0,0,1]],dtype=bool))
        bad=summary(np.array([[.02,.03,.005],[.02,.03,.005]]),
                    np.array([[0,0,1],[0,0,1]],dtype=bool))
        self.assertGreater(rank(good),rank(bad))

    def test_nonarrival_paths_contribute_gradients(self):
        v=torch.tensor([[.1,.1,.1]],requires_grad=True)
        d=torch.zeros_like(v)
        objective(v,d).backward()
        self.assertTrue(torch.isfinite(v.grad).all())
        self.assertGreater(v.grad.abs().sum().item(),0)

    def test_max_term_penalizes_worst_transition(self):
        v=torch.tensor([[.1,.2,.19]],requires_grad=True)
        d=torch.zeros_like(v)
        self.assertGreater(objective(v,d,path_weight=4).item(),
                           objective(v,d,path_weight=0).item())


if __name__=='__main__': unittest.main()
