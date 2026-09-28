import unittest
import numpy as np
import torch
from experiments.safedreamer_b_fit100_train import labels,sampling_weights,BoundedReturnNet,select_candidate


class Fit100Tests(unittest.TestCase):
    def test_latency_bins_are_not_remaining_time(self):
        path=dict(monitor=dict(event_times=[200],pairs=[dict(low_entry_timestep=50)]))
        rows=labels(path)
        self.assertEqual(len(rows),300)
        self.assertEqual(rows[0]['bucket'],2)
        self.assertEqual(rows[199]['bucket'],2)
        self.assertAlmostEqual(rows[199]['target'],.01)
        self.assertEqual(rows[199]['low_to_high_latency'],150)
        self.assertFalse(rows[199]['censored'])
        self.assertTrue(rows[200]['censored'])
        self.assertEqual(rows[200]['bucket'],4)

    def test_stratum_weights_long_oversampling(self):
        rows=[dict(bucket=i) for i,n in enumerate([10,20,30,40,50]) for _ in range(n)]
        w=sampling_weights(rows)
        masses=[sum(v for r,v in zip(rows,w) if r['bucket']==i) for i in range(5)]
        np.testing.assert_allclose(masses,[1,1,3,5,1])

    def test_bound_and_residual_do_not_escape_cap(self):
        net=BoundedReturnNet(np.zeros(3),np.ones(3),8,2.4)
        y=net(torch.randn(100,3)*100).detach().numpy()
        self.assertTrue(((y>=0)&(y<=2.4)).all())
        total=np.minimum(2.4,y+100)
        np.testing.assert_array_equal(total,np.full_like(y,2.4))

    def test_censored_path_is_not_fake_success(self):
        rows=labels(dict(monitor=dict(event_times=[],pairs=[])))
        self.assertTrue(all(r['censored'] for r in rows))
        self.assertEqual(rows[0]['target'],3.)
        self.assertIsNone(rows[0]['low_to_high_latency'])

    def test_selection_uses_only_three_validation_metrics(self):
        def s(name,u,m,p):return dict(name=name,internal_validation=dict(
            reset_completed_raw_error=dict(mean_underestimation=u,MAE=m),full_pass_rate=p))
        i,info=select_candidate([s('bad',1,2,.1),s('good',.2,.3,.8)])
        self.assertEqual(i,1);self.assertEqual(info['pareto_candidates'],['good'])


if __name__=='__main__':unittest.main()
