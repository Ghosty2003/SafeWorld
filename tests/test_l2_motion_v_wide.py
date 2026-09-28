import unittest
import numpy as np
import torch
from experiments import l2_motion_v_wide_search as w


class Tests(unittest.TestCase):
    def test_grid_is_fixed_unique48(self):
        cfg=w.configs()
        self.assertEqual(len(cfg),48)
        self.assertEqual(len({c['id'] for c in cfg}),48)
        self.assertEqual(len({c['seed'] for c in cfg}),48)

    def test_sparse_matches_pointwise_all_families(self):
        rng=np.random.RandomState(123)
        x=rng.normal(size=(2,4,5));q=np.array([[0,1,2,2],[1,1,0,2]])
        for cfg in w.configs()[::4]:
            net=w.Value(5,cfg).double()
            model=dict(config=cfg,mean=np.zeros(5),scale=np.ones(5),weights=net.state_dict())
            direct=net(torch.tensor(x),torch.tensor(q)).detach().numpy()
            np.testing.assert_allclose(w.predict(model,x,q),direct,rtol=1e-12,atol=1e-12)
            self.assertTrue((direct[q!=2]>=.01).all());self.assertTrue((direct[q==2]==0.).all())

    def test_combination_preserves_training_drift(self):
        v=np.array([[.09,.06,.03,0.]])
        old=np.array([[.12,.08,.04,0.]])
        q=np.array([[0,1,1,2]]);b=q!=2;d=~b
        for cfg in [dict(kind='single',gain=2),dict(kind='blend',weight=.25),dict(kind='min'),dict(kind='max'),dict(kind='baseline')]:
            stat=w.base.statistics(w.combine(v,old,cfg),b,d)
            self.assertEqual(stat['p1_violations'],0);self.assertEqual(stat['p2_violations'],0)


if __name__=='__main__':unittest.main()
