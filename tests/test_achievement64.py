import unittest
import numpy as np
import torch
import json
from pathlib import Path
from unittest.mock import patch

from experiments.l2_achievement64 import AchievementValue, done_for_radius, summary
from experiments import l2_achievement64 as experiment


class Achievement64Tests(unittest.TestCase):
    def test_radius_conversion_and_history(self):
        aps = np.zeros((1, 4, 3))
        aps[0, :, 0] = [.8, .2, -.1, .9]
        np.testing.assert_array_equal(done_for_radius(aps, .3), [[False, False, True, True]])
        np.testing.assert_array_equal(done_for_radius(aps, .6), [[False, True, True, True]])

    def test_acceptance_starts_outside_sublevel_and_is_trainable(self):
        net = AchievementValue('anchored_accepting_scalar', width=8, latent_dim=2)
        x = torch.randn(1, 3, 2, requires_grad=True)
        done = torch.ones(1, 3)
        v = net(x, done)
        self.assertTrue(torch.all(v > .01))
        self.assertTrue(torch.all(v == v[0, 0]))
        v.mean().backward()
        self.assertGreater(float(net.accepting_raw.grad), 0)

    def test_initial_success_separated_from_later_arrival(self):
        v = np.array([[.005,.005,.005],[.04,.02,.005],[.04,.03,.02]])
        d = np.array([[1,1,1],[0,0,1],[0,0,0]],dtype=bool)
        a = summary(v,d)
        self.assertEqual(a['initial_goal_paths'],1)
        self.assertEqual(a['noninitial_certificate']['successes'],1)
        self.assertEqual(a['arrival_given_initially_pending']['trials'],2)
        self.assertEqual(a['clean_observed_region_paths']['successes'],1)
        self.assertEqual(a['terminal_only_entry_paths'],1)

    def test_full_64_edges_audited(self):
        a = summary(np.full((2,65),.005),np.ones((2,65),dtype=bool))
        self.assertEqual(a['p1']['checked'],128)
        self.assertEqual(a['p2']['checked'],0)
        self.assertEqual(a['zfree_sources']['checked'],128)

    def test_resume_restores_checkpoint_spec_and_seeds(self):
        plan=dict(horizon=64,eta=.01,counts=experiment.COUNTS,radii=[.3],action_source='random',
                  checkpoint='/tmp/test-point5.ckpt',seeds=dict(train=17501,validation=17502,
                  calibration=17503,test=17504))
        with patch.multiple(experiment,OUT=Path('/tmp/mock-run'),CHECKPOINT=None,RADII=(1.,),SEEDS={},ACTION_SOURCE='policy',DEVICE='gpu'):
            with patch.object(Path,'read_text',return_value=json.dumps(plan)):
                experiment.configure(restore=True)
                self.assertEqual(experiment.CHECKPOINT,plan['checkpoint'])
                self.assertEqual(experiment.RADII,(.3,))
                self.assertEqual(experiment.SEEDS,plan['seeds'])
                self.assertEqual(experiment.ACTION_SOURCE,'random')
                self.assertEqual(experiment.DEVICE,'cpu')
                with self.assertRaises(ValueError):
                    experiment.configure(device='gpu',restore=True)
                with self.assertRaises(ValueError):
                    experiment.configure(action_source='policy',restore=True)
                with self.assertRaises(ValueError):
                    experiment.configure(radii=[1.],restore=True)
                with self.assertRaises(ValueError):
                    experiment.configure(checkpoint='/tmp/different.ckpt',restore=True)
                with self.assertRaises(ValueError):
                    experiment.configure(seed_base=42,restore=True)

    def test_strict_original_goal_boundary(self):
        aps=np.zeros((1,3,3)); aps[0,:,0]=[.01,0.,-.01]
        np.testing.assert_array_equal(done_for_radius(aps,.3),[[False,False,True]])

    def test_policy_features_preserve_planner_memory(self):
        a=dict(latent=np.zeros((2,65,512)),planner_action_mean=np.ones((2,65,16,2)),
               planner_action_std=np.ones((2,65,16,2))*2,planner_initialized=np.ones((2,65,1)))
        x=experiment.value_features(a,'policy')
        self.assertEqual(x.shape,(2,65,577))
        self.assertEqual(x[0,0,512],1.)
        self.assertEqual(x[0,0,544],2.)
        self.assertEqual(experiment.value_features(a,'random').shape,(2,65,512))
        del a['planner_action_mean']
        with self.assertRaises(KeyError): experiment.value_features(a,'policy')

    def test_policy_input_schema_prediction_roundtrip(self):
        torch.manual_seed(7)
        net=AchievementValue('anchored_accepting_scalar',width=8,latent_dim=577)
        z=np.zeros((2,3,577),dtype=np.float64)
        d=np.array([[False,False,True],[False,True,True]])
        model=dict(variant='anchored_accepting_scalar',width=8,weights=net.state_dict(),
                   mean=np.zeros(577),scale=np.ones(577))
        expected=net(torch.tensor(z,dtype=torch.float32),torch.tensor(d,dtype=torch.float32)).detach().numpy()
        np.testing.assert_array_equal(experiment.predict(model,z,d),expected)


if __name__ == '__main__':
    unittest.main()
