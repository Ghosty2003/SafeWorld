import copy
import unittest
import numpy as np
from experiments.safedreamer_recurrence_screening import signals,augmented_gate,characterize,status_for,choose
from experiments.safedreamer_high_clearance_pilot import monitor


class ScreeningTests(unittest.TestCase):
    def test_signals_use_only_past(self):
        a={'decoded':np.random.default_rng(9).normal(size=(301,29))}
        b=copy.deepcopy(a); b['decoded'][150:]*=20
        for name,s in signals(a).items():
            np.testing.assert_array_equal(s[:150],signals(b)[name][:150])
        self.assertEqual(signals(a)['progress'][0],.5)

    def test_approach_orientation(self):
        a={'decoded':np.zeros((301,29))}; a['decoded'][:,7]=2
        a['decoded'][10:,7]=.5
        s=signals(a)
        self.assertLess(s['approach'][9],s['approach'][10])
        self.assertGreater(s['progress'][10],.5)
        self.assertEqual(monitor(s['approach'],.4,.6)['event_times'],[10])

    def test_safety_is_extra_and_not_override(self):
        good={k:True for k in ['drift_pass','delta_margin_pass','region_gate_pass',
            'transition_soundness_pass','event_source_pass']}
        self.assertTrue(augmented_gate(good,[2,4],False,False)['fit_preview_C_rec'])
        self.assertFalse(augmented_gate(good,[2,4],True,False)['fit_preview_C_rec'])
        good['drift_pass']=False
        self.assertFalse(augmented_gate(good,[2,4],True,True)['fit_preview_C_rec'])

    def test_characterize_allpass_not_same_counts(self):
        s=characterize([2,3,4,6]); self.assertTrue(s['all_paths_count_pass'])
        self.assertGreater(s['variance'],0)
        self.assertEqual(s['zero_fraction'],0)

    def test_status_does_not_reward_event_saturation(self):
        a=characterize([20,30,40,50])
        self.assertEqual(status_for(a,a,.01,1.,0.)[0],'BORDERLINE')
        a=characterize([0,0,0,0])
        self.assertEqual(status_for(a,a,.01,1.,0.)[0],'REJECT')

    def test_select_by_declared_priority_not_rate(self):
        rows=[dict(id='motion',status='BORDERLINE',semantic_priority=2,safety_required=False,
            threshold_label='q25_75',signal='speed',pass_rate=1.),
            dict(id='clear',status='BORDERLINE',semantic_priority=0,safety_required=False,
            threshold_label='q25_75',signal='clearance',pass_rate=.1)]
        p,b=choose(rows); self.assertEqual(p['id'],'clear'); self.assertEqual(b['id'],'motion')


if __name__=='__main__': unittest.main()
