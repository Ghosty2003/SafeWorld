import unittest
import numpy as np
from core.finite_recurrence import safe_window_monitor,reference_events,path_score,budget_diagnostics,combine_gates,FAILURES
from experiments.safedreamer_finite_recurrence_pilot import seed_pair,require_fit_stage,SeededResetRecorder


class FiniteRecurrenceTests(unittest.TestCase):
    def test_300_safe_states_six_disjoint_events(self):
        s=np.ones(301,dtype=bool); m=safe_window_monitor(s)
        self.assertEqual(m['event_times'],[48,96,144,192,240,288])
        self.assertEqual(m['event_times'],reference_events(s))
        self.assertEqual(m['hazard_interruption_count'],0)
        self.assertEqual(m['counter'][-1],12)

    def test_initial_state_is_not_counted(self):
        self.assertEqual(safe_window_monitor(np.ones(48,dtype=bool))['event_count'],0)

    def test_hazard_break_does_not_reset_budget(self):
        s=np.ones(97,dtype=bool); s[48]=False
        m=safe_window_monitor(s)
        self.assertEqual(m['event_times'],[96]); self.assertEqual(m['hazard_interruption_count'],1)
        f=budget_diagnostics(np.full(97,.48),[96],[96],0.,np.ones(97,bool))
        self.assertTrue(f['drift_pass']); self.assertFalse(f['transition_soundness_pass'])
        self.assertFalse(f['delta_margin_pass'])

    def test_final_arrival_edge_checked_before_reset(self):
        f=budget_diagnostics(np.full(49,.47),[48],[48],0.,np.ones(49,bool))
        self.assertEqual(f['n_transitions'],48)
        self.assertFalse(f['transition_soundness_pass'])
        self.assertFalse(f['delta_margin_pass'])

    def test_exact_budget_and_observed_tail(self):
        e=[48,96,144,192,240,288]
        f=budget_diagnostics(np.full(301,.48),e,e,0.,np.ones(301,bool))
        self.assertTrue(f['transition_soundness_pass']); self.assertTrue(f['delta_margin_pass'])
        self.assertEqual(f['segments'][-1]['length'],12)
        self.assertFalse(f['segments'][-1]['completed'])
        self.assertTrue(combine_gates(f,6,6)['fit_preview_C_rec'])
        self.assertIsNone(combine_gates(f,6,6)['certificate_event'])

    def test_empty_event_path_is_not_dropped(self):
        f=budget_diagnostics(np.full(301,3.),[],[],0.,np.ones(301,bool))
        r=combine_gates(f,0,3)
        self.assertIn('INSUFFICIENT_EVENTS',r['failure_reasons'])
        self.assertEqual(f['n_transitions'],300)

    def test_every_active_gate_is_in_and(self):
        good={k:True for k in FAILURES}
        for gate,reason in FAILURES.items():
            if gate=='count_pass': continue
            bad=dict(good); bad[gate]=False
            r=combine_gates(bad,6,3,cv_enabled=True,sliding_enabled=True)
            self.assertFalse(r['fit_preview_C_rec']); self.assertIn(reason,r['failure_reasons'])
        good['region_gate_pass']=None
        self.assertFalse(combine_gates(good,6,3)['fit_preview_C_rec'])

    def test_same_count_wrong_event_time_fails(self):
        f=budget_diagnostics(np.full(101,3.),[48,96],[49,97],0.,np.ones(101,bool))
        self.assertFalse(f['event_source_pass']); self.assertFalse(f['transition_soundness_pass'])

    def test_detector_reference_random_traces(self):
        rng=np.random.default_rng(17)
        for _ in range(100):
            s=rng.random(301)>.03
            self.assertEqual(safe_window_monitor(s)['event_times'],reference_events(s))

    def test_score_includes_censored_tail(self):
        self.assertAlmostEqual(path_score(np.full(301,.1),[48]),2.42)

    def test_four_seed_namespaces_and_locked_roles(self):
        pools=[{v for i in range(1000) for v in seed_pair(r,i)} for r in ('fit','cal_delta','cal_CP','test')]
        self.assertEqual(len(set.union(*pools)),sum(map(len,pools)))
        for role in ('cal_delta','cal_CP','test'):
            with self.assertRaises(RuntimeError): require_fit_stage(role)

    def test_seeded_reset_rejects_unseeded_calls_and_restores(self):
        class Gym:
            def reset(self,seed=None): return seed
        gym=Gym(); original=gym.reset
        env=type('SafetyGymCoor',(),{})(); env._dmenv=gym
        outer=type('EmbodiedWrapper',(),{})(); outer.env=env
        recorder=SeededResetRecorder(outer)
        with self.assertRaises(RuntimeError): gym.reset()
        recorder.pending=123
        self.assertEqual(gym.reset(),123); self.assertIsNone(recorder.pending)
        recorder.restore(); self.assertEqual(gym.reset,original)


if __name__=='__main__': unittest.main()
