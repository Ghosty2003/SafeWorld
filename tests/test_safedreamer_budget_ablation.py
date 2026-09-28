import unittest
import numpy as np
from experiments.safedreamer_budget_ablation import minimal_scale,scaled_diagnostics


class BudgetScaleTests(unittest.TestCase):
    def setUp(self):
        self.pred=np.full(301,.3); self.events=[48,96,144,192,240,288]; self.region=np.ones(301,bool)

    def test_distinct_scaling_definitions(self):
        self.assertAlmostEqual(minimal_scale(self.pred,self.events,.1,'g_only'),.38/.3)
        self.assertAlmostEqual(minimal_scale(self.pred,self.events,.1,'whole_budget'),.48/.4)

    def test_first_failure_and_margin(self):
        r=scaled_diagnostics(self.pred,self.events,self.events,.1,self.region,1.,'g_only')
        self.assertEqual(r['first_failure_time'],41)
        self.assertAlmostEqual(r['minimum_budget_margin'],-.08)

    def test_scale_cannot_repair_region_or_event_failure(self):
        self.region[12]=False
        r=scaled_diagnostics(self.pred,self.events,self.events,.1,self.region,10.,'g_only')
        self.assertFalse(r['fit_preview_C_rec']); self.assertEqual(r['first_failure_time'],12)
        self.assertIn('REGION_GATE_FAIL',r['failure_reasons'])

    def test_boundary_and_monotonicity(self):
        rates=[scaled_diagnostics(self.pred,self.events,self.events,.1,self.region,s,'g_only')['fit_preview_C_rec']
               for s in [.5,.75,1.,1.25,1.5,2.]]
        self.assertEqual(rates,sorted(rates))
        b=minimal_scale(self.pred,self.events,.1,'g_only')
        self.assertTrue(scaled_diagnostics(self.pred,self.events,self.events,.1,self.region,b,'g_only')['fit_preview_C_rec'])


if __name__=='__main__': unittest.main()
