import unittest
import numpy as np
from experiments.safedreamer_b_budget_diagnosis import prediction_metrics, failure_detail
from core.finite_recurrence import budget_diagnostics


class BudgetDiagnosisTests(unittest.TestCase):
    def test_censored_tail_not_a_return_label(self):
        rows=[dict(completed=True,required_budget=1.,raw_g=.8),
              dict(completed=False,required_budget=99.,raw_g=0.)]
        result=prediction_metrics(rows,1.)
        self.assertEqual(result['n'],1)
        self.assertAlmostEqual(result['learned_mae'],.2)
        self.assertEqual(result['constant_mae'],0.)

    def test_failure_after_success_is_distinct(self):
        f=budget_diagnostics(np.full(11,.04),[2],[2],0.,np.ones(11,bool))
        d=failure_detail(f,[2],1)
        self.assertEqual(d['first_budget_failure'],7)
        self.assertTrue(d['budget_failure_after_Mth_event'])
        self.assertIsNone(failure_detail(f,[2],2)['Mth_event_time'])

    def test_no_budget_failure(self):
        f=budget_diagnostics(np.ones(11),[],[],0.,np.ones(11,bool))
        d=failure_detail(f,[],1)
        self.assertIsNone(d['first_budget_failure'])
        self.assertFalse(d['budget_failure_after_Mth_event'])


if __name__=='__main__': unittest.main()
