"""Post-counterexample diagnostics must preserve evidence and disjoint splits."""

import copy
import io
import unittest
from contextlib import redirect_stdout

from core.safety_evidence import ENVIRONMENT_VIOLATION
from experiments.l2_safedreamer import diagnose_after_violation
from main import VerifyConfig, verify
from specs import get_spec_by_id


def _case():
    spec = copy.deepcopy(get_spec_by_id("ltl_hazard_avoidance"))
    train = [[{"hazard_dist": 0.5}, {"hazard_dist": -0.1}]]
    calibration = [[{"hazard_dist": 0.4}, {"hazard_dist": 0.3}]]
    env = [{"hazard_dist": 0.4}, {"hazard_dist": -0.2}]
    cfg = VerifyConfig(
        paired_rollouts=[(calibration[0], env)],
        fit_lppm_params=True, lppm_train_trajectories=train,
        lppm_epochs=2, verbose=False,
    )
    result = verify(calibration, spec, cfg)
    return spec, train, calibration, cfg, result


class DiagnosticTests(unittest.TestCase):
    def test_neural_training_preserves_verdict_and_witness(self):
        spec, train, calibration, cfg, result = _case()
        original_summary = result.summary()
        output = io.StringIO()
        with redirect_stdout(output):
            diagnostics = diagnose_after_violation(result, train, calibration, spec, cfg)
        self.assertEqual(diagnostics.training_info["backend"], "torch_mlp")
        self.assertEqual(len(diagnostics.pathwise), len(calibration))
        self.assertIsNotNone(diagnostics.zfree_closure)
        self.assertEqual(result.safety_verdict.verdict, ENVIRONMENT_VIOLATION)
        self.assertIsNone(result.lppm)
        self.assertEqual(result.summary(), original_summary)
        self.assertIn("L2 DIAGNOSTIC ONLY", output.getvalue())
        self.assertIn("L2 certificate:", output.getvalue())
        self.assertIn("eta: 0.010000", output.getvalue())
        self.assertRegex(output.getvalue(), r"P1 violating transitions: \d+ / \d+ p1_checked_transitions")
        self.assertRegex(output.getvalue(), r"P2 violating transitions: \d+ / \d+ bad_transitions")
        self.assertIn("certificate_event_successes:", output.getvalue())
        self.assertIn("z_free_sampled_closure:", output.getvalue())
        self.assertIn("deductive_status: NOT_ESTABLISHED", output.getvalue())
        self.assertIn("result: NO_WARRANT", output.getvalue())

    def test_shared_training_calibration_data_rejected(self):
        spec, _, calibration, cfg, result = _case()
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "disjoint"):
            diagnose_after_violation(result, calibration, calibration, spec, cfg)
