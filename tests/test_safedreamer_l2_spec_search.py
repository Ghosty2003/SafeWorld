import unittest
import numpy as np
from experiments.safedreamer_l2_spec_search import candidates, monitor, graph_feasibility, statistics


class TestL2SpecSearch(unittest.TestCase):
    def test_return_is_causal_and_absorbing(self):
        spec=candidates(.4,.8)[1]
        q,b,d=monitor([.9,.3,.6,.9,.1],np.ones(5),spec)
        np.testing.assert_array_equal(q,[0,1,1,2,2])
        for end in range(1,6):
            np.testing.assert_array_equal(monitor([.9,.3,.6,.9,.1][:end],np.ones(end),spec)[0],q[:end])

    def test_recovery_requires_eight_high_states(self):
        spec=candidates(.4,.8)[3]
        q,b,d=monitor([.3]+[.9]*7+[.5]+[.9]*8,np.ones(17),spec)
        self.assertFalse(d[:-1].any());self.assertTrue(d[-1])

    def test_persistence_never_finite_completion(self):
        spec=candidates(.4,.8)[4]
        q,b,d=monitor([.3,.4,.1,.9],np.ones(4),spec)
        self.assertFalse(d.any());np.testing.assert_array_equal(b,[False,False,True,False])

    def test_positive_cycle_rejected_zero_cycle_allowed(self):
        x=np.array([[[0.],[1.],[0.]]]);q=np.zeros((1,3),int)
        result,_=graph_feasibility(x,q,np.array([[True,False,True]]))
        self.assertFalse(result['feasible'])
        result,v=graph_feasibility(x,q,np.zeros((1,3),bool))
        self.assertTrue(result['feasible']);self.assertEqual(v.max(),0.)

    def test_no_artificial_last_edge_and_strict_source_bad(self):
        x=np.arange(4.).reshape(1,4,1);q=np.array([[0,0,0,1]])
        result,v=graph_feasibility(x,q,np.array([[True,True,True,False]]))
        self.assertEqual(result['edges'],3);self.assertAlmostEqual(v[0,0],.03)
        s=statistics(np.array([[.06,.04,.02,0.]]),q==0,q==1)
        self.assertEqual(s['bad_transitions'],3);self.assertEqual(s['certificate_preview'],1)
        s=statistics(np.array([[.06,.04,.02,.02]]),q==0,q==1)
        self.assertEqual(s['p2_violations'],1);self.assertEqual(s['certificate_preview'],0)

    def test_all_gate_and(self):
        v=np.array([[.04,.02,0.]]);b=np.array([[True,True,False]]);d=~b
        self.assertEqual(statistics(v,b,d)['certificate_preview'],1)
        self.assertEqual(statistics(v,b,d,region=np.array([[True,False,True]]))['certificate_preview'],0)
        self.assertEqual(statistics(v,b,np.zeros_like(d))['certificate_preview'],0)

    def test_constant_nonvacuous_p2_fails(self):
        b=np.array([[True,True,False]])
        self.assertEqual(statistics(np.ones((1,3)),b,~b)['p2_violations'],2)

    def test_same_latent_different_mode_not_false_cycle(self):
        x=np.array([[[0.],[0.],[0.]]]);q=np.array([[0,1,2]])
        result,v=graph_feasibility(x,q,np.array([[True,True,False]]))
        self.assertTrue(result['feasible']);self.assertEqual(result['nodes'],3)
        np.testing.assert_allclose(v,[[.02,.01,0.]])

    def test_branch_uses_shared_node_consistently(self):
        x=np.array([[[0.],[1.],[2.]],[[0.],[3.],[4.]]]);q=np.zeros((2,3),int)
        result,v=graph_feasibility(x,q,np.array([[True,False,False],[True,True,False]]))
        self.assertTrue(result['feasible']);self.assertEqual(v[0,0],v[1,0])
        self.assertAlmostEqual(v[0,0],.02)

    def test_p2_checks_source_not_target(self):
        v=np.array([[.02,.02]])
        self.assertEqual(statistics(v,np.array([[False,True]]),np.zeros((1,2),bool))['p2_violations'],0)
        self.assertEqual(statistics(v,np.array([[True,False]]),np.zeros((1,2),bool))['p2_violations'],1)


if __name__=='__main__':unittest.main()
