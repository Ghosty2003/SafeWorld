import numpy as np
from experiments import l2_spec_continuation as exp


def spec(length=3):
    return dict(kind='window',length=length,threshold=.2)


def test_window_causal_reset_and_absorbing():
    c=np.array([.3,.1,.3,.3,.3,.1])
    q,b,d=exp.monitor(c,np.zeros(len(c)),spec())
    np.testing.assert_array_equal(q,[1,0,1,2,3,3])
    np.testing.assert_array_equal(d,[False,False,False,False,True,True])
    for end in range(1,len(c)+1):
        np.testing.assert_array_equal(exp.monitor(c[:end],np.zeros(end),spec())[0],q[:end])


def test_window_endpoint_and_initial_count():
    q,b,d=exp.monitor(np.ones(301),np.zeros(301),spec(48))
    assert q[0]==1 and not d[46] and d[47] and d[-1]
    c=np.zeros(301);c[253:]=1
    assert exp.monitor(c,np.zeros(301),spec(48))[2][-1]
    c[253]=0
    assert not exp.monitor(c,np.zeros(301),spec(48))[2][-1]


def test_strict_p1p2_and_nonvacuous_controls():
    b=np.array([[True,True,False,False]])
    d=~b
    v=np.array([[.04,.02,0,0]])
    good=exp.base.statistics(v,b,d)
    constant=exp.base.statistics(np.ones_like(v),b,d)
    assert good['p1p2_paths']==1 and constant['p1p2_paths']==0
    assert exp.qualification([good,good],{'constant':constant})
    assert not exp.qualification([good,good],{'shuffled':good})
    empty=dict(good,bad_transitions=0)
    assert not exp.qualification([empty,good],{'constant':constant})


def test_product_graph_positive_cycle_rejected():
    x=np.array([[[0.],[0.]]]);q=np.zeros((1,2),int);b=np.ones((1,2),bool)
    row,witness=exp.base.graph_feasibility(x,q,b)
    assert not row['feasible'] and witness is None
