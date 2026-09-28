"""Paper-aligned H1 candidate training and pointwise drift diagnostics.

This module deliberately does NOT promote pointwise validation to a global
L3 warrant. A SafeDreamer certified covering/containment backend is absent.
Finite samples (even all passing) are not a substitute for that backend.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn

from .operator_model import (assert_anchor_splits_disjoint,
                             assert_structural_certificate_bound,
                             certificate_successor_means,
                             evaluate_torch_scalar_fn)


class NormalizedBoundedCertificate(nn.Module):
    def __init__(self, mean, scale, bound=1., hidden=64):
        super().__init__()
        if bound <= 0:
            raise ValueError('Positive structural bound required')
        mean, scale = np.asarray(mean), np.asarray(scale)
        if mean.ndim != 1 or mean.shape != scale.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Invalid normalization')
        self.B = float(bound)
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.tensor(scale,dtype=torch.float32))
        self.net=nn.Sequential(nn.Linear(len(mean),hidden),nn.Tanh(),
                               nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))

    def forward(self,x):
        return self.B * torch.sigmoid(self.net((x-self.mean)/self.scale).squeeze(-1))


def fit_h1(train, core, *, epochs=500, seed=41, eps_w=.01, eps_u=.001):
    """Train only on fit-role data. U drift penalized INCLUDING acceptance.

    Counter-derived shaping is a fitting heuristic, not a region certificate.
    W and U are evaluated afresh at each product state, never a countdown.
    """
    if not train or epochs < 1 or eps_w <= 0 or eps_u < 0:
        raise ValueError('Invalid training configuration')
    train=[b.validated() for b in train]
    core=np.asarray(core,dtype=np.float32)
    if core.ndim != 2 or not len(core) or not np.isfinite(core).all():
        raise ValueError('Nonempty finite training core required')
    torch.manual_seed(seed)
    mean=core.mean(0); scale=np.maximum(core.std(0),.05)
    # Counter and planner-initialized bit have stable, declared units.
    mean[-2:]=0; scale[-2:]=1
    W=NormalizedBoundedCertificate(mean,scale)
    U=NormalizedBoundedCertificate(mean,scale)
    x=torch.tensor(np.stack([b.anchor for b in train]))
    y=torch.tensor(np.stack([b.successors for b in train]))
    c=torch.tensor(core)
    non_f=(x[:,-1] < 1.)
    target_w=eps_w*48*(1-c[:,-1])
    target_u=.1+.4*(1-c[:,-1])
    opt=torch.optim.Adam(list(W.parameters())+list(U.parameters()),lr=.001)
    history=[]
    for epoch in range(epochs):
        dw=W(y.flatten(0,1)).reshape(y.shape[:2]).mean(1)-W(x)
        du=U(y.flatten(0,1)).reshape(y.shape[:2]).mean(1)-U(x)
        loss_w=torch.relu(dw[non_f]+eps_w).mean() if non_f.any() else dw.sum()*0
        loss_u=torch.relu(du+eps_u).mean()  # no acceptance/retention bypass
        shape=((W(c)-target_w)**2).mean()+((U(c)-target_u)**2).mean()
        loss=loss_w+loss_u+.25*shape
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch==0 or (epoch+1)%50==0 or epoch+1==epochs:
            history.append(dict(epoch=epoch+1,loss=float(loss.item()),
                                W_drift_loss=float(loss_w.item()),U_drift_loss=float(loss_u.item()),
                                shaping_loss=float(shape.item())))
    W.eval(); U.eval()
    return W,U,history


def drift_diagnostics(W,U,batches,*,fit_batches,ell=.8,eps_w=.01,delta=.05,independent=True):
    if not batches or not 0 < delta < 1 or ell <= 0 or eps_w <= 0:
        raise ValueError('Invalid validation configuration')
    batches=[b.validated() for b in batches]
    if independent:
        assert_anchor_splits_disjoint(fit_batches,batches,left_name='fit',right_name='validation')
    for net,name in ((W,'W'),(U,'U')):
        assert_structural_certificate_bound(net,net.B,name)
    x=np.stack([b.anchor for b in batches])
    w=evaluate_torch_scalar_fn(W,x); u=evaluate_torch_scalar_fn(U,x)
    gw=certificate_successor_means(W,batches,bound=W.B,certificate_name='W')-w
    gu=certificate_successor_means(U,batches,bound=U.B,certificate_name='U')-u
    if (w<0).any() or (w>W.B).any() or (u<0).any() or (u>U.B).any():
        raise ValueError('Structural bound violated')
    kappas=np.array([len(b.successors) for b in batches])
    # Union budget across BOTH functions and all validation anchors.
    delta_each=delta/(2*len(batches))
    factor=np.sqrt(math.log(1/delta_each)/(2*kappas))
    in_c=u<=ell; non_f=x[:,-1]<1
    required_w=in_c & non_f; required_u=in_c
    ew=W.B*factor; eu=U.B*factor
    rows=[]
    for i,b in enumerate(batches):
        rows.append(dict(id=b.sample_id,in_candidate_C=bool(in_c[i]),accepting=bool(not non_f[i]),
                         W=float(w[i]),U=float(u[i]),mean_drift_W=float(gw[i]),mean_drift_U=float(gu[i]),
                         mc_radius_W=float(ew[i]),mc_radius_U=float(eu[i]),
                         upper_drift_W_at_anchor=float(gw[i]+ew[i]),
                         upper_drift_U_at_anchor=float(gu[i]+eu[i])))
    # In-sample diagnostics must never be described as a concentration certificate.
    return dict(n_anchors=len(batches),n_in_candidate_C=int(in_c.sum()),
                n_W_required=int(required_w.sum()),n_U_required=int(required_u.sum()),
                n_accepting=int((~non_f).sum()),
                n_W_empirical_pass=int((required_w & (gw<=-eps_w)).sum()),
                n_U_empirical_pass=int((required_u & (gu<=0)).sum()),
                n_W_pointwise_ucb_pass=int((required_w & (gw+ew<=-eps_w)).sum()) if independent else None,
                n_U_pointwise_ucb_pass=int((required_u & (gu+eu<=0)).sum()) if independent else None,
                pointwise_confidence=1-delta if independent else None,
                per_function_per_anchor_delta=delta_each if independent else None,
                assumption='independent successor draws after certificate freeze' if independent else 'IN_SAMPLE_ONLY',
                scope='FINITE_ANCHORS_ONLY_NO_REGION_COVERAGE',rows=rows)


def formal_status(diagnostics, *, independent_frozen=False):
    """Honest capability gate, not a boolean-based external proof importer.

    NO positive global branch is implemented for this RSSM/planner backend.
    This is explicit so callers cannot mistake the audit for a full prover.
    """
    return dict(verdict='ABSTAIN',global_infinite_status='NOT_ESTABLISHED',
                probability_lower_bound=None,confidence=None,
                reason='SAFEDREAMER_GLOBAL_PROOF_BACKEND_UNAVAILABLE',
                obligations={
                    'frozen_certificate_independent_successors': 'CHECKED' if independent_frozen else 'NOT_CHECKED',
                    'pointwise_drift': 'CHECKED_AT_SAMPLED_ANCHORS_ONLY',
                    'closed_loop_cellwise_envelopes': 'NOT_ESTABLISHED',
                    'covering_of_operative_reachable_region': 'NOT_ESTABLISHED',
                    'sublevel_containment_in_covered_region': 'NOT_ESTABLISHED',
                    'H1_U_nonincrease_over_entire_C_including_F': 'NOT_ESTABLISHED',
                    'initial_distribution_support_in_C': 'NOT_ESTABLISHED',
                    'collar_entry_probability_bound': 'NOT_ESTABLISHED',
                    'initial_U_expectation_upper_bound': 'NOT_ESTABLISHED'},
                pointwise_W_pass=diagnostics['n_W_pointwise_ucb_pass'],
                pointwise_U_pass=diagnostics['n_U_pointwise_ucb_pass'],
                note='Failed or missing proof is not evidence that GF is false.')
