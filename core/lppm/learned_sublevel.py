"""Paper Eq. (3), (4), (10)-(12), audited on finite observed prefixes.

No acceptance-dependent output clamp and no synthetic terminal transition.
A low value is only a CANDIDATE sublevel membership, not a validity proof.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import beta


class ProductValue(torch.nn.Module):
    """Nonnegative trainable V(z,q), including the accepting branch."""
    def __init__(self, latent_dim=512, width=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim+2, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, 1))

    def forward(self, normalized_z, done):
        q = torch.stack((1-done, done), dim=-1)
        raw = self.net(torch.cat((normalized_z, q), dim=-1)).squeeze(-1)
        return torch.nn.functional.softplus(raw)


def predict(model, z, done):
    net = ProductValue(z.shape[-1], model['width'])
    net.load_state_dict(model['weights'])
    net.eval()
    a = torch.tensor((z-model['mean'])/model['scale'], dtype=torch.float32)
    with torch.no_grad():
        return net(a, torch.tensor(done, dtype=torch.float32)).numpy().astype(np.float64)


def paper_loss(values, done, normalized_z, scale, eta, regularization):
    """Source-state gradient penalty converted back to raw latent units."""
    increase = values[:, 1:]-values[:, :-1]
    l1 = torch.relu(increase).mean()  # P1 on ALL transitions
    l2 = (torch.relu(increase+eta)*(1-done[:, :-1])).mean()
    grad = torch.autograd.grad(values[:, :-1].sum(), normalized_z, create_graph=True)[0]
    penalty = (grad[:, :-1]/scale).square().sum(-1).mean()
    return l1+l2+regularization*penalty, (l1, l2, penalty)


def rate(event, inferential=False):
    event = np.asarray(event, dtype=bool)
    n, k = event.size, int(event.sum())
    row = dict(successes=k, trials=n, rate=k/n if n else None)
    if inferential:
        row.update(cp_lower=float(beta.ppf(.05, k, n-k+1)) if k else (0. if n else None),
                   confidence=.95 if n else None)
    return row


def audit(values, done, eta=.01, inferential=False):
    v, done = np.asarray(values), np.asarray(done, dtype=bool)
    if v.shape != done.shape or v.ndim != 2 or v.shape[1] < 2:
        raise ValueError('Expected aligned N by H+1 arrays')
    if not np.isfinite(v).all() or np.any(v < 0):
        raise ValueError('V must be finite and nonnegative')
    if np.any(done[:, :-1] & ~done[:, 1:]):
        raise ValueError('F(goal) accepting state must be absorbing')
    delta = v[:, :-1]-v[:, 1:]
    waiting = ~done[:, :-1]
    bad1, bad2 = delta < 0, waiting & (delta < eta)
    path_pass = ~(bad1 | bad2).any(1)
    zfree = v < eta  # EXACTLY Eq. (3); no done predicate or floor.
    entries = zfree.any(1)
    source = zfree[:, :-1]
    eligible = source.any(1)
    exits = source & ~zfree[:, 1:]
    closure_bad = (source & (bad1 | bad2 | exits)).any(1)
    candidate_c = path_pass & zfree[:, -1]  # Eq. (4), finite-prefix event only
    goal = done[:, -1]
    first = np.where(entries, zfree.argmax(1), -1)
    first_goal = np.where(goal, done.argmax(1), -1)
    suffix_inside = np.maximum.accumulate(zfree, axis=1)
    stayed = ~((~zfree) & suffix_inside).any(1)
    row = dict(
        n_paths=len(v), transitions_per_path=v.shape[1]-1,
        p1=dict(violations=int(bad1.sum()), checked=bad1.size),
        p2=dict(violations=int(bad2.sum()), checked=int(waiting.sum())),
        mean_paper_transition_loss=float(np.maximum(-delta, 0).mean()
                                        +(np.maximum(eta-delta, 0)*waiting).mean()),
        p1p2_paths=rate(path_pass, inferential),
        zfree_entered=rate(entries, inferential),
        zfree_endpoint=rate(zfree[:, -1], inferential),
        goal_reached=rate(goal, inferential),
        candidate_certificate_event=rate(candidate_c, inferential),
        finite_goal_and_certificate=rate(candidate_c & goal, inferential),
        goal_given_zfree_entry=rate(goal[entries], inferential),
        sampled_closure=rate((~closure_bad)[eligible], inferential),
        stayed_in_zfree_observed_suffix=rate(stayed[eligible], inferential),
        zfree_sources=dict(checked=int(source.sum()), p1_violations=int((source & bad1).sum()),
                          p2_violations=int((source & bad2).sum()), exits=int(exits.sum())),
        initial_zfree_paths=int(zfree[:, 0].sum()), initial_goal_paths=int(done[:, 0].sum()),
        post_initial_entries=int((entries & ~zfree[:, 0]).sum()),
        terminal_only_entry_paths=int((entries & ~eligible).sum()),
        pending_sublevel_states=int((zfree & ~done).sum()),
        pending_sublevel_paths=int((zfree & ~done).any(1).sum()),
        candidate_event_without_goal=int((candidate_c & ~goal).sum()),
        entry_without_goal_by_horizon=int((entries & ~goal).sum()),
        v_min=float(v.min()), v_max=float(v.max()), exact_zero_states=int((v == 0).sum()),
        waiting_value_range=[float(v[~done].min()), float(v[~done].max())] if (~done).any() else None,
        accepting_value_range=[float(v[done].min()), float(v[done].max())] if done.any() else None,
        first_entry_step=first.tolist(), first_goal_step=first_goal.tolist(),
        per_path_candidate_event=candidate_c.tolist(), per_path_goal=goal.tolist(),
        finite_test_only=True, support_wide_validity='NOT_ESTABLISHED',
    )
    row['sampled_region_status'] = ('NO_OBSERVED_MEMBERS' if not entries.any() else
        'COUNTEREVIDENCE' if (zfree & ~done).any() or closure_bad.any() else
        'NO_SOURCE_TRANSITIONS' if not eligible.any() else 'NO_VIOLATION_ON_OBSERVED_SOURCES')
    return row
