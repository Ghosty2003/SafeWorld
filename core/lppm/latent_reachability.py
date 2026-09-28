"""Shared training/evaluation semantics for full-latent F(goal) candidates.

Restricted to reachability. The product state q_t has consumed label(z_t).
Thus H+1 observations provide H genuine dynamic product transitions, and a
goal at the final observation is recognized without inventing a successor.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import beta


def make_features(latent, aps):
    latent, aps = np.asarray(latent), np.asarray(aps)
    if latent.ndim != 3 or aps.ndim != 3 or latent.shape[:2] != aps.shape[:2]:
        raise ValueError('Latent/AP arrays must align as (paths, H+1 states, features)')
    if latent.shape[1] < 2 or aps.shape[-1] != 3:
        raise ValueError('Expected >=2 states and goal_dist/hazard_dist/velocity')
    if not np.isfinite(latent).all() or not np.isfinite(aps).all():
        raise ValueError('Nonfinite inputs')
    x = np.concatenate([latent, aps], axis=-1).astype(np.float64)
    done = np.maximum.accumulate(aps[:, :, 0] < 0, axis=1)
    return x, done


def training_targets(done, step=0.02):
    """Finite observed-graph ranking labels, NOT predicted arrival times.

    Censored endpoints retain positive residual budget. Time is used only
    to construct training constraints/auxiliary labels, never as V input.
    """
    y = np.zeros(done.shape, dtype=np.float64)
    for i in range(len(done)):
        indices = np.flatnonzero(done[i])
        end = int(indices[0]) if indices.size else done.shape[1]-1
        for t in range(end+1):
            if not done[i, t]:
                y[i, t] = step*(end-t+1)
    return y


def fit_scaler(x):
    flat = x.reshape(-1, x.shape[-1])
    return flat.mean(0), np.maximum(flat.std(0), 0.05)


def normalize(x, mean, scale):
    if x.shape[-1] != len(mean) or len(mean) != len(scale):
        raise ValueError('Feature schema differs from fitted preprocessing')
    return (x-mean)/scale


def kernel_predict(x, model, chunk=512):
    shape = x.shape[:-1]
    z = normalize(x, model['mean'], model['scale']).reshape(-1, x.shape[-1])
    values = []
    for start in range(0, len(z), chunk):
        distance2 = cdist(z[start:start+chunk], model['centers'], 'sqeuclidean')
        kernel = np.exp(-distance2/(2*float(model['bandwidth'])**2))
        values.append(kernel @ model['alpha'])
    # Positive waiting branch; q=done is applied identically in audit_values.
    return np.maximum(float(model['floor']), np.concatenate(values)).reshape(shape)


def binomial_rate(k, n, gamma=0.05):
    return dict(successes=int(k), trials=int(n), rate=float(k/n) if n else None,
                cp_lower=float(beta.ppf(gamma, k, n-k+1)) if k else 0.0,
                confidence=1-gamma)


def audit_values(waiting_values, done, eta=0.01):
    if waiting_values.shape != done.shape or not np.isfinite(waiting_values).all():
        raise ValueError('Invalid prediction shape or nonfinite values')
    v = np.where(done, 0.0, waiting_values)
    if np.any(v < 0):
        raise ValueError('Negative certificate values')
    # Exactly H edges from H+1 observed states. No synthetic edge.
    delta = v[:, :-1]-v[:, 1:]
    p2 = ~done[:, :-1]
    p1 = ~p2
    bad2 = p2 & (delta < eta)
    bad1 = p1 & (delta < 0)
    path_pass = ~(bad1 | bad2).any(1)
    zfree = done & (v < eta)
    event = path_pass & zfree[:, -1]
    closure_sources = zfree[:, :-1]
    closure_seen = closure_sources.any(1)
    closure_bad = (closure_sources & (~zfree[:, 1:] | bad1 | bad2)).any(1)
    return dict(
        n_paths=len(done), states_per_path=done.shape[1], transitions_per_path=done.shape[1]-1,
        p1=dict(violations=int(bad1.sum()), checked=int(p1.sum())),
        p2=dict(violations=int(bad2.sum()), checked=int(p2.sum())),
        p1p2_paths=binomial_rate(path_pass.sum(), len(done)),
        goal_reached=binomial_rate(done[:, -1].sum(), len(done)),
        certificate_event=binomial_rate(event.sum(), len(done)),
        initial_goal_paths=int(done[:, 0].sum()),
        certificate_successes_initial_goal=int((event & done[:, 0]).sum()),
        certificate_successes_after_start=int((event & ~done[:, 0]).sum()),
        sampled_closure=binomial_rate((closure_seen & ~closure_bad).sum(), closure_seen.sum()),
        v_min=float(v.min()), v_max=float(v.max()),
        min_p2_margin=float(delta[p2].min()) if p2.any() else None,
        per_path_event=event.tolist())
