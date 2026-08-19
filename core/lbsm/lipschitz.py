"""
core/lbsm/lipschitz.py

Spectral-norm-based Lipschitz constant computation.

Why this exists now (Stage 2), not Stage 3
--------------------------------------------
The Stage-0 investigation flagged this file as "a Theorem D.5 white-box
envelope prerequisite" and filed it under Stage 3. That placement was too
late: Theorem D.3's validation-complexity correction,

    eps_eff = eps - eta_conc - (L + L_phi) * Delta

needs TWO different Lipschitz constants, not one:

  - L      : the post-expectation map's Lipschitz envelope contributed by the
             DYNAMICS (Theorem D.5's white-box computation from a trained
             transition network's weights). Genuinely Stage-3 work -- Stage 2
             still trains against the Stage-1 analytic kernel z'=gamma*z+xi,
             for which L=gamma EXACTLY (no weight-based computation needed;
             see core/lbsm/verifier.py::verify_trained_recurrence_warrant()).
  - L_phi  : the certificate's OWN design Lipschitz constant (L_V or L_U --
             Theorem D.3: "W and U are LV- and LU-Lipschitz by design"). This
             is needed as soon as W/U are trained NEURAL NETWORKS with their
             own weights, which is exactly what Stage 2 introduces. THIS is
             what this module computes.

So: this file computes L_phi (small, simple MLP, no RSSM/recurrence, no
dynamics involved) now, in Stage 2. Stage 3 will reuse the same
spectral_norm_bound() primitive for the harder L computation, but that is a
separate, harder problem (a trained world model's transition head, not a
certificate network) -- not attempted here.

Kept independent of core/lbsm/trainer.py's training loop, per the Stage-0
review note: this is a pure numerical utility, not coupled to any specific
network's training code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only if torch is absent
    torch = None
    nn = None


# Known Lipschitz constants for the activations this project's LBSM networks
# use. Exact (not approximated): these are the textbook max-derivative
# values, not fit or estimated from data.
ACTIVATION_LIPSCHITZ = {
    "relu": 1.0,
    "sigmoid": 0.25,
    "softplus": 1.0,
    "tanh": 1.0,
    "identity": 1.0,
}


def spectral_norm_bound(weight) -> float:
    """
    Exact spectral norm (largest singular value) of a weight matrix, via SVD.

    Exact rather than power-iteration-approximated: the networks this module
    targets (small MLPs, a few hundred hidden units at most) make a full SVD
    cheap and exact, so there is no accuracy/cost tradeoff to make here.
    Power iteration would be the natural fallback for weight matrices too
    large for a full SVD (e.g. a large transition-network head in Stage 3) --
    documented here as the reason it is NOT implemented in this file: writing
    an approximate method now, for a case this project does not yet exercise,
    would be untested code.
    """
    if torch is not None and isinstance(weight, torch.Tensor):
        return float(torch.linalg.matrix_norm(weight.detach(), ord=2))
    import numpy as np

    return float(np.linalg.svd(weight, compute_uv=False)[0])


def mlp_lipschitz_bound(layer_weights: list, activations: list[str]) -> float:
    """
    Theorem D.5-style composition bound for a feedforward network:

        L <= product_i ( ||W_i||_2 * Lip(activation_i) )

    `activations[i]` is the activation applied AFTER layer i's weight matrix
    (use "identity" for a layer with no following activation -- e.g. a final
    linear layer, before any separate output-bounding transform the caller
    applies afterward; see lbsm_net_design_lipschitz() below, which adds that
    factor explicitly for this project's actual W/U heads).
    """
    if len(layer_weights) != len(activations):
        raise ValueError("layer_weights and activations must have the same length")
    bound = 1.0
    for w, act in zip(layer_weights, activations):
        if act not in ACTIVATION_LIPSCHITZ:
            raise ValueError(f"unknown activation Lipschitz constant for {act!r}")
        bound *= spectral_norm_bound(w) * ACTIVATION_LIPSCHITZ[act]
    return bound


def layernorm_lipschitz_bound(
    gamma,
    sigma_min: float,
    epsilon: float = 1e-5,
    safety_factor: float = 1.0,
) -> float:
    """
    Lipschitz upper bound for a LayerNorm layer y = gamma * (x-mu(x))/sqrt(sigma^2(x)+eps) + beta,
    GIVEN a declared floor sigma_min on the per-sample pre-normalization std
    (not a fixed constant Theorem D.5 supplies -- this is the Stage-4 patch
    the paper itself doesn't have; see the module-level note below and
    measure_layernorm_variance_tdmpc2.py for how sigma_min is chosen and
    checked against real data before this function is trusted).

    Derivation (why gamma_max_abs / sigma_min, not something looser or
    tighter): write LayerNorm as two stages.

    1. Centering, x -> Px where P = I - (1/n) 1 1^T is an orthogonal
       projection (operator norm exactly 1).
    2. Normalize-and-scale: with sigma^2 = ||Px||^2 / n (population variance,
       matching PyTorch's convention) and epsilon negligible,
       (Px)/sqrt(sigma^2) = sqrt(n) * (Px / ||Px||) -- i.e. exactly sqrt(n)
       times the sphere-normalize map u -> u/||u||. That map's Jacobian at a
       point of norm r is (I - u_hat u_hat^T)/r, an orthogonal-projection
       scaled by 1/r, so its OPERATOR NORM IS EXACTLY 1/r (not a padded
       constant) -- and since ||Px|| = sqrt(n)*sigma, the sqrt(n) factors
       from the two stages cancel exactly, leaving Lipschitz(normalize) = 1/sigma.

    Composing with the diagonal gamma-scale (operator norm = max_i|gamma_i|,
    NOT the Euclidean norm ||gamma||_2, since diag(gamma) is a diagonal
    linear map) gives max|gamma_i| / sigma as the bound once sigma >= sigma_min.

    Caveat (why safety_factor exists, default 1.0 i.e. off): the derivation
    above is a TIGHT LOCAL bound (exact Jacobian operator norm at each
    point). Turning a pointwise Jacobian bound into a GLOBAL Lipschitz
    constant over the region {sigma >= sigma_min} formally needs that region
    to be geodesically well-behaved (e.g. convex, so the mean-value theorem
    applies along straight lines) -- {sigma >= sigma_min} is the exterior of
    a ball, which is NOT convex, so this module has not carried out a fully
    rigorous global proof. safety_factor is a caller-supplied multiplicative
    margin for hedging that gap; left at 1.0 (no hedge) by default because
    Stage 3/4's headline feasibility conclusion is off by 9+ orders of
    magnitude, making a 2x-10x constant-factor uncertainty here practically
    immaterial -- this is flagged honestly, not silently assumed away.
    """
    if sigma_min <= 0:
        raise ValueError("sigma_min must be positive")
    gamma_inf_norm = float(_abs_max(gamma))
    return safety_factor * gamma_inf_norm / math.sqrt(sigma_min ** 2 + epsilon)


def _abs_max(gamma):
    if torch is not None and isinstance(gamma, torch.Tensor):
        return gamma.detach().abs().max().item()
    import numpy as np

    return np.abs(np.asarray(gamma)).max()


@dataclass(frozen=True)
class VarianceFloorResult:
    candidate_sigma_min_sq: float
    percentile_value: float
    percentile: float
    discount: float
    observed_min: float
    observed_max: float
    n_samples: int
    n_below_candidate: int
    validated: bool


def choose_and_validate_variance_floor(
    observed_variances,
    percentile: float = 1.0,
    discount: float = 0.5,
) -> VarianceFloorResult:
    """
    Stage-4 recipe: given an array of REAL, MEASURED per-sample pre-LayerNorm
    variances (see measure_layernorm_variance_tdmpc2.py for how these are
    obtained -- real rollouts, real forward passes, not synthetic data),
    pick a candidate sigma_min^2 from the low tail of the empirical
    distribution, discounted further for conservatism, and then EXPLICITLY
    CHECK whether the true observed minimum in the SAME sample still clears
    that candidate.

    This does NOT claim the check is a proof that sigma_min holds for all of
    Z (an unmeasured input could always have smaller variance) -- it only
    reports whether the specific measured sample is consistent with the
    chosen floor, honestly, including reporting FALSE when it is not.
    """
    import numpy as np

    arr = np.asarray(observed_variances, dtype=float)
    if arr.size == 0:
        raise ValueError("observed_variances is empty")
    p = float(np.percentile(arr, percentile))
    candidate = p * discount
    observed_min = float(arr.min())
    validated = observed_min >= candidate
    return VarianceFloorResult(
        candidate_sigma_min_sq=candidate,
        percentile_value=p,
        percentile=percentile,
        discount=discount,
        observed_min=observed_min,
        observed_max=float(arr.max()),
        n_samples=int(arr.size),
        n_below_candidate=int((arr < candidate).sum()),
        validated=validated,
    )


def lbsm_net_design_lipschitz(net, B: float) -> float:
    """
    Design Lipschitz constant L_phi for this project's LBSMNet architecture
    (core/lbsm/trainer.py::LBSMNet): Linear -> ReLU -> Linear -> ReLU ->
    Linear -> (B * sigmoid) applied in forward(), not as an nn.Sequential
    member. The output-bounding B*sigmoid contributes an extra factor of
    B * 0.25 (sigmoid's own Lipschitz constant) on top of the MLP body's
    composition bound.
    """
    if nn is None:
        raise RuntimeError("lbsm_net_design_lipschitz() requires torch")
    weights = [layer.weight for layer in net.mlp if isinstance(layer, nn.Linear)]
    if not weights:
        raise ValueError("net.mlp contains no nn.Linear layers")
    # One activation follows every linear layer except the last (which feeds
    # directly into the B*sigmoid output transform, accounted for below).
    activations = ["relu"] * (len(weights) - 1) + ["identity"]
    body_bound = mlp_lipschitz_bound(weights, activations)
    return body_bound * B * ACTIVATION_LIPSCHITZ["sigmoid"]
