"""
tests/test_lbsm_trained.py

Stage-2 acceptance tests for core/lbsm/trainer.py + core/lbsm/verifier.py's
Theorem D.3 sample-based path. Kept separate from tests/test_lbsm_analytic.py
(Stage 1, exact/closed-form -- untouched by this file).

Per the task corrections for Stage 2:
  1. The Theorem D.3 eps_eff = eps - eta_conc - (L+L_phi)*Delta correction is
     mandatory here, not the eta_conc=Delta=0 shortcut Stage 1's exact
     construction legitimately used. test_epsilon_eff_correction_catches_...
     below is the assertion-based (not documentation-based) proof this is
     actually enforced.
  2. core/lbsm/trainer.py::fit_lbsm()'s train/val anchor split is MANDATORY
     (raises), stricter than L2's opt-in warn_if_overlap.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from core.lbsm.trainer import fit_lbsm
from core.lbsm.verifier import (
    epsilon_effective,
    generate_annulus_anchors,
    hoeffding_eta_conc,
    is_anchor_certified,
    verify_trained_recurrence_warrant,
)

# Small, fast-training parameterization -- these tests validate MECHANICS
# (does the pipeline run and enforce what it should), not paper-matching
# numerics. Same E.2-style geometry as tests/test_lbsm_analytic.py.
GAMMA = 0.5
SIGMA = 0.1
D = 2
K = 3
R_F = 0.30
ELL = 10.0
B = ELL


def _make_anchors(seed: int, n_radial: int, n_angular: int) -> list[np.ndarray]:
    b = K * SIGMA * (D ** 0.5)
    r_inv = b / (1 - GAMMA)
    inner = generate_annulus_anchors(R_F, r_inv, n_radial, n_angular)
    outer = generate_annulus_anchors(r_inv, ELL ** 0.5, n_radial, n_angular)
    return inner + outer


# ─── Mandatory disjoint split ───────────────────────────────────────────────

def test_mandatory_disjoint_split_raises_on_overlap():
    anchors = _make_anchors(seed=0, n_radial=2, n_angular=4)
    train = anchors[:5]
    val = [anchors[2]]  # object-identical to one of train's entries

    with pytest.raises(ValueError, match="shared between train_anchors and val_anchors"):
        fit_lbsm(
            train_anchors=train, val_anchors=val, seed=0,
            r_F=R_F, ell=ELL, gamma=GAMMA, sigma=SIGMA, d=D, k=K, B=B,
            n_epochs=1,
        )


def test_disjoint_split_with_no_overlap_does_not_raise():
    anchors = _make_anchors(seed=0, n_radial=2, n_angular=4)
    train, val = anchors[:5], anchors[5:8]
    # Should run without raising (n_epochs=1 for speed -- this test is about
    # the guard, not training quality).
    result = fit_lbsm(
        train_anchors=train, val_anchors=val, seed=0,
        r_F=R_F, ell=ELL, gamma=GAMMA, sigma=SIGMA, d=D, k=K, B=B,
        n_epochs=1,
    )
    assert result.n_train_anchors == 5
    assert result.n_val_anchors == 3


def test_seed_is_required_no_default():
    """fit_lbsm's `seed` parameter must have no default -- inspect the live
    signature rather than just trying a call, so a future edit that adds a
    default is caught even if every existing call site still passes seed
    explicitly."""
    sig = inspect.signature(fit_lbsm)
    assert sig.parameters["seed"].default is inspect.Parameter.empty, (
        "fit_lbsm()'s seed parameter must remain required (no default) -- see "
        "module docstring re: the L2 walker-walk asymmetric-seeding gap this "
        "is deliberately not repeating."
    )


# ─── Theorem D.3's eps_eff correction: mandatory, and provably load-bearing ─

def test_epsilon_eff_correction_catches_false_positive_naive_check_misses():
    """
    Concrete numeric scenario where a NAIVE check (raw margin minus only the
    statistical concentration term, ignoring the Lipschitz-lift term) would
    certify an anchor, while the correct eps_eff = eps - eta_conc -
    (L+L_phi)*Delta -- exercised via the REAL is_anchor_certified() used by
    verify_trained_recurrence_warrant() -- correctly refuses it.

    Numbers: kappa=2000, B_bar=1.0, delta'=0.05 -> eta_conc ~= 0.0304.
    observed_margin=0.05 -> naive check: 0.05 - 0.0304 = 0.0196 > 0 (would
    certify). With L=0.5 (a plausible dynamics contraction constant) and
    L_phi=2.0 (a plausible under-trained network's design Lipschitz
    constant) at mesh_delta=0.02: (L+L_phi)*Delta = 0.05, so
    eps_eff = 0.0196 - 0.05 = -0.0304 < 0 -- correctly NOT certified.
    """
    kappa, B_bar, delta_prime = 2000, 1.0, 0.05
    observed_margin = 0.05
    L, L_phi, mesh_delta = 0.5, 2.0, 0.02

    eta_conc = hoeffding_eta_conc(kappa, B_bar, delta_prime)
    assert eta_conc == pytest.approx(0.03037, abs=1e-4)

    naive_margin = observed_margin - eta_conc
    assert naive_margin > 0, "test setup error: naive check should APPEAR to pass here"

    eps_eff = epsilon_effective(observed_margin, eta_conc, L, L_phi, mesh_delta)
    assert eps_eff < 0, "test setup error: full correction should REFUSE here"

    # The function the real verifier actually calls must agree with eps_eff,
    # not with the naive margin -- this is the assertion that would fail if
    # a future edit accidentally reverted to the naive check.
    assert is_anchor_certified(observed_margin, eta_conc, L, L_phi, mesh_delta) is False
    assert (naive_margin > 0) != is_anchor_certified(observed_margin, eta_conc, L, L_phi, mesh_delta), (
        "naive and correct checks must diverge on this scenario, or the test is vacuous"
    )


def test_epsilon_eff_correction_certifies_when_margin_genuinely_survives():
    """Sanity check in the other direction: a comfortably large observed
    margin, tiny Delta, small L_phi -- eps_eff should stay positive and
    is_anchor_certified() should return True."""
    eta_conc = hoeffding_eta_conc(kappa=2000, B_bar=1.0, delta_prime=0.05)
    eps_eff = epsilon_effective(epsilon=0.5, eta_conc=eta_conc, L=0.5, L_phi=0.5, mesh_delta=0.01)
    assert eps_eff > 0
    assert is_anchor_certified(observed_margin=0.5, eta_conc=eta_conc, L=0.5, L_phi=0.5, mesh_delta=0.01) is True


# ─── L_core prevents degenerate boundary collapse ──────────────────────────

def test_l_core_shapes_boundary_measurably_vs_omitting_it():
    """
    Train twice (same seed, same anchors, same epoch budget), once with
    L_core enabled and once disabled (use_l_core=False, a lever that exists
    ONLY for this test -- see trainer.py's docstring). L_core's boundary term
    explicitly pushes U(boundary-region points) upward toward ell+m; nothing
    else in the loss does. So U evaluated on a fixed held-out boundary-region
    probe set should come out measurably higher WITH L_core than without it.
    """
    anchors = _make_anchors(seed=1, n_radial=3, n_angular=6)
    train, val = anchors[:20], anchors[20:26]

    common_kwargs = dict(
        train_anchors=train, val_anchors=val, seed=42,
        r_F=R_F, ell=ELL, gamma=GAMMA, sigma=SIGMA, d=D, k=K, B=B,
        hidden_dim=32, n_epochs=150, kappa_train=4,
    )
    result_with_core = fit_lbsm(use_l_core=True, **common_kwargs)
    result_without_core = fit_lbsm(use_l_core=False, **common_kwargs)

    # Fixed, independent boundary-region probe set (not seen during training).
    probe_rng = np.random.default_rng(999)
    probe = []
    for _ in range(40):
        direction = probe_rng.normal(size=D)
        direction /= np.linalg.norm(direction)
        probe.append(direction * (ELL ** 0.5))
    probe_arr = np.stack(probe)

    import torch
    with torch.no_grad():
        probe_t = torch.as_tensor(probe_arr, dtype=torch.float32)
        u_with_core = result_with_core.U_net(probe_t).mean().item()
        u_without_core = result_without_core.U_net(probe_t).mean().item()

    assert u_with_core > u_without_core + 1.0, (
        f"U(boundary) with L_core ({u_with_core:.3f}) is not measurably higher than "
        f"without it ({u_without_core:.3f}) -- L_core's boundary-shaping term should "
        "produce a clear, not marginal, difference here."
    )


# ─── End-to-end: train then verify via the Theorem D.3 sample-based path ───

def test_trained_network_end_to_end_verification_runs_and_is_self_consistent():
    """
    Not a paper-number-matching test (Stage 2 has none to match -- the task
    spec is explicit that the trained warrant need only be "close to, not
    equal to" Stage 1's 0.92, and mainly that the verification MECHANICS run
    and self-report honestly). Trains a small network briefly, then runs the
    full sample-based Theorem D.3 verifier end to end, and checks internal
    consistency of whatever comes out (rather than asserting a specific
    warrant value, which would be flaky against training-convergence noise).
    """
    anchors = _make_anchors(seed=2, n_radial=3, n_angular=6)
    train, val = anchors[:20], anchors[20:26]

    training = fit_lbsm(
        train_anchors=train, val_anchors=val, seed=7,
        r_F=R_F, ell=ELL, gamma=GAMMA, sigma=SIGMA, d=D, k=K, B=B,
        hidden_dim=32, n_epochs=300, kappa_train=4,
    )
    assert len(training.loss_history) == 300
    assert all(np.isfinite(training.loss_history))

    rng = np.random.default_rng(123)
    result = verify_trained_recurrence_warrant(
        W_net=training.W_net, U_net=training.U_net, B=B,
        gamma=GAMMA, sigma=SIGMA, d=D, k=K, r_F=R_F, ell=ELL, z0_norm_sq=0.8,
        eps_tr=0.02, eps_U=0.02, kappa=200, delta_prime=0.05,
        n_radial=3, n_angular=6, rng=rng,
    )

    assert result.n_anchors == result.n_certified + result.n_failed
    assert result.n_anchors > 0
    assert result.eta_conc > 0
    assert result.mesh_delta > 0
    assert result.L == pytest.approx(GAMMA)
    assert result.L_V > 0 and result.L_U > 0

    if result.all_anchors_certified:
        assert result.warrant is not None
        assert 0.0 < result.warrant <= 1.0
        assert result.alpha_collar == 0.0
    else:
        assert result.warrant is None
        assert result.n_failed > 0
        assert len(result.failed_anchor_examples) > 0

    # Reported, not asserted to a specific value -- see docstring.
    print(
        f"\n[Stage-2 cross-check] certified={result.all_anchors_certified} "
        f"n_certified={result.n_certified}/{result.n_anchors} "
        f"mesh_delta={result.mesh_delta:.5f} L_V={result.L_V:.3f} L_U={result.L_U:.3f} "
        f"warrant={result.warrant}"
    )
