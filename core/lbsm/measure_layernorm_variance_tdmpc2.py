"""
core/lbsm/measure_layernorm_variance_tdmpc2.py

Stage-4 diagnostic (not part of the core reusable core/lbsm API, not exported
via __init__.py -- a one-off measurement script, matching this project's
existing convention for investigation scripts like
tdmpc2_anchor_distribution_check.py referenced in wrappers/tdmpc2_wrapper.py).

Loads the REAL TD-MPC2 walker-walk checkpoint via wrappers/tdmpc2_wrapper.py
(the same load path used everywhere else in this project -- no separate,
untested loading logic), runs REAL MPC-planned rollouts, and hooks each of
the 3 `_dynamics` NormedLinear layers' `ln` (nn.LayerNorm) submodules to
capture the actual pre-normalization activation on every real forward call.
This is the "先测出真实分布" step the Stage-4 task requires -- sigma_min must
come from real data, not an assumption.

Run: conda activate dyno && python -m core.lbsm.measure_layernorm_variance_tdmpc2
"""

from __future__ import annotations

import numpy as np

from configs.settings import RolloutConfig
from core.lbsm.lipschitz import choose_and_validate_variance_floor, layernorm_lipschitz_bound
from wrappers.tdmpc2_wrapper import TDMPC2Wrapper


def measure(
    n_rollouts: int = 5, horizon: int = 30, seed: int = 0,
) -> tuple[dict[int, np.ndarray], list["torch.Tensor"]]:
    """Returns ({layer_index: np.ndarray of per-sample pre-LN variances}, [gamma per layer])."""
    w = TDMPC2Wrapper()
    w.load()

    dynamics = w._agent.model._dynamics  # nn.Sequential of 3 NormedLinear
    gammas = [normed_linear.ln.weight.detach().clone() for normed_linear in dynamics]
    captured: dict[int, list[float]] = {0: [], 1: [], 2: []}

    def make_hook(layer_idx: int):
        def hook(module, inputs):
            x = inputs[0]  # pre-LayerNorm activation, shape (batch, features)
            # Population variance over the feature dim, matching PyTorch
            # LayerNorm's own (unbiased=False) convention exactly.
            var = x.detach().var(dim=-1, unbiased=False)
            captured[layer_idx].extend(var.cpu().numpy().tolist())
        return hook

    handles = []
    for i, normed_linear in enumerate(dynamics):
        handles.append(normed_linear.ln.register_forward_pre_hook(make_hook(i)))

    try:
        cfg = RolloutConfig(horizon=horizon, n_rollouts=n_rollouts, seed=seed)
        w.sample_latent_rollouts(cfg, action_source="mpc_plan")
    finally:
        for h in handles:
            h.remove()
        w.close()

    return {i: np.array(v) for i, v in captured.items()}, gammas


def main(seeds: list[int] = (0, 1, 7, 42), discount: float = 0.4):
    """
    The validated Stage-4 procedure. A single seed=0 run (discount=0.5) was
    tried FIRST and is worth recording honestly: layer 2 FAILED validation
    (4 of 12,849,318 samples fell below the candidate floor) -- exactly the
    "explicit check must be able to fail" case the task specified, not a
    hypothetical. Investigating further (a second seed=0 rerun already gave a
    different observed minimum -- CEM's internal sampling is not fully
    reproducible across process invocations even at "the same" seed -- and a
    seed=7 rerun) showed the true minimum is stable in the 34-42 range across
    independent seeds, not a runaway/unbounded tail, so a MORE conservative
    discount (0.4, applied uniformly across all 3 layers, on data POOLED
    across multiple seeds rather than trusting any single run) does validate
    robustly. That is what this function runs and reports -- not the first
    (failed) attempt dressed up as if it had worked.
    """
    print(f"Loading real TD-MPC2 walker-walk-3.pt and pooling real MPC-rollout "
          f"pre-LayerNorm variance data across seeds {list(seeds)}...")

    pooled: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: []}
    gammas = None
    for seed in seeds:
        variances, g = measure(seed=seed)
        gammas = g  # weights are seed-independent; keep the last load's tensors
        for i in range(3):
            pooled[i].append(variances[i])
        print(f"  seed={seed}: " + ", ".join(f"L{i} min={variances[i].min():.3f}" for i in range(3)))

    results = {}
    print(f"\n=== Pooled result, discount={discount} ===")
    for layer_idx in range(3):
        combined = np.concatenate(pooled[layer_idx])
        r = choose_and_validate_variance_floor(combined, percentile=1.0, discount=discount)
        print(f"\n_dynamics.{layer_idx}.ln  (n={len(combined)} pooled samples)")
        print(f"  true_min={r.observed_min:.4f}  p1={r.percentile_value:.4f}  "
              f"candidate_sigma_min_sq={r.candidate_sigma_min_sq:.4f}")
        print(f"  VALIDATED={r.validated}  n_below_candidate={r.n_below_candidate}/{r.n_samples}")
        if not r.validated:
            print("  STOPPING: envelope argument does not hold for this layer at this discount -- "
                  "do not proceed to compute a Lipschitz bound from an invalidated floor.")
            results[layer_idx] = {"validated": False, "result": r}
            continue
        sigma_min = r.candidate_sigma_min_sq ** 0.5
        gamma_max = gammas[layer_idx].abs().max().item()
        bound = layernorm_lipschitz_bound(gammas[layer_idx], sigma_min=sigma_min)
        print(f"  sigma_min={sigma_min:.4f}  gamma_max_abs={gamma_max:.4f}  "
              f"layernorm_lipschitz_bound={bound:.3f}")
        results[layer_idx] = {"validated": True, "sigma_min": sigma_min, "gamma_max_abs": gamma_max,
                               "ln_bound": bound, "result": r}

    return results


if __name__ == "__main__":
    main()
