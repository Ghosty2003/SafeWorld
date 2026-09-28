from __future__ import annotations

import warnings
from typing import Any

from .config import DEFAULT_LPPM_CONFIG
from .loss import heuristic_epoch_loss, p1_loss, p2_loss, smoothness_penalty
from .model import NeuralLPPM, infer_feature_keys
from .verifier import find_trajectory_overlap, iter_transitions, run_product_trajectory

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


def fit_lppm(
    trajectories: list[list[dict[str, float]]],
    dpa,
    spec: dict,
    eta: float = DEFAULT_LPPM_CONFIG.eta,
    n_epochs: int = 300,
    lr: float = 1e-3,
    lambda_reg: float = 0.01,
    calib_trajectories: list[list[dict[str, float]]] | None = None,
    allow_overlap: bool = False,
    overlap_reason: str | None = None,
) -> dict[str, Any]:
    """
    calib_trajectories : opt-in disjointness check. Pass the calibration
      split here to verify it shares no trajectory object with `trajectories`
      -- Theorem 5.4's PAC bound assumes the calibration draw is exchangeable
      with, and disjoint from, whatever fit V_phi. Off by default (None): no
      check, no behavior change for callers that don't pass it. See
      EXPERIMENT_CONFIG.md §8.8/§8.12 for why this was added (main.py::
      verify()'s Step 3 was found to reuse one trajectory list for both fit
      and calibrate).

    allow_overlap, overlap_reason : External-review fix -- overlap detected
      when calib_trajectories is passed now RAISES ValueError by default
      (previously only warned, despite main.py's own comment at its one real
      call site incorrectly claiming this "fails loudly"). Pass
      allow_overlap=True with a non-empty overlap_reason to proceed anyway
      (e.g. a deliberate ablation) -- this is a forced, auditable opt-out:
      omitting overlap_reason while allow_overlap=True is itself an error.
    """
    if calib_trajectories is not None:
        n_overlap = find_trajectory_overlap(trajectories, calib_trajectories)
        if n_overlap:
            if not allow_overlap:
                raise ValueError(
                    f"fit_lppm(): {n_overlap} trajectory object(s) also present in "
                    "calib_trajectories -- the calibration split is not disjoint "
                    "from the training split, which breaks the exchangeability "
                    "premise Theorem 5.4's p_hat_gamma relies on. Pass "
                    "allow_overlap=True with a non-empty overlap_reason to proceed "
                    "anyway (e.g. a deliberate ablation)."
                )
            if not overlap_reason:
                raise ValueError(
                    "fit_lppm(): allow_overlap=True requires a non-empty "
                    "overlap_reason explaining why this violation is intentional "
                    "-- silent overrides are not permitted."
                )
            warnings.warn(
                f"fit_lppm(): {n_overlap} trajectory object(s) also present in "
                f"calib_trajectories, allowed via allow_overlap=True: {overlap_reason}",
                stacklevel=2,
            )

    odd_prios = dpa.odd_priorities
    all_transitions = []
    for traj in trajectories:
        path = run_product_trajectory(traj, dpa, spec)
        # External-review fix: see iter_transitions()'s docstring -- this
        # previously zipped adjacent path entries directly, silently
        # dropping the FINAL transition (on the last observed AP) of every
        # trajectory from training data entirely (the same bug fixed in
        # check_pathwise_conditions()/verify_zfree_closure() -- training was
        # blind to it too, for every trajectory ever fit).
        all_transitions.extend(iter_transitions(path, dpa))

    if not all_transitions:
        return {
            "loss_history": [],
            "final_loss": 0.0,
            "n_transitions": 0,
            "epochs_trained": 0,
            "odd_priorities": odd_prios,
            "spec_id": spec.get("id", ""),
            "weights": None,
            "backend": "heuristic",
            "reason": "No transitions available for LPPM fitting.",
        }

    if torch is None or NeuralLPPM is None or F is None:
        return _fit_lppm_heuristic_fallback(all_transitions, dpa, spec, eta, n_epochs)

    feature_keys = infer_feature_keys(trajectories)
    state_to_idx = {state: idx for idx, state in enumerate(dpa.states)}
    odd_to_idx = {prio: idx for idx, prio in enumerate(odd_prios)}

    z_curr = torch.tensor(
        [[float(curr.z.get(key, 0.0)) for key in feature_keys] for curr, _ in all_transitions],
        dtype=torch.float32,
    )
    z_next = torch.tensor(
        [[float(nxt.z.get(key, 0.0)) for key in feature_keys] for _, nxt in all_transitions],
        dtype=torch.float32,
    )
    q_curr = torch.tensor([state_to_idx[curr.q] for curr, _ in all_transitions], dtype=torch.long)
    q_next = torch.tensor([state_to_idx[nxt.q] for _, nxt in all_transitions], dtype=torch.long)
    curr_priorities = torch.tensor([curr.priority for curr, _ in all_transitions], dtype=torch.long)

    hidden_dim = int(spec.get("lppm_hidden_dim", 128))
    q_embed_dim = int(spec.get("lppm_q_embed_dim", 16))
    device = torch.device("cpu")
    model = NeuralLPPM(
        latent_dim=len(feature_keys),
        n_states=len(dpa.states),
        n_heads=max(len(odd_prios), 1),
        hidden_dim=hidden_dim,
        q_embed_dim=q_embed_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    z_next = z_next.to(device)
    q_curr = q_curr.to(device)
    q_next = q_next.to(device)
    curr_priorities = curr_priorities.to(device)
    loss_history: list[float] = []

    for _epoch in range(n_epochs):
        optimizer.zero_grad()
        z_curr_epoch = z_curr.clone().to(device).requires_grad_(True)
        v_curr = model(z_curr_epoch, q_curr)
        v_next = model(z_next, q_next)
        loss = p1_loss(v_curr, v_next, curr_priorities, odd_to_idx)
        loss = loss + p2_loss(v_curr, v_next, curr_priorities, odd_to_idx, eta)
        loss = loss + lambda_reg * smoothness_penalty(v_curr, z_curr_epoch)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        loss_history.append(loss_value)
        if loss_value < 1e-6:
            break

    # A small training loss is not by itself evidence that a useful ranking
    # function was learned.  In particular, P1 admits every constant
    # function, and the nonnegative softplus head can saturate at zero.  If
    # P2 examples are present, a candidate whose *entire sampled value range*
    # is narrower than eta cannot possibly realize even one eta-sized drop.
    # Record an explicit post-fit audit so downstream calibration cannot
    # mistake this degenerate numerical solution for a meaningful V_phi.
    with torch.no_grad():
        v_curr_final = model(z_curr, q_curr)
        v_next_final = model(z_next, q_next)

    p1_checked = p1_violations = 0
    p2_checked = p2_violations = 0
    for odd_priority, head_index in odd_to_idx.items():
        p1_mask = curr_priorities < odd_priority
        if torch.any(p1_mask):
            p1_gaps = v_next_final[p1_mask, head_index] - v_curr_final[p1_mask, head_index]
            p1_checked += int(p1_gaps.numel())
            p1_violations += int((p1_gaps > 0.0).sum().item())

        p2_mask = curr_priorities == odd_priority
        if torch.any(p2_mask):
            p2_margins = v_curr_final[p2_mask, head_index] - v_next_final[p2_mask, head_index]
            p2_checked += int(p2_margins.numel())
            p2_violations += int((p2_margins < eta).sum().item())

    sampled_values = torch.cat((v_curr_final.reshape(-1), v_next_final.reshape(-1)))
    value_min = float(sampled_values.min().cpu().item())
    value_max = float(sampled_values.max().cpu().item())
    value_mean = float(sampled_values.mean().cpu().item())
    value_std = float(sampled_values.std(unbiased=False).cpu().item())
    value_span = value_max - value_min
    collapse_detected = bool(
        p2_checked > 0
        and p2_violations == p2_checked
        and value_span + 1e-12 < eta
    )
    if collapse_detected:
        training_status = "collapsed_invalid"
        failure_reason = (
            "All sampled P2 checks fail and the fitted V_phi value span is smaller "
            "than eta, so the candidate cannot realize one required P2 descent."
        )
    elif p1_violations or p2_violations:
        training_status = "residual_violations"
        failure_reason = (
            "The fitted candidate retains sampled P1/P2 violations; held-out "
            "validation is required and no deductive claim is available."
        )
    elif p2_checked == 0:
        training_status = "no_p2_coverage"
        failure_reason = (
            "The training split contains no P2-triggering transition, so it provides "
            "no empirical evidence that the required bad-state descent was learned."
        )
    else:
        training_status = "sampled_fit_pass"
        failure_reason = None

    return {
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "n_transitions": len(all_transitions),
        "epochs_trained": len(loss_history),
        "odd_priorities": odd_prios,
        "spec_id": spec.get("id", ""),
        "weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "backend": "torch_mlp",
        "feature_keys": feature_keys,
        "state_to_idx": state_to_idx,
        "odd_to_idx": odd_to_idx,
        "hidden_dim": hidden_dim,
        "q_embed_dim": q_embed_dim,
        "training_status": training_status,
        "failure_reason": failure_reason,
        "collapse_detected": collapse_detected,
        "candidate_valid_on_training_sample": p1_violations == 0 and p2_violations == 0,
        "train_p1_checked": p1_checked,
        "train_p1_violations": p1_violations,
        "train_p2_checked": p2_checked,
        "train_p2_violations": p2_violations,
        "train_value_min": value_min,
        "train_value_max": value_max,
        "train_value_mean": value_mean,
        "train_value_std": value_std,
        "train_value_span": value_span,
    }


def _fit_lppm_heuristic_fallback(
    all_transitions,
    dpa,
    spec: dict[str, Any],
    eta: float,
    n_epochs: int,
) -> dict[str, Any]:
    odd_prios = dpa.odd_priorities
    loss_history: list[float] = []
    for _epoch in range(n_epochs):
        avg_loss = heuristic_epoch_loss(all_transitions, odd_prios, spec, dpa, eta)
        loss_history.append(avg_loss)
        if avg_loss < 1e-6:
            break
    return {
        "loss_history": loss_history,
        "final_loss": loss_history[-1] if loss_history else 0.0,
        "n_transitions": len(all_transitions),
        "epochs_trained": len(loss_history),
        "odd_priorities": odd_prios,
        "spec_id": spec.get("id", ""),
        "weights": None,
        "backend": "heuristic",
        "reason": "PyTorch unavailable; used heuristic LPPM fallback.",
    }
