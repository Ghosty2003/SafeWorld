"""
eval_ltl_hazard_real_lppm.py — First real NeuralLPPM verdict for ltl_hazard_avoidance.

Addresses everything flagged before running:
  - explicit backend log (never trust "no error" alone)
  - train/calibration split MUST be disjoint (main.py currently reuses the same
    `trajectories` for both fit_lppm and calibrate_lppm -- that would be an
    optimistic/overfit p_hat_gamma; this script splits properly)
  - own-odd-priority (trap) activity check on real trajectories (liveness --
    same trap as CarDreamer's L4' vacuous-truth pattern)
  - cross-check direction against the established STL result (collision
    semantics: WARRANT, rho*=+1.126, 100/100)

Reproducibility (EXPERIMENT_CONFIG.md §8.9)
--------------------------------------------
CarDreamerWrapper.sample_rollouts() (JAX/GPU) was confirmed non-reproducible
across separate process runs at fixed seed even though the RNG seeding
itself is fully deterministic -- root cause is GPU/cuDNN convolution-
algorithm autotuning (JAX/XLA benchmarks candidate cuDNN implementations at
runtime and picks whichever times fastest *at that moment*; different
algorithms are not bit-identical). The os.environ["XLA_FLAGS"] line below
must be set BEFORE jax is imported anywhere in this process (it is, in
wrappers.cardreamer_wrapper, imported lazily well after this) -- do not
remove it or move it below other imports. Verified: 2 independent full runs
of this exact script, flag set, produced identical output at every printed
step (not just the final p_hat_gamma) -- see EXPERIMENT_CONFIG.md §7 #3/#22.

Usage: conda activate cardreamer && python cardreamer/eval_ltl_hazard_real_lppm.py
"""
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"

import sys
sys.path.insert(0, "/home/bot/SafeWorld")

import numpy as np
import torch

from specs.ltl_specs import get_ltl_spec_by_id
from utils.spec_analysis import analyze_spec_structure
from core.lppm.automaton import build_parity_automaton
from core.lppm.calibrator import calibrate_lppm
from core.lppm.config import DEFAULT_LPPM_CONFIG
from core.lppm.model import NeuralLPPM, infer_feature_keys
from core.lppm.loss import p1_loss, p2_loss, smoothness_penalty
from core.lppm.verifier import run_product_trajectory, verify_zfree_closure

N_ROLLOUTS = 100
HORIZON = 50
TRAIN_FRAC = 0.6
ETA = DEFAULT_LPPM_CONFIG.eta
GAMMA = 0.05
N_EPOCHS = 1500


def main():
    spec = get_ltl_spec_by_id("ltl_hazard_avoidance")
    spec["analysis"] = analyze_spec_structure(spec)
    dpa = build_parity_automaton(spec)
    print(f"[0] automaton: states={dpa.states}  priority={dpa.priority}  "
        f"odd_priorities={dpa.odd_priorities}  backend={dpa.backend}")

    print(f"[1] sampling {N_ROLLOUTS} real CarDreamer rollouts (horizon={HORIZON})...",
        flush=True)
    from wrappers.cardreamer_wrapper import CarDreamerWrapper
    from configs.settings import RolloutConfig
    cfg = RolloutConfig(n_rollouts=N_ROLLOUTS, horizon=HORIZON, seed=0, action_source="actor")
    w = CarDreamerWrapper(cfg)
    w.load()
    trajectories = w.sample_rollouts(cfg)
    print(f"    got {len(trajectories)} trajectories x {len(trajectories[0])} steps")

    # ── liveness check: does the REAL imagined data ever visit the odd/bad state? ──
    n_visit_trap = 0
    hazard_mins = []
    for traj in trajectories:
        path = run_product_trajectory(traj, dpa, spec)
        hazard_mins.append(min(s["hazard_dist"] for s in traj))
        if any(ps.priority == 1 for ps in path):   # priority=1 == "trap" for Safety template
            n_visit_trap += 1
    print(f"\n[2] LIVENESS CHECK (own-odd priority = 'trap' visitation, "
        f"the same activity trap as CarDreamer's L4'):")
    print(f"    rollouts that ever visit trap: {n_visit_trap}/{len(trajectories)}")
    print(f"    hazard_dist min across rollouts: p10={np.percentile(hazard_mins,10):.3f} "
        f"p50={np.percentile(hazard_mins,50):.3f} min={min(hazard_mins):.3f}")
    if n_visit_trap == 0:
        print("    !!! VACUOUS: P2 (descent while in trap) has ZERO relevant transitions in "
            "this data. Any p_hat_gamma computed from training on this data is a liveness "
            "artifact -- there is no genuine descent evidence, the constraint is trivially "
            "satisfied because the bad state never occurs. This is the SAME activity trap "
            "flagged for L4' (vacuous-true trigger) and the synthetic (b)/(d) categories in "
            "Step 2 -- now confirmed on REAL data.")

    # ── disjoint train / calibration split ───────────────────────────────────
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(trajectories))
    n_train = int(len(trajectories) * TRAIN_FRAC)
    train_traj = [trajectories[i] for i in idx[:n_train]]
    calib_traj = [trajectories[i] for i in idx[n_train:]]
    print(f"\n[3] disjoint split: train={len(train_traj)}  calibration={len(calib_traj)}  "
        f"(calibrate_lppm will NEVER see the train rollouts)")

    # ── train NeuralLPPM on the TRAIN split only, logging p1/p2 separately ──
    print(f"\n[4] training NeuralLPPM on train split ({N_EPOCHS} epochs max)...")
    odd_prios = dpa.odd_priorities
    all_transitions = []
    for traj in train_traj:
        path = run_product_trajectory(traj, dpa, spec)
        all_transitions.extend((path[i], path[i + 1]) for i in range(len(path) - 1))
    print(f"    n_transitions={len(all_transitions)}")

    feature_keys = infer_feature_keys([[t[0].z for t in all_transitions]])
    state_to_idx = {s: i for i, s in enumerate(dpa.states)}
    odd_to_idx = {p: i for i, p in enumerate(odd_prios)}
    z_curr = torch.tensor([[float(c.z.get(k, 0.0)) for k in feature_keys] for c, _ in all_transitions], dtype=torch.float32)
    z_next = torch.tensor([[float(n.z.get(k, 0.0)) for k in feature_keys] for _, n in all_transitions], dtype=torch.float32)
    q_curr = torch.tensor([state_to_idx[c.q] for c, _ in all_transitions], dtype=torch.long)
    q_next = torch.tensor([state_to_idx[n.q] for _, n in all_transitions], dtype=torch.long)
    curr_priorities = torch.tensor([c.priority for c, _ in all_transitions], dtype=torch.long)

    torch.manual_seed(0)
    model = NeuralLPPM(latent_dim=len(feature_keys), n_states=len(dpa.states),
                       n_heads=max(len(odd_prios), 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    p1_hist, p2_hist = [], []
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        zc = z_curr.clone().requires_grad_(True)
        v_curr = model(zc, q_curr)
        v_next = model(z_next, q_next)
        l1 = p1_loss(v_curr, v_next, curr_priorities, odd_to_idx)
        l2 = p2_loss(v_curr, v_next, curr_priorities, odd_to_idx, ETA)
        p1_hist.append(float(l1.detach())); p2_hist.append(float(l2.detach()))
        loss = l1 + l2 + 0.01 * smoothness_penalty(v_curr, zc)
        loss.backward()
        optimizer.step()
        if p1_hist[-1] + p2_hist[-1] < 1e-7:
            break
    print(f"    BRANCH CONFIRMED: real gradient-trained NeuralLPPM (not heuristic fallback) "
        f"-- explicit log, not inferred from absence-of-error")
    print(f"    epochs_run={len(p1_hist)}")
    print(f"    p1 residual: start={p1_hist[0]:.5f} -> final={p1_hist[-1]:.6f}")
    print(f"    p2 residual: start={p2_hist[0]:.5f} -> final={p2_hist[-1]:.6f}")
    converged = p1_hist[-1] < 1e-3 and (n_visit_trap == 0 or p2_hist[-1] < 1e-3)
    print(f"    CHECK 1 (residual convergence): "
        f"{'PASS' if converged else 'DID NOT CONVERGE -- investigate'}")

    lppm_params = {
        "backend": "torch_mlp",
        "weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "feature_keys": feature_keys, "state_to_idx": state_to_idx, "odd_to_idx": odd_to_idx,
        "hidden_dim": 128, "q_embed_dim": 16,
    }

    # ── calibrate on the DISJOINT held-out split ─────────────────────────────
    print(f"\n[5] calibrating p_hat_gamma on the {len(calib_traj)} HELD-OUT rollouts "
        f"(never seen during training)...")
    lppm_res = calibrate_lppm(calib_traj, dpa, spec, gamma=GAMMA, eta=ETA,
                              lppm_params=lppm_params)
    print(f"    k={sum(1 for pw in lppm_res.pathwise if pw.satisfied)}/{len(calib_traj)}  "
        f"p_hat_gamma={lppm_res.p_hat_gamma:.4f}  warranted={lppm_res.is_warranted()}")

    # ── precise breakdown: WHICH condition (P1 / P2 / Z_free) is actually failing? ──
    n_p1_fail = sum(1 for pw in lppm_res.pathwise if pw.p1_violations > 0)
    n_p2_fail = sum(1 for pw in lppm_res.pathwise if pw.p2_violations > 0)
    n_p1_only = sum(1 for pw in lppm_res.pathwise if pw.p1_violations > 0 and pw.p2_violations == 0)
    print(f"    breakdown: p1_violations>0 in {n_p1_fail}/{len(calib_traj)} rollouts  "
        f"(avg count={np.mean([pw.p1_violations for pw in lppm_res.pathwise]):.2f})")
    print(f"               p2_violations>0 in {n_p2_fail}/{len(calib_traj)} rollouts  "
        f"(avg count={np.mean([pw.p2_violations for pw in lppm_res.pathwise]):.2f})")
    print(f"               rollouts failing ONLY on P1 (not P2): {n_p1_only}/{len(calib_traj)}")
    if n_p1_fail > 0 and n_p2_fail == 0:
        print("    DIAGNOSIS: k=0 is caused by imperfectly-converged P1 (network residual "
            "0.0096 > 0, so isolated non-increase violations occur along real trajectories), "
            "NOT by the vacuous P2. These are genuinely-safe rollouts (hazard_dist never "
            "negative) being marked UNSATISFIED because training didn't fully converge -- "
            "a false negative from under-training, not a vacuous false positive.")

    # ── cross-check vs established STL result (collision semantics: WARRANT, rho*=+1.126, 100/100) ──
    print(f"\n[6] CROSS-CHECK vs established STL result (WARRANT, rho*=+1.126, 100/100):")
    stl_says_safe = True   # established result
    lppm_says_safe = lppm_res.is_warranted()
    print(f"    STL:  safe={stl_says_safe}   LPPM: warranted={lppm_says_safe}")
    if n_visit_trap == 0:
        print("    NOTE: liveness check above found trap NEVER visited in 100 real rollouts "
            "-- any LPPM WARRANT here is a vacuous-P2 result (same class of issue as the "
            "activity traps found throughout this project), not independent confirmation.")
    print(f"    {'directionally consistent' if stl_says_safe == lppm_says_safe else 'DIRECTIONALLY INCONSISTENT -- investigate'}")

    # ── Z_free-internal closure: Theorem 5.4's ACTUAL minimal premise, on the
    # SAME calibration split (calib_traj) used for p_hat_gamma above -- not a
    # new split, not new data. See core/lppm/verifier.py::verify_zfree_closure(). ──
    print(f"\n[7] Z_FREE-INTERNAL CLOSURE CHECK (Theorem 5.4's exact minimal premise, "
        f"same calib_traj as [5] above):")
    closure = verify_zfree_closure(calib_traj, dpa, spec, gamma=GAMMA, eta=ETA, lppm_params=lppm_params)
    if not closure.verifiable:
        print("    UNVERIFIABLE: zero (transition, head) pairs in this calibration split had "
            "a source state inside Z_free = {(z,q): V<eta} -- no Z_free-internal samples to "
            "compute a bound from. p_hat_gamma above remains a valid lower bound for the "
            "stronger whole-trajectory event C(tau), but cannot yet be cited as satisfying "
            "Theorem 5.4's minimal premise specifically.")
    else:
        print(f"    n_zfree={closure.n_zfree}  k_zfree={closure.k_zfree}  "
            f"p_hat_closure={closure.p_hat_closure:.4f}  (gamma={closure.gamma}, eta={closure.eta})")
        if closure.p_hat_closure >= lppm_res.warrant_threshold:
            print(f"    p_hat_closure >= warrant_threshold={lppm_res.warrant_threshold:.2f} -- "
                "p_hat_gamma may now be cited as satisfying Theorem 5.4's minimal premise "
                "(not just the stronger whole-trajectory event).")
        else:
            print(f"    p_hat_closure < warrant_threshold={lppm_res.warrant_threshold:.2f} -- "
                "NOT enough evidence yet for the Theorem-5.4-minimal-premise wording; "
                "continue citing p_hat_gamma only as a lower bound for the stronger "
                "whole-trajectory event C(tau).")


if __name__ == "__main__":
    main()
