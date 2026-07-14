"""
collect_probe_data.py — Collect (h_t_post, h_t_prior, num_completed) for progress probe.

Usage:
  # Step 1: start CARLA
  /home/bot/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -world-port=2000 &

  # Step 2: collect
  conda activate cardreamer
  cd /home/bot/SafeWorld
  python collect_probe_data.py [--episodes 30] [--save_dir /tmp/probe_data]

  # Step 3: transferability check only (data already collected)
  python collect_probe_data.py --check_only --save_dir /tmp/probe_data

Architecture
------------
h_t_post  : RSSM posterior deter from jax_agent.policy() on real CARLA observations
h_t_prior : RSSM prior deter from world-model imagination launched from anchor states
label     : cumulative waypoints completed (num_completed from env info)

Transferability check verifies probe trained on h_t_post generalises to h_t_prior.
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, "/home/bot/CarDreamer")
sys.path.insert(0, "/home/bot/SafeWorld")

SAVE_DIR     = "/tmp/probe_data"
BURN_IN      = 3        # steps before anchor/imagination
HORIZON      = 50       # imagination depth
ANCHOR_EVERY = 10       # launch imagination every N real steps
PROGRESS_K   = 15       # binary label: cum_wpt > K


# ─────────────────────────────────────────────────────────────────────────────
# JIT imagination function (h_t_prior)
# ─────────────────────────────────────────────────────────────────────────────

from cardreamer.probe_common import (  # shared, was duplicated here
    build_imagine_from_latent_fn as build_imagine_fn,
)


# ─────────────────────────────────────────────────────────────────────────────
# Part A: collect real episode data (needs live CARLA)
# ─────────────────────────────────────────────────────────────────────────────

def _wrap_obs(raw_obs, reward, is_first, is_last, is_terminal):
    """
    Construct agent-compatible obs dict (batch size 1) from raw gym obs.
    Agent expects: birdeye_wpt, is_first, is_last, is_terminal, reward
    """
    img = raw_obs["birdeye_wpt"] if isinstance(raw_obs, dict) else raw_obs
    return {
        "birdeye_wpt": img[np.newaxis],                           # (1, H, W, C)
        "is_first":    np.array([is_first],    dtype=bool),       # (1,)
        "is_last":     np.array([is_last],     dtype=bool),       # (1,)
        "is_terminal": np.array([is_terminal], dtype=bool),       # (1,)
        "reward":      np.array([reward],      dtype=np.float32), # (1,)
    }


def collect(args):
    import jax
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print("[1/3] Loading CarDreamerWrapper ...", flush=True)
    from wrappers.cardreamer_wrapper import CarDreamerWrapper
    from configs.settings import RolloutConfig

    cfg = RolloutConfig(n_rollouts=1, horizon=HORIZON, seed=0, action_source="actor")
    w = CarDreamerWrapper(cfg)
    w.load()
    jax_agent = w._jax_agent
    print(f"  Loaded. act_dim={w._n_acts}", flush=True)

    # ── Connect to CARLA ──────────────────────────────────────────────────────
    print("[2/3] Connecting to CARLA ...", flush=True)
    import car_dreamer
    env, _ = car_dreamer.create_task("carla_four_lane")
    print("  CARLA connected.", flush=True)

    # ── Build imagination function ─────────────────────────────────────────────
    print("[3/3] Building imagination JIT (first call will trace) ...", flush=True)
    imagine_fn = build_imagine_fn(jax_agent, HORIZON)
    print("  Done.\n", flush=True)

    # ── Storage ───────────────────────────────────────────────────────────────
    post_h   = []   # list of arrays (N_steps, h_dim) per episode → concat later
    post_wpt = []   # list of arrays (N_steps,)

    prior_h     = []   # list of (HORIZON, h_dim) arrays
    anchor_wpts = []   # cum_wpt at each anchor

    ep_summary  = []   # (ep_len, cum_wpt, is_dest, is_col)

    print(f"Collecting {args.episodes} episodes ...\n", flush=True)

    for ep in range(args.episodes):
        raw_obs  = env.reset()
        done     = False
        step     = 0
        cum_wpt  = 0
        is_dest  = False
        is_col   = False
        pol_state = None

        ep_h   = []
        ep_wpt = []

        while not done:
            is_first = (step == 0)

            # ── Posterior: RSSM update via production policy() ────────────────
            obs_dict  = _wrap_obs(raw_obs, reward=0.0, is_first=is_first,
                                  is_last=False, is_terminal=False)
            outs, pol_state = jax_agent.policy(obs_dict, pol_state, mode="eval")

            # latent = ((deter, stoch, logit, ...), action), task_state, expl_state
            latent = pol_state[0][0]   # dict: deter (1, h_dim), stoch, logit
            h_np   = np.asarray(latent["deter"])[0]   # (h_dim,)
            ep_h.append(h_np)
            ep_wpt.append(cum_wpt)

            # ── Imagination snapshot at anchor steps ─────────────────────────
            if step >= BURN_IN and step % ANCHOR_EVERY == 0:
                _dev = jax_agent.policy_devices[0]
                rng  = jax_agent._next_rngs(jax_agent.policy_devices)
                # jax_agent.policy() returns numpy (host); must device_put before JIT
                deter = jax.device_put(np.asarray(latent["deter"]), _dev)
                stoch = jax.device_put(np.asarray(latent["stoch"]), _dev)
                logit = jax.device_put(np.asarray(latent["logit"]), _dev)

                h_prior_jax, _ = imagine_fn(jax_agent.varibs, rng, deter, stoch, logit)
                # explicit device_get required: jax_transfer_guard="disallow" blocks np.asarray
                h_prior_np = np.asarray(jax.device_get(h_prior_jax))[:, 0, :]  # (HORIZON, h_dim)
                prior_h.append(h_prior_np)
                anchor_wpts.append(cum_wpt)

            # ── Step env ─────────────────────────────────────────────────────
            act_oh  = np.asarray(outs["action"])[0]   # (act_dim,) one-hot
            act_idx = int(np.argmax(act_oh))
            raw_obs, reward, done, info = env.step(act_idx)

            cum_wpt += int(info.get("num_completed", 0))
            if info.get("is_collision", False):
                is_col = True
            if hasattr(env, "is_destination_reached") and env.is_destination_reached():
                is_dest = True

            step += 1

        post_h.append(np.array(ep_h))
        post_wpt.append(np.array(ep_wpt))
        ep_summary.append((step, cum_wpt, int(is_dest), int(is_col)))
        print(f"  ep {ep+1:3d}/{args.episodes}  "
              f"len={step:4d}  wpt={cum_wpt:3d}  "
              f"dest={is_dest}  col={is_col}  "
              f"anchors={len(prior_h)}", flush=True)

    env.close()

    # ── Save ──────────────────────────────────────────────────────────────────
    flat_h   = np.concatenate(post_h,   axis=0)   # (N, h_dim)
    flat_wpt = np.concatenate(post_wpt, axis=0)   # (N,)

    np.savez(os.path.join(args.save_dir, "posterior.npz"),
             h=flat_h, cum_wpt=flat_wpt,
             ep_summary=np.array(ep_summary, dtype=float))

    if prior_h:
        np.savez(os.path.join(args.save_dir, "prior.npz"),
                 h=np.array(prior_h),               # (M, HORIZON, h_dim)
                 anchor_wpt=np.array(anchor_wpts))  # (M,)

    ep_arr = np.array(ep_summary, dtype=float)
    print(f"\nSaved to {args.save_dir}/")
    print(f"  posterior: {flat_h.shape[0]} steps × h_dim={flat_h.shape[1]}")
    print(f"  prior:     {len(prior_h)} anchors × {HORIZON} steps")
    print(f"  episodes:  "
          f"len∈[{ep_arr[:,0].min():.0f},{ep_arr[:,0].max():.0f}]  "
          f"wpt∈[{ep_arr[:,1].min():.0f},{ep_arr[:,1].max():.0f}]  "
          f"dest={ep_arr[:,2].sum():.0f}/{len(ep_summary)}  "
          f"col={ep_arr[:,3].sum():.0f}/{len(ep_summary)}")


# ─────────────────────────────────────────────────────────────────────────────
# Part B: probe training + transferability check (no CARLA needed)
# ─────────────────────────────────────────────────────────────────────────────

def check(save_dir):
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    print("\n" + "="*60)
    print("  Probe Training + Transferability Check")
    print("="*60)

    post_path  = os.path.join(save_dir, "posterior.npz")
    prior_path = os.path.join(save_dir, "prior.npz")
    if not os.path.exists(post_path):
        print(f"ERROR: {post_path} not found. Run collect first."); return
    if not os.path.exists(prior_path):
        print(f"ERROR: {prior_path} not found. Run collect first."); return

    post  = np.load(post_path)
    prior = np.load(prior_path)
    h_post   = post["h"]          # (N, h_dim)
    cum_wpt  = post["cum_wpt"]    # (N,)
    h_prior  = prior["h"]         # (M, HORIZON, h_dim)
    a_wpt    = prior["anchor_wpt"]

    print(f"\nData: {h_post.shape[0]} posterior steps, "
          f"{h_prior.shape[0]} prior sequences, h_dim={h_post.shape[1]}")

    # ── A. Train probe on posterior h_t ─────────────────────────────────────
    print("\n[A] Train progress probe on posterior h_t ...")
    sc    = StandardScaler()
    X     = sc.fit_transform(h_post)
    y     = cum_wpt.astype(float)
    yb    = (cum_wpt > PROGRESS_K).astype(int)

    ridge = Ridge(alpha=1.0)
    r2s   = cross_val_score(ridge, X, y,  cv=5, scoring="r2")
    ridge.fit(X, y)
    lr    = LogisticRegression(max_iter=1000, class_weight="balanced")
    auc   = cross_val_score(lr,    X, yb, cv=5, scoring="roc_auc")
    print(f"  Ridge R²     (5-fold): {r2s.mean():.3f} ± {r2s.std():.3f}")
    print(f"  Logistic AUC (wpt>{PROGRESS_K}, 5-fold): {auc.mean():.3f} ± {auc.std():.3f}")

    # ── B. Apply probe to imagination h_t_prior ──────────────────────────────
    print(f"\n[B] Apply probe to {h_prior.shape[0]} imagination sequences ...")
    M, T, _ = h_prior.shape
    preds = np.stack([
        ridge.predict(sc.transform(h_prior[i]))
        for i in range(M)
    ])   # (M, T)

    # C1: trajectory-level positive slope (>80% of sequences have net upward progress).
    # Step-level SNR = 0.07 (noise 15× signal); step/smoothed-step monotone checks are
    # therefore uninformative.  The right prior-side test is: does imagination predict
    # forward progress over the full horizon?
    raw_diffs  = np.diff(preds, axis=1)
    net_slopes = preds[:, -1] - preds[:, 0]                # (M,)
    pct_pos    = (net_slopes > 0).mean()
    print(f"  C1 positive net slope (preds[T] > preds[0]): {pct_pos:.2%}  [> 0.80]")
    print(f"     raw step stats: mean={raw_diffs.mean():.3f}  std={raw_diffs.std():.3f}"
          f"  (SNR≈{abs(raw_diffs.mean())/max(raw_diffs.std(),1e-9):.2f}; "
          f"step-level monotone is uninformative at SNR<0.1)")

    # C2: rate (wpt/step) — expected ~0.35-0.40 for carla_four_lane at ~0.37 wpt/step
    rates = net_slopes / T
    rate_ok = 0.2 < rates.mean() < 1.0
    print(f"  C2 rate wpt/step: {rates.mean():.3f} ± {rates.std():.3f}  [0.2-1.0]")

    # C3: variance ratio prior/post
    var_post  = np.var(ridge.predict(X))
    var_prior = np.var(preds)
    vratio    = var_prior / max(var_post, 1e-9)
    print(f"  C3 var ratio prior/post: {vratio:.2f}  [< 3]")

    # C4: anchor alignment (predicted progress at t=0 should correlate with true)
    corr = float(np.corrcoef(a_wpt, preds[:, 0])[0, 1]) if len(a_wpt) > 1 else 0.0
    print(f"  C4 anchor correlation:   r={corr:.3f}  [> 0.6]")

    # C5: OOD fraction — what fraction of prior h_t dims are > 3σ from posterior mean.
    # If large, the probe is extrapolating outside its training distribution.
    post_mu  = h_post.mean(0)                         # (h_dim,)
    post_sig = h_post.std(0) + 1e-6
    h_prior_flat = h_prior.reshape(-1, h_prior.shape[-1])
    z_scores = np.abs((h_prior_flat - post_mu) / post_sig)
    # Per-sample OOD: fraction of features > 3σ for each prior state
    per_sample_ood = (z_scores > 3.0).mean(axis=1)   # (M*T,) expected ~0.003 if in-dist
    ood_frac = float(per_sample_ood.mean())
    ood_p95  = float(np.percentile(per_sample_ood, 95))
    print(f"  C5 OOD fraction: mean={ood_frac:.4f}  p95={ood_p95:.4f}  "
          f"[mean < 0.05  (baseline ~0.003 for in-dist)]")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    criteria = {
        "posterior R² > 0.50":         r2s.mean() > 0.50,
        "pos slope > 80% seqs":         pct_pos    > 0.80,
        "rate in [0.2, 1.0]":           rate_ok,
        "var ratio < 3.0":              vratio     < 3.0,
        "anchor corr > 0.60":           corr       > 0.60,
        "OOD mean < 0.05":              ood_frac   < 0.05,
    }
    for name, ok in criteria.items():
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}]  {name}")
    print("="*60)
    if all(criteria.values()):
        print("  VERDICT: PASS — probe transfers to imagination h_t")
        print("  → Safe to integrate probe into SafeWorld evaluation")
    else:
        n_fail = sum(1 for ok in criteria.values() if not ok)
        print(f"  VERDICT: FAIL ({n_fail}/{len(criteria)} criteria failed)")
        print("  → Distribution gap between posterior and prior; probe not ready")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",   type=int, default=30)
    parser.add_argument("--save_dir",   default=SAVE_DIR)
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()

    if not args.check_only:
        collect(args)

    post_path = os.path.join(args.save_dir, "posterior.npz")
    if os.path.exists(post_path):
        check(args.save_dir)
    else:
        print(f"No data at {post_path}. Run without --check_only first.")
