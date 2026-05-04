"""
wrappers/pointbutton_wrapper.py

SAFEWORLD wrapper for PointButton1 DreamerV3 (PyTorch, sw_bench) checkpoints.

Checkpoint format
─────────────────
  agent_state_dict:
    _wm.encoder._mlp.layers.Encoder_linear{i}.weight   [hidden, in]  bias=False
    _wm.encoder._mlp.layers.Encoder_norm{i}.weight/bias [hidden]
    _wm.dynamics.W                                       [1, deter_dim]
    _wm.dynamics._img_in_layers.{0,1}.weight/bias        linear + LN
    _wm.dynamics._cell.layers.GRU_linear.weight          [3D, 2D]
    _wm.dynamics._cell.layers.GRU_norm.weight/bias       [3D]
    _wm.dynamics._img_out_layers.{0,1}.weight/bias       linear + LN
    _wm.dynamics._imgs_stat_layer.weight/bias            [stoch_flat, D]
    _wm.heads.decoder._mlp.layers.Decoder_linear{i}.weight
    _wm.heads.decoder._mlp.layers.Decoder_norm{i}.weight/bias
    _wm.heads.decoder._mlp.mean_layer.obs.weight/bias    [obs_dim, hidden]

These checkpoints have NO auxiliary AP decoder head — every trained aux key
(cost, speed, goal_button_distance, nearest_hazard_distance, ...) is absent.

AP extraction
─────────────
  Without aux decoder, APs are derived from the deterministic GRU state h via
  data-driven variance statistics (same approach as DreamerV3Wrapper "stats"
  mode — Section 4.7 of the paper):
    hazard_dim  = argmax_{d} Var(h_d)         over N×T rollout steps
    goal_dim    = second-highest variance dim
    hazard_dist = mean[d_haz] + σ·std[d_haz] − h[d_haz]   (positive = safe)
    goal_dist   = h[d_goal] − (mean[d_goal] − σ·std[d_goal])  (positive = reached)

  Unsupported APs (velocity, near_obstacle, near_human, zone_a/b/c, carrying)
  are always 0.0 — specs using them will produce degenerate (N/A) verdicts.

RSSM prior step
───────────────
  GRU formula used (standard "minimal DreamerV3 GRU"):
    gru_in = cat(h, img_in)
    gates  = LN(Linear(gru_in, bias=False))     # [3D]
    r, u, c = split(gates, 3)
    h_next = sigmoid(u) * h + (1 − sigmoid(u)) * tanh(c)
  (reset gate r is produced but not applied separately — absorbed into c logit.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from configs.settings import RolloutConfig
from .base import WorldModelWrapper


# ─── pure-functional helpers ─────────────────────────────────────────────────

def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _ln(
    x:   torch.Tensor,
    w:   torch.Tensor,
    b:   torch.Tensor,
) -> torch.Tensor:
    return F.layer_norm(x, [w.shape[0]], w, b)


# ─── wrapper ──────────────────────────────────────────────────────────────────

class PointButtonWrapper(WorldModelWrapper):
    """
    SAFEWORLD wrapper for sw_bench PointButton1 DreamerV3 checkpoints.

    Quick start
    -----------
    from wrappers import PointButtonWrapper
    from configs.settings import RolloutConfig
    from specs.spec_calibrator import load_env_config

    cfg = RolloutConfig(horizon=50, n_rollouts=30, seed=42)
    env_config = load_env_config("configs/environments/largedim_usingdata.json")
    w = PointButtonWrapper(cfg)
    w.load(env_config=env_config)
    trajs = w.sample_rollouts()
    """

    def __init__(self, config: RolloutConfig | None = None):
        super().__init__(config)
        self.device:     torch.device = torch.device("cpu")
        self.env_config: dict         = {}

        # Extracted weight tensors (keyed without the _wm. prefix)
        self._wm: dict[str, torch.Tensor] | None = None

        # Architecture dimensions (resolved at load())
        self._deter_dim:     int = 512
        self._stoch_flat:    int = 256
        self._stoch_k:       int = 32
        self._stoch_classes: int = 8
        self._act_dim:       int = 2

        # Data-driven AP stats (populated by first sample_rollouts())
        self._ap_stats: dict[str, Any] | None = None

    # ── load ─────────────────────────────────────────────────────────────────

    def load(self, **kwargs) -> None:
        """
        Load checkpoint and extract world-model weights.

        Parameters
        ----------
        checkpoint_path : str | Path   path to .pt file (or set in env_config)
        env_config      : dict         loaded environment JSON
        device          : str          "cpu" | "cuda:0"
        """
        env_config  = kwargs.get("env_config") or {}
        device_str  = kwargs.get("device") or self.config.extra.get("device", "cpu")
        ckpt_path   = (
            kwargs.get("checkpoint_path")
            or env_config.get("checkpoint_path")
            or self.config.extra.get("checkpoint_path")
        )
        if ckpt_path is None:
            raise ValueError(
                "PointButtonWrapper.load() requires checkpoint_path. "
                "Pass it as kwarg or set env_config['checkpoint_path']."
            )

        self.device     = torch.device(device_str)
        self.env_config = env_config

        arch = env_config.get("model_arch", {})
        self._deter_dim     = int(arch.get("deter_dim",     512))
        self._stoch_flat    = int(arch.get("stoch_flat",    256))
        self._stoch_k       = int(arch.get("stoch_k",       32))
        self._stoch_classes = int(arch.get("stoch_classes", 8))
        self._act_dim       = int(arch.get("act_dim",       2))

        # Load full agent checkpoint and extract _wm.* prefix
        ckpt = torch.load(
            Path(ckpt_path).expanduser(),
            map_location=self.device,
            weights_only=False,
        )
        sd = ckpt.get("agent_state_dict", ckpt)
        self._wm = {
            k[4:]: v.to(self.device).float()
            for k, v in sd.items()
            if k.startswith("_wm.")
        }
        if not self._wm:
            raise RuntimeError(
                f"No '_wm.*' keys found in checkpoint {ckpt_path}. "
                "Expected keys like '_wm.encoder._mlp.layers.Encoder_linear0.weight'."
            )

    # ── RSSM helpers ──────────────────────────────────────────────────────────

    def _init_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Initial (h, z) using the learnable W param."""
        W = self._wm.get("dynamics.W")  # [1, deter_dim]
        h = W.expand(1, -1).clone() if W is not None else torch.zeros(
            1, self._deter_dim, device=self.device)
        z = torch.zeros(1, self._stoch_flat, device=self.device)
        return h, z

    def _prior_step(
        self,
        h:      torch.Tensor,
        z_flat: torch.Tensor,
        action: torch.Tensor,
        rng:    np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One-step RSSM prior (imagination). Returns (h_next, z_next)."""
        wm = self._wm
        D  = self._deter_dim

        # img_in: cat(z_flat, action) [stoch_flat+act] → [D]
        img_in = torch.cat([z_flat, action], dim=-1)
        img_in = F.linear(img_in, wm["dynamics._img_in_layers.0.weight"])
        img_in = _ln(img_in,
                     wm["dynamics._img_in_layers.1.weight"],
                     wm["dynamics._img_in_layers.1.bias"])
        img_in = _silu(img_in)

        # GRU: cat(h, img_in) [2D] → [3D] gates → h_next [D]
        gru_in = torch.cat([h, img_in], dim=-1)
        gates  = F.linear(gru_in, wm["dynamics._cell.layers.GRU_linear.weight"])
        gates  = _ln(gates,
                     wm["dynamics._cell.layers.GRU_norm.weight"],
                     wm["dynamics._cell.layers.GRU_norm.bias"])
        r, u, c = gates.chunk(3, dim=-1)
        h_next = torch.sigmoid(u) * h + (1.0 - torch.sigmoid(u)) * torch.tanh(c)

        # img_out [D] → stoch logits [stoch_flat]
        img_out = F.linear(h_next, wm["dynamics._img_out_layers.0.weight"])
        img_out = _ln(img_out,
                      wm["dynamics._img_out_layers.1.weight"],
                      wm["dynamics._img_out_layers.1.bias"])
        img_out = _silu(img_out)

        logits = F.linear(
            img_out,
            wm["dynamics._imgs_stat_layer.weight"],
            wm["dynamics._imgs_stat_layer.bias"],
        )  # [1, stoch_flat]

        # Categorical sampling per stoch_k groups
        logits_2d = logits.reshape(self._stoch_k, self._stoch_classes)  # [K, C]
        if rng is not None:
            noise    = torch.tensor(
                rng.exponential(size=(self._stoch_k, self._stoch_classes)).astype(np.float32),
                device=self.device,
            )
            probs = torch.softmax(logits_2d - torch.log(noise + 1e-10), dim=-1)
        else:
            probs = torch.softmax(logits_2d, dim=-1)

        z_idx  = probs.argmax(dim=-1)                              # [K]
        z_1hot = F.one_hot(z_idx, num_classes=self._stoch_classes).float()  # [K, C]
        z_next = z_1hot.reshape(1, self._stoch_flat)               # [1, stoch_flat]

        return h_next, z_next

    def _decode_obs(self, h: torch.Tensor, z: torch.Tensor) -> np.ndarray:
        """Decode feat = cat(h, z) → reconstructed 76-dim obs (numpy)."""
        wm   = self._wm
        feat = torch.cat([h, z], dim=-1)
        x    = F.linear(feat, wm["heads.decoder._mlp.layers.Decoder_linear0.weight"])
        x    = _ln(x,
                   wm["heads.decoder._mlp.layers.Decoder_norm0.weight"],
                   wm["heads.decoder._mlp.layers.Decoder_norm0.bias"])
        x    = _silu(x)
        x    = F.linear(x, wm["heads.decoder._mlp.layers.Decoder_linear1.weight"])
        x    = _ln(x,
                   wm["heads.decoder._mlp.layers.Decoder_norm1.weight"],
                   wm["heads.decoder._mlp.layers.Decoder_norm1.bias"])
        x    = _silu(x)
        obs  = F.linear(
            x,
            wm["heads.decoder._mlp.mean_layer.obs.weight"],
            wm["heads.decoder._mlp.mean_layer.obs.bias"],
        )
        return obs.squeeze(0).detach().cpu().numpy()

    # ── AP extraction ─────────────────────────────────────────────────────────

    def _compute_ap_stats(self, h_flat: np.ndarray) -> None:
        """
        Compute data-driven AP thresholds from (N*T, deter_dim) h matrix.

        Finds the two highest-variance dimensions and assigns them to
        hazard_dist (d0) and goal_dist (d1).
        """
        thr        = self.env_config.get("ap_thresholds", {})
        haz_sigma  = float(thr.get("hazard_sigma", 1.5))
        goal_sigma = float(thr.get("goal_sigma",   1.0))

        means = h_flat.mean(axis=0)
        stds  = h_flat.std(axis=0) + 1e-8
        order = np.argsort(h_flat.var(axis=0))[::-1]
        d0, d1 = int(order[0]), int(order[1])

        self._ap_stats = {
            "hazard_dim": d0,
            "hazard_thr": float(means[d0] + haz_sigma  * stds[d0]),
            "goal_dim":   d1,
            "goal_thr":   float(means[d1] - goal_sigma * stds[d1]),
        }

    def _h_to_aps(self, h_np: np.ndarray) -> dict[str, float]:
        """Convert single h vector to SAFEWORLD AP dict."""
        st = self._ap_stats
        return {
            # positive → safe (agent is far from hazard)
            "hazard_dist":   float(st["hazard_thr"] - h_np[st["hazard_dim"]]),
            # positive → goal reached (h is in the "high" zone for goal dim)
            "goal_dist":     float(h_np[st["goal_dim"]] - st["goal_thr"]),
            # unsupported — always 0.0
            "velocity":      0.0,
            "near_obstacle": 0.0,
            "near_human":    0.0,
            "zone_a":        0.0,
            "zone_b":        0.0,
            "zone_c":        0.0,
            "carrying":      0.0,
            "model_cost":    0.0,
        }

    # ── rollout ───────────────────────────────────────────────────────────────

    def sample_rollouts(
        self,
        config: RolloutConfig | None = None,
    ) -> list[list[dict[str, float]]]:
        """
        Sample N pure-imagination RSSM rollouts.

        On first call, computes data-driven h-stats for AP thresholds.
        Subsequent calls reuse the same stats so thresholds are consistent.
        """
        self._ensure_loaded()
        cfg = config or self.config

        all_h_seqs: list[np.ndarray] = []

        with torch.no_grad():
            for i in range(cfg.n_rollouts):
                rng     = np.random.default_rng(cfg.seed + i)
                h, z    = self._init_state()
                h_steps: list[np.ndarray] = []

                for _ in range(cfg.horizon):
                    h_steps.append(h.squeeze(0).cpu().numpy())
                    action_np = rng.uniform(-1.0, 1.0, size=self._act_dim).astype(np.float32)
                    action_t  = torch.tensor(action_np, device=self.device).unsqueeze(0)
                    h, z = self._prior_step(h, z, action_t, rng=rng)

                all_h_seqs.append(np.stack(h_steps))  # (T, deter_dim)

        # Compute AP stats from all collected h vectors (once per wrapper lifetime)
        if self._ap_stats is None:
            h_flat = np.concatenate(all_h_seqs, axis=0)  # (N*T, D)
            self._compute_ap_stats(h_flat)

        # Build AP trajectories
        return [
            [self._h_to_aps(h_t) for h_t in h_seq]
            for h_seq in all_h_seqs
        ]

    def sample_paired_rollouts(
        self,
        config: RolloutConfig | None = None,
    ) -> list[tuple[list[dict[str, float]], list[dict[str, float]]]]:
        raise NotImplementedError(
            "PointButtonWrapper: paired rollouts require a running "
            "SafetyPointButton1-v0 environment.  Not implemented."
        )

    def decode_and_replay(
        self,
        config:      RolloutConfig | None = None,
        closed_loop: bool                 = False,
    ) -> list[list]:
        raise NotImplementedError(
            "PointButtonWrapper.decode_and_replay() not implemented."
        )

    # ── public helpers ────────────────────────────────────────────────────────

    def ap_keys(self) -> list[str]:
        return [
            "hazard_dist", "goal_dist",
            "velocity", "near_obstacle", "near_human",
            "zone_a", "zone_b", "zone_c", "carrying",
            "model_cost",
        ]

    def get_ap_stats(self) -> dict | None:
        """Return data-driven AP statistics (populated after first sample_rollouts)."""
        return self._ap_stats

    # ── internals ─────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._wm is None:
            raise RuntimeError(
                "PointButtonWrapper: call load() before sampling rollouts."
            )

    def close(self) -> None:
        self._wm = None
