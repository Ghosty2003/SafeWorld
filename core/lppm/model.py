from __future__ import annotations

from typing import Any

from utils.spec_analysis import analyze_spec_structure

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


NeuralLPPM = None

if nn is not None:
    class NeuralLPPM(nn.Module):
        def __init__(
            self,
            latent_dim: int,
            n_states: int,
            n_heads: int,
            hidden_dim: int = 128,
            q_embed_dim: int = 16,
        ):
            super().__init__()
            self.q_embedding = nn.Embedding(n_states, q_embed_dim)
            self.mlp = nn.Sequential(
                nn.Linear(latent_dim + q_embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.head = nn.Linear(hidden_dim, n_heads)

        def forward(self, z: torch.Tensor, q_idx: torch.Tensor) -> torch.Tensor:
            q_embed = self.q_embedding(q_idx)
            feat = torch.cat([z, q_embed], dim=-1)
            hidden = self.mlp(feat)
            return F.softplus(self.head(hidden))


def _nonneg(value: float) -> float:
    """
    Batch-4 audit fix: several heuristic-fallback branches below subtract an
    unclamped z-feature value (progress/zone_val/margin) from a small
    constant -- only the lower end of that feature was ever clamped (via the
    per-branch max(0.0, ...) on the raw AP reading), not the resulting V
    itself, so a large raw feature value could still drive the whole
    expression negative. V is meant to be a nonnegative certificate value:
    both Z_free = {(z,q): V_phi(z,q) < eta} and the P2 descent check assume
    V >= 0. Clamp explicitly at every heuristic branch's return point,
    rather than relying on each branch's arithmetic happening to stay
    nonnegative on its own.
    """
    return max(0.0, value)


def compute_lppm_value(
    z: dict[str, float],
    q: str,
    r: int,
    spec: dict[str, Any],
    t: int,
    T: int,
    lppm_params: dict | None = None,
    dpa=None,
) -> float:
    if lppm_params is not None:
        learned = predict_learned_lppm_value(z, q, r, lppm_params)
        if learned is not None:
            return learned

    analysis = spec.get("analysis") or analyze_spec_structure(spec)
    spec["analysis"] = analysis
    mp = analysis["mp_class"]
    rem = max(0.0, (T - t) / T)
    objectives = analysis["objectives"]
    meta = dpa.state_meta.get(q, {}) if dpa is not None else {}

    if mp == "Safety":
        if q == "trap":
            return 0.0
        safety_margin = min((z.get(ap, 0.0) for ap in objectives["safety"]), default=0.0)
        return _nonneg(rem * (1.0 + max(0.0, safety_margin)))

    if mp == "Guarantee":
        # External-review fix: this branch previously read ONLY
        # meta["remaining_goals"] (objectives["guarantee"]'s UNORDERED
        # plain-goal bucket) -- for a state built from the
        # guarantee_sequences (ordered F(A and F(B and F(C...)))) branch of
        # core/lppm/automaton.py, remaining_goals is the empty list for
        # EVERY state whenever the spec has no plain unordered goals at all
        # (e.g. ltl_sequential_goals, ltl_three_stage), so V collapsed to
        # the unconditional 0.0 branch everywhere -- completely blind to
        # sequence_progress, the field that actually tracks how far a
        # purely-sequential spec has advanced.
        remaining = meta.get("remaining_goals", objectives["guarantee"])
        sequences = objectives.get("guarantee_sequences", [])
        seq_progress = meta.get("sequence_progress")
        seq_remaining = (
            sum(len(seq) - p for seq, p in zip(sequences, seq_progress))
            if seq_progress is not None else 0
        )
        total_remaining = len(remaining) + seq_remaining
        if total_remaining == 0:
            return 0.0
        progress = max((z.get(ap, 0.0) for ap in objectives["guarantee"]), default=0.0)
        return _nonneg(rem * (1.0 + total_remaining - max(0.0, progress)))

    if mp == "Obligation":
        if q == "trap":
            return 0.0
        remaining = meta.get("remaining_goals", objectives["guarantee"])
        if not remaining:
            return rem * 0.1
        safety_margin = min((z.get(ap, 0.0) for ap in objectives["safety"]), default=0.0)
        progress = max((z.get(ap, 0.0) for ap in objectives["guarantee"]), default=0.0)
        return _nonneg(rem * (1.0 + len(remaining) - progress + max(0.0, safety_margin)))

    if mp == "Recurrence":
        remaining = meta.get("remaining_recur", objectives["recurrence"])
        if not remaining:
            return rem * 0.25
        zone_val = max((z.get(ap, 0.0) for ap in objectives["recurrence"]), default=0.0)
        return _nonneg(rem * (1.0 + len(remaining) - zone_val))

    if mp == "Persistence":
        if q == "absorbed":
            return 0.0
        stability = min((z.get(ap, 0.0) for ap in objectives["persistence"]), default=0.0)
        return _nonneg(rem * (1.0 + max(0.0, stability)))

    if mp in {"Reactivity", "Streett"}:
        if q == "trap":
            return 0.0
        pending = meta.get(
            "pending",
            [f"{item['trigger']}->{item['response']}" for item in objectives["responses"]],
        )
        if not pending:
            return rem * 0.25
        margin = 0.0
        for item in objectives["responses"]:
            margin += max(0.0, z.get(item["response"], 0.0)) - max(0.0, z.get(item["trigger"], 0.0))
        return _nonneg(rem * (1.0 + len(pending) - margin))

    return rem


def predict_learned_lppm_value(
    z: dict[str, float],
    q: str,
    r: int,
    lppm_params: dict[str, Any],
) -> float | None:
    if torch is None or nn is None or F is None:
        return None
    if lppm_params.get("backend") != "torch_mlp":
        return None
    weights = lppm_params.get("weights")
    feature_keys = lppm_params.get("feature_keys")
    state_to_idx = lppm_params.get("state_to_idx")
    odd_to_idx = lppm_params.get("odd_to_idx")
    if not weights or not feature_keys or not state_to_idx or not odd_to_idx:
        return None
    if r not in odd_to_idx or q not in state_to_idx:
        return None

    model = get_or_build_lppm_model(lppm_params)
    if model is None:
        return None
    z_vec = torch.tensor([[float(z.get(key, 0.0)) for key in feature_keys]], dtype=torch.float32)
    q_idx = torch.tensor([state_to_idx[q]], dtype=torch.long)
    with torch.no_grad():
        values = model(z_vec, q_idx)
    return float(values[0, odd_to_idx[r]].item())


def get_or_build_lppm_model(lppm_params: dict[str, Any]):
    cached = lppm_params.get("_model_cache")
    if cached is not None:
        return cached
    if torch is None or nn is None or F is None:
        return None
    weights = lppm_params.get("weights")
    feature_keys = lppm_params.get("feature_keys")
    state_to_idx = lppm_params.get("state_to_idx")
    odd_to_idx = lppm_params.get("odd_to_idx")
    if not weights or feature_keys is None or state_to_idx is None or odd_to_idx is None:
        return None
    model = NeuralLPPM(
        latent_dim=len(feature_keys),
        n_states=len(state_to_idx),
        n_heads=max(len(odd_to_idx), 1),
        hidden_dim=int(lppm_params.get("hidden_dim", 128)),
        q_embed_dim=int(lppm_params.get("q_embed_dim", 16)),
    )
    model.load_state_dict(weights)
    model.eval()
    lppm_params["_model_cache"] = model
    return model


def infer_feature_keys(trajectories: list[list[dict[str, float]]]) -> list[str]:
    keys: set[str] = set()
    for traj in trajectories:
        for state in traj:
            keys.update(state.keys())
    return sorted(keys)
