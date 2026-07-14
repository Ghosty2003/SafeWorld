# Writing a wrapper for a new world model

A wrapper adapts one world model to the SafeWorld verification pipeline.
Subclass `WorldModelWrapper` in `base.py` and implement:

| Method | Contract |
|---|---|
| `load(checkpoint_path, ...)` | load weights; no side effects before this |
| `sample_rollouts(config)` | N imagination rollouts → `list[list[dict[str, float]]]` — one AP dict per step |
| `ap_keys()` | the AP dimension names this wrapper can provide |
| `validate_trajectories(...)` | inherited; warns about missing AP keys |
| `sample_paired_rollouts(config)` | *optional* — paired (model, real-env) rollouts for cerr calibration; base raises `NotImplementedError` |

Existing wrappers: `random_wrapper.py` (minimal reference),
`dreamerv3_wrapper.py` (Safety-Gymnasium), `cardreamer_wrapper.py`
(CARLA/DreamerV3 — the most complete example, with CV-based AP extraction
from decoded images and replay-anchored burn-in).

## Hard-earned rules (violating any of these produced wrong verdicts for us)

1. **AP honesty beats AP coverage.** If a quantity cannot be extracted
   faithfully from your model's outputs, emit `UNCERTAIN_SENTINEL` (-999.0)
   and let the pipeline route the spec to INCONCLUSIVE_perception — do NOT
   substitute a proxy. (Example: velocity from a non-monotone reward head is
   anti-conservative → false-SAFE.) Sentinels must be gated *before* STL
   robustness: -999 fed into `G`/`F` produces garbage in both directions.
2. **Validate your extraction before reporting any number**: channel order
   (multi-frame vote, not "looks plausible"), detection completeness on
   frames claimed empty, physical sanity of a witness trajectory.
3. **Start states must approximate the deployment distribution** (replay
   anchors + a few real burn-in steps), not the model's learned initial
   distribution — otherwise coverage assumptions of the transfer theory fail.
4. **Bound GPU memory by chunking imagination batches**; decoder activations
   scale linearly with batch × horizon.
5. **Paired rollouts must fork from the same posterior state.** Pairing a
   replay-anchored imagination with a fresh env reset measures distribution
   distance, not model error (Definition 3.5). See
   `cardreamer/calibrate_cerr.py` for the correct construction.
6. If you add **latent probes** (reading task quantities out of hidden
   states), read `cardreamer/README.md` — probe reliability must be measured
   per probe and per decision scale, never assumed.
