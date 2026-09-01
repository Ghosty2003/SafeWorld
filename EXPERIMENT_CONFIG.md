# SAFEWORLD Experiment Configuration & Provenance Ledger

**Note on skeleton provenance**: the user referenced a prior "v1.0" document with
this section structure; that file is not present in this repository (searched
`*.md` for "Capability matrix" / "DeepReach" / "Tier 1" / "Baseline" — no
matches). This document is built from the section outline given verbatim in
conversation, not recovered from a v1.0 file. If a v1.0 file exists elsewhere,
reconcile against it before treating this as authoritative.

This document covers **Sections 0, 3, 7, 8 only** (the "minimal viable version"
per the agreed starting scope — facts that decay if not written down now).
Sections 1, 2, 4, 5, 6, 9, 10 are NOT started; they are organizational/write-up
scaffolding that can be assembled later from existing material (capability
matrix discussion, metric design, etc.) and are lower urgency.

---

## 0. Meta

| Field | Value |
|---|---|
| Submission target | ICLR |
| Paper title (current version) | **GAP** — not available in this session; owner to fill |
| Document version | v0.1 |
| Document date | 2026-08-03 |
| Owner | **GAP** — not assigned; owner to fill |
| Source-of-truth contract | This document is authoritative for **implementation-state facts**: what code actually does, what was actually run, what numbers actually came out, and where code comments/citations were found to be stale. The paper LaTeX (not present in this repo/session) is authoritative for **theorem statements and numbering**. Where the two diverge, this doc flags the divergence explicitly (see the theorem table below and Section 8) rather than picking a side — reconciliation requires the actual paper source, which this session did not have access to. |

### Theorem numbering cross-reference

Built from the citation audit performed this session (grep across `main.py`,
`core/lppm/*.py`, `core/transfer_calibrator.py`) and corrected in code (commit:
uncommitted as of this doc's writing — see Section 8 item 3 for the full
before/after list). Three-way mapping requested (current paper ↔ code-comment
↔ old NeurIPS numbering); **the old-NeurIPS-numbering column is a gap** — this
session was never given that mapping and did not guess it.

| Current paper (per user's methodology doc, this session) | Code comment BEFORE this session's fix | Old NeurIPS numbering | Status |
|---|---|---|---|
| Theorem 5.1 (L1 two-level transfer guarantee) | "Theorem 5.5" (main.py, core/transfer_calibrator.py, 8 sites) | **GAP** | Fixed in code this session |
| Theorem 5.4 (L2 statistical/calibrated branch) | "Theorem 5.5" (main.py, core/lppm/verifier.py, 3 sites) | **GAP** | Fixed in code this session |
| Theorem 5.5 (L3 idealized LBSM, exact-everywhere, optional stopping ⇒ a.s.) | not cited anywhere in code (no L3/LBSM implementation exists yet) | **GAP** | N/A — unimplemented |
| Theorem 5.6 (L3 real/sample-based, Lipschitz extrapolation + collar) | not cited anywhere in code | **GAP** | N/A — unimplemented |
| Corollary 5.2 | correctly cited already at `main.py` (delta_err field) and `core/transfer_calibrator.py:179` | **GAP** | **Unverified** — left as-is with a TODO comment; do not assume correct without checking full paper |
| Theorem D.4 | correctly cited already at `main.py` (delta_cp field) | **GAP** | **Unverified** — left as-is with a TODO comment |

---

## 3. Data Construction

### 3.1 TD-MPC2 / walker-walk (checkpoint seed=3) — `G(height>0.6)` calibration set

| Field | Value |
|---|---|
| Anchor source | 200 REAL physics states from 8 full 500-step MPC-driven `walker-walk` episodes (seed=0 dm_control task seed), taken every 20 steps. Anchors are **exact** replays via `physics.get_state()`/`set_state()` (confirmed lossless, max diff 5.08e-08 across 5 spot-checked anchors) — **not** the approximate `burn_in` step-count path in `wrappers/tdmpc2_wrapper.py`. This exactness is load-bearing for Theorem 5.4's exchangeability premise (calibration and future-deployment states must be drawn from the same distribution); the approximate `burn_in=(0,480)` path was bootstrap-tested and found genuinely outside sampling noise (std=0.0306 vs bootstrap max 0.0297 at n=200) — it is documented in the wrapper as approximate-only, not used for this calibration set. |
| action_source | `"mpc_plan"` only (explicit required kwarg, no default in `TDMPC2Wrapper.sample_rollouts()`/`sample_latent_rollouts_from_states()`). This is the only (anchor distribution, action-mechanism) pairing C0/C1 validated — see Section 7 items C0/C1(C). |
| Rollout length H | 100 steps (matches the validated C1 depth; do not extend without re-validating drift at longer depth). |
| N | 200 (all saved anchors used; no subsampling). |
| Stratification | **None currently.** Anchors are a natural every-20-step sample across 8 real MPC episodes, not stratified by violation/boundary/safe. This is a known gap: the height_safety k=0 diagnosis (Section 7) found the calibration split has **zero** trap-visiting (height≤0.6) rollouts, i.e. no boundary/violation stratum at all in the current construction. A stratified resample (deliberately including low-height real states) is the proposed remediation, not yet implemented. |
| Seed | dm_control task seed=0 (env construction); `TDMPC2Wrapper.load(seed=3)` is the checkpoint identifier, not an RNG seed; `np.random.default_rng(0)` for the train/calibration permutation. |
| params_source sidecar | **None persisted.** `fit_lppm()`'s `training_info` dict (weights, feature_keys, etc.) exists only in-process during each script run; it is not saved to disk. Gap: no reproducibility artifact currently exists for a trained NeuralLPPM — rerunning `tdmpc2/eval_ltl_height_safety_lppm.py` retrains from scratch (deterministic given the fixed seeds, but not literally a cached artifact). |
| Physics-state anchor file | `/tmp/tdmpc2_anchors_with_physics_state.npz` (scratchpad, ephemeral — **not durable**; must be moved into the repo/a persistent data directory before this becomes a citable artifact). Produced by `tdmpc2_recollect_with_physics_state.py` (scratchpad). |

### 3.2 CarDreamer / carla_four_lane — `ltl_hazard_avoidance` calibration set

| Field | Value |
|---|---|
| Anchor source | Replay-buffer real (obs, action) sequences, burn-in per `CarDreamerWrapper`'s `use_replay_start=True` (deployment-distribution start states, Def 3.9), NOT `rssm.initial()` cold-start. |
| action_source | `"actor"` (trained deployment policy). |
| Rollout length H | 50. |
| N | 100. |
| Stratification | None recorded in `cardreamer/eval_ltl_hazard_real_lppm.py` — natural replay-buffer sample. |
| Seed | 0 (`RolloutConfig(seed=0)`). |
| params_source sidecar | Same gap as 3.1 — `training_info` not persisted to disk. |

### 3.3 Disjoint split declaration (both 3.1 and 3.2)

`fit_lppm()` trains ONLY on the train split; `calibrate_lppm()` is run ONLY on
the held-out calibration split, which the trained model never saw. This was
already a fixed bug in the CarDreamer script (see its own header comment:
"main.py currently reuses the same trajectories for both fit_lppm and
calibrate_lppm -- that would be an optimistic/overfit p_hat_gamma"); the
TD-MPC2 script (`tdmpc2/eval_ltl_height_safety_lppm.py`) was written with this
fix already in place, not retrofitted.

| | Train frac | Train N | Calib N | Split seed |
|---|---|---|---|---|
| hazard_avoidance | 0.6 | 60 | 40 | `np.random.default_rng(0)` |
| height_safety | 0.6 | 120 | 80 | `np.random.default_rng(0)` |

**No independent third split exists yet** for the Z_free-closure check (Section
7/8) — it currently reuses the calibration split. Whether that's acceptable or
needs its own disjoint split is an open methodological question, not yet
decided.

### 3.4 Imagined-vs-real validation division of labor (TD-MPC2 height probe)

Three distinct curves were computed, deliberately NOT merged into one table —
each answers a different question:

| Curve | Action mechanism (real / imagined) | Distribution | Validity |
|---|---|---|---|
| (A) pi(z)-real vs pi(z)-imagined | matched (bare policy both sides) | WRONG — pi(z)-driven real episodes visit a fall-prone region (min height 0.04–0.13) never seen under MPC | Confounded by distribution mismatch; depth-100 p50≈0.34m is not usable as "true" dynamics drift |
| (B) MPC-real vs pi(z)-imagined | mismatched (CEM vs bare policy) | correct (real MPC episodes) | Confounded by action-mechanism mismatch; reframed as "does bare-policy imagination represent the physical consequences of real MPC deployment" — a different, valid question, not a dynamics-drift measurement |
| (C) MPC-real vs MPC-imagined | matched (CEM both sides, via `plan_from_z`) | correct (200 real MPC anchors) | **The only methodologically clean measurement.** Depth-100: p50=0.0218m, p90=0.0637m, max=0.1123m. Consistent across all 8 contributing episodes (per-episode breakdown, no tail domination). This is the curve underlying the height probe's use in the L2 calibration set. |

`plan_from_z()` (the CEM/MPPI planner entered directly at a latent z instead of
an obs) was regression-tested against the official, unmodified
`TDMPC2._plan(obs)`: fixed real-obs sequence, 6 steps, t0=False chained
(exercising the `_prev_mean` warm-start state), same seed before each step,
both eval_mode=True and eval_mode=False. Result: action and `_prev_mean`
matched **exactly** (0.00e+00 max abs diff) at every step. It is a literal copy
of `_plan()`'s body minus the initial `encode()` call, not a reconstruction.

---

## 7. Per-Reported-Number Provenance Ledger

Tier and caveat fields are my own judgment (per instruction, not delegated).
**Tier-1** = clean methodology, reproducible script exists and ran
successfully, no known unresolved confound. **Tier-2** = number exists and was
computed correctly on its own terms, but has a known, currently-unresolved gap
that must be closed (or explicitly caveated) before citing it under a specific
theorem's guarantee. **Blocked** = conflicting/ambiguous provenance, do not
cite until reconciled.

| # | Number | Layer/Theorem | Checkpoint+seed | Dataset (§3) | Script | Tier | Caveat |
|---|---|---|---|---|---|---|---|
| 1 | ρ\*=+1.126, 100/100, WARRANT (hazard_avoidance, collision semantics) | L1 / STL robustness | CarDreamer carla_four_lane | — (pre-session result, cited as "established" inside `eval_ltl_hazard_real_lppm.py`'s own header comment) | not run this session | Tier-2 | **Conflicts with a second number for the same spec** (see #2). Not independently reproduced this session. |
| 2 | ρ\*=−0.500, VIOLATION (hazard_avoidance) | L1 / STL robustness | CarDreamer carla_four_lane | — | referenced only in this session's methodology-comparison message, not run | **Blocked** | Directly contradicts #1 (WARRANT +1.126 vs VIOLATION −0.500) for what is described as the same spec. Almost certainly reflects two different semantic versions (VERIFICATION_CARDREAMER.md §I documents a "3m safety circle → collision" semantic change) that got conflated in conversation. **Must be reconciled against which semantic each number belongs to before either is cited.** |
| 3 | **⚠ L2 PROGRESS-MEASURE PATH CONFIRMED STRUCTURALLY INFEASIBLE FOR THIS SPEC — see #27 for the replacement trajectory-level statistical safety rate (`p_hat_safety=0.9243, k=97/100`). This row's `p̂_γ` is kept for provenance/diagnostic history ONLY, not as a citable result.** p̂_γ=0.1227, k=9/40, NOT warranted (threshold 0.80). The NUMBER is reproducible; the CERTIFICATE it comes from is not meaningful — see the full chain below and §8.9/§8.16/§8.17/§8.18/§8.19/§8.20/§8.21. | L2 / Theorem 5.4 (statistical) | CarDreamer carla_four_lane | §3.2 | `cardreamer/eval_ltl_hazard_real_lppm.py`, run with `XLA_FLAGS` set | **Tier-2** (downgraded from Tier-1 once the certificate itself was shown degenerate — reproducibility alone no longer implies citability) | **Full chain of custody, in order (do not skip steps when citing this number elsewhere):** (1) sampling reproducibility fixed — root cause was GPU/cuDNN conv-algorithm autotuning, not RNG seeding; `XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"` (now baked into the eval script itself) makes the full pipeline byte-reproducible, verified via 2 independent full reruns matching on every intermediate, not just the final number (§8.9). (2) That gave `p̂_γ=0.1227, k=9/40` as the first reproducible real result for this spec. (3) η sensitivity swept over B5's range {0.001,0.005,0.01,0.05,0.1}: NOT monotonic, swings ~30% relative between adjacent grid points, every value far below `warrant_threshold=0.80` (§8.16) — η=0.01's only documented justification is "matches original plan" (`experimental_setup.md:525`), no physical derivation. (4) P1-violation attribution: trained V collapsed to a near-constant ≈7×10⁻¹⁴, essentially independent of `hazard_dist`; root cause is only 2/60 (3.3%) training trajectories ever visiting `trap`, starving `p2_loss`'s gradient signal (§8.17). (5) Three remediation strategies tested (λ_P2 reweighting, trap-transition oversampling, smoothness-penalty removal) — NONE resolved the collapse (§8.18); cross-checkpoint comparison against the SAME failure mode already seen in `height_safety` in §8.19. **Bottom-line conclusion**: the certificate underlying this `p̂_γ` has not learned a meaningful dependency on `hazard_dist` — the number is a real, reproducible output of the pipeline, but it does not certify what Theorem 5.4 claims it certifies, because the V it's built from is not a real ranking function for this property. Do not cite `p̂_γ=0.1227` (or any of the historical values below) as representative of this spec's verifiability until retrained against data with meaningfully better trap coverage (§8.19's identified-but-not-executed follow-up). **Historical superseded values, kept for provenance per project convention (§8.9)**: 0.7856 (pre-root-cause-fix), 0.1831 and 0.1036 (unflagged reruns during root-cause investigation, §7 #19), an unflagged 0.1227 (§7 #21) that coincidentally matched this run's headline number without being byte-identical (differs in the 4th decimal of its own p1-residual trajectory) — none of these, including the reproducible 0.1227, should be read as "the" verified `p̂_γ` for this spec given step (5)'s finding. |
| 4 | ĉ_err ≈ 0.30 (saturated) | L1 / Transfer Calibrator | CarDreamer carla_four_lane | — | not identified this session | Tier-2 | Referenced from prior project knowledge in conversation, not reproduced or located in this session. Needs a script+commit pointer from existing project history before citing. |
| 5 | C0: R²=0.9986, MAE=0.0132m | probe validity (pre-L2) | TD-MPC2 walker-walk seed=3 | 15-episode pilot (8 MPC + 7 random, episode-grouped CV) | `tdmpc2_pilot.py` (scratchpad) | **Tier-1** | Clean. Height is a directly-encoded obs component, not pixel-inferred — probe accuracy is expected to be high and is. |
| 6 | C1 curve (A): pi(z)-real vs pi(z)-imagined, depth100 p50≈0.34m | diagnostic only, NOT a validity number | TD-MPC2 walker-walk seed=3 | new 8-episode pi(z)-driven collection | `tdmpc2_c1_decompose.py` (scratchpad) | Tier-1 (as a diagnostic) | Explicitly confounded by distribution (fall-prone region); must never be reported as "dynamics drift" — see §3.4. |
| 7 | C1 curve (B): MPC-real vs pi(z)-imagined, depth100 p50≈0.5m (original pilot curve) | diagnostic only, reframed | TD-MPC2 walker-walk seed=3 | 15-episode pilot | `tdmpc2_pilot.py` (scratchpad) | Tier-1 (as reframed diagnostic) | Confounded by action-mechanism mismatch; valid ONLY under the reframed question stated in §3.4, not as a dynamics-drift number. |
| 8 | C1 curve (C): MPC-real vs MPC-imagined, depth100 p50=0.0218m p90=0.0637m max=0.1123m | probe validity (the load-bearing one) | TD-MPC2 walker-walk seed=3 | 200 real-MPC anchors, matched mechanism+distribution | `tdmpc2_mpc_imagine_full.py` (scratchpad) | **Tier-1** | Per-episode breakdown confirmed no tail domination (all 8 episodes same order of magnitude at every depth 1–100). This is the number the L2 calibration set's probe validity rests on. |
| 9 | `plan_from_z` vs official `_plan()`: max diff 0.00e+00 (6 steps, eval_mode True/False) | infra regression, not a paper number | TD-MPC2 walker-walk seed=3 | fixed real-obs sequence | `tdmpc2_plan_consistency_test.py` (scratchpad) | **Tier-1** | Exact match, both eval_mode settings, multi-step `_prev_mean` chain exercised. |
| 10 | Physics-state restoration: max diff 5.08e-08 (5 spot-checked anchors) | infra regression | TD-MPC2 walker-walk seed=3 | recollected anchors | `tdmpc2_recollect_with_physics_state.py` (scratchpad) | **Tier-1** | Lossless to float precision. |
| 11 | Anchor-distribution bootstrap check: burn_in std=0.0306 vs bootstrap [5%,95%]=[0.0219,0.0297] at n=200 (outside range) | infra validity, not a paper number directly | TD-MPC2 walker-walk seed=3 | burn_in-sampled anchors vs C1's 200 real anchors | `tdmpc2_anchor_distribution_check4.py` + `tdmpc2_anchor_bootstrap_check.py` (scratchpad) | **Tier-1** | Confirms burn_in path is a real (not noise-explainable) distribution mismatch — motivated switching to exact physics-state anchors for the calibration set (§3.1). The bootstrap min-test in the same run is noted as **mechanically invalid** (a fixed-pool bootstrap can never sample below its own observed minimum) — do not reuse that specific test. |
| 12 | **⚠ L2 PROGRESS-MEASURE PATH CONFIRMED STRUCTURALLY INFEASIBLE — see #27 for the replacement trajectory-level statistical safety rate (`p_hat_safety=0.9765, k=199/200`). This row kept for provenance/diagnostic history ONLY.** ltl_height_safety: p̂_γ=0.0000, k=0/80, NOT warranted | L2 / Theorem 5.4 (statistical) | TD-MPC2 walker-walk seed=3 | §3.1 | `tdmpc2/eval_ltl_height_safety_lppm.py` (**repo file, not scratchpad**) | Tier-2 | See #13, #14. Genuine (1/200 total, 0/80 in this specific calib split) trap-visitation exists in the full 200 but not the calib split — see STL cross-check ρ\*=−0.2483, 199/200 satisfied, 1 genuine violation traced to a real near-fall episode. Z_free-interior closure checked (#18, §8.2): UNVERIFIABLE — "Pr[τ⊨φ]≥p̂_γ" not licensed under Theorem 5.4's minimal premise, continue citing only against C(τ). |
| 13 | height_safety k=0 attribution: V≈0.5882–0.5884 (near-constant, close to softplus(0)≈0.693 untrained baseline), P1 "violations" 3494/7920 transitions (magnitude ≤1e-4), P2 violations 0/7920, terminal Z_free 0/80 | diagnostic | TD-MPC2 walker-walk seed=3 | §3.1 calib split | `tdmpc2_height_safety_k0_diagnosis.py` (scratchpad) | **Tier-1** (as a diagnostic; the k=0 result it explains is Tier-2, see #12) | V collapsed to a near-untrained constant because the calib split has zero trap-visiting rollouts → zero P2 gradient signal → P1-only training has a trivial (constant) global optimum. This is being recorded as a documented degenerate/negative case (property-liveness failure), not something to be tuned away. |
| 14 | height_safety with data-derived P1 tolerance (3×std=0.000028): P1-failing rollouts drop from 80/80 to 11/80; k stays 0/80; p̂_γ stays 0.0000 | diagnostic, confirms root cause isolation | TD-MPC2 walker-walk seed=3 | §3.1 calib split | `tdmpc2_height_safety_k0_diagnosis_v2.py` (scratchpad), using new `p1_tol` param in `core/lppm/verifier.py` (repo file) | **Tier-1** | Confirms the P1 noise cleanup does NOT change p̂_γ — the entire k=0 result is attributable to the terminal-Z_free condition, not P1 noise. `p1_tol` defaults to 0.0 (old strict behavior preserved) elsewhere in the project; regression-confirmed. |
| 15 | hazard_avoidance Z_free-interior closure: 0/1960 calib transitions have source V<eta=0.01 (source V range 0.6633–0.9013) | Theorem 5.4 structural-premise check | CarDreamer carla_four_lane | §3.2 calib split (reused, no independent split) | `hazard_zfree_closure_check.py` (scratchpad) | **Tier-1** (as a "insufficient sample, correctly stopped" result) | Confirms #3's caveat is real and not TD-MPC2-specific: the SAME collapsed-V pattern occurs independently on a different checkpoint/spec/codebase-path. Strengthens the case that this is a systemic NeuralLPPM training-objective issue (P1-only trivial optimum under trap-sparse data), not an isolated quirk. |
| 16 | GPU vs CPU TD-MPC2: 17.9x speedup, 403 MiB peak GPU memory | infra decision | TD-MPC2 walker-walk seed=3 | — | **script name not preserved** (pre-compaction in this session) | Tier-2 | Numbers are trusted (were reported carefully with a tight polling loop after an earlier misleading near-zero reading), but the exact script is not locatable in current scratchpad listing — rerun/relocate before citing in a reproducibility appendix. |
| 17 | `support_level`/`automaton_translation` label fix; `coverage_verified` marker; P1 tolerance mechanism; theorem citation corrections | code correctness, not an experimental number | — | — | `main.py`, `utils/spec_analysis.py`, `core/cegar/loop.py`, `core/lppm/verifier.py`, `core/lppm/calibrator.py`, `core/transfer_calibrator.py` (all repo files, this session) | **Tier-1** | All regression-confirmed (numeric outputs byte-identical pre/post for existing callers; new fields default to values that preserve old behavior). See Section 8 for the full list. |
| 18 | height_safety Z_free-interior closure: n_zfree=0 → UNVERIFIABLE (reproduced #12's calib split exactly: p̂_γ=0.0000, k=0/80, byte-identical) | Theorem 5.4 structural-premise check | TD-MPC2 walker-walk seed=3 | §3.1 calib split (reused, no independent split) | `tdmpc2/eval_ltl_height_safety_lppm.py` step [8] (**repo file**, calls new `core/lppm/verifier.py::verify_zfree_closure()`) | **Tier-1** (as a "insufficient sample, correctly stopped" result) | Confirms #15/#13's collapsed-V pattern via the new dedicated, tested function (`tests/test_zfree_closure.py`) rather than an ad hoc script — same conclusion, now on reusable infrastructure. |
| 19 | `cardreamer/eval_ltl_hazard_real_lppm.py` reruns (this round, 2×) did NOT reproduce #3's ledgered p̂_γ=0.7856: got p̂_γ=0.1831 (1500 epochs) and p̂_γ=0.1036 (8000 epochs), both with V collapsed to ≈0 on 100% of calib transitions | infra reproducibility finding, not a paper number | CarDreamer carla_four_lane | §3.2, freshly re-sampled (NOT #3's original draw) | `cardreamer/eval_ltl_hazard_real_lppm.py` (repo file) + `hazard_longer_training_check.py` (scratchpad, 8000-epoch variant) | **Tier-1** (as a reproducibility finding) | See §8.9. Rules out under-training as the cause (5.3× more epochs, same collapse, stable from epoch≈1000). Points to `CarDreamerWrapper.sample_rollouts()` (JAX/GPU) not reproducing its own fixed seed. Blocks re-verifying #3/#15 with fresh runs until root-caused or the original run's data/weights are cached. |
| 20 | `ltl_height_safety` post-§8.11-fix rerun: p̂_γ=0.0000, k=0/80 -- exactly unchanged from #12 | L2 / Theorem 5.4 (statistical) | TD-MPC2 walker-walk seed=3 | §3.1, SAME reproducible pipeline as #12 | `tdmpc2/eval_ltl_height_safety_lppm.py` (repo file) | **Tier-1** (as a clean before/after comparison) | This spec's pipeline is exactly reproducible (unlike CarDreamer's, #19) -- confirms §8.11's terminal-transition bug fix had ZERO effect on this specific result (all 80 calib rollouts were already failing via P1 violations spread throughout, not concentrated on the final transition). The bug itself is still real and still fixed; this number just wasn't sensitive to it. |
| 21 | `ltl_hazard_avoidance` post-§8.11-fix rerun: p̂_γ=0.1227, k=9/40 | L2 / Theorem 5.4 (statistical) | CarDreamer carla_four_lane | §3.2, freshly re-sampled (3rd independent draw this session, see #19) | `cardreamer/eval_ltl_hazard_real_lppm.py` (repo file) | Tier-2 | Do NOT read this as "the bug-fixed 0.7856" -- per #19/§8.9, this pipeline's calibration draw is not reproducible across runs at all, so this number, #19's two reruns, and the original 0.7856 are four independent, mutually incomparable draws. The §8.11 fix is confirmed correct via synthetic counterexamples (not via this number) -- this spec offers no clean before/after signal either way. |
| 22 | CarDreamer sampling non-reproducibility (§8.9/#19): root cause = GPU/cuDNN conv-algorithm autotuning (NOT RNG seeding, which was confirmed byte-identical across runs). Fix `XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"` verified at the sampling layer (2 runs, identical SHA-256 output hash) AND end-to-end through the full `fit_lppm`/`calibrate_lppm` pipeline (3 total runs of `cardreamer/eval_ltl_hazard_real_lppm.py` with the fix -- 2 with the env var set externally, 1 relying on the fix now being baked into the script itself via `os.environ` -- all 3 produced identical `p̂_γ=0.1227, k=9/40` and identical full intermediate output) | infra root-cause + fix, now load-bearing for a real reported number | CarDreamer carla_four_lane | §3.2 | `cardreamer_rng_diagnosis.py` (scratchpad, sampling-layer isolation) + `cardreamer/eval_ltl_hazard_real_lppm.py` (repo file, now self-contained -- sets the flag itself, no external env var needed) | **Tier-1** | This is what makes Section 7 #3's `p̂_γ=0.1227` a Tier-1, reproducible number -- the first one this project has had for `ltl_hazard_avoidance`. Applies identically to `carla_roundabout` if/when that checkpoint is ever wired into the LPPM path (see §8.9's roundabout note); that script would need the same `os.environ["XLA_FLAGS"]` line added. |
| 23 | `ltl_hazard_avoidance` η (B5) sensitivity sweep, η∈{0.001,0.005,0.01,0.05,0.1}, NeuralLPPM retrained per η on the identical §7 #3 calibration data: p̂_γ = 0.1036, 0.1036, **0.1227 (current default)**, 0.0851, 0.0000 respectively (k=8,8,9,7,0 out of 40) | L2 / Theorem 5.4, sensitivity ablation (B5) | CarDreamer carla_four_lane | §3.2, same split as §7 #3 (reused, not resampled) | `hazard_eta_sensitivity_sweep.py` (scratchpad) | **Tier-1** (as a sensitivity finding) | See §8.16. Result is NOT monotonic in η -- P1 failures dominate and are η-insensitive (31-33/40 throughout); the η=0.1 collapse to 0 is a training-convergence failure (final P1 loss ≠ 0), not evidence "larger η is worse" per se. Every tested η gives NOT-warranted (all far below 0.80 threshold); the ~30% relative swing between adjacent grid points means the current default's specific `p̂_γ=0.1227` should not be cited without this context. η=0.01's only documented justification (`experimental_setup.md:525`) is "matches original plan" -- no physical/theoretical derivation exists. |
| 24 | P1-violation attribution for `ltl_hazard_avoidance` (η=0.01): trained V collapsed to a near-constant ≈7×10⁻¹⁴ everywhere (raw logits mean=-30.41, std=0.35, essentially independent of `hazard_dist`∈[-0.25,32.0]); flagged P1 "violations" are float-noise-scale deltas (~1e-16), 6-8 orders of magnitude below V itself; root cause: only 2/60 (3.3%) TRAINING trajectories ever visit trap, so only 60/2940 (2.0%) of training transitions carry any P2 gradient signal, leaving P1's trivial-constant global optimum to dominate | diagnostic, confirms/quantifies §7 #15's suspected systemic issue | CarDreamer carla_four_lane | §3.2, same split as §7 #3/#23 | `hazard_p1_violation_attribution.py`, `hazard_v_precision_check.py` (both scratchpad) | **Tier-1** (as a diagnostic; the low-p̂_γ result it explains is Tier-1/reliable but low, §7 #3) | See §8.17. CONFIRMS this is a training-configuration/data-sparsity problem (P1-only trivial optimum under trap-sparse data), NOT evidence that no valid LPPM exists for this spec under Corollary 5.3 -- same mechanism already diagnosed for `height_safety` (§7 #13/#14), now directly evidenced (not just inferred) on a second, independent checkpoint. |
| 25 | Remediation test for §7 #24's diagnosis: λ_P2 reweighting {1,5,10}, trap-transition oversampling {5x,10x,20x}+combined, and smoothness-penalty removal -- ALL tested, NONE produced a network with both converged P1 and a strong/consistent V-dependency on `hazard_dist` (correlation stayed weak and sign-inconsistent: -0.29 to +0.24 across conditions with P1 converged; the one strong correlation, -0.57 at λ=10, broke P1 convergence instead) | negative result, training-remediation ablation | CarDreamer carla_four_lane | §3.2, same split as §7 #3/#24 (no new data collection, per explicit scoping) | `hazard_p2_starvation_remediation.py`, `hazard_p2_starvation_oversample.py`, `hazard_smoothness_confound_check.py` (all scratchpad) | **Tier-1** (as a negative result) | See §8.18. Explicitly bounded conclusion: cheap/standard training-configuration fixes did not resolve the collapse WITHIN THE "no new data" CONSTRAINT this round was scoped under -- this is NOT sufficient grounds to conclude no valid LPPM exists (Corollary 5.3 sense), since architecture/hyperparameter search was not exhaustive and the most promising untested lever (genuinely new trap-visiting rollouts, not duplicates of the existing ~60) requires relaxing that constraint. Do not cite this as "LPPM is impossible for this spec" -- it isn't shown. |
| 26 | Trap dwell/recovery diagnosis on RAW physical signal (both real checkpoints, no automaton abstraction): hazard_avoidance 6 violation episodes across 3/100 trajectories (lengths 1,1,2,2,3,6,11 steps -- mean 4.17, median 2.5), height_safety 1 episode across 1/200 trajectories (length exactly 4 steps). **100% of episodes (7/7 total) recovered above threshold within the same trajectory; 0% stayed violated until the rollout horizon ended** | diagnostic, informs (does not resolve) the §8.19 structural-tension framing | both real checkpoints | §3.1 (height_safety) + §3.2 (hazard_avoidance), same data as §7 #3/#12, no new collection | `trap_dwell_diagnosis_hazard.py`, `trap_dwell_diagnosis_height.py` (both scratchpad) | **Tier-1** | See §8.20. Confirms `core/lppm/automaton.py`'s Safety-branch `trap` has no escape transition BY CONSTRUCTION (deliberate `G(not p)` semantics, not a bug) -- but the real physical systems, on both checkpoints, never actually dwell in violation for a sustained/self-reinforcing period; every observed excursion resolves on its own. Reported as evidence for the reader to weigh, not used here to choose between the "trap is genuinely reachable-and-self-looping, retreat to trajectory-level statistical safety rate" framing vs. "trap-sparsity/short-excursion is the dominant practical driver" framing -- that choice is explicitly left open. |
| 27 | `ltl_hazard_avoidance`: **trajectory-level statistical safety rate** `p_hat_safety=0.9243, k=97/100, gamma=0.05` -- the REPLACEMENT result after #3's L2 progress-measure path was confirmed structurally infeasible | Trajectory-level statistical safety rate (independent, WEAKER guarantee type than Theorem 5.4 -- see §8.21) | CarDreamer carla_four_lane | §3.2, SAME reproducible sampling as #3 (seed=0, `XLA_FLAGS`), no new data collection | `core/trajectory_calibration.py` (repo file, new this round) + `hazard_trajectory_safety_rate.py` (scratchpad runner) | **Tier-1** | See §8.21. NOT a Theorem-5.4 warrant -- `C(tau)=1[G(hazard_dist>0) holds at every step]` checked directly on raw trajectories, no certificate network, no automaton, no absorption-witness/progress-measure structure. Reuses `core.lppm.calibrator._clopper_pearson_lower` (not reimplemented). k=97/100 matches exactly the 3/100 trajectories with a raw violation identified in §7 #26's dwell diagnosis (same data). Do not compare `p_hat_safety` against `warrant_threshold=0.80` as if it were a `p_hat_gamma` -- the two are different guarantee types measuring different events. |
| 28 | `ltl_height_safety`: **trajectory-level statistical safety rate** `p_hat_safety=0.9765, k=199/200, gamma=0.05` -- the REPLACEMENT result after #12's L2 progress-measure path was confirmed structurally infeasible | Trajectory-level statistical safety rate (independent, WEAKER guarantee type than Theorem 5.4 -- see §8.21) | TD-MPC2 walker-walk seed=3 | §3.1, SAME 200-anchor real-MPC pool as #12, no new data collection | `core/trajectory_calibration.py` (repo file, new this round) + `height_trajectory_safety_rate.py` (scratchpad runner) | **Tier-1** | See §8.21. NOT a Theorem-5.4 warrant -- same methodology as #27, `C(tau)=1[G(height>0.6) holds at every step]`. k=199/200 matches exactly the 1/200 trajectories with a raw violation identified in §7 #26's dwell diagnosis (same data) and the STL cross-check (#12's own caveat: "199/200 satisfied"). |

---

## 8. Known Gaps & Code-Paper Consistency

### 8.1 Deductive L2 branch: not implemented anywhere

The methodology describes L2 as having two branches: deductive (NN
verification, support-wide, zero-miscoverage) and statistical (Theorem 5.4,
CP-calibrated). Only the statistical branch exists in this codebase.
`core/lppm/trainer.py::fit_lppm` trains via gradient descent on **sampled**
transitions; `core/lppm/calibrator.py::calibrate_lppm` calibrates via
Clopper-Pearson on a **held-out sample**. `core/cegar/*` was checked as a
candidate deductive branch and ruled out: its abstraction is built from
**sampled trajectory data**, and even its own counterexample-concretization
check (`counterexample.py`) is itself CP-calibrated, not a formal
(bound-propagation/SMT/abstract-interpretation) proof over network weights.

**Status this session**: sealed off, not implemented. `main.py`'s
`support_level` can no longer silently become the reserved "deductive" value
anywhere (see 8.3) — every current p̂_γ in this project is a Theorem 5.4
statistical-branch number, and this is now enforced by an assertion, not just
documentation.

### 8.2 Z_free-interior closure gap — VERIFIED this round, not yet licensable for either existing reported number

Theorem 5.4's structural premise — "(P1)-(P2) hold at every admissible
transition whose SOURCE lies in Z_free" — is strictly weaker than what
`core/lppm/verifier.py::check_pathwise_conditions()`'s C(τ) actually checks
(P1/P2 across the **entire finite rollout**, plus Z_free membership **only at
the terminal step**). Existing p̂_γ numbers computed against C(τ) remain
valid lower bounds for this weaker Theorem-5.4-exact event (a stronger
event's probability lower-bounds a weaker one's) but could not, until this
round, be cited as satisfying the theorem's *minimal* premise specifically —
no code isolated the Z_free-restricted subset to check.

**Verified this round.** `core/lppm/verifier.py::verify_zfree_closure()` (new,
independent of `check_pathwise_conditions()` by design — does not share its
loop) isolates exactly the (transition, head) pairs whose source lies in
Z_free = {(z,q): V<η}, checks (P1)-(P2) on that subset alone, and reports a
Clopper-Pearson lower bound `p̂_closure` (reusing `calibrator.py`'s existing CP
function, not reimplementing it) — or, when the subset is empty, explicitly
`verifiable=False, p̂_closure=None` rather than a number that would look
statistically meaningful but isn't. Test-driven (`tests/test_zfree_closure.py`,
4 tests, all confirmed to fail before the function existed and pass after).

Run against both specs' existing calibration data:

- **`ltl_height_safety`** (TD-MPC2 walker-walk): `n_zfree=0` →
  **UNVERIFIABLE**, on a fresh reproduction of the exact calibration split
  that produced the ledgered `p̂_γ=0.0000` (Section 7 #12) — reproduced that
  number exactly (`k=0/80`), confirming this is the same run. Consistent with
  the pre-existing V-collapse diagnosis (#13): V never gets anywhere near η.
- **`ltl_hazard_avoidance`** (CarDreamer carla_four_lane): the pre-existing
  ad hoc diagnostic behind ledger #15 (`0/1960`, source V∈[0.66,0.90], same
  calibration draw that produced the ledgered `p̂_γ=0.7856`) is retained as
  the answer for this spec — **not** re-derived with the new function this
  round. A fresh attempt to rerun the full pipeline (needed to call the new
  function against real data) could not reproduce the ledgered 0.7856: it
  landed on `p̂_γ≈0.10–0.18` across two separate reruns (one at the
  production script's 1500 training epochs, one extended to 8000 epochs to
  rule out under-training — same outcome both times, see 8.9), with V
  collapsed to ≈0 on 100% of transitions rather than the historical
  0.66–0.90 range. Root cause: `CarDreamerWrapper.sample_rollouts()`'s
  JAX/GPU-based world-model imagination step does not appear to reproduce
  identical trajectories run-to-run even at fixed `seed=0`, so a fresh
  training run can land in a qualitatively different (here, degenerate
  collapsed-V) local optimum than the one behind the ledgered number. Per
  explicit decision: cite the existing ad hoc diagnostic (#15) rather than
  this round's non-reproducing reruns, since #15 is the one actually computed
  on the calibration draw the ledgered 0.7856 came from.

**Net conclusion — neither existing number is upgraded to the precise
"satisfies Theorem 5.4's minimal premise" wording.** Both come back
`UNVERIFIABLE` (zero Z_free-source samples), not "verified below threshold" —
there is no evidence either way, so the more precise citation is not
licensed. Continue citing both `p̂_γ=0.7856` (#3) and `p̂_γ=0.0000` (#12) only
as Clopper-Pearson lower bounds on the strictly stronger whole-trajectory
event C(τ), not as satisfying Theorem 5.4's minimal premise. This is the same
conclusion the pre-existing ad hoc diagnostic (#15) already pointed to for
hazard_avoidance — this round adds height_safety's confirmation, reusable
tested infrastructure (`verify_zfree_closure()`), and the CarDreamer
reproducibility finding (8.9) as a distinct, newly-discovered blocker on ever
closing this gap for that spec via more reruns of the current pipeline.

Remediation path (reported, not yet executed): dedicated Z_free-interior
anchor resampling via the resettable-generative oracle (Definition 4.1's
off-support reset explicitly licenses resetting to any z, including
off-support states engineered to already satisfy V<eta) — this is the
cheapest way to get qualifying samples without waiting for V to naturally
visit that region. A sampled closure check would introduce its own δ term
(closure confidence × reachability confidence, union bound) into the overall
guarantee; alternatively, since Z_free is a compact sublevel set, this is
flagged as the sub-problem best suited to an NN-verification upgrade if the
deductive branch (8.1) is ever built. (For hazard_avoidance specifically,
this remediation is blocked on 8.9's reproducibility issue first — resampling
more anchors from a pipeline that doesn't reproduce its own seed doesn't
obviously help.)

### 8.3 `support_level` label conflation — fixed this session

`main.py:571-580` (pre-fix) upgraded `support_level` to `"sound"` whenever the
LTL→parity-automaton translation was exact (Spot backend), conflating
"automaton encoding is correct" with "certificate is deductively verified" —
two different things per 8.1. Fixed:
- New `automaton_translation` field (independent, tracks exact/template).
- `support_level` can only be `SUPPORT_DEDUCTIVE` / `SUPPORT_CALIBRATED` /
  `SUPPORT_APPROXIMATE` (new enum in `utils/spec_analysis.py`); bounded STL's
  previous default of `"sound"` was renamed to `SUPPORT_CALIBRATED` (Theorem
  5.1's guarantee is `1-δ_cp-δ_err`, not deterministic — same conflation,
  lighter version, at the L1 default).
- An assertion in `main.py`'s LPPM step fails loud if `support_level` is ever
  `SUPPORT_DEDUCTIVE` without a real deductive verifier attached (none exists,
  so this should never fire).
- Dormant-bug scope check: `spot` is not installed in any of this project's
  conda environments (`dyno`, `cardreamer`, `base`) — this bug never actually
  fired for any historical reported number, confirmed by checking that neither
  `ltl_hazard_avoidance`'s nor `ltl_height_safety`'s reported numbers ever went
  through `main.py::verify()` (both were produced by standalone scripts calling
  `build_parity_automaton`/`fit_lppm`/`calibrate_lppm` directly).
- Regression: all numeric outputs confirmed byte-identical pre/post-fix; only
  the label string and note text changed.

### 8.4 P1 numeric tolerance — added this session

`check_pathwise_conditions`'s P1 check previously used a fixed `1e-6` epsilon,
far below the ~1e-4 float-noise floor observed when a trained V collapses to
a near-constant output (see 8.2's root cause chain). Added an optional
`p1_tol` parameter (default 0.0, preserving old strict behavior for existing
callers — regression-confirmed), recorded on both `PathwiseResult` and
`LPPMResult` for auditability. Guidance documented in the function's docstring:
derive `p1_tol` from the calibration run's own V distribution, not a hardcoded
constant, and report the value alongside any p̂_γ it affected.

### 8.5 CEGAR `coverage_verified` — docstring honesty fix

`CegarResult`'s `SAFE` verdict docstring previously asserted "Sound when the
sample covers all reachable cells" with no corresponding checked field. Per
instruction, implementing the actual coverage check is backlog (not done);
this session instead (a) rewrote the docstring to state plainly that coverage
is NOT verified, and (b) added an explicit `coverage_verified: bool = False`
field (always False — no code path sets it True) so callers cannot mistake
`verdict==SAFE` for "coverage checked." `main.py`'s `summary()` now appends
`[reachable-cell coverage NOT verified]` next to any SAFE verdict. Regression-
confirmed via `run_oneshot()`.

### 8.6 Silent-fallback inventory

| Location | Fallback | Currently surfaced? |
|---|---|---|
| `core/lppm/trainer.py::_fit_lppm_heuristic_fallback` | If torch/NeuralLPPM/F unavailable, `fit_lppm` silently switches to a heuristic epoch-loss trainer instead of gradient-trained NeuralLPPM. | Partially — `main.py` prints an explicit WARNING when `backend != "torch_mlp"`, but only if `cfg.verbose=True`; a caller reading only the numeric `p_hat_gamma` with verbose off would miss it. |
| `core/lppm/model.py::compute_lppm_value` | If `lppm_params` is None or `predict_learned_lppm_value` returns None (missing keys), silently falls back to a hand-written heuristic value function per mp_class. | Same as above — gated behind `cfg.verbose`. |
| `hazard_zfree_closure_check.py` (this session's script) | Reimplements the Clopper-Pearson formula inline (`beta_dist.ppf(...)`) instead of importing `core/lppm/calibrator.py::_clopper_pearson_lower`. | N/A — not a project-code issue, a minor duplication in a scratchpad diagnostic script; flag for cleanup if this script is promoted to a permanent artifact. No second CP implementation exists inside `core/` itself — confirmed by grep, only one call site. |

Whether any of these should become a documented "methodological contribution"
(e.g., explicit three-valued fallback semantics) or just stay a
verbose-gated warning is a decision left to the owner — not made here.

### 8.7 `main.py::verify()` Step 3 disjoint-split violation — confirmed dormant, now guarded

`main.py::verify()`'s Step 3 (`fit_lppm_params=True` branch) calls
`fit_lppm(trajectories=trajectories, ...)` and then
`calibrate_lppm(trajectories=trajectories, ...)` with the **same**
`trajectories` list — no train/calibration split at all. This is not a new
finding in the sense of being previously unknown to the project: the
`cardreamer/eval_ltl_hazard_real_lppm.py` header comment already names this
exact bug verbatim ("train/calibration split MUST be disjoint (main.py
currently reuses the same `trajectories` for both fit_lppm and calibrate_lppm
— that would be an optimistic/overfit p_hat_gamma; this script splits
properly)") as the reason that script hand-rolls its own split instead of
calling `verify()`. What was missing was (a) promoting it out of a script
header aside into the gap ledger with its theorem-relevance made explicit,
and (b) a runtime guard.

**Why this is more basic than the other §8 items**: exchangeability of the
calibration draw (and hence disjointness from whatever fit `V_phi`) is a
load-bearing premise of Theorem 5.4's PAC bound itself, not an
implementation-completeness gap like the missing deductive branch (§8.1) or
an unverified structural premise like Z_free interior closure (§8.2) — if
this path had ever produced a reported number, that number's `p_hat_gamma`
would not carry the claimed coverage guarantee at all, not just carry it
provisionally.

**Confirmed dormant** (not just "the two real eval scripts don't use it" —
verified via import-graph grep, not just reading): `fit_lppm_params` as a
string appears nowhere in the repo except inside `main.py` itself — no CLI
flag exposes it, no `VerifyConfig(...)` construction site (CLI entry point,
`verify_from_wrapper`, or elsewhere) sets it, and grepping for any caller of
`main.py::verify()`/`from main import` outside `main.py`'s own docstring
example returns nothing. `cardreamer/eval_ltl_hazard_real_lppm.py` and
`tdmpc2/eval_ltl_height_safety_lppm.py` — the only two scripts that have ever
produced a reported `p_hat_gamma` (0.7856, 0.0000) — both import directly
from `core.lppm.*` and bypass `main.py` entirely. **The two existing reported
numbers are unaffected**; no rerun needed.

**Fix applied this session**: added an opt-in disjointness check —
`find_trajectory_overlap()` (`core/lppm/verifier.py`, identity-based via
`id()`, not content-equality, since content-identical-but-independently-drawn
trajectories are exactly the degenerate case — e.g. height_safety's k=0 run —
where a content-based check would false-positive) — surfaced through new
`calib_trajectories`/`warn_if_overlap` params on `fit_lppm()` and
`train_trajectories`/`warn_if_overlap` on `calibrate_lppm()`, both defaulting
to off (zero behavior change for existing callers). `main.py`'s Step 3 now
passes `warn_if_overlap=True` so any future accidental use of
`fit_lppm_params=True` emits a `UserWarning` instead of silently violating
Theorem 5.4's premise. This does **not** fix the underlying API — `verify()`
still has no way to accept a real second split; that needs a `VerifyConfig`
signature change and is left as backlog, not attempted here.

### 8.7b TD-MPC2 `cfg.seed` does not control CEM/MPPI action sampling — touches the reported `height_safety` result

Discovered as a side effect of Stage-4 LayerNorm variance measurement
(`core/lbsm/measure_layernorm_variance_tdmpc2.py`), not the original goal of
that work: running the same nominal seed twice, in separate process
invocations, produced different measured minima. Traced to root cause in the
upstream TD-MPC2 source (`tdmpc2_src/tdmpc2`):

- `common/seed.py::set_seed(seed)` exists and seeds `random`, `numpy`,
  `torch`, and `torch.cuda` comprehensively — and the *official* upstream
  entry point `train.py` calls it (`train.py:49`).
- **This project's `wrappers/tdmpc2_wrapper.py::load()` never calls
  `set_seed()`, `torch.manual_seed()`, `np.random.seed()`, or
  `random.seed()` anywhere** (confirmed by grep — zero hits).
- The only place `cfg.seed` is actually consumed is
  `tdmpc2_src/tdmpc2/envs/dmcontrol.py:104`:
  `suite.load(domain, task, task_kwargs={'random': cfg.seed}, ...)` — this
  seeds *only* dm_control's own environment-reset randomization (e.g.
  initial joint angles at `initialize_episode()`).
- CEM/MPPI's actual action sampling (`tdmpc2.py::_plan()`, the
  `r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, ...)`
  line inside the MPPI iteration loop) draws directly from the **global**
  torch RNG stream, with no local generator and no re-seeding — so it is
  governed by whatever process-level torch RNG state happens to exist, not
  by `cfg.seed`.

**Net effect**: passing `seed=N` to `TDMPC2Wrapper.load()` (or to
`RolloutConfig`) gives a false impression of full reproducibility. It pins
dm_control's episode-reset randomization only; it does **not** pin the CEM
action-sampling that generates the bulk of a rollout's actual content
whenever `action_source="mpc_plan"` — which is this project's documented
*primary* TD-MPC2 action mechanism (`EXPERIMENT_PARAMETERS.md` §1).

**Scope of impact, checked directly, not assumed**:
- **Does NOT touch CarDreamer/`hazard_avoidance`'s `p̂_γ`** — CarDreamer uses
  a trained deployment policy (`action_source="actor"`), never CEM; this gap
  is TD-MPC2-specific. (The parenthetical in this entry's originating
  discussion floated hazard_avoidance as a hypothetical example — checked
  and ruled out.)
- **Does touch the reported `height_safety` result** (`p̂_γ=0.0000`,
  Section 7 ledger): `tdmpc2/eval_ltl_height_safety_lppm.py:81` calls
  `w.load(checkpoint=CKPT, task="walker-walk", seed=3)`, and that script's
  rollout collection uses `action_source="mpc_plan"` throughout — so the
  *trajectories themselves*, not just the LPPM network weights, are not
  reproducible from the documented `seed=3`. This is **broader than, and
  additional to**, the already-documented `EXPERIMENT_PARAMETERS.md` §5 gap
  ("`fit_lppm()` never seeds torch") — that entry describes the LPPM
  *network init* as unseeded; this entry additionally covers the *rollout
  data generation* itself being unseeded, a separate variance source. Grepped
  the script directly: it has `np.random.default_rng(0)` for the train/calib
  split only, no `torch.manual_seed()` call anywhere.
- Does **not** invalidate the reported `0.0000` number (it is a real
  measurement of what that specific run actually produced) — it means the
  number is not exactly reproducible by "rerun with seed=3" as the script's
  own call signature implies, and any future rerun should not assume
  agreement with the original run just because the same `seed=` argument was
  passed.

**Not fixed here** (out of scope for a diagnostic aside) — flagged so a
future session doesn't assume "TD-MPC2 seed=N is fully reproducible." A real
fix would call `common.seed.set_seed(cfg.seed)` (or at minimum
`torch.manual_seed(...)`) inside `TDMPC2Wrapper.load()`, before any CEM
planning occurs.

### 8.7c Declared `mp_class="Response"` never matches `infer_mp_class()`'s output vocabulary — label-only, status: initial judgment, not verified

Found during the L2 automaton-audit batches (batch 2's 30-spec classification
table). Four STL specs (`stl_gap_recovery`, `stl_gap_response`,
`stl_obstacle_response_human_task`, `stl_human_proximity_response`) declare
`"mp_class": "Response"` in their spec dict, but
`utils/spec_analysis.py::infer_mp_class()` **can never return the string
`"Response"`** — its output vocabulary is `{Recurrence, Streett, Reactivity,
Persistence, Obligation, Guarantee, Safety, Unclassified}`. At runtime these
four specs compute `mp_class="Recurrence"` (confirmed by direct computation,
not assumed).

**Current judgment (initial, not independently re-verified): this is a
labeling-vocabulary mismatch, not a routing bug.** The declared field is
never read by `build_parity_automaton()` or `main.py::verify()` — both
recompute `analysis = spec.get("analysis") or analyze_spec_structure(spec)`
and use ONLY the freshly-inferred `mp_class`, so the declared string never
drives a real dispatch decision. Additionally, all four specs are bounded
(`verification_mode="finite_stl"`), so they never reach the LPPM/L2 pipeline
at all regardless of `mp_class` — `main.py::verify()`'s bounded-path early
return (STL-only) happens before Step 3's automaton construction.

**Why this still belongs in the gap list despite not affecting L2**: any
future script that reads a spec's *declared* `"mp_class"` field directly
(e.g. a paper-table generator, or a coverage-audit script) rather than
calling `analyze_spec_structure()` would see `"Response"` and could
misclassify these four specs in that downstream artifact, even though the
verification pipeline itself is unaffected. Not fixed here — recorded so a
future session doesn't have to rediscover it, and so "initial judgment" does
not silently harden into "verified" without anyone re-checking it.

### 8.8 Not yet checked this session (explicitly out of scope, not silently skipped)

- Whether `ltl_hazard_avoidance`'s ρ\* conflict (#1 vs #2 in Section 7) reflects
  a genuine semantic-version mixup or a transcription error in conversation.
- ~~Whether the CarDreamer Z_free-closure result (0/1960) would change under a
  larger or independently-drawn calibration split (only one split was tried).~~
  Answered (partially, unexpectedly) this round — see 8.9: an independently-*
  drawn* split isn't even reachable on demand, because the sampling pipeline
  itself doesn't reproduce its own fixed seed. Whether a **successfully
  reproduced** 0.66–0.90-range V distribution would show closure under a
  different held-out split remains untested.
- `Corollary 5.2` and `Theorem D.4` citations — left with TODO markers,
  unverified against the full paper text.

### 8.9 `CarDreamerWrapper.sample_rollouts()` does not reproduce across runs at fixed `seed=0` — RESOLVED this round: root cause found (GPU/cuDNN conv-algorithm autotuning, NOT an unseeded RNG), fix verified end-to-end through the full `fit_lppm`/`calibrate_lppm` pipeline, `ltl_hazard_avoidance` now has its first reproducible real result (§7 #3)

**Status as of the root-cause round**: root cause is now identified and the
fix is verified at the sampling layer (see "Root cause, confirmed" below).
Three runs of the identical script/checkpoint/seed had previously produced
three different `p̂_γ` numbers (0.7856 / 0.1831 / 0.1227, §7 #3/#19/#21).
Investigation was deferred one round past when it was first requested
(deprioritized in favor of six correctness fixes, without prior user
sign-off) before being done this round.

**Root cause, confirmed by direct experiment (methodology: same as the
TD-MPC2 CEM-seed investigation this project did earlier)**:

1. Traced `RolloutConfig.seed` end to end. It DOES fully and correctly seed
   the numpy-side data selection: `wrappers/cardreamer_wrapper.py::
   CarDreamerWrapper._sample_replay_batch()`'s `np.random.default_rng(seed)`
   controls which replay episodes/burn-in start points get used, and is NOT
   the problem.
2. The actual per-step POLICY ACTION sampling during `wm.imagine(policy,
   start, horizon)` (`_imagine_and_decode()`'s `policy(state): return
   actor(state).sample(seed=nj.rng())`) draws from ninjax's RNG threading,
   which is seeded per-call from `self._jax_agent._next_rngs(...)` —
   `dreamerv3/jaxagent.py::JAXAgent.__init__` sets `self.rng =
   np.random.default_rng(config.seed)`, where `config.seed` is the value
   baked into the CHECKPOINT's own saved training config YAML (fixed,
   unrelated to `RolloutConfig.seed`, but *also* fixed/deterministic run to
   run for a given checkpoint file). **Empirically confirmed this numpy
   `PCG64` state (`self._jax_agent.rng`) is byte-identical across separate
   process runs** (same `bit_generator` state, same next-draw value,
   verified via a diagnostic script printing it right after `w.load()`,
   before any sampling) — so despite superficially resembling the TD-MPC2
   case (a checkpoint-config seed shadowing the caller's seed), **the RNG
   seeding itself is NOT the source of nondeterminism here** — it is fully
   deterministic and reproducible on its own.
3. With RNG seeding ruled out, tested at small scale (n=2, horizon=3): two
   independent process runs produced a byte-identical SHA-256 hash of the
   full sampled output. Nondeterminism did NOT reproduce at this scale.
4. Tested at the real eval script's actual scale (n=100, horizon=50, real
   trained actor policy — matching `cardreamer/eval_ltl_hazard_real_lppm.py`
   exactly): two independent runs, IDENTICAL RNG state confirmed on both,
   produced DIFFERENT output hashes (`469cb2ef...` vs `b2fceff1...`).
   Nondeterminism reproduces at this scale despite provably identical RNG
   state feeding the computation.
5. This isolates the cause to the GPU computation itself, not seeding. The
   earlier full eval script logs show XLA autotuning cuDNN convolution
   algorithms at runtime (`"Trying algorithm eng23{...} for conv ... is
   taking a while"`) — JAX/XLA's default behavior benchmarks several
   candidate cuDNN conv implementations at trace time and picks whichever
   times fastest *at that moment*; different algorithms are not required to
   (and empirically do not) produce bit-identical floating-point results for
   the same inputs. This is a well-known cuDNN/XLA behavior, not specific to
   this codebase.

**Fix, verified**: setting `XLA_FLAGS="--xla_gpu_autotune_level=0
--xla_gpu_deterministic_ops=true"` before the JAX/CarDreamer process starts
disables autotuning search (forces a fixed algorithm choice) and requests
deterministic reduction ops. Two independent full-scale (n=100, horizon=50,
real actor policy) runs WITH this flag set produced a byte-identical SHA-256
output hash (`3bc2db72...` both times) — confirming the fix at the sampling
layer.

**Follow-up completed (same round, after explicit go-ahead)**: reran the
full `fit_lppm`/`calibrate_lppm` pipeline via `cardreamer/eval_ltl_hazard_
real_lppm.py` with `XLA_FLAGS` set, twice independently, specifically to
check whether the sampling-layer fix survives being composed with the
downstream torch-based NeuralLPPM training (a DIFFERENT process/framework
the `XLA_FLAGS` fix does not touch — `torch.manual_seed(0)`-seeded,
CPU-only, full-batch with no dataloader/shuffling, so a priori unlikely to
introduce its own nondeterminism, but not assumed without checking). Result:
**every intermediate value printed by the script matched exactly between
the two runs** — not just the final `p̂_γ`, but the full training-residual
trajectory, the violation breakdown, and the Z_free-closure numbers too.
`p̂_γ=0.1227, k=9/40` is now the project's first empirically-reproducible
real-world `ltl_hazard_avoidance` result (Section 7 #3, now Tier-1, WITH the
`XLA_FLAGS` precondition stated as part of its own reliability claim, not as
a separate footnote).

**Mode of nondeterminism**: not checked exhaustively (would require many
repeated runs), but the mechanism itself (autotuning selects among a small,
fixed set of registered cuDNN algorithm implementations for a given op
shape) structurally implies a small finite set of possible outcomes per run,
not unbounded randomness — consistent with, though not proof of, the
"collapsed vs not-collapsed" two-mode pattern observed across the three
historical training outcomes (0.7856's V∈[0.66,0.90] vs 0.1831/0.1036's
V-collapse).

**Consequence for `carla_roundabout`**: the root cause (XLA/cuDNN conv
autotuning) is a property of the JAX/GPU execution environment, not specific
to `carla_four_lane`'s checkpoint — it would affect `carla_roundabout`
identically once/if that checkpoint is ever run through
`sample_rollouts()`+`fit_lppm`/`calibrate_lppm`. The same `XLA_FLAGS` fix
should be applied preemptively for any future roundabout LPPM work, not
rediscovered from scratch.

**Scope note, recorded preemptively**: this issue has NOT yet been checked
against `carla_roundabout` — not because it's known to be unaffected, but
because `carla_roundabout` has never gone through `fit_lppm`/`calibrate_lppm`
at all (see §3.4: roundabout currently only has STL/monitor-path results, no
LPPM `p̂_γ` number exists for it yet). If/when roundabout is ever wired into
the LPPM calibration path in the future, this same seed issue must be
checked BEFORE trusting any resulting `p̂_γ` — do not assume it's fine just
because `carla_four_lane` was the checkpoint where it was discovered; both
share the same `CarDreamerWrapper.sample_rollouts()` code path.

**Discovery history** (for provenance — root cause is now known, see above;
this table is the evidence trail that led there). While attempting to run
the new `verify_zfree_closure()` (8.2) against `ltl_hazard_avoidance`'s real
calibration data, two independent reruns of `cardreamer/eval_ltl_hazard_
real_lppm.py` — identical code, identical `seed=0` — produced a materially
different result each time, and neither matched the ledgered `p̂_γ=0.7856`
(Section 7 #3):

| Run | p̂_γ | k | Z_free-source V range |
|---|---|---|---|
| Ledgered (earlier session phase, `hazard_zfree_closure_check.py`) | 0.7856 | — | 0.66–0.90 (never enters Z_free) |
| Rerun 1 (1500 training epochs, production default) | 0.1831 | 12/40 | ≈0 on 1960/1960 transitions (fully collapsed into Z_free) |
| Rerun 2 (8000 training epochs, ruling out under-training) | 0.1036 | 8/40 | ≈0 on 1960/1960 transitions (same collapse, stable from epoch ≈1000 onward) |

Training duration was ruled out as the cause at the time (rerun 2 extended
training 5.3× with no qualitative change) — correctly, as it turned out:
the real cause (GPU/cuDNN conv-algorithm autotuning, confirmed above) acts
on the SAMPLED DATA itself, before training ever starts, so no amount of
extra training epochs could have fixed it.

**Consequence, updated**: no cached copy of the trajectories/trained weights
behind the ledgered 0.7856 exists, so it specifically cannot be reproduced
even now. But with the `XLA_FLAGS` fix verified (above), any FUTURE run of
this pipeline — with the flag set — should itself be internally reproducible
going forward, which is the more important practical outcome: this class of
"three different numbers from the identical script" surprise should not
recur for new results, even though the specific historical 0.7856 remains
permanently unrecoverable.

### 8.10 `main.py::verify()`'s Lemma E.6 gate only refused Unclassified — Recurrence/Reactivity sailed through into the LPPM pipeline — FIXED this round

External-review finding. The Step-3 gate added in Batch 2 of the L2 audit
(8.2-adjacent work, this session) refused `UNCLASSIFIED_MP_CLASS` /
`classification_uncertain`, but never refused `mp_class in {"Recurrence",
"Reactivity"}` -- these classes are NOT co-Büchi-expressible (Lemma E.6), yet
`build_parity_automaton()` DOES have a construction branch for them (it can
build *some* automaton, just not one the co-Büchi P1/P2 Theorem-5.4
certificate is valid for), so they sailed straight through the entire LPPM
pipeline and could fabricate an unsupported WARRANT/VIOLATION verdict.

Confirmed via direct execution: `ltl_patrol`, `ltl_dual_patrol`,
`ltl_safe_patrol` (all `mp_class="Recurrence"`, `verification_mode=
"infinite_parity"` -- i.e. real, registered specs that DO reach
`build_parity_automaton()`, not diverted to the bounded-STL path) previously
produced a `VERIFICATION` verdict through `main.py::verify()` end-to-end.

**Fixed**: added `LEMMA_E6_EXCLUDED_MP_CLASSES = {"Recurrence", "Reactivity"}`
gate immediately after the existing Unclassified gate, routing to
INCONCLUSIVE with its own `support_note` (does NOT reuse the "classification
uncertain" wording -- these specs' classification is confident; the
exclusion reason is structural, not epistemic) explaining: not co-Büchi-
expressible per Lemma E.6, correct home is L3, L3 confirmed infeasible on
real checkpoints as of this session, so INCONCLUSIVE rather than a silent
"route to L3" that doesn't functionally exist yet. Test-driven
(`tests/test_l2_lemma_e6_exclusion.py`, 3 tests, confirmed to fail against
the pre-fix gate first -- `ltl_patrol` returned `verdict='VIOLATION'`
end-to-end pre-fix -- then pass post-fix).

**Placeholder-ledger check**: `ltl_patrol`/`ltl_dual_patrol`/`ltl_safe_patrol`
were checked against `experimental_setup.md`'s coverage tables --
`zone_a`/`zone_b` are structurally unavailable in the CarDreamer wrapper, so
all three are marked `BLOCKED`/`N/A` (CarDreamer coverage table) and
`UNTESTED` (LTL coverage table) there. **No number needs to be marked
deprecated, because none ever existed** for any of these three specs --
same precise wording this project has used for every prior audit fix that
turned out to affect zero real reported results.

### 8.11 `run_product_trajectory()`'s consumers were blind to the FINAL transition of every trajectory — FIXED this round (highest-priority external-review finding)

External-review finding, confirmed the most serious of this round.
`check_pathwise_conditions()`, `verify_zfree_closure()` (8.2, this session),
and `trainer.py::fit_lppm()`'s `all_transitions` construction all built
their transition list by zipping ADJACENT `product_path` entries directly
(`(path[i], path[i+1])` for `i in range(T-1)`) -- product_path[t].q always
equals product_path[t-1].q_next, so this correctly captures transitions
0..T-2, but the FINAL transition (on the last observed AP, from
product_path[T-1].q to product_path[T-1].q_next -- the TRUE state after the
entire trajectory) was never included anywhere, because there is no
product_path[T] entry to pair it with. This silently made every pathwise
check, Z_free-source detection, and every training example blind to
whatever happened on the last observation of every trajectory ever
processed by this pipeline -- systemic, not a boundary edge case, affecting
every historical L2 result including `ltl_hazard_avoidance` and
`ltl_height_safety`.

**A second, interacting bug was found while verifying the fix against the
reviewer's own worked example** (a 2-step G(p) trajectory violating p only
on the last observation): naively swapping `last.q` for `last.q_next` in the
terminal check ALONE is actively WRONG for Safety-class specs --
`core/lppm/model.py`'s Safety branch returns `V=0.0` unconditionally for
`q=="trap"`, so a trajectory ending inside "trap" would trivially satisfy
Z_free (`0.0 < eta`) regardless of how it got there; entering trap is also a
V-DECREASE (never flagged by P1) sourced from an even-priority state (never
routed to P2 either). The pre-fix code happened to reject the reviewer's
example anyway, but by numeric accident (the stale pre-transition state
paired with the violating z produced a large V), not because the check was
examining the true final state.

**Fixed together** (both required for correctness, confirmed via 3 concrete
worked examples before writing any code -- see the conversation for the
full derivation):
1. New `core/lppm/verifier.py::iter_transitions(product_path, dpa)` helper:
   yields all T (curr, nxt) transition pairs for a T-observation trajectory,
   synthesizing a terminal `ProductState` for the final pair (t=T, z reused
   from the last real observation, q=q_next=the true final automaton state,
   priority=that state's own priority) -- `len(product_path)` itself is
   untouched, so `rem=(T-t)/T` computations elsewhere are unaffected.
2. `check_pathwise_conditions()`'s terminal Z_free check now uses the true
   final state (via `iter_transitions()`'s last pair) AND treats landing in
   an ODD-priority state as an automatic violation, not deferred to
   `V<eta` -- closing the trap/V=0 interaction.
3. `check_pathwise_conditions()`, `verify_zfree_closure()`, and
   `trainer.py::fit_lppm()`'s transition-list construction all switched to
   `iter_transitions()`.

Test-driven (`tests/test_l2_terminal_transition_fix.py`, 5 tests): 3 of 5
confirmed to FAIL against the pre-fix code (a Guarantee spec achieving its
goal only on the last observation was wrongly REJECTED; a Safety spec
violating mid-trajectory-then-"recovering" was wrongly ACCEPTED via the
missing final-transition P2 check; `verify_zfree_closure()` missed a P1
violation placed on the final transition) -- the other 2 (the reviewer's own
Safety example, and a "goal never achieved" regression guard) already
passed pre-fix, for the reasons above. All 5 pass post-fix. Full suite:
regression-confirmed, zero failures across all pre-existing tests.

**Real results rerun, per explicit instruction — do not silently replace,
mark both**:
- `ltl_hazard_avoidance` (CarDreamer carla_four_lane): pre-fix ledgered
  `p̂_γ=0.7856` (Section 7 #3) was computed under this bug. Post-fix rerun:
  `p̂_γ=0.1227, k=9/40`. Note (§8.9): this pipeline does not reproduce its
  own calibration draw across runs regardless of this fix (confirmed again:
  a THIRD independent rerun this round, still not matching 0.7856, still
  not matching either of the two §8.9 reruns' own numbers either), so
  isolating "effect of this bug fix" from "resampling variance" is **not
  possible** for this spec -- the post-fix number is a fresh, independent
  draw, not a controlled before/after comparison. Do not cite `0.1227` as
  "the bug-fixed 0.7856" -- it isn't comparable to it at all.
- `ltl_height_safety` (TD-MPC2 walker-walk): pre-fix ledgered `p̂_γ=0.0000,
  k=0/80` (Section 7 #12) was computed under this bug. Post-fix rerun:
  `p̂_γ=0.0000, k=0/80` -- **exactly unchanged**. This pipeline IS exactly
  reproducible (confirmed earlier this session, §8.2/§8.9), so this
  comparison DOES isolate the bug fix's actual effect for this spec: zero
  effect, because every one of this calibration split's 80 rollouts was
  already failing via P1 violations spread across many transitions (not
  concentrated on the final one) -- the previously-invisible final
  transition was never the deciding factor for this particular result. The
  underlying bug is still real and still fixed (confirmed via the synthetic
  worked examples above); this spec's specific number just wasn't sensitive
  to it.

**Net conclusion**: this fix is confirmed correct and necessary (3 concrete
synthetic counterexamples, one directly reproducing the external reviewer's
own worked example), and is now permanently regression-guarded
(`tests/test_l2_terminal_transition_fix.py`). Its real-world numeric impact
could only be cleanly measured for `ltl_height_safety` (no change) --
`ltl_hazard_avoidance`'s pipeline nondeterminism (§8.9) makes any before/after
comparison for that spec uninterpretable, independent of this fix.

### 8.12 `fit_lppm()`/`calibrate_lppm()`'s disjoint-split violation only ever warned, despite a comment claiming it "fails loudly" — FIXED this round

External-review finding #3. `core/lppm/trainer.py::fit_lppm()` and
`core/lppm/calibrator.py::calibrate_lppm()`'s opt-in disjointness check
(`warn_if_overlap=True`) only ever called `warnings.warn()` on detected
overlap -- structurally incapable of raising anything. `main.py::verify()`'s
one real call site (the `fit_lppm_params=True` branch) passed the SAME
`trajectories` list as both the training set and `calib_trajectories`,
guaranteeing 100% overlap every time that branch ran, and its own comment
claimed this was "guarded here so it fails loudly rather than silently" --
simply incorrect; passing `warn_if_overlap=True` cannot raise, by
construction. This is worse than no protection at all: the comment would
mislead a future reader into trusting a check that never actually blocked
anything.

**Fixed**: overlap detected once a caller opts in (by passing
`calib_trajectories`/`train_trajectories` at all) now RAISES `ValueError` by
default, unless the caller explicitly passes `allow_overlap=True` with a
non-empty `overlap_reason` (a forced, auditable opt-out -- omitting the
reason while `allow_overlap=True` is itself an error). `main.py::verify()`'s
`fit_lppm_params=True` branch now requires a genuinely separate
`VerifyConfig.lppm_train_trajectories` -- reusing `trajectories` for both
roles is no longer reachable even by accident; the branch raises immediately
if this field is unset, with a message pointing at exactly what's missing.
`core/lbsm/trainer.py`'s comments (which referenced L2's old opt-in-warn
behavior as a point of comparison for L3's own mandatory-raise check) updated
to describe the new behavior accurately.

Test-driven (`tests/test_l2_disjoint_split_enforcement.py`, 8 tests, all
confirmed to fail against the pre-fix code first -- including one that
directly observed the `UserWarning` `main.py` was emitting where its own
comment promised a hard failure).

**Placeholder-ledger check**: `fit_lppm_params=True` was already confirmed
dormant before this round (no real caller anywhere sets it) -- this fix
changes zero real reported numbers, only closes a latent trap for any future
caller.

### 8.13 `verify_zfree_closure()` treated autocorrelated transitions within one trajectory as independent Bernoulli samples — FIXED this round; result now surfaced on `LPPMResult`

External-review finding #4. The Z_free-closure diagnostic added this session
(§8.2) computed its Clopper-Pearson bound over individual (transition, head)
pairs -- but multiple transitions from the SAME trajectory share one
underlying `V_phi` realization and autocorrelated dynamics, violating the
exchangeability premise Clopper-Pearson's exact interval requires. This
could badly overstate `p_hat_closure`: a single genuine violation, diluted
across many passing transitions from the same trajectory, barely moved the
per-transition rate.

**Fixed**: the statistical unit is now the TRAJECTORY. Each trajectory
contributes exactly one trial to `n_zfree` (if it has at least one
Z_free-source transition) and exactly one success to `k_zfree` **iff every
one of its Z_free-source transitions satisfies P1/P2** -- mirroring
`PathwiseResult.satisfied`'s own whole-trajectory `C(tau)` construction (an
AND across all checked conditions), restricted to the Z_free-source subset.
A trajectory with zero Z_free-source transitions contributes to neither
count (it cannot be counted as "passing" a condition it never triggered).
Test-driven (`tests/test_zfree_closure.py`, new test constructing two
trajectories -- one that stays fully safe, one that violates only on its
final transition -- confirmed the pre-fix per-transition counting gave
`n_zfree=1998` for a 2-trajectory batch; post-fix gives the correct `n=2,
k=1`).

**Also fixed (same finding, second half)**: `verify_zfree_closure()`'s
result was previously only visible to a caller that separately remembered
to call it themselves -- `calibrate_lppm()` never surfaced it. Now
`LPPMResult` has its own `zfree_closure: ZFreeClosureResult | None` field,
populated automatically by `calibrate_lppm()` (kept strictly separate from
`p_hat_gamma` -- never blended into it), and both `LPPMResult.summary()` and
`main.py`'s `VerificationResult.summary()` report it. Any real caller of
`calibrate_lppm()`/`verify()` now sees this diagnostic without extra effort.

**Note**: `cardreamer/eval_ltl_hazard_real_lppm.py` and
`tdmpc2/eval_ltl_height_safety_lppm.py`'s own explicit `verify_zfree_closure()`
calls (steps [7]/[8], added earlier this session before `calibrate_lppm()`
computed it internally) are now redundant with what `calibrate_lppm()`
computes on the same data -- not incorrect (both use the identical
`calib_traj`, so they agree), just duplicated work. Left as-is this round;
worth simplifying next time either script is touched.

### 8.14 Obligation automaton construction silently drops disjunctive safety / sequential guarantee clauses — confirmed unreachable by all 30 registered specs, documented as a known limitation (not fixed)

External-review finding #5. `core/lppm/automaton.py`'s Obligation branch
reads only `objectives["safety"]` (the plain conjunctive-atom bucket) and
`objectives["guarantee"]` (the plain unordered-goal bucket) -- it never
reads `objectives["safety_disjunctions"]` (Batch 3a, G(a v b) recognition)
or `objectives["guarantee_sequences"]` (Batch 3b, ordered-chain
recognition). Since Batch 3a taught `infer_mp_class()` to treat
`safety_disjunctions` as satisfying `has_safety`, a spec shaped like
`G(a v b) & F(goal)` is now correctly CLASSIFIED as Obligation but then
silently mis-CONSTRUCTED: `safe_aps = objectives["safety"] = []` (empty,
since this spec's safety is entirely disjunctive), producing zero
trap-entry transitions -- the safety requirement is dropped from the
automaton entirely. Confirmed via direct construction: a trajectory
violating the disjunction while achieving the guarantee goal reaches an
ACCEPTING state.

**Decision: not fixed this round.** Confirmed via direct enumeration that
none of the 30 currently-registered specs trigger this path -- all 3 real
`mp_class="Obligation"` specs (`ltl_safe_goal`, `ltl_safe_slow_goal`,
`stl_safe_goal_reach`) use plain conjunctive safety and plain unordered
guarantee; `safety_disjunctions`/`guarantee_sequences` are empty for all
three. Per explicit scoping decision, downgraded to a documented known
limitation rather than fixed now. **Permanently scope-guarded**
(`tests/test_l2_obligation_disjunctive_scope_guard.py`): one test asserts
none of the 30 registered specs currently trigger the broken path (fails
loudly if a future spec addition would), the other pins the exact
reproduction of the current (wrong) behavior for whoever picks this up. If
a future spec needs disjunctive-safety or sequential Obligation, the
Obligation branch must be extended first (mirroring how the Safety and
Guarantee branches already handle their own disjunction/sequence buckets)
-- do not register such a spec against this pipeline until then.

### 8.15 Sequential Guarantee (F(A and F(B and F(C...)))): simultaneous satisfaction under-advanced by one step; heuristic V blind to sequence progress — FIXED this round

External-review finding #6, two independent bugs in the Batch-3b
`guarantee_sequences` construction.

1. `core/lppm/automaton.py`'s `advance_progress()` only advanced a
   sequence's progress index by ONE step per observation even when MULTIPLE
   consecutive required atoms were active SIMULTANEOUSLY -- e.g. F(A and
   F(B)) with A and B both true at the same instant t IS satisfied (F(B)
   only needs B true at some time >= t, including t itself), but the pre-fix
   automaton only consumed A at t, leaving it waiting for a future
   observation of B that might never come again. Fixed: `while` instead of
   `if`, advancing through as many consecutive required atoms as are
   simultaneously active (order is still enforced -- B alone, without A
   ever having held, still does not advance anything; confirmed via a
   dedicated regression test).

2. `core/lppm/model.py::compute_lppm_value()`'s Guarantee branch read only
   `meta["remaining_goals"]` (`objectives["guarantee"]`'s plain unordered
   bucket) -- for a PURELY sequential spec (`guarantee_sequences` non-empty,
   `guarantee` EMPTY, e.g. `ltl_sequential_goals`, `ltl_three_stage`),
   `remaining_goals` is `[]` for EVERY state in the automaton, so V
   collapsed to the unconditional `0.0` branch everywhere -- the real
   progress signal (`sequence_progress`, correctly populated in
   `state_meta` since Batch 3b) was never read. Fixed: total remaining
   "distance" now combines both the unordered `remaining_goals` count and
   the sum of each sequence's outstanding steps
   (`len(seq) - sequence_progress[i]`), falling back to the old behavior
   exactly when `sequence_progress` is absent (plain, non-sequential
   Guarantee specs -- zero behavior change for them).

Test-driven (`tests/test_l2_sequential_guarantee_fixes.py`, 4 tests, 3
confirmed to fail against the pre-fix code first: simultaneous satisfaction
on `ltl_sequential_goals` and `ltl_three_stage` both wrongly stayed in an
odd/rejecting state; heuristic V was confirmed frozen at exactly `0.0`
regardless of `sequence_progress`). All pass post-fix. Full suite
regression-confirmed.

**Placeholder-ledger check**: `ltl_sequential_goals` and `ltl_three_stage`
are both `BLOCKED`/`UNTESTED` in `experimental_setup.md`'s coverage tables
(`zone_a`/`zone_b`/`zone_c` structurally unavailable) -- **no number needs
to be marked deprecated, because none ever existed** for either spec.

### 8.16 η (margin parameter) sensitivity sweep for `ltl_hazard_avoidance` (B5) — result is NOT monotonic in η; `η=0.01`'s only stated justification is "matches original plan"

**Purpose, stated explicitly**: NOT a search for an η that makes `p̂_γ`
look better. A sensitivity analysis, reported honestly regardless of
outcome — the reader should be able to judge for themselves how much this
result depends on the η choice.

**Setup**: B5's own documented range (`experimental_setup.md:489`) is
`η ∈ {0.001, 0.005, 0.01, 0.05, 0.1}` — used as-is (a slightly different
range was proposed when this sweep was requested; deferred to B5's existing
range per explicit instruction to stay consistent rather than start a new
one). The SAME confirmed-reproducible sampling call (§8.9/§7 #3, `seed=0`,
`XLA_FLAGS` fix) was invoked exactly ONCE, and the same disjoint
train=60/calib=40 split reused across all 5 conditions — only η varies
between them. **NeuralLPPM was retrained from scratch for each η** (fresh
model, `torch.manual_seed(0)` reset each time) — η enters `p2_loss`'s
margin target directly, so sweeping only the verification-side threshold
while training under a fixed/different implicit target would reproduce
exactly the train/verify inconsistency Batch 4 of the L2 audit fixed (§8.4).

**Result — NOT monotonic**:

| η | k | N | p̂_γ | rollouts w/ P1 failure | rollouts w/ P2 failure | training converged? |
|---|---|---|---|---|---|---|
| 0.001 | 8 | 40 | 0.1036 | 32/40 | 1/40 | yes (p1→0, p2→η exactly) |
| 0.005 | 8 | 40 | 0.1036 | 32/40 | 1/40 | yes |
| 0.01 (current default) | 9 | 40 | 0.1227 | 31/40 | 1/40 | yes |
| 0.05 | 7 | 40 | 0.0851 | 33/40 | 1/40 | yes |
| 0.1 | 0 | 40 | 0.0000 | 32/40 | 1/40 | **NO** — final p1 loss 0.000923, not 0 |

The naive expectation stated when this sweep was requested — "if `p̂_γ`
rises monotonically with η, that itself shows the result is sensitive to
the η choice" — did not materialize in that exact form, but the actual
result is arguably a more serious finding: **`p̂_γ` is low (0.08–0.12) and
non-monotonic across the entire "reasonable" range (η≤0.05), then
collapses to exactly 0 at η=0.1 because training itself fails to converge**
(P1 loss does not reach 0, the only condition in this whole sweep where
that happens). This is NOT a case where one η choice happens to look good
and the others look bad by comparison — every η in B5's range gives a
NOT-warranted result (all `p̂_γ` values are far below the 0.80
`warrant_threshold`), and the specific value is unstable enough (0.1036 →
0.1227 → 0.0851, a ~30% relative swing between adjacent order-of-magnitude
η values) that citing any single one of them, including the current
default 0.01, as "the" `p̂_γ` for `ltl_hazard_avoidance` overstates the
precision this measurement actually has.

**Root driver identified**: P1 (non-increase) failures dominate and are
essentially η-INSENSITIVE, sitting at 31–33/40 across every η tested
(η doesn't appear in the P1 check at all, and only weakly influences it
indirectly through training). P2 failures stay at exactly 1/40 throughout.
The variation in `k`/`p̂_γ` across η∈{0.001,...,0.05} is being driven by the
terminal Z_free-membership check (`V<η`), not by P1 or P2 directly — larger
η makes that specific check structurally easier to satisfy, but training
under a larger η target also pushes the network toward larger absolute V
values (to have room for a bigger descent margin), so the two effects
partially cancel, producing the observed non-monotonic wobble rather than a
clean trend either direction — until η=0.1, where the training objective
itself becomes too demanding for this network/data to satisfy at all, and
`p̂_γ` collapses to 0 for the unrelated reason of outright training failure,
not a "large η is inherently bad for Z_free" effect.

**η=0.01's justification, checked as requested**: `experimental_setup.md`
line 525 records it as *"0.01 default (matches original plan)"* — that is
the ONLY documented justification found anywhere in the repo. There is no
physical, domain, or theoretical derivation of why 0.01 specifically (as
opposed to any other order-of-magnitude value) is the right descent-margin
requirement for this spec/checkpoint pair. Given this sweep shows the
result swinging by ~30% relative across the immediately adjacent B5 grid
points, "matches the original plan" is not a strong enough justification to
present 0.01's specific number as load-bearing without this sensitivity
context alongside it. This is a real limitation to state explicitly in any
paper section that cites `ltl_hazard_avoidance`'s `p̂_γ` — not something a
different η choice can fix, since NONE of the tested values clear
`warrant_threshold=0.80` and the specific value is demonstrably unstable in
the region around the current default.

Script: `hazard_eta_sensitivity_sweep.py` (scratchpad, not yet promoted to
a repo file — promote before citing this sweep as a standing, rerunnable
experiment rather than a one-off diagnostic).

### 8.17 P1-violation attribution for `ltl_hazard_avoidance` (η=0.01): trained V has collapsed to a near-constant ≈7×10⁻¹⁴ everywhere — CONFIRMED training-configuration/data-sparsity problem, NOT evidence of a Corollary-5.3-sense structural non-existence

**Question this answers**: is the low `p̂_γ`/high-P1-failure-rate result for
`ltl_hazard_avoidance` (§7 #3/#23, k=9/40 at η=0.01, P1 failing in 31/40
rollouts) telling us (a) this specific training run/checkpoint pairing is
poorly configured, or (b) no valid LPPM ranking function exists for this
spec on this checkpoint at all (a Corollary 5.3-sense structural claim)?
Diagnosed the same way as the `height_safety` k=0 case (§7 #13/#14):
inspect the trained V directly, not just the pass/fail counts.

**Method**: same reproducible sampling/split as §7 #3/#23 (η=0.01 training).
For each of the 40 calibration trajectories, recorded per-transition V
(computed via the trained `NeuralLPPM`), whether P1/P2 was violated, and
trajectory-level features (`hazard_dist` min/mean, trap visitation).
Followed up by inspecting the model's raw PRE-softplus logits directly
(not just the post-softplus V) to distinguish "genuinely collapsed" from
"float32 underflow."

**Finding 1 — V is collapsed to a near-constant, not float32 underflow,
across ALL 40 calibration trajectories, uniformly**:

```
Raw logit stats across every (z,q) pair in the calib split:
  mean=-30.4143  std=0.3470  min=-32.0881  max=-30.1692
```

An extremely narrow logit range (std=0.347 on a mean of -30.4) regardless
of `hazard_dist`, which itself ranges from -0.25 (colliding) to 32.0
(maximally safe) across these same states — the network has learned to
almost completely ignore its input and output a near-constant logit. This
gives `V = softplus(logit) ≈ 6–8 × 10⁻¹⁴` everywhere (displays as exactly
`0.0000` at 4 decimals in the earlier sweep's table, §7 #23) — NOT float32
underflow (which needs logit ≲ -87; this network sits at -30, comfortably
representable) — a genuine, trained collapse to a value astronomically
smaller than η=0.01 (12 orders of magnitude smaller), just not literally
the bit-pattern zero.

**Finding 2 — the flagged "P1 violations" are float-noise-scale, not
meaningful non-monotonicity**:

```
traj=0 t=2   V_curr=7.197353471674259e-14  V_next=7.223442764076049e-14  delta=2.61e-16
traj=0 t=4   V_curr=7.223442764076049e-14  V_next=7.231576313248764e-14  delta=8.13e-17
traj=0 t=5   V_curr=7.231576313248764e-14  V_next=7.246902866209562e-14  delta=1.53e-16
```

Every flagged delta is on the order of 1e-16–1e-15 — 6–8 orders of
magnitude smaller than V itself (~7e-14), which is itself 12 orders of
magnitude below η. `check_pathwise_conditions()`'s default `p1_tol=0.0`
(strict, exact zero tolerance — Batch 4's own fix, §8.4/§8.5, deliberately
removed the old implicit 1e-6 floor) is, AS DESIGNED, flagging every one of
these as a violation. That design choice is correct in general (see §8.5's
reasoning for why the implicit floor was wrong to have) — but it also means
these 31/40 "P1-failing" rollouts are failing on numerical noise in the
15th significant digit of an already-collapsed near-zero constant, not on
any physically meaningful increase in "distance to violating safety."

**Finding 3 — root cause: P2 gradient signal is almost entirely absent
from the training split**: only 2/60 (3.3%) TRAINING trajectories ever
visit `trap` at all, meaning only 60/2940 (2.0%) of training transitions
have `curr.priority == r` (the condition that activates `p2_loss`'s
gradient term). 98% of the training signal comes from `p1_loss` alone,
whose global optimum is trivially satisfied by ANY constant function (a
constant never increases). With essentially no counter-pressure from `p2`
to make V depend meaningfully on `hazard_dist`, the optimizer settles into
exactly the degenerate solution observed: a near-constant output, with only
a faint residual dependence on the input (std=0.347 in logit space) left
over — just enough to produce the float-noise-scale wiggles that trip the
strict P1 check on rollouts with more `hazard_dist` variation (this also
explains Finding 4 below).

**Finding 4 — violations correlate with hazard-proximity, but as a
noise-density effect, not a distinct failure subpopulation**:

| group | n | hazard_min mean | hazard_mean mean | visits_trap |
|---|---|---|---|---|
| P1-FAILING | 31 | 13.665 | 19.487 | 1/31 |
| P1-CLEAN | 9 | 27.169 | 31.734 | 0/9 |

P1-failing rollouts skew toward lower `hazard_min` (closer to the vehicle
at some point) — but EVERY trajectory in both groups shows the identical
collapsed per-trajectory `V_std≈0.0000` (at working precision) profile;
there is no genuinely distinct "these rollouts have a different V behavior"
subpopulation. The correlation is exactly what "noise density scales with
feature variation" predicts: rollouts that get closer to the hazard have
more `hazard_dist` movement, hence slightly more logit movement (since the
residual ~0.35-std dependency on input isn't literally zero), hence more
opportunities for the collapsed-but-not-perfectly-flat function to produce
a positive-then-negative wiggle that trips the strict zero-tolerance check.
This is NOT evidence of violations concentrated in a specific, meaningfully
different initial-condition sub-distribution — it's the same degenerate
solution everywhere, sampled slightly more often where there's more input
variation to sample noise from.

**Conclusion, directly answering the question this diagnosis was run
for**: this is a **training-configuration / data-sparsity problem**, not
evidence of a Corollary-5.3-sense structural non-existence of a valid LPPM
for this spec. The mechanism (P1-only trivial optimum when P2's gradient
signal is starved by trap-sparse training data) is now confirmed by DIRECT
evidence (raw logits, exact per-transition deltas) on `hazard_avoidance`,
not just inferred from aggregate pass/fail counts — and it is the exact
same mechanism already diagnosed for `ltl_height_safety`'s k=0 result (§7
#13/#14) and flagged as a suspected systemic issue in §7 #15's caveat
("systemic NeuralLPPM training-objective issue... not an isolated quirk").
This diagnosis now makes that suspicion a confirmed, quantitatively
evidenced finding across BOTH real checkpoints this project has. Nothing
here demonstrates or even suggests that G(hazard_dist>0) lacks a valid
ranking function in principle — only that THIS training run, with THIS
much trap-sparse data, cannot find one. A training regime that
oversamples/upweights trap-visiting rollouts (or otherwise directly
addresses the P2 gradient-starvation problem) is the natural next step
before drawing any conclusion about the spec itself being unverifiable
under this framework.

Scripts: `hazard_p1_violation_attribution.py`, `hazard_v_precision_check.py`
(both scratchpad).

### 8.18 P2-gradient-starvation remediation test for `ltl_hazard_avoidance` — three training-strategy interventions tested, NONE resolved the collapse; NOT sufficient grounds to conclude no valid LPPM exists

**Purpose**: §8.17 diagnosed the low-`p̂_γ` result as a training-configuration
problem (P2's gradient signal starved by only 2/60 trap-visiting training
trajectories), explicitly NOT evidence of a Corollary-5.3-sense structural
non-existence. This round tests whether that diagnosis is actionable —
same reproducible sampling/split as §7 #3 throughout, no data collection
changes.

**Diagnostic used for every condition** (not just `p̂_γ`, per explicit
instruction that a better final number alone does not establish the
certificate is meaningful): raw pre-softplus logit mean/std/range,
correlation between logit and `hazard_dist` across the calibration set
(near 0 = still flat/collapsed; meaningfully negative = V tracking real
danger), whether P1 training actually converged (final P1 loss ≈ 0, not
just a better `p̂_γ` achieved by sacrificing P1), and P1-violation delta
magnitude (noise-scale vs genuine).

**Step 1 — loss reweighting** (`λ_P2 ∈ {1, 5, 10}`, multiplying `p2_loss`
before the sum with `p1_loss`):

| λ_P2 | logit mean | logit std | corr(logit, hazard_dist) | P1 converged? | k/N | p̂_γ |
|---|---|---|---|---|---|---|
| 1 (baseline) | -30.41 | 0.35 | -0.29 | yes | 9/40 | 0.1227 |
| 5 | -17.05 | 1.18 | +0.15 (wrong-ish sign, weak) | yes | 8/40 | 0.1036 |
| 10 | +10.57 | 0.10 | -0.57 (strongest, but...) | **NO** (final P1 loss 0.0012) | 0/40 | 0.0000 |

λ_P2=10 shows the strongest correlation but achieves it by breaking P1
convergence — genuine (not noise-scale) P1 violations appear (median delta
7.4e-4, vs ~1e-16 at baseline), a different failure mode, not a fix. None
of the three conditions produced BOTH P1 convergence AND a strong,
consistent correlation.

**Step 2 — oversample trap-adjacent transitions within the training split
only** (duplication of the SAME 60 existing trap-adjacent transitions;
`find_trajectory_overlap(train_traj, calib_traj)` checked and asserted 0
both before and unaffected by oversampling, since only the training-side
transition multiset changes — the calibration set and its exchangeability
with the training draw are untouched):

| condition | trap-adjacent % | corr(logit, hazard_dist) | P1 converged? | k/N | p̂_γ |
|---|---|---|---|---|---|
| 5x oversample, λ=1 | 9.4% | +0.05 | yes | 8/40 | 0.1036 |
| 10x oversample, λ=1 | 17.2% | -0.13 | yes | 8/40 | 0.1036 |
| 20x oversample, λ=1 | 29.4% | +0.24 | yes | 8/40 | 0.1036 |
| 10x oversample, λ=3 (combined) | 17.2% | -0.25 | yes | 8/40 | 0.1036 |

All four kept P1 converged (better than λ_P2=10 alone) but correlation
stayed weak and INCONSISTENT IN SIGN across oversampling factors — not a
clean trend toward a real, physically-sensible dependency. `p̂_γ` is flat
at 0.1036 (mildly worse than the 0.1227 baseline) across every condition
tested. **Likely reason oversampling didn't help**: duplication of the SAME
60 unique trap-adjacent transitions provides more gradient updates on
identical information, not new information about the hazard_dist→safety
relationship — it cannot manufacture generalizable signal that isn't
present in the underlying ~60-transition sample to begin with.

**Additional confound check (not originally requested, added because
neither Step 1 nor Step 2 worked)**: `smoothness_penalty(v_curr, z_curr)`
(weight 0.01, present in every run above including baseline) directly
penalizes the squared gradient of V w.r.t. z — i.e. it actively works
against learning any real z-dependency, which is exactly what was being
tested for. Reran the baseline (λ_P2=1, no oversampling) with this weight
set to 0: correlation got slightly WEAKER (-0.11 vs -0.29), not stronger,
while logit variance increased (std 1.61 vs 0.35) — the network moved
around more without tracking `hazard_dist` any more consistently. Ruled
out as the primary blocker.

**Conclusion — explicitly bounded, not overclaiming**: three plausible,
cheap training-configuration interventions (loss reweighting up to 10x,
transition-duplication oversampling up to 20x combined with 3x reweighting,
and removing an actively-working-against-the-goal regularizer) were tested
and NONE produced a network with both converged P1 and a strong, consistent
V-dependency on `hazard_dist`. Per the explicit instruction this round was
scoped under: **this is NOT yet sufficient grounds to conclude no valid
LPPM exists for this spec** (a genuine Corollary-5.3-sense structural
claim would need either a formal argument or a much more exhaustive search
— different architectures, learning rates, training curricula, or,
relaxing the "no new data collection" constraint this round was
deliberately scoped under, genuinely NEW trap-visiting rollouts rather than
duplicating the same ~60 existing transitions). What CAN be said: within
the "don't touch data collection" constraint, the cheap/standard training
adjustments available did not resolve it, and there's a principled reason
duplication-based oversampling specifically was unlikely to (no new
information content). The most promising still-untested next step is
collecting genuinely additional trap-visiting rollouts, which requires
relaxing that constraint — a decision for a separate, explicit follow-up.

Scripts: `hazard_p2_starvation_remediation.py`,
`hazard_p2_starvation_oversample.py`, `hazard_smoothness_confound_check.py`
(all scratchpad).

### 8.19 Trap-sparsity failure mode — unified record across both real checkpoints (height_safety + hazard_avoidance)

**Purpose**: §7 #13/#14 (`height_safety`) and §7 #24/#25 (`hazard_avoidance`)
independently diagnosed the same underlying failure mode on two different
checkpoints, two different specs, two different simulators/action spaces.
Collected here as one unified record, since the pattern is now confirmed
general rather than checkpoint-specific, and future work on either
checkpoint (or a new one, e.g. `carla_roundabout`) should check against
this table before re-diagnosing from scratch.

**The failure mode**: when the calibration/training data contains too few
trap-visiting rollouts, `p2_loss`'s gradient signal is too sparse to
counteract `p1_loss`'s trivial global optimum (any constant function
trivially satisfies "non-increasing"). `NeuralLPPM` collapses to a
near-constant output that is essentially independent of the real AP
feature (`height` / `hazard_dist`), producing a "certificate" that carries
no real safety information despite training loss converging cleanly.

| | `ltl_height_safety` (TD-MPC2 walker-walk) | `ltl_hazard_avoidance` (CarDreamer carla_four_lane) |
|---|---|---|
| Trap-visiting rollouts, full dataset | 1/200 (0.5%) | 3/100 (3%) |
| Trap-visiting rollouts, TRAINING split specifically | 1/120 (0.8%, inferred — calib has confirmed 0/80) | 2/60 (3.3%, directly counted) |
| Trap-adjacent TRAINING transitions | not separately counted | 60/2940 (2.0%) |
| Collapsed V value | ≈0.5882–0.5884 (near the UNTRAINED softplus(0)≈0.693 baseline — barely moved from initialization) | ≈6–8×10⁻¹⁴ (far BELOW the untrained baseline — moved a great deal, just converged to near-zero) |
| P1 "violation" magnitude | ≤1e-4 (float noise around the near-constant) | ~1e-16 (float noise, even smaller in absolute terms) |
| P1-failing rollouts (strict p1_tol=0) | 3494/7920 transitions | 31/40 rollouts |
| Does a data-derived P1 tolerance fix it? | NO — P1-failing rollouts drop from 80/80 to 11/80 (§7 #14) but `k` stays 0/80, `p̂_γ` stays 0.0000. Confirms the k=0 result is driven by the TERMINAL Z_free condition, not P1 noise. | Not separately retested this round, but the same conclusion is expected to hold given #25's finding that the underlying V-collapse (not just the P1-tolerance nuisance) is the real blocker. |
| Remediation attempted this round | Not attempted (out of scope at diagnosis time) | λ_P2 reweighting {1,5,10}, transition-duplication oversampling {5x,10x,20x}+combined, smoothness-penalty removal — ALL FAILED (§8.18) |
| Conclusion reached | Documented degenerate/negative case; k=0 is a genuine liveness failure of the calibration split, not a code bug (§7 #13 caveat) | Same class of failure, now with a confirmed root cause (trap-sparse gradient starvation) AND a confirmed negative result on cheap remediations (§8.18) — the certificate has not learned a meaningful dependency on `hazard_dist` |

**Common thread**: BOTH real checkpoints this project has ever calibrated an
LPPM certificate against exhibit this exact failure mode. This is now
strong evidence the pattern is a structural weakness of the current
training setup (P1-only trivial optimum whenever trap-visitation is rare in
the calibration data — which, for genuinely SAFE policies being evaluated
against a safety property, is the EXPECTED common case, not an edge case:
a policy that rarely enters the trap region is, definitionally, the
scenario this whole verification apparatus exists to certify). This is a
paper-level point worth making explicitly: the current NeuralLPPM training
objective is poorly suited to exactly the regime (mostly-safe policies)
that L2 verification is supposed to be useful for.

**Identified but NOT executed follow-up: collect genuinely new
trap-visiting rollouts (data collection, not reweighting)**

Both this round's oversampling experiment (§8.18) and the underlying
mechanism argue that reweighting/duplicating the EXISTING scarce
trap-adjacent data cannot manufacture new information — the fix most
likely to work requires genuinely new, diverse trap-proximate trajectories.
Sketched here for a future round to decide whether to invest in, NOT
implemented:

- **CarDreamer / `carla_four_lane`**: `CarDreamerWrapper.sample_rollouts()`
  currently uses `action_source="actor"` (the trained policy) exclusively
  for the real result (§3.2) — by construction, a competent trained policy
  rarely approaches the hazard region, which is exactly why trap-visitation
  is only 3/100. Candidate directions: (a) mix in a fraction of
  `action_source="random"` rollouts (already supported —
  `_build_imagine_random_fn()` exists in the wrapper, currently used "for
  coverage / falsification passes only" per its own docstring, not wired
  into the main calibration pipeline) specifically to bias the TRAINING
  split toward more hazard-proximate trajectories, while keeping the
  CALIBRATION split drawn from the true actor-policy distribution
  (important: don't let training-distribution augmentation change what
  distribution `p̂_γ` is actually being calibrated against — this itself
  would need a careful exchangeability argument before trusting the
  resulting number); (b) initialize a larger raw `N_ROLLOUTS` (say 500–1000
  instead of 100) so that even at a rare-event rate of ~3%, enough
  trap-visiting rollouts land in the training split by chance alone.
- **TD-MPC2 / walker-walk**: the 200-anchor pool
  (`tdmpc2_recollect_with_physics_state.py`) was collected from real MPC
  episodes without deliberately targeting near-fall states; the ledger (§7
  #12) already references "a real near-fall episode" existing somewhere in
  the broader dataset, suggesting near-boundary physics states ARE
  reachable, just under-sampled by the current anchor-selection procedure.
  Candidate direction: bias anchor selection toward physics states with
  lower `height` margin (still genuinely sampled from real MPC rollouts,
  not synthesized) rather than uniform sampling across all recollected
  states.
- **General principle for whichever is pursued**: the new data must still
  go through the SAME disjoint train/calib split and overlap checks already
  enforced (`find_trajectory_overlap`, §8.12) — augmenting the TRAINING
  side with more trap-proximate rollouts does not, by itself, break
  anything already in place, but the calibration side's distributional
  assumptions (what population `p̂_γ` is a statement about) need explicit
  re-examination before citing any resulting number, since Theorem 5.4's
  guarantee is about the SAME distribution the calibration draw comes from.

This is recorded as an identified, reasoned-through, NOT-yet-approved
direction — a decision for a future round, not an implicit next step.

**Paper-level framing — the structural tension, not just an empirical
coincidence** (elevated on review from "the same bug happened twice" to a
methodological limitation worth stating explicitly; suitable to lift
directly into a paper's L2 limitations discussion, e.g. a §7-style section):

> LPPM's statistical branch trains its certificate `V_φ` via gradient
> descent on two loss terms, `p1_loss` and `p2_loss` (Section 4.3). Only
> `p2_loss` provides gradient signal that makes `V_φ` depend meaningfully
> on the underlying state — it is the term that fires specifically on
> transitions sourced at odd-priority ("bad-set"/trap) automaton states.
> `p1_loss` alone has a trivial global optimum: any constant function
> satisfies "non-increasing." This creates a structural tension with what
> co-Büchi verification is most useful for. The deployment case where a
> co-Büchi certificate has the most practical value is exactly certifying
> a policy that performs well — one that rarely, if ever, visits the
> bad-set region under its normal operating distribution. But "rarely
> visits the bad-set" is precisely the condition that starves `p2_loss` of
> gradient signal during training, since bad-set-adjacent transitions are
> sparse or absent in trajectories sampled from that same policy. The
> policies LPPM would be most valuable for certifying are, by
> construction, the hardest to train a non-degenerate certificate for. This
> is not a coincidental implementation defect: it follows directly from
> a basic supervised/self-supervised learning fact (a loss term needs
> examples where it is active to shape the learned function) applied to
> the specific asymmetry of what "safe" and "certifiable" mean for this
> training objective. It was observed and quantitatively confirmed on two
> independent real checkpoints, two different simulators, and two different
> action spaces (`ltl_height_safety`/TD-MPC2 walker-walk: V collapsed to
> ≈0.588, `ltl_hazard_avoidance`/CarDreamer carla_four_lane: V collapsed to
> ≈7×10⁻¹⁴ — Section [X] table), not a single implementation's quirk.
> Practically, this means naive rollout collection under the policy being
> certified is not sufficient to produce a trustworthy `p̂_γ` — deliberate
> bad-set-stratified or exploratory/adversarial data collection is a
> necessary part of deploying this method, not a training hyperparameter
> that can be tuned away after the fact (Section 8.18 tested three cheap
> training-side interventions — loss reweighting, transition oversampling,
> regularizer removal — against data collected under the certified policy
> alone; none resolved the collapse, consistent with this being a data
> characteristic rather than an optimization pathology).

**Logic chain, for direct citation**:
1. LPPM training gets its only real gradient signal on `V_φ`'s dependence
   on state from `p2_loss`, which only activates on bad-set-adjacent
   (odd-priority-source) transitions.
2. Co-Büchi verification's deployment value is specifically in certifying
   well-performing, rarely-violating policies — the most practically
   useful verification target.
3. "Well-performing" means bad-set transitions are inherently sparse in
   naturally-collected rollouts from that policy, directly starving
   `p2_loss`'s gradient and biasing training toward `p1_loss`'s trivial
   constant-function optimum.
4. Observed and quantitatively confirmed on two independent checkpoints
   (different simulator, different architecture) — not a single
   implementation's coincidence (§8.19's comparison table).
5. Consequence: asset/data collection for this method needs deliberate
   design (bad-set-stratified sampling, or adversarial/exploratory policy
   mixing during data collection) — it cannot be assumed to emerge from
   passively collected deployment-policy rollouts, and this is a
   deployment-planning concern, not a training-hyperparameter one.

### 8.20 Trap dwell/recovery diagnosis — is trap genuinely reachable-and-self-looping, or transient? (raw physical signal, both real checkpoints)

**Question this answers**: independent of the automaton's own modeling
choice (see below), does the REAL underlying physical signal, once it dips
into violation, stay violated for multiple consecutive steps (genuine
sustained dwell), or does it recover within a step or two (transient
excursion)? This bears directly on which of two framings is more
appropriate: "trap is genuinely reachable and self-looping — treat
reachability itself as a prerequisite and fall back to a
trajectory-level statistical safety rate" vs. "observed trap-sparsity is
closer to the original training-data-starvation explanation, and the
theoretical tension is real but practically closer to an edge case."
**This diagnosis reports the data; it does not choose between those two
framings — that decision is explicitly left to the reader/user.**

**Method**: same reproducible sampling as §7 #3 (`ltl_hazard_avoidance`,
CarDreamer, seed=0+`XLA_FLAGS`) and the same 200-anchor real-MPC pool as §7
#12 (`ltl_height_safety`, TD-MPC2) — no new data collection. For every
trajectory, scanned the RAW AP time series (`hazard_dist`, `height` —
before automaton abstraction) for maximal consecutive runs below the
safety threshold ("violation episodes"), and recorded each episode's
length and whether the signal recovered above threshold within the same
trajectory or was still violating when the rollout ended.

**Preliminary fact, stated for context (not itself the answer)**:
`core/lppm/automaton.py`'s Safety-branch automaton has NO transition out of
`trap` anywhere in its construction — every transition from `trap` maps
back to `trap`, unconditionally (`transitions = {("trap", frozenset()):
"trap"}` plus a `("trap", {not_ap}): "trap"` entry per safety atom, and
never a `trap`→`ok` entry). This is a deliberate encoding of `G(not p)`'s
own semantics (once violated, permanently failed, by definition) — NOT an
automaton-construction bug. The question this diagnosis actually probes is
whether the REAL system's behavior gives this permanent-failure semantics
practical teeth (sustained dwell) or whether it's a purely definitional
commitment that the physical system itself never really exhibits
(transient excursions only).

**Result — `ltl_hazard_avoidance` (100 trajectories, 50 steps each,
CarDreamer)**:

- 3/100 trajectories have ≥1 raw `hazard_dist≤0` step; 6 total violation
  episodes.
- Episode length distribution: `1:1, 2:2, 3:1, 6:1, 11:1` (steps) — min=1,
  max=11, mean=4.17, median=2.5.
- **Recovered: 6/6 episodes (100%). Terminal-at-horizon: 0/6 (0%).**
- Full detail:
  `traj=23 start=35 len=11 min=-0.050 RECOVERED`,
  `traj=50 start=1 len=2 min=-0.050 RECOVERED`,
  `traj=50 start=6 len=1 min=-0.050 RECOVERED`,
  `traj=50 start=10 len=3 min=-0.050 RECOVERED`,
  `traj=50 start=20 len=2 min=-0.050 RECOVERED`,
  `traj=73 start=32 len=6 min=-0.250 RECOVERED`.

**Result — `ltl_height_safety` (200 trajectories, 100 steps each, TD-MPC2)**:

- 1/200 trajectories has ≥1 raw `height≤0.6` step; 1 total violation
  episode.
- Episode length: exactly 4 steps (min=max=mean=median=4.0).
- **Recovered: 1/1 (100%). Terminal-at-horizon: 0/1 (0%).**
- Detail: `traj=50 start=7 len=4 min=+0.3662 RECOVERED`.

**Both checkpoints show the identical qualitative pattern**: every single
observed raw-signal violation, on both real checkpoints, recovered above
threshold within the same trajectory. Zero episodes (0 out of 7 total
across both checkpoints) stayed violated until the rollout horizon ended.
Dwell lengths are short but not trivially "one step and gone" — up to 11
consecutive steps observed for hazard_avoidance — so this is not a claim
that violations are infinitesimally brief, only that none of them, in this
data, are sustained/self-reinforcing episodes that the system fails to
escape from on its own.

**What this evidence supports and does NOT resolve (left for the
reader)**: the observed real behavior is consistent with option 3's
framing (transient excursions, not genuine sustained self-looping dwell) —
nothing in either dataset shows the physical system getting "stuck" in the
bad-set region for an extended, self-reinforcing period; every excursion
observed resolves on its own within a handful of steps. Under this
reading, the automaton's permanent-absorption semantics for `trap` is
correctly implementing `G(not p)`'s definition, but that definition is
setting a bar the real system's recoverable dynamics don't naturally
exercise in a sustained way — the mathematical tension described in §8.19
remains real (it follows from `p2_loss`'s gradient dependence on
odd-priority-sourced transitions, which is a fact about the LOSS
construction, not about how long the real system dwells in violation), but
this data suggests trap-SPARSITY (few, and specifically SHORT, excursions)
is the dominant practical driver observed here, more so than "genuine
extended self-looping" being common and simply undertrained-for. A
secondary, practically important implication either way: even a
successful "collect more trap-visiting rollouts" intervention (§8.19's
follow-up sketch) would still yield relatively few genuine `trap`→`trap`
self-loop transitions per episode if real excursions stay this short —
collecting MORE brief excursions is not the same lever as collecting
LONGER-DWELLING ones, and the two would need different data-collection
designs (more rollouts vs. deliberately adversarial/sustained-violation
rollouts) if pursued.

Scripts: `trap_dwell_diagnosis_hazard.py`, `trap_dwell_diagnosis_height.py`
(both scratchpad).

### 8.21 Trajectory-level statistical safety rate — independent, weaker fallback certificate path adopted for `ltl_hazard_avoidance` and `ltl_height_safety` after L2's progress-measure path was confirmed structurally infeasible

**Why this exists**: §8.17–§8.20's full diagnostic chain (V-collapse
attribution, three independent remediation strategies all failing, and the
trap dwell/recovery check) together establish that the L2 co-Büchi
progress-measure certificate path (Theorem 5.2/5.4) is not viable for
either real checkpoint this project has — not because of a fixable
training-configuration bug, but because of a structural tension between
what `p2_loss` needs (odd-priority/trap-adjacent transitions) and what a
competent policy naturally provides (very few of them). This is the
adopted fallback: a genuinely different, explicitly WEAKER certificate
type, not a patched-up version of the same one.

**Implementation**: new `core/trajectory_calibration.py`, deliberately
NOT placed under `core/lppm/` — it shares no code with the progress-measure
path (no automaton, no certificate network, no per-step ranking function)
except reusing `core.lppm.calibrator._clopper_pearson_lower` for the
statistical bound itself (not reimplemented). `compute_trajectory_safety_rate()`
computes `C(τ) = 1[predicate holds at every timestep of τ]` directly on raw
AP values and reports the exact Clopper-Pearson lower bound on
`Pr[C(τ)=1]`. Test-covered (`tests/test_trajectory_calibration.py`, 5
tests: correctness of the all-safe/one-bad-step cases, CP-reuse check, and
an explicit assertion that the summary text never reads like a Theorem-5.4
warrant).

**Guarantee-strength wording discipline (enforced, not just documented)**:
`TrajectorySafetyRateResult.summary()` always states "NOT a Theorem-5.4
warrant" and spells out exactly what the number does and doesn't mean —
this is deliberate, not incidental, so that this weaker result can never be
silently quoted with the same "warrant" language reserved for Theorem
5.4's actual guarantee. **`p_hat_safety` says only**: "under the SAME
deployment distribution these trajectories were drawn from, the empirical
fraction of ENTIRE sampled trajectories with zero violations, lower-bounded
at confidence `1-γ`, is `p_hat_safety`." It carries NO per-step
ranking-function structure, NO finite-prefix-absorption-witness argument,
and NO extension beyond the sampled horizon — a fundamentally different
(and weaker) claim than Theorem 5.4's.

**Results, computed on the SAME already-reproducible sampled data (no new
data collection, no retraining, run in minutes)**:

| spec | n | k | p̂_safety (γ=0.05) |
|---|---|---|---|
| `ltl_hazard_avoidance` | 100 | 97 | 0.9243 |
| `ltl_height_safety` | 200 | 199 | 0.9765 |

Both `k` values match exactly the violation counts already established by
§7 #26's raw-signal dwell diagnosis (3/100 and 1/200 violating
trajectories respectively) — same data, independently cross-checked via a
different computation path.

**These numbers replace §7 #3/#12's `p̂_γ` as the citable result for each
spec** (see §7 #27/#28) — the old rows are kept intact for provenance and
diagnostic history, not deleted or overwritten, per this project's
established convention.
