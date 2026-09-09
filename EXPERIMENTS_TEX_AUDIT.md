# `experiments.tex` Audit — Discrepancy List (Step 1 only, no rewrite)

**Source audited:** `/home/bot/Downloads/_ICLR_27_SafeWorld/paper/sections/experiments.tex` (41 lines, `\section{Experiments}`).
**Reference for "V2 reality":** `/home/bot/SafeWorld` (SAFEWORLD V2) as described in `experimental_setup.md` §2–§4 and `EXPERIMENT_CONFIG.md`.
**Scope of this pass:** list every discrepancy between what the tex describes and what V2 actually is/has, and recommend **REWRITE** (describes something real that needs updating to V2's actual facts) vs. **MECHANISM ABSENT** (describes something that does not exist anywhere in V2 — needs a decision to delete or keep as an explicitly-labeled separate analytic/toy construction). Per instruction: **no rewriting happens in this pass** — this is the list to confirm against before any edit.

---

## Line-by-line / paragraph-by-paragraph

### L3 (intro paragraph)

> "The learned-model runs are pending: quantitative magnitudes here are controlled-dynamics Tier-2 illustrative design targets, not measured results."

- **Discrepancy:** False relative to V2. Learned-model runs are **not** pending — V2 has real rollouts and real (if incompletely-attributed) numbers from two trained checkpoints (CarDreamer DreamerV3 on carla_four_lane/carla_roundabout; TD-MPC2 on walker-walk).
- **Recommendation: REWRITE.** Needs to state plainly that learned-model results now exist for two checkpoint families, with the caveats already established in `EXPERIMENT_CONFIG.md` (Theorem 5.4's Z_free-closure premise unverified; height_safety is a documented degenerate case). Do not delete — this is exactly where the paper's real evidence should be introduced.

### L7 ("Environments and specifications")

> "We use four Safety-Gymnasium tasks (SafetyPointGoal1/2-v0, SafetyCarGoal1-v0, SafetyPointButton1-v0)..."

- **Discrepancy:** None of these environments exist in V2. V2's real environments are CarDreamer (`carla_four_lane`, `carla_roundabout`) and TD-MPC2 (`walker-walk`; sibling checkpoints `walker-run`/`walker-stand`/`-backwards` exist but are unevaluated).
- **Recommendation: REWRITE.** Environment names/count must become the real ones. Note the real count is smaller (2 CarDreamer tasks + 1 TD-MPC2 task, not "four").

> "...evaluate the 15-spec taxonomy plus two extended-coverage specs (Persistence, Reactivity) across the Manna–Pnueli hierarchy."

- **Discrepancy:** V2's registry currently has **30 specs** (16 LTL + 14 STL — the original 15+8, plus `ltl_height_safety`/`stl_height_safety`, three roundabout-motivated STL variants, and two human-task specs). Per `experimental_setup.md` §4.0, only a small subset of (checkpoint, spec) cells are actually RUNNABLE.
- **Recommendation: REWRITE.** Spec count and coverage description need to match §4.0's real per-checkpoint RUNNABLE/BLOCKED breakdown, not a blanket "15+2."

> "The canonical headline rollout horizon is $T=50$..."

- **Discrepancy:** V2 uses different horizons per (checkpoint, spec): 50 (CarDreamer hazard_avoidance), 55 (`stl_safe_flow_patrol` CV variant), 80 (roundabout L3 formal verdict), 100 (TD-MPC2 height_safety). There is no single canonical $T$.
- **Recommendation: REWRITE.** Either report horizon per (checkpoint, spec) or explicitly state horizon is not yet standardized across V2's real pipelines.

> "...bounded lasso-language check against a reference LTL semantics as their intended (prototype, not-yet-completed) validation criterion..."

- **Discrepancy:** **No such lasso-language check exists anywhere in V2.** `core/lppm/automaton.py` builds automata either via Spot (unavailable — not installed in any V2 conda env, confirmed this session) or via hand-written template automata (Safety/Guarantee/Obligation/Recurrence patterns) with no separate formal-language validation step against a reference semantics.
- **Recommendation: MECHANISM ABSENT.** Needs a decision: drop this claim entirely, or replace with what V2 actually does (template automaton construction, `dpa.exact`/`automaton_translation` flag distinguishing Spot-exact vs. template, as fixed this session).

> "...operational support is scoped to the five validated-in-intent shapes (the Obligation automaton is a known language-incorrect draft, excluded from operational scope)..."

- **Discrepancy — confirmed by direct grep, not just unverified:** `core/lppm/automaton.py:116` **has** an `if mp == "Obligation":` branch (a real, present automaton-construction template), and it is not flagged anywhere in V2's code or docs as "known language-incorrect" or excluded from operational scope — `ltl_safe_goal`/similar Obligation-class specs are treated as ordinarily runnable (subject to the same AP-availability gating as everything else, see §4.0).
- **Recommendation: MECHANISM ABSENT (contradicted, not just unmatched).** Either this refers to a real, specific bug in the Obligation template that was found and fixed at some point (not documented anywhere findable this session) and the claim is now stale, or it describes a different, non-V2 implementation. Do not silently drop or silently keep — confirm which, since V2's actual Obligation branch does not carry any such warning today.

### L9 ("World models and baselines")

> "Verification runs through a common `WorldModel` interface."

- **Discrepancy:** False. `experimental_setup.md` §2 already establishes: **no shared `WorldModel` Protocol class exists** in V2. The real interface is `wrappers/base.py`'s abstract class plus one hand-written wrapper per model, with materially different methods per wrapper (e.g. `TDMPC2Wrapper.sample_latent_rollouts_from_states()` has no CarDreamer equivalent).
- **Recommendation: REWRITE.** Must describe the real per-model wrapper situation, not a unified interface.

> "Current experiments use a controlled linear-dynamics model (`RandomWorldModel`) to study convergence/timing/scaling..., plus the small trained models of Section~\ref{sec:exp_l2l3}; DreamerV3/TD-MPC2 integration is deferred (GPU-pending...)."

- **Discrepancy:** Directly contradicted by V2. DreamerV3 (via CarDreamer) and TD-MPC2 integration is **not deferred** — both are integrated, checkpoint-loaded, and have real rollout-derived results. `wrappers/random_wrapper.py` (`RandomWorldModelWrapper`) does exist in V2 and is described in its own docstring as a controlled baseline for validating verification algorithms independent of GPU training variability — that part is plausibly reusable — but the framing that trained models are "deferred" is false.
- **Recommendation: REWRITE.** Keep the `RandomWorldModel` framing (real, matches V2), but replace "integration is deferred" with the real state: two trained checkpoint families are integrated, with the specific gaps that exist (deductive-branch absence, Z_free-closure gap) stated honestly instead of a blanket "deferred."

> "The verification peer is predicate-abstraction CEGAR, whose Safe verdict is sound only over the sampled behavior set, not kernel-wide..."

- **Discrepancy:** `core/cegar/*` does exist in V2 (confirmed this session, six files: abstraction/buchi/counterexample/loop/product/scc) and this description of CEGAR's `Safe` verdict is actually **consistent** with what this session found and fixed (`CegarResult.coverage_verified` field added specifically because "sound only over the sampled behavior set, not kernel-wide" was previously an unenforced docstring claim). However, I found no evidence CEGAR has been run as an actual **comparison baseline** against CarDreamer/TD-MPC2 rollout data in V2 — only synthetic/smoke-test trajectories were used when this session regression-tested it.
- **Recommendation: REWRITE (partial match).** The conceptual description of CEGAR's guarantee is reusable and matches V2's real semantics after this session's fix. But "is the verification peer" implies a real comparison was run — confirm whether one exists anywhere in V2 (I did not find one) before keeping that framing.

> "Heuristic proxies of [Abate et al. 2024, Henzinger et al. 2025, Ansaripour et al. 2023]... are labeled as proxies and not compared on false-warrant rate; the load-bearing L3 comparison is..."

- **Discrepancy:** **No implementation of these baselines exists anywhere in V2.** No L3/LBSM implementation exists in V2 at all (established earlier this session: Manna-Pnueli recurrence-class properties are flagged as "structurally uncertifiable by LPPM, requires unimplemented L3/LBSM").
- **Recommendation: MECHANISM ABSENT.** This entire baseline-comparison framing describes work that doesn't exist in V2. Needs a decision: is this future work to be described as such, or should it be removed until an actual L3 implementation exists?

> "Policy learners (SafeDreamer, CPO, DreamerV3, Shielding) are context, not verification competitors."

- **Discrepancy:** None of these are implemented or run in V2.
- **Recommendation: MECHANISM ABSENT** (as a comparison) — if kept, should be reframed as pure related-work citation, not something evaluated in "this section."

> "Ablations: No-Certificate (monitor only), Class-Agnostic (one CBF-style scalar regardless of parity), STL-Only (no calibration)."

- **Discrepancy — confirmed by repo-wide grep:** `grep -rn "No-Certificate\|Class-Agnostic\|STL-Only\|no_certificate\|class_agnostic\|stl_only"` across all of `/home/bot/SafeWorld`'s Python files returns **zero matches**. None of these three ablations exist in V2 under these names or any obvious variant.
- **Recommendation: MECHANISM ABSENT (confirmed).**

> "Main tables use uniform random actions (broad exploration, not deployed-policy safety); policy-conditioned verification is in Appendix."

- **Discrepancy:** **Inverted relative to V2's actual practice.** This whole session's TD-MPC2 work established the opposite emphasis: `action_source="mpc_plan"` (deployment-policy-driven) is the only validated, primary-result path; `action_source="pi_prior"`/random is explicitly secondary/coverage-only and every trajectory produced with it is tagged `_approx_pi_prior` specifically so it cannot be mistaken for a primary result. CarDreamer's `action_source="actor"` is likewise the primary path used for `stl_hazard_avoidance`'s real LPPM result.
- **Recommendation: REWRITE.** Framing needs to flip: policy-conditioned/deployment-driven rollouts are primary in V2; random-action rollouts are the auxiliary coverage-only path.

### L11 ("Metrics and units of analysis")

> "...across the 15 shared trained seeds (seed-level warrant rate $k/15$..."

- **Discrepancy:** V2 has **1 seed per checkpoint today** (per `experimental_setup.md` §8.3: "no multi-seed sweep exists yet").
- **Recommendation: REWRITE or explicitly mark aspirational.** "15 seeds" is a target-design number, not V2's current reality.

> Clopper–Pearson / Ville bounds, Theorem references `thm:lppm_conformal`/`thm:lbsm_warrant`.

- **Discrepancy:** Structurally consistent with V2's real `_clopper_pearson_lower` (exact one-sided CP, confirmed this session) — but the LaTeX theorem labels here need to be checked against the corrected numbering established this session (L1=Theorem 5.1, L2 statistical=Theorem 5.4; `thm:lbsm_warrant` would be L3, which V2 does not implement at all).
- **Recommendation: REWRITE (partial match) — the CP mechanics are real and reusable; the theorem cross-references need updating, and any `thm:lbsm_warrant` citation needs to be scoped as "not yet implemented in V2," not silently kept as if it were.**

> "On analytically controlled systems with known $\Pr[\tau\models\spec]$... false-warrant rate, coverage, abstention, and tightness... it is there that the heuristic proxies are run."

- **Discrepancy:** Depends on the "heuristic proxies" (flagged MECHANISM ABSENT above). The `RandomWorldModel`-based false-warrant-rate/coverage sweep concept is plausible and may partially exist (`RandomWorldModelWrapper`'s own docstring mentions exactly this use case — "validate verification algorithms independently of GPU training variability"), but I found no evidence such a sweep has actually been run and reported in V2.
- **Recommendation: REWRITE (partial match) — needs confirmation whether this sweep has ever actually been executed and produced numbers anywhere in V2; if not, mark pending rather than implying it's done.**

### L13 ("Implementation status")

> "...several theorem premises (support-wide NN verification, certified containment/collar, decoder-based AP extraction) are not yet implemented..."

- **Discrepancy:** This is **directionally correct and closely matches** what this session independently found and documented in `EXPERIMENT_CONFIG.md` §8 (deductive/support-wide NN-verification branch absent everywhere; Z_free-interior closure/"containment" premise unverified). This is the one paragraph in the whole file that's already roughly aligned with V2 reality, modulo exact wording.
- **Recommendation: REWRITE (light touch) — replace the generic phrasing with the session's specific, named findings (no deductive L2 branch anywhere; Z_free-closure gap confirmed on two independent (checkpoint, spec) pairs; `support_level` label-conflation bug found and fixed) rather than deleting this paragraph.**

> "Tier 1 (analytic constructions): the closed-form co-Büchi invariant-ball construction and the worked recurrence construction of Section~\ref{sec:exp_l2l3}..."; entire §19-40 (the analytic L2/L3 construction subsection and its table).

- **Discrepancy:** This entire subsection (structured-residual invariant ball, $\rho^\star$ closed form, LBSM recurrence construction, Table `tab:l2l3`) describes an **analytic, controlled-linear-dynamics demonstration that has no counterpart anywhere in V2.** V2 has no invariant-ball/spectral-norm-envelope construction, no L3/LBSM implementation at all, and no `RandomWorldModel`-based analytic L2 demonstration matching this description (V2's L2 is the real, data-driven `fit_lppm`/`calibrate_lppm` statistical branch on real CarDreamer/TD-MPC2 rollouts — categorically different from a closed-form analytic argument on a synthetic linear system).
- **Recommendation: MECHANISM ABSENT — the biggest single decision point in this audit.** This is a large, self-contained, internally-consistent piece of writing (not scattered discrepancies like elsewhere). Needs an explicit decision: (a) keep it as a clearly-labeled *separate* analytic toy-example section (if the paper wants a controlled, fully-solved demonstration alongside the real CarDreamer/TD-MPC2 evidence), or (b) remove it and rely solely on V2's real (data-driven, no-L3) results. Do not merge or reconcile it with V2's numbers — it is not describing the same experiment.

### L15–17 ("Main results")

> "8 headline specs × 4 envs"; "CPO/SafeDreamer/Shielding neither express nor certify Levels 5–8"; the whole verification-rate-schema/ablation/Appendix cross-reference paragraph.

- **Discrepancy:** References the same non-existent baselines (CPO/SafeDreamer/Shielding — MECHANISM ABSENT, see above) and the same non-existent "4 envs × 8 headline specs" grid (REWRITE, see L7 above). Internally consistent with the rest of the file's aspirational framing, not with V2.
- **Recommendation:** Same two dispositions as already listed above, applied here; no new discrepancy type.

---

## Summary table (for the confirmation pass)

| Region | Disposition needed | New discrepancy, or same as another entry? |
|---|---|---|
| L3 intro | REWRITE | new |
| L7 env list | REWRITE | new |
| L7 spec taxonomy count | REWRITE | new |
| L7 horizon $T=50$ | REWRITE | new |
| L7 lasso-language check | MECHANISM ABSENT | new |
| L7 Obligation-automaton exclusion | MECHANISM ABSENT (confirmed contradicted) | new |
| L9 common `WorldModel` interface | REWRITE | new |
| L9 "DreamerV3/TD-MPC2 deferred" | REWRITE | new |
| L9 CEGAR-as-peer | REWRITE (partial) | new |
| L9 heuristic-proxy baselines | MECHANISM ABSENT | new |
| L9 policy-learner baselines | MECHANISM ABSENT | same class as above |
| L9 three named ablations | MECHANISM ABSENT (confirmed) | new |
| L9 random-vs-policy-conditioned framing | REWRITE (inverted) | new |
| L11 "15 shared seeds" | REWRITE | new |
| L11 theorem cross-references | REWRITE (partial) | new |
| L11 false-warrant-rate sweep | REWRITE (partial, confirm run status) | new |
| L13 implementation-status paragraph | REWRITE (light touch, already close) | new |
| L19–40 analytic L2/L3 construction + table | **MECHANISM ABSENT — biggest decision point** | new |
| L15–17 main-results paragraph | REWRITE + MECHANISM ABSENT | reuses above |

**Not yet done (explicitly out of scope for this pass, per instruction):** no rewriting, no placeholder insertion, no deletion. This is the change-scope list only, awaiting confirmation on each MECHANISM ABSENT item (especially the L19–40 analytic construction, which is the largest single call to make) before any edit happens.
