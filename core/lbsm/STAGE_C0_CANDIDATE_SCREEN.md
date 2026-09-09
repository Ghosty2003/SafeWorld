# Candidate screen: does anything escape the spectral-norm / conditional-variance double bind?

Analysis only, no data run — per task scope. All three candidates rejected
on theoretical grounds alone; none warrant a Stage-5A-style empirical probe.

| # | Candidate | Escapes spectral-norm problem? | Escapes conditional-variance problem? | Recommendation |
|---|---|---|---|---|
| 1 | Action-space recurrence | **No** | **No** | Reject, no data needed |
| 2 | Boolified AP/automaton-label reduction | **No** (question doesn't apply — d_eff untouched) | N/A | Reject, no data needed |
| 3 | Single joint angle (local low-dim DOF) | **No** | **Almost certainly no** (a priori) | Reject, no data needed |

## 1. Action-space recurrence — rejected on a Theorem-D.3 technicality, not a dimensionality argument

`action_dim = 6` confirmed directly from the checkpoint (`_pi.2.weight` shape
`(12, 512)` = 2×6 for mean+logstd; matches dm_control walker's 6 actuated
joints). Small, as expected — but this doesn't matter, because of what `L`
actually is.

Theorem D.5's `L` is the Lipschitz constant of the **post-expectation map
`G_φ(z) := E_{a~π(·|z), ξ}[φ(f_θ(z,a)+ξ)]`, as a function of `z`** — the
action is *integrated out* via the expectation over `π(·|z)`, not a
dimension the covering-net ever indexes. The covering net (Theorem D.3)
places anchors and Δ-balls in `z`-space (or the product `X=Z×Q`); `a` never
appears as an axis to cover. So "the action space is low-dimensional" is a
true but irrelevant fact — it doesn't touch `L` or `d_eff`, both of which
are about `z`. Worse: any certificate actually built to depend on `a` would
have to route through `_pi` (the policy network) to relate it back to `z`,
and `_pi` is architecturally identical to `_dynamics` (3-layer
`NormedLinear`, same LayerNorm/Mish pattern confirmed above) — composing
through it would very plausibly *add* another large spectral-norm factor
rather than removing one. Also worth noting: "actions recur" isn't a
meaningful safety property for this system in the first place (nothing
about the fall-avoidance goal is naturally stated over actions) — a second,
independent reason not to pursue this.

**No data run needed** — the objection is that the covering-net's cost
structure never depended on action-space dimension to begin with, so
shrinking it (however small `action_dim` already is) cannot help.

## 2. AP/automaton-label boolification — rejected by re-reading Remark D.4's own formula

Remark D.4 (already quoted once in Stage 3, worth re-confirming precisely
because this candidate's premise depends on it): the anchor count is
`n = Θ(|Q_reach| · M · (D/Δ)^{d_eff})`, and **`d_eff` is explicitly defined
as "the box-counting dimension of the *continuous factor* of the product
reachable set."** `|Q_reach|` (reachable automaton modes) and `M` (label
cells) are separate **multiplicative prefactors**, not part of the exponent,
and for any fixed small automaton (e.g. Stage 1's 2-state Büchi machine)
they are already `O(1)`.

Coarsening or boolifying the AP labeling changes `M` (possibly shrinking an
already-`O(1)` constant further) but cannot touch `d_eff`, which is defined
purely in terms of the continuous `z`-factor's covering dimension —
independent of how the labels are structured. This is a direct, formula-level
contradiction of the candidate's premise, not an empirical question.

**No data run needed** — this is a definitional fact about which symbol in
the formula controls what; no measurement could change the conclusion.

## 3. A single "closed" joint-angle subsystem — rejected on structural/physical grounds

Two independent objections, one that generalizes beyond this candidate:

**(a) The general argument (subsumes this candidate, and in retrospect also
explains why Stages 5/5B were structurally doomed regardless of *which*
projection was chosen):** any candidate of the form "linearly project `z`
onto a low-dim physical readout, then verify recurrence there" produces
`L_proj = ||M_readout||_2 · L_dynamics_full`. `L_dynamics_full ≈ 3,286,776`
(Stage 4, validated) is a property of the **trained network's weights** —
fixed, and unaffected by which output direction is read off afterward. A
projection matrix's operator norm for any physically-meaningful readout
(height, velocity, a joint angle, or any bounded combination) is `O(1)`
(Stage 5 got 0.54, Stage 5B's 2-row matrix got 0.71) — nowhere close to the
~10⁷-10⁸ shrinkage that would be needed to make `c = D·L/ε_min` land near
the feasible range. **No choice of linear readout can escape this**, because
`L_dynamics_full` is not a property of the readout at all. This is the
single biggest takeaway of this screening round: it is not that height and
velocity happened to be the wrong two directions — the mechanism rejects
*every* member of this candidate family at once.

**(b) The physical-plausibility argument specific to "a single joint":**
dm_control's `walker` domain is a 9-DOF planar biped (root x/z/rotation + 6
leg joints) whose whole point is tight dynamical coupling between joints —
balance and gait depend on whole-body coordination (contact forces,
center-of-mass dynamics), especially in exactly the fall/recovery regime
that already broke the 1D and 2D projections (Stage 5 finding: `h_t ∈
[0.385,1.241]` — the region containing `h_min=0.6` — is precisely where
whole-body state matters most). A single joint angle carries *less*
information about global state than `(height, velocity)` did, and that
combination already failed at 66%/84% conditional-variance ratios in that
same regime. There is no dm_control-documented "quasi-autonomous" joint
subsystem for this domain (unlike, say, a system with a genuinely
weakly-coupled appendage) — this is a deductive judgment from the domain's
known structure, not yet empirically confirmed, and is offered at somewhat
lower confidence than (a) or than candidates 1-2's formula-level rejections.
Given (a) already rejects this candidate independently and more decisively,
the empirical gap in (b) does not seem worth closing.

## Bottom line

All three candidates fail before reaching data. Two (action-space,
AP-boolification) fail on direct formula/definition grounds — `L` and
`d_eff` simply are not functions of the quantities these candidates propose
to shrink. The third (joint angle) reduces to the same general obstruction
already identified in (a): **no linear projection of this trained network's
output can shrink `L_dynamics_full`, because that quantity belongs to the
network's weights, not to any choice of readout.** This is a stronger,
more general conclusion than "three specific ideas didn't work" — it says
the entire *linear-projection-based dimensionality-reduction* strategy is
closed off for this checkpoint, not just the three or four specific
directions tried so far. Escaping it would require either retraining
`_dynamics` under an explicit Lipschitz constraint (shrinking
`L_dynamics_full` itself, at the weights level) or a certification strategy
that does not route through a generic Theorem-D.3 covering-net at all.
Neither is attempted here. No Stage-5A-style empirical probe is recommended
for any of the three candidates.
