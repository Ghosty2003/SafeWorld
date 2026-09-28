# Original goal <0.30: three checkpoint comparison, 64 steps

Completed for OSRP-Vector SafetyPointGoal1 checkpoint seeds 0, 5, 10.
Each model used 120 fresh training, 60 development, 100 calibration and 100
test paths. Every path has 64 real imagination transitions / 65 states.
World-model weights were frozen; a separate V and training scaler were fitted
for each checkpoint. Existing experiments were not overwritten.

The fixed property is F_[0,64](norm(decoded goal vector) < 0.30). The radius
matches the original environment goal, with requested strict `<` rather than
the environment's `<=` at the boundary. No radius relaxation was used.
Actions remain random, NOT SafeDreamer's deployed planner: matching the goal
does not mean reproducing the original deployment policy. Only model futures
were evaluated, not real-environment future trajectories.

## Training and selection

Two variants per checkpoint were trained for 1,500 updates, evaluating every
300 updates: original Eq.12/ProductValue and the previously declared learned
accepting-scalar plus low-value-anchor extension. Each V was selected only
on its own development set, then frozen before calibration. Forecasts were
saved before collecting independent test paths. All checkpoints are reported;
no checkpoint was selected or refitted using holdout results.

All three selected V models are the extension at update 900. Their accepting
value is 0.00816008 < eta=0.01, learned from an initial value 0.01815 > eta.
The original-loss candidates had zero observed development sublevel members
at all inspected checkpoints for all three world models. Thus a nonempty
region here is NOT a demonstrated benefit of swapping checkpoints alone.

| Checkpoint | Training arrivals | Development arrivals | Development full C & goal |
|---|---|---|---|
| _0 | 23/120 | 5/60 | 2/60 |
| _5 | 20/120 | 8/60 | 6/60 |
| _10 | 22/120 | 7/60 | 4/60 |

## Frozen predictions and fresh test outcomes

| Checkpoint | Calibration arrival / Z_free entry | Test arrival / Z_free entry | Calibration full C & goal | Individual 95% lower for C & goal | Test full C & goal |
|---|---|---|---|---|---|
| _0 | 18/100 | 10/100 | 12/100 | 7.07% | 2/100 |
| _5 | 16/100 | 13/100 | 6/100 | 2.64% | 8/100 |
| _10 | 14/100 | 16/100 | 9/100 | 4.78% | 7/100 |

Goal probability individual calibration 95% lower bounds are 11.97%, 10.30%,
8.67%, respectively. All CP intervals are pathwise, not transitionwise.
Across the six primary claims (goal and C&goal for three checkpoints), the
Bonferroni simultaneous 95% lower bounds are:

| Checkpoint | Goal lower | C & goal lower |
|---|---|---|
| _0 | 9.82% | 5.45% |
| _5 | 8.31% | 1.74% |
| _10 | 6.85% | 3.48% |

These bound population event probabilities under iid same-distribution
assumptions, NOT realized proportions in the next 100 paths. There is a large
calibration/test discrepancy for _0's certificate event (12% versus 2%):
do not describe this prediction as accurate or stable. Its test two-sided
95% interval is approximately [0.24%, 7.04%]. The observed ranking alone does
not establish statistical superiority of one world model.

## Candidate region versus whole-path certificate

| Checkpoint | Test initial members | Later entries | Observed Z_free-source transitions | Region-source violations / exits | All-source P1 violations | Waiting-source P2 violations |
|---|---|---|---|---|---|---|
| _0 | 0 | 10 | 337 | 0 / 0 | 1590/6400 | 1966/6063 |
| _5 | 1 | 12 | 618 | 0 / 0 | 1623/6400 | 2032/5782 |
| _10 | 0 | 16 | 642 | 0 / 0 | 1596/6400 | 1981/5758 |

No low-valued pending states or certificate events without observed goal
were found on the collected data. All test entries have an observed successor.
Conditional observed-suffix closure rates are 10/10, 13/13 and 16/16, but their
individual one-sided 95% lower bounds are only 74.11%, 79.42%, 82.93%.

In these samples Z_free coincides with the absorbing completion-memory branch:
the goal has been reached at least once. It is not a claim that the robot
stays physically inside the goal circle. The learned constant accepting branch
makes P1 equality structural after completion; P2 is not required in that
accepting branch. This is an architecture-assisted region, not discovery of
an unrestricted physical invariant through the original loss alone.

Unseen waiting states could still enter the learned sublevel. No support-wide
boundary verification has been done. Many pre-completion P1/P2 constraints
fail, so region existence is not equivalent to whole-path certification.

Conclusion: nonempty sampled candidate regions are possible at the ORIGINAL
0.30 threshold with this explicit training extension. However, complete
certificate-event coverage remains low and probability prediction is not
uniformly accurate. No global or infinite-horizon warrant is established;
the report is SAMPLED_DIAGNOSTICS_ONLY, not a high-success safety guarantee.

## Reproduction and verification

The run's sampling rules and seeds match across models, but initial reset RNG
is not fully controlled; this is not asserted to be an exactly paired
initial-state/action-sequence experiment. Each split's complete latent/RSSM,
decoded observations/APs, unique fingerprints and hashes are saved.

- All three worker exit codes: 0.
- Each saved per-model report reproduced exactly by re-evaluation.
- comparison.json reproduced exactly with `--compare-only`.
- 19 relevant unit tests passed; `git diff --check` passed.
- EGL destructor warnings occurred at shutdown, after reports were saved;
  re-evaluation required no simulator and succeeded.

For each seed directory:
`python experiments/l2_achievement64.py --phase evaluate --output artifacts/safedreamer_l2_goal03_checkpoint64/seed0`
(substitute seed5 or seed10). To reproduce the comparison:
`python experiments/l2_goal03_checkpoint64.py --compare-only`.
