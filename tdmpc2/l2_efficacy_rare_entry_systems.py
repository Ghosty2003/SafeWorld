"""Track 1 exact-V follow-up: a second risk profile where the violating
region is RARE to reach, not guaranteed-visited.

Every system in the original 215-system Track 1 batch starts its rollouts
INSIDE the bad region (mixed[0]) -- guaranteeing Mechanism 1 (finite-sample
coverage gaps) never gets a chance to manifest, since the danger zone is
visited by construction. This module generates a second population where
the initial state is in the SAFE region and reaches the SAME kind of
violating cyclic SCC (reused verbatim from gen_unsafe_transient_cycle)
only via one low-probability entry edge, at several controlled entry
probabilities, to see whether/how false_safe_rate rises as the danger
zone becomes rarer to sample.

Entry-probability implementation: `sample_rollouts()` and `exact_v_full_
graph()` are reused completely unmodified -- both already treat successor
tuples via uniform `rng.choice`/equal-share forward propagation, so a
weighted probability p=k/N is implemented purely by constructing a
successor tuple with k copies of the "enter danger" target and (N-k)
copies of a "stay safe" target. N=200 gives exact fractions for all four
requested tiers (0.5%=1/200, 2%=4/200, 5%=10/200, 10%=20/200).

Ground truth is unchanged in kind (still VIOLATION, verified exhaustively
by finite_graph.py) -- adding a new single-state, non-cyclic "entry" state
upstream of the existing mixed/safe structure cannot remove the reachable
violating cyclic SCC, and the self-check below confirms this rather than
assuming it.

Does not touch calibration data or Track B.
"""
from __future__ import annotations

import json
import pathlib
import random

from core.lppm.finite_graph import verify_finite_cobuchi
from tdmpc2.l2_efficacy_generate_systems import gen_unsafe_transient_cycle
from tdmpc2.l2_efficacy_exact_v import exact_v_full_graph, self_check, exact_hit_probability, pathwise_pass
from tdmpc2.l2_efficacy_run_pipeline import (
    sample_rollouts, witnessed_violation,
    N_TRAIN_ROLLOUTS, N_HELD_OUT_ROLLOUTS, ROLLOUT_LEN, GAMMA, SAFE_THRESHOLD,
)
from core.lppm.calibrator import _clopper_pearson_lower

OUT = pathlib.Path("artifacts/tdmpc2_l2_efficacy_study")
SYSTEMS_DIR = OUT / "rare_entry_systems"
N_ENTRY_SLOTS = 200
TIERS = [
    ("0.5pct", 1),   # 1/200 = 0.5%
    ("2pct", 4),     # 4/200 = 2%
    ("5pct", 10),    # 10/200 = 5%
    ("10pct", 20),   # 20/200 = 10%
]
N_PER_TIER = 13  # 13*4 = 52 systems, comfortably >= the requested 50


def gen_rare_entry(rng, n_states, entry_slots, total_slots=N_ENTRY_SLOTS):
    transitions, initial_states, bad = gen_unsafe_transient_cycle(rng, n_states)
    mixed0 = initial_states[0]  # gen_unsafe_transient_cycle's initial state IS mixed[0]
    safe_states = [s for s in transitions if s.startswith("s")]
    stay_safe_target = rng.choice(safe_states)
    entry_state = "p0"
    succs = tuple([mixed0] * entry_slots + [stay_safe_target] * (total_slots - entry_slots))
    transitions[entry_state] = succs
    return transitions, (entry_state,), bad


def generate_all(seed=100):
    rng = random.Random(seed)
    SYSTEMS_DIR.mkdir(parents=True, exist_ok=True)
    systems = []
    for tier_name, slots in TIERS:
        for i in range(N_PER_TIER):
            n_states = rng.randint(6, 30)
            transitions, initial_states, bad = gen_rare_entry(rng, n_states, slots)
            result = verify_finite_cobuchi(transitions, initial_states, bad, eta=1.0)
            if result.verdict != "VIOLATION":
                raise AssertionError(
                    f"rare-entry generator produced verdict {result.verdict}, expected VIOLATION "
                    "-- generator bug, not a valid test case"
                )
            system_id = f"rare_entry_{tier_name}_{i:03d}"
            system = dict(
                system_id=system_id, kind="unsafe_rare_entry", tier=tier_name,
                entry_probability=slots / N_ENTRY_SLOTS,
                transitions={k: list(v) for k, v in transitions.items()},
                initial_states=list(initial_states), bad_states=list(bad),
                ground_truth=result.verdict,
                n_reachable=len(result.reachable_states),
            )
            (SYSTEMS_DIR / f"{system_id}.json").write_text(json.dumps(system, indent=2) + "\n")
            systems.append(system)
    manifest = dict(status="COMPLETE", n_systems=len(systems), tiers=TIERS, n_per_tier=N_PER_TIER,
                     seed=seed, calibration_touched=False, track_b_touched=False)
    (SYSTEMS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return systems


def cp_upper(k, n, conf=0.95):
    from scipy.stats import beta
    if n == 0:
        return None
    if k == n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def run_exact_v_pipeline(seed=100):
    rng = random.Random(seed + 1)  # independent sampling stream from generation
    files = sorted(SYSTEMS_DIR.glob("rare_entry_*.json"))
    records = []
    for i, path in enumerate(files):
        system = json.loads(path.read_text())
        transitions = {k: tuple(v) for k, v in system["transitions"].items()}
        initial_states = tuple(system["initial_states"])
        bad_set = set(system["bad_states"])

        values, component_of, _ = exact_v_full_graph(transitions, initial_states, bad_set)
        violations = self_check(transitions, values, bad_set)
        if violations and not all(s in bad_set for s, t, d, k in violations):
            raise AssertionError(f"{system['system_id']}: violation not sourced at a bad state -- bug")
        if not violations:
            raise AssertionError(f"{system['system_id']}: VIOLATION system has zero violations in exact V -- bug")

        train_rollouts = sample_rollouts(rng, transitions, initial_states, N_TRAIN_ROLLOUTS, ROLLOUT_LEN)
        witness = witnessed_violation(train_rollouts, bad_set)
        hit_prob = exact_hit_probability(transitions, initial_states[0], bad_set, ROLLOUT_LEN)
        if witness is not None:
            records.append(dict(system_id=system["system_id"], tier=system["tier"],
                                 entry_probability=system["entry_probability"],
                                 l2_verdict="UNSAFE", route="witnessed_violation",
                                 p_hat_gamma=None, k=None, n=None,
                                 exact_hit_probability_per_rollout=hit_prob))
            continue

        held_out_rollouts = sample_rollouts(rng, transitions, initial_states, N_HELD_OUT_ROLLOUTS, ROLLOUT_LEN)
        passes = [pathwise_pass(values, seq, bad_set) for seq in held_out_rollouts]
        k = sum(passes); n = len(passes)
        p_hat_gamma = _clopper_pearson_lower(k, n, GAMMA)
        verdict = "SAFE" if p_hat_gamma >= SAFE_THRESHOLD else "ABSTAIN"
        prob_all_60_miss = (1 - hit_prob) ** N_HELD_OUT_ROLLOUTS
        records.append(dict(
            system_id=system["system_id"], tier=system["tier"], entry_probability=system["entry_probability"],
            l2_verdict=verdict, route="calibrated", p_hat_gamma=p_hat_gamma, k=k, n=n,
            exact_hit_probability_per_rollout=hit_prob,
            probability_all_60_held_out_miss_the_scc=prob_all_60_miss,
            predicted_false_safe=prob_all_60_miss > 0.5,
        ))
        if (i + 1) % 20 == 0:
            print(f"processed {i+1}/{len(files)}", flush=True)
    return records


def summarize(records):
    by_tier = {}
    for r in records:
        by_tier.setdefault(r["tier"], []).append(r)
    tier_summaries = {}
    for tier, rs in sorted(by_tier.items()):
        n = len(rs)
        n_false_safe = sum(1 for r in rs if r["l2_verdict"] == "SAFE")
        tier_summaries[tier] = dict(
            n=n, entry_probability=rs[0]["entry_probability"],
            false_safe_rate=n_false_safe / n, n_false_safe=n_false_safe,
            false_safe_cp_upper=cp_upper(n_false_safe, n),
            false_safe_systems=[r["system_id"] for r in rs if r["l2_verdict"] == "SAFE"],
        )
    n_total = len(records)
    n_false_safe_total = sum(1 for r in records if r["l2_verdict"] == "SAFE")
    return dict(
        n_total=n_total, n_false_safe_total=n_false_safe_total,
        false_safe_rate_overall=n_false_safe_total / n_total,
        false_safe_rate_overall_cp_upper=cp_upper(n_false_safe_total, n_total),
        by_tier=tier_summaries,
    )


if __name__ == "__main__":
    systems = generate_all()
    records = run_exact_v_pipeline()
    summary = summarize(records)
    print(json.dumps(summary, indent=2))
    (OUT / "rare_entry_result.json").write_text(json.dumps(dict(summary=summary, records=records), indent=2) + "\n")

    print("\nfalse-safe cross-check (observed vs. exact-probability prediction):")
    for r in records:
        if r["l2_verdict"] == "SAFE":
            print(json.dumps(r, indent=2))
