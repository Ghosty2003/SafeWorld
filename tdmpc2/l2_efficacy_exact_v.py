"""Track 1 control group 4 (final): exact, hand-constructed V via full-graph
SCC analysis, to isolate Mechanism 1 (finite-sample coverage gaps) from
BOTH free-MLP approximation noise (Finding 1) and the hinge-loss cyclic-
equality degeneracy discovered in the tabular-V control group.

Pre-registered expectations (written before running):
  - SAFE systems: true_safe_rate should be ~100% -- an exact V has zero
    defects on a genuinely safe system by construction (Theorem 5.2); any
    abstain here would indicate an implementation bug, to be investigated,
    not accepted.
  - UNSAFE systems (the 50 calibrated-route ones): false_safe_rate is
    expected to become nonzero for the first time, candidates drawn from
    the 23 systems previously attributed to Mechanism 1 (comfortable
    bad-source margin under the free MLP, i.e. no genuine violation was
    ever sampled). Each false-safe must be paired with the exact
    probability that a 40-step rollout from the fixed initial state hits
    the violating structure.
  - Judgment: false-safes concentrated in systems where that hit
    probability is low (<5%) -> Theorem 5.4's probabilistic claim is
    honest, the apparent false-safe is a Boolean-vs-probabilistic framing
    artifact, not a calibration failure. Any false-safe with hit
    probability >10% would be a genuine calibration failure, reported
    separately.

Construction: reuses core.lppm.finite_graph's own `_reachable` and
`_tarjan` helpers (imported directly, not reimplemented) and replicates
its SCC-condensation `remaining_bad_visits` recursion UNCONDITIONALLY
(verify_finite_cobuchi only computes it in the SAFE branch and raises
before doing so if a violating component exists). This recursion is
well-founded regardless of whether the reachable graph contains a
violating cyclic component, because successors_by_component explicitly
excludes same-component edges -- the recursion only ever traverses the
(acyclic, by definition) SCC-condensation DAG, never the possibly-cyclic
original graph. V(s) = eta * remaining_bad_visits(component_of(s)) is
therefore well-defined for every reachable state on every system,
SAFE or VIOLATION, with no approximation and no training.

For VIOLATION systems this V is (and by Corollary 5.3, mathematically
MUST be) not P1/P2-valid everywhere on the full graph: states within the
violating cyclic SCC keep a CONSTANT V for as long as a path stays inside
that SCC (successors_by_component never routes within the same
component), so a bad-source transition that does not leave the SCC has
delta=0, violating the required eta descent. This is the expected,
mathematically necessary signature of "no valid LPPM exists here" -- the
self-check (item 1) verifies violations occur ONLY there, nowhere in the
safe region, for both SAFE and VIOLATION systems.

Calibration then uses ONLY the same 60 held-out rollouts as every other
Track 1 run (same seed/RNG sequence) -- the full graph is used solely to
construct V and to compute exact hit probabilities for false-safe
attribution, never consulted during the pathwise pass/fail check itself.

Does not touch calibration data or Track B. Does not modify any prior
result file.
"""
from __future__ import annotations

import json
import pathlib
import random

from core.lppm.calibrator import _clopper_pearson_lower
from core.lppm.finite_graph import _reachable, _tarjan
from tdmpc2.l2_efficacy_run_pipeline import (
    sample_rollouts, witnessed_violation,
    N_TRAIN_ROLLOUTS, N_HELD_OUT_ROLLOUTS, ROLLOUT_LEN, ETA, GAMMA, SAFE_THRESHOLD,
)

SYSTEMS_DIR = pathlib.Path("artifacts/tdmpc2_l2_efficacy_study/systems")
OUT = pathlib.Path("artifacts/tdmpc2_l2_efficacy_study")


def exact_v_full_graph(transitions, initial_states, bad_states, eta=ETA):
    """V(s) = eta * remaining_bad_visits(component_of(s)), computed
    unconditionally (unlike verify_finite_cobuchi, which raises before this
    point if a violating component exists)."""
    graph = {s: tuple(succ) for s, succ in transitions.items()}
    bad = set(bad_states)
    reachable = _reachable(graph, initial_states)
    components = _tarjan(graph, reachable)
    component_of = {s: idx for idx, comp in enumerate(components) for s in comp}

    successors_by_component = {idx: set() for idx in range(len(components))}
    for s in reachable:
        source = component_of[s]
        for succ in graph[s]:
            target = component_of[succ]
            if source != target:
                successors_by_component[source].add(target)

    memo = {}

    def remaining_bad_visits(idx):
        if idx in memo:
            return memo[idx]
        component = components[idx]
        own = int(bool(bad.intersection(component)))
        downstream = max((remaining_bad_visits(t) for t in successors_by_component[idx]), default=0)
        memo[idx] = own + downstream
        return memo[idx]

    values = {s: eta * remaining_bad_visits(component_of[s]) for s in reachable}
    return values, component_of, components


def self_check(transitions, values, bad_states, eta=ETA):
    """Return list of (source, target, delta, kind) violations on the FULL
    graph -- expected empty for SAFE systems, expected nonempty (and
    confined to the violating region) for VIOLATION systems."""
    bad = set(bad_states)
    violations = []
    for s, succs in transitions.items():
        if s not in values:
            continue
        for t in succs:
            if t not in values:
                continue
            delta = values[t] - values[s]
            if delta > 1e-9:
                violations.append((s, t, delta, "P1"))
            elif s in bad and delta > -eta + 1e-9:
                violations.append((s, t, delta, "P2"))
    return violations


def pathwise_pass(values, seq, bad_set):
    for i in range(len(seq) - 1):
        delta = values[seq[i + 1]] - values[seq[i]]
        if delta > 1e-9:
            return False
        if seq[i] in bad_set and delta > -ETA + 1e-9:
            return False
    return True


def exact_hit_probability(transitions, initial_state, bad_set, horizon):
    """Exact probability (via forward probability propagation over the
    known transition structure, matching sample_rollouts' uniform-random-
    successor-choice model exactly) that a `horizon`-step rollout from
    initial_state EVER traverses a transition where the source state is
    bad AND stays within the same SCC as its successor (i.e. hits the
    'no valid descent possible' structure that a correct V must flag) AT
    LEAST ONCE. Computed by exact dynamic programming over the "still
    clean" state distribution: once a probability mass hits the event, it
    is credited to p_hit_at_least_once exactly once and removed from
    further propagation (it no longer matters what that mass does
    afterward) -- this avoids double/multi-counting a single trajectory
    that dwells through several consecutive bad-same-SCC hops, which an
    earlier, buggy version of this function did (summing P(hit at step t)
    over t without deduplication, which can and did exceed the true
    entry probability itself on dwelling-heavy systems)."""
    values, component_of, _ = exact_v_full_graph(transitions, (initial_state,), bad_set)
    dist = {initial_state: 1.0}  # probability mass that has NOT hit the event yet
    p_hit_at_least_once = 0.0
    for _ in range(horizon):
        new_dist = {}
        for s, p in dist.items():
            succs = transitions.get(s, ())
            if not succs:
                continue
            share = p / len(succs)
            for t in succs:
                if s in bad_set and component_of.get(t) == component_of.get(s):
                    p_hit_at_least_once += share  # credited once, not propagated further
                else:
                    new_dist[t] = new_dist.get(t, 0.0) + share
        dist = new_dist
    return p_hit_at_least_once


def run():
    rng = random.Random(42)
    files = sorted(SYSTEMS_DIR.glob("*.json"))
    records = []
    self_check_summary = []

    for i, path in enumerate(files):
        system = json.loads(path.read_text())
        transitions = {k: tuple(v) for k, v in system["transitions"].items()}
        initial_states = tuple(system["initial_states"])
        bad_set = set(system["bad_states"])

        values, component_of, components = exact_v_full_graph(transitions, initial_states, bad_set)
        violations = self_check(transitions, values, bad_set)
        self_check_summary.append(dict(
            system_id=system["system_id"], kind=system["kind"], ground_truth=system["ground_truth"],
            n_violations=len(violations),
            violations_all_bad_source=all(s in bad_set for s, t, d, k in violations),
        ))
        if system["ground_truth"] == "SAFE" and violations:
            raise AssertionError(
                f"{system['system_id']}: SAFE system has {len(violations)} violations in the exact V "
                "-- implementation bug, must be fixed before proceeding"
            )
        if system["ground_truth"] == "VIOLATION" and violations and not all(s in bad_set for s, t, d, k in violations):
            raise AssertionError(
                f"{system['system_id']}: VIOLATION system has a violation NOT sourced at a bad state "
                "-- construction bug, must be investigated"
            )

        # No torch.manual_seed call needed here (unlike the MLP/tabular runs):
        # exact V uses no torch RNG at all, and sample_rollouts() only consumes
        # the shared `rng` (random.Random) instance, so train/held-out rollouts
        # below are bit-identical to every prior experiment regardless.
        train_rollouts = sample_rollouts(rng, transitions, initial_states, N_TRAIN_ROLLOUTS, ROLLOUT_LEN)
        witness = witnessed_violation(train_rollouts, bad_set)
        if witness is not None:
            records.append(dict(system_id=system["system_id"], kind=system["kind"],
                                 ground_truth=system["ground_truth"], l2_verdict="UNSAFE",
                                 route="witnessed_violation", p_hat_gamma=None, k=None, n=None))
            continue

        held_out_rollouts = sample_rollouts(rng, transitions, initial_states, N_HELD_OUT_ROLLOUTS, ROLLOUT_LEN)
        passes = [pathwise_pass(values, seq, bad_set) for seq in held_out_rollouts]
        k = sum(passes); n = len(passes)
        p_hat_gamma = _clopper_pearson_lower(k, n, GAMMA)
        verdict = "SAFE" if p_hat_gamma >= SAFE_THRESHOLD else "ABSTAIN"

        record = dict(
            system_id=system["system_id"], kind=system["kind"], ground_truth=system["ground_truth"],
            l2_verdict=verdict, route="calibrated", p_hat_gamma=p_hat_gamma, k=k, n=n,
        )
        if system["ground_truth"] == "VIOLATION" and verdict == "SAFE":
            hit_prob = exact_hit_probability(transitions, initial_states[0], bad_set, ROLLOUT_LEN)
            record["exact_violating_structure_hit_probability_per_rollout"] = hit_prob
            record["probability_all_60_held_out_miss_it"] = (1 - hit_prob) ** N_HELD_OUT_ROLLOUTS
        records.append(record)

        if (i + 1) % 40 == 0:
            print(f"processed {i+1}/{len(files)}", flush=True)

    return records, self_check_summary


def cp_upper(k, n, conf=0.95):
    from scipy.stats import beta
    if n == 0:
        return None
    if k == n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def summarize(records):
    unsafe = [r for r in records if r["ground_truth"] == "VIOLATION"]
    safe = [r for r in records if r["ground_truth"] == "SAFE"]
    calib_unsafe = [r for r in unsafe if r["route"] == "calibrated"]
    calib_safe = [r for r in safe if r["route"] == "calibrated"]
    n_false_safe = sum(1 for r in unsafe if r["l2_verdict"] == "SAFE")
    n_true_safe = sum(1 for r in safe if r["l2_verdict"] == "SAFE")

    def phat_stats(rs):
        vals = [r["p_hat_gamma"] for r in rs if r["p_hat_gamma"] is not None]
        if not vals:
            return dict(n=0)
        vs = sorted(vals)
        return dict(n=len(vals), min=vs[0], median=vs[len(vs) // 2], max=vs[-1],
                    mean=sum(vals) / len(vals), n_ge_0_95=sum(1 for v in vals if v >= 0.95),
                    n_zero=sum(1 for v in vals if v == 0.0))

    false_safe_records = [r for r in unsafe if r["l2_verdict"] == "SAFE"]
    return dict(
        n_unsafe=len(unsafe), n_safe=len(safe),
        n_calib_unsafe=len(calib_unsafe), n_calib_safe=len(calib_safe),
        false_safe_rate=n_false_safe / len(unsafe) if unsafe else None,
        false_safe_rate_cp_upper_all=cp_upper(n_false_safe, len(unsafe)),
        false_safe_rate_cp_upper_calibrated_only=cp_upper(
            sum(1 for r in calib_unsafe if r["l2_verdict"] == "SAFE"), len(calib_unsafe)),
        n_false_safe=n_false_safe,
        false_safe_details=false_safe_records,
        true_safe_rate=n_true_safe / len(safe) if safe else None,
        p_hat_gamma_on_safe_systems=phat_stats(calib_safe),
        p_hat_gamma_on_calibrated_unsafe_systems=phat_stats(calib_unsafe),
    )


if __name__ == "__main__":
    records, self_check_summary = run()
    n_safe_with_violations = sum(1 for s in self_check_summary if s["ground_truth"] == "SAFE" and s["n_violations"] > 0)
    n_unsafe_with_violations = sum(1 for s in self_check_summary if s["ground_truth"] == "VIOLATION" and s["n_violations"] > 0)
    n_unsafe_total_scc = sum(1 for s in self_check_summary if s["ground_truth"] == "VIOLATION")
    print(f"\nself-check: SAFE systems with any violation: {n_safe_with_violations} (must be 0)")
    print(f"self-check: VIOLATION systems with violations confined to bad-source: "
          f"{n_unsafe_with_violations}/{n_unsafe_total_scc}")

    summary = summarize(records)
    print(json.dumps({k: v for k, v in summary.items() if k != "false_safe_details"}, indent=2))
    print("\nfalse-safe details:")
    print(json.dumps(summary["false_safe_details"], indent=2))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exact_v_result.json").write_text(json.dumps(
        dict(summary=summary, records=records, self_check_summary=self_check_summary), indent=2) + "\n")
