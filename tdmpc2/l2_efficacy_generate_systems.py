"""Generate random small finite transition systems for the L2-efficacy study
(Track 1), get their EXACT ground truth via core/lppm/finite_graph.py's
verify_finite_cobuchi (exhaustive, universal-over-all-paths semantics), and
save them for the sampled-access pipeline to consume without ever looking at
the exhaustive result.

Four constructive KIND categories, chosen so ground truth is correct BY
CONSTRUCTION (verified afterward against the exact checker as a sanity
check, not trusted blindly):

  safe_unreachable: bad states exist but have no incoming edge from the
      reachable component. True SAFE, trivial difficulty.
  safe_recoverable: bad states are reachable via one-way "excursion" edges
      out of a self-contained safe home cycle, and every excursion path
      returns into the home cycle within a bounded number of steps and
      never leads back to another bad state. True SAFE via a
      recovery/co-Buchi structure (finitely many bad visits on every path),
      not because bad is unreachable.
  unsafe_trap: a reachable cyclic SCC exists all of whose states are bad
      (absorbing, no escape). True VIOLATION, the easy/obvious kind.
  unsafe_transient_cycle: a reachable cyclic SCC contains AT LEAST ONE bad
      state but also non-bad states and/or edges that COULD escape --
      still True VIOLATION under the universal (worst-path) co-Buchi
      semantics, because at least one infinite path can keep revisiting the
      bad state, even though other paths from the same states could avoid
      it. This is deliberately the harder/less obvious VIOLATION case,
      testing whether the sampled pipeline (which only sees SOME paths) can
      still catch it.

Does not touch calibration data or Track B.
"""
from __future__ import annotations

import json
import pathlib
import random
from dataclasses import asdict

from core.lppm.finite_graph import verify_finite_cobuchi

OUT = pathlib.Path("artifacts/tdmpc2_l2_efficacy_study/systems")


def _state(prefix, i):
    return f"{prefix}{i}"


def gen_safe_unreachable(rng, n_states):
    n_reach = max(3, int(n_states * rng.uniform(0.5, 0.8)))
    n_bad = n_states - n_reach
    n_bad = max(1, n_bad)
    reach = [_state("s", i) for i in range(n_reach)]
    bad = [_state("b", i) for i in range(n_bad)]
    transitions = {}
    for i, s in enumerate(reach):
        # Minimum out-degree 2 (when the reach set is large enough) so no
        # reachable state is ever a deterministic single-successor link --
        # survey found systems with n_reach small enough that randint(1,..)
        # occasionally landed on out_deg=1 for every state, collapsing all
        # sampled rollouts onto a single deterministic cycle (min observed:
        # 1 unique sequence out of 60 samples).
        lo = min(2, n_reach)
        out_deg = rng.randint(lo, min(3, n_reach))
        succs = rng.sample(reach, out_deg)
        if s not in succs and rng.random() < 0.3:
            succs[0] = s  # allow self-loop sometimes for cycle variety
        transitions[s] = tuple(succs)
    for i, b in enumerate(bad):
        # bad states form their own disconnected cyclic component (unreachable)
        transitions[b] = (bad[(i + 1) % len(bad)],)
    return transitions, (reach[0],), tuple(bad)


def gen_safe_recoverable(rng, n_states):
    """Initial state starts INSIDE a one-way excursion of bad states that
    empties into a self-contained safe home cycle. Home states have NO edge
    back into the excursion (unlike an earlier, buggy version of this
    generator that unioned an excursion edge onto an existing home-ring
    state -- that made the excursion perpetually reachable again on every
    lap, which under universal co-Buchi semantics is already a VIOLATION,
    not a recovery). Bad is therefore visited only finitely (once, on the
    way in) on every possible infinite path: true SAFE."""
    n_home = max(3, int(n_states * rng.uniform(0.4, 0.6)))
    n_excursion = max(2, n_states - n_home)
    home = [_state("h", i) for i in range(n_home)]
    excursion = [_state("e", i) for i in range(n_excursion)]
    transitions = {}
    # home is a strongly-connected safe cycle (ring plus random chords among home only)
    for i, s in enumerate(home):
        succs = {home[(i + 1) % n_home]}
        for _ in range(rng.randint(0, 2)):
            succs.add(rng.choice(home))
        transitions[s] = tuple(succs)
    # excursion is a one-way chain, entered only at t=0 (the initial state),
    # never re-enterable from home.
    chain = excursion
    for i in range(len(chain) - 1):
        transitions[chain[i]] = (chain[i + 1],)
    transitions[chain[-1]] = (rng.choice(home),)
    bad = tuple(excursion)  # all excursion states are bad; home is entirely safe
    return transitions, (chain[0],), bad


def gen_unsafe_trap(rng, n_states):
    n_safe = max(2, int(n_states * rng.uniform(0.4, 0.7)))
    n_trap = max(2, n_states - n_safe)
    safe = [_state("s", i) for i in range(n_safe)]
    trap = [_state("t", i) for i in range(n_trap)]
    transitions = {}
    for i, s in enumerate(safe):
        # Ring plus random chords (matches the home-ring style used
        # elsewhere) -- a bare deterministic ring collapsed sampled
        # rollouts' pre-trap prefix onto a single sequence (survey found
        # systems as low as 4 unique rollouts out of 60).
        succs = {safe[(i + 1) % n_safe]}
        for _ in range(rng.randint(0, 2)):
            succs.add(rng.choice(safe))
        transitions[s] = tuple(succs)
    entry = rng.choice(safe)
    transitions[entry] = tuple(set(transitions[entry]) | {trap[0]})
    for i, t in enumerate(trap):
        # trap is absorbing: every trap state's successors stay within trap
        out_deg = rng.randint(1, min(2, n_trap))
        succs = rng.sample(trap, out_deg)
        transitions[t] = tuple(succs) if succs else (t,)
    return transitions, (safe[0],), tuple(trap)


def gen_unsafe_transient_cycle(rng, n_states, escape_prob=None):
    """Starts INSIDE the mixed ring (not re-entered from safe on every lap,
    unlike an earlier version of this generator which re-offered the
    excursion from an existing safe-ring state on every pass -- that made a
    full non-escaping lap trivially likely to be sampled within a handful of
    rollouts, since the walk got many independent attempts. Here, `safe` is
    a separate, self-contained absorbing region with NO edge back into
    `mixed`, so escaping is a genuine one-way exit: the ONLY way to keep
    revisiting bad forever is to never take the escape edge, exactly the
    adversarial path the universal co-Buchi semantics considers. A full lap
    around the n_mixed-state ring without ever escaping has probability
    (1-escape_prob)**n_mixed per attempt, which for moderate escape_prob and
    n_mixed can be made small enough that many sampled short rollouts fail
    to witness a cycle directly, forcing genuine reliance on the calibrated
    route -- this is the deliberately harder case."""
    n_safe = max(2, int(n_states * rng.uniform(0.2, 0.4)))
    n_mixed = max(4, n_states - n_safe)
    safe = [_state("s", i) for i in range(n_safe)]
    mixed = [_state("m", i) for i in range(n_mixed)]
    bad = tuple(mixed[: max(1, n_mixed // 2)])
    ep = escape_prob if escape_prob is not None else rng.uniform(0.35, 0.75)
    transitions = {}
    for i, s in enumerate(safe):
        # Ring plus random chords AMONG SAFE STATES ONLY (never back into
        # mixed -- that one-way property must be preserved). A bare
        # deterministic ring made the post-escape tail fully determined by
        # which mixed state was escaped from, collapsing sampled rollouts
        # onto a handful of outcomes (survey found median 6/60 unique).
        succs = {safe[(i + 1) % n_safe]}
        for _ in range(rng.randint(0, 2)):
            succs.add(rng.choice(safe))
        transitions[s] = tuple(succs)
    for i, m in enumerate(mixed):
        succs = {mixed[(i + 1) % n_mixed]}
        if rng.random() < ep:
            succs.add(rng.choice(safe))
        transitions[m] = tuple(succs)
    return transitions, (mixed[0],), bad


GENERATORS = {
    "safe_unreachable": gen_safe_unreachable,
    "safe_recoverable": gen_safe_recoverable,
    "unsafe_trap": gen_unsafe_trap,
    "unsafe_transient_cycle": gen_unsafe_transient_cycle,
}
EXPECTED_VERDICT = {
    "safe_unreachable": "SAFE", "safe_recoverable": "SAFE",
    "unsafe_trap": "VIOLATION", "unsafe_transient_cycle": "VIOLATION",
}


def generate_one(rng, kind, n_states):
    transitions, initials, bad = GENERATORS[kind](rng, n_states)
    result = verify_finite_cobuchi(transitions, initials, bad, eta=1.0)
    if result.verdict != EXPECTED_VERDICT[kind]:
        raise AssertionError(
            f"generator {kind} produced a system whose exact verdict "
            f"{result.verdict} does not match the intended {EXPECTED_VERDICT[kind]} "
            "-- generator logic bug, not a valid test case"
        )
    return dict(
        kind=kind, n_states_requested=n_states,
        transitions={k: list(v) for k, v in transitions.items()},
        initial_states=list(initials), bad_states=list(bad),
        ground_truth=result.verdict,
        n_reachable=len(result.reachable_states), n_bad_reachable=len(result.bad_states),
        bad_fraction_reachable=len(result.bad_states) / max(1, len(result.reachable_states)),
    )


def run(n_per_kind=25, seed=0, per_kind_override=None):
    rng = random.Random(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    systems = []
    for kind in GENERATORS:
        n_this_kind = (per_kind_override or {}).get(kind, n_per_kind)
        for i in range(n_this_kind):
            n_states = rng.randint(6, 30)
            system = generate_one(rng, kind, n_states)
            system["system_id"] = f"{kind}_{i:03d}"
            systems.append(system)
            (OUT / f"{system['system_id']}.json").write_text(json.dumps(system, indent=2) + "\n")
    n_per_kind_actual = {kind: (per_kind_override or {}).get(kind, n_per_kind) for kind in GENERATORS}
    manifest = dict(
        status="COMPLETE", n_systems=len(systems), n_per_kind=n_per_kind_actual,
        kinds=list(GENERATORS.keys()), seed=seed,
        ground_truth_source="core.lppm.finite_graph.verify_finite_cobuchi (exhaustive)",
        calibration_touched=False, track_b_touched=False,
    )
    (OUT.parent / "systems_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return systems


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-kind", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.n_per_kind, args.seed)
