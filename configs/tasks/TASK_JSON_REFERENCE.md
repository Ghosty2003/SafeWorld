# Task JSON Reference

This folder stores task-level JSON specs for SAFEWORLD V2.

Each task JSON follows this top-level structure:

```json
{
  "task_name": "my_task",
  "predicates": [
    {
      "name": "predicate_name",
      "type": "distance",
      "source": "hazard_dist",
      "threshold": 0.5,
      "operator": ">"
    }
  ],
  "specification": {
    "type": "STL",
    "formula": "G[0,49] predicate_name"
  },
  "rollout": {
    "horizon": 50,
    "num_samples": 100
  },
  "calibration": {
    "delta": 0.05
  }
}
```

## Field meanings

- `task_name`: human-readable task ID.
- `predicates`: named task predicates computed from trajectory state.
- `predicates[].name`: symbolic name used inside the formula.
- `predicates[].type`: currently `scalar`, `distance`, `region`, or `decoded_scalar`.
- `predicates[].source`: state key read from each trajectory step.
- `predicates[].threshold`: task-dependent boundary.
- `predicates[].operator`: usually `>` or `<`.
- `specification.type`: usually `STL` for bounded specs or `LTL` for unbounded specs.
- `specification.formula`: temporal logic formula over predicate names.
- `rollout.horizon`: trajectory length `T`.
- `rollout.num_samples`: rollout count `N`.
- `calibration.delta`: confidence/calibration failure probability.

## Confidence presets

Typical `num_samples` choices:

- `20-50`: quick check
- `100-500`: moderate verification
- `1000+`: high-confidence safety

## L1: Simple Safety

Meaning: always stay safe.

Formula shape:

```json
{
  "task_name": "simple_safety",
  "predicates": [
    {
      "name": "safe",
      "type": "distance",
      "source": "hazard_dist",
      "threshold": 0.5,
      "operator": ">"
    }
  ],
  "specification": {
    "type": "STL",
    "formula": "G[0,49] safe"
  },
  "rollout": {
    "horizon": 50,
    "num_samples": 100
  },
  "calibration": {
    "delta": 0.05
  }
}
```

## L2: Reachability

Meaning: eventually reach a goal.

Formula shape:

```json
{
  "task_name": "reach_goal",
  "predicates": [
    {
      "name": "goal_reached",
      "type": "distance",
      "source": "goal_dist",
      "threshold": 0.0,
      "operator": "<"
    }
  ],
  "specification": {
    "type": "STL",
    "formula": "F[0,49] goal_reached"
  },
  "rollout": {
    "horizon": 50,
    "num_samples": 100
  },
  "calibration": {
    "delta": 0.05
  }
}
```

## L3: Bounded STL

Meaning: safety or reachability within a bounded window.

Formula shape:

```json
{
  "task_name": "bounded_window_task",
  "predicates": [
    {
      "name": "safe",
      "type": "distance",
      "source": "hazard_dist",
      "threshold": 0.3,
      "operator": ">"
    },
    {
      "name": "goal_reached",
      "type": "distance",
      "source": "goal_dist",
      "threshold": 0.0,
      "operator": "<"
    }
  ],
  "specification": {
    "type": "STL",
    "formula": "G[0,20] safe & F[10,30] goal_reached"
  },
  "rollout": {
    "horizon": 40,
    "num_samples": 200
  },
  "calibration": {
    "delta": 0.05
  }
}
```

## L4: Conjunction or Multi-condition

Meaning: satisfy multiple finite-horizon conditions together.

Formula shape:

```json
{
  "task_name": "safe_and_reach_goal",
  "predicates": [
    {
      "name": "safe",
      "type": "distance",
      "source": "hazard_dist",
      "threshold": 0.5,
      "operator": ">"
    },
    {
      "name": "goal_reached",
      "type": "distance",
      "source": "goal_dist",
      "threshold": 0.0,
      "operator": "<"
    }
  ],
  "specification": {
    "type": "STL",
    "formula": "G[0,49] safe & F[0,49] goal_reached"
  },
  "rollout": {
    "horizon": 50,
    "num_samples": 200
  },
  "calibration": {
    "delta": 0.05
  }
}
```

## L5: Recurrence

Meaning: visit a condition repeatedly.

Formula shape:

```json
{
  "task_name": "patrol_region",
  "predicates": [
    {
      "name": "patrol",
      "type": "region",
      "source": "zone_a",
      "threshold": 0.5,
      "operator": ">"
    }
  ],
  "specification": {
    "type": "LTL",
    "formula": "G(F patrol)"
  },
  "rollout": {
    "horizon": 80,
    "num_samples": 300
  },
  "calibration": {
    "delta": 0.05
  }
}
```

Note:
- This routes to the infinite-parity/LPPM path.
- Current support is marked `approximate`.

## L6: Persistence

Meaning: eventually enter a condition and stay there forever.

Formula shape:

```json
{
  "task_name": "stabilize_in_safe_region",
  "predicates": [
    {
      "name": "stable_safe",
      "type": "distance",
      "source": "hazard_dist",
      "threshold": 0.4,
      "operator": ">"
    }
  ],
  "specification": {
    "type": "LTL",
    "formula": "F(G stable_safe)"
  },
  "rollout": {
    "horizon": 80,
    "num_samples": 300
  },
  "calibration": {
    "delta": 0.05
  }
}
```

Note:
- This is also handled by the infinite-parity/LPPM path.
- Current support is `approximate`.

## L7: Response or Reactivity

Meaning: whenever trigger happens, response must eventually happen.

Formula shape:

```json
{
  "task_name": "react_to_obstacle",
  "predicates": [
    {
      "name": "trigger",
      "type": "scalar",
      "source": "near_obstacle",
      "threshold": 0.0,
      "operator": "<"
    },
    {
      "name": "response",
      "type": "scalar",
      "source": "velocity",
      "threshold": 0.3,
      "operator": "<"
    }
  ],
  "specification": {
    "type": "LTL",
    "formula": "G(trigger -> F response)"
  },
  "rollout": {
    "horizon": 100,
    "num_samples": 400
  },
  "calibration": {
    "delta": 0.05
  }
}
```

Note:
- `->` is now supported by the parser.
- Current support is `approximate`.

## L8: Full Multi-predicate Reactive Task

Meaning: several obligations at once, often safety plus multiple reactive or recurrent conditions.

Formula shape:

```json
{
  "task_name": "full_reactive_mission",
  "predicates": [
    {
      "name": "safe",
      "type": "distance",
      "source": "hazard_dist",
      "threshold": 0.5,
      "operator": ">"
    },
    {
      "name": "patrol_a",
      "type": "region",
      "source": "zone_a",
      "threshold": 0.5,
      "operator": ">"
    },
    {
      "name": "trigger",
      "type": "scalar",
      "source": "near_obstacle",
      "threshold": 0.0,
      "operator": "<"
    },
    {
      "name": "response",
      "type": "scalar",
      "source": "velocity",
      "threshold": 0.3,
      "operator": "<"
    }
  ],
  "specification": {
    "type": "LTL",
    "formula": "G safe & G(F patrol_a) & G(trigger -> F response)"
  },
  "rollout": {
    "horizon": 120,
    "num_samples": 500
  },
  "calibration": {
    "delta": 0.05
  }
}
```

Note:
- This maps to the hardest ladder level.
- Current support is `approximate`, because the parity/LPPM backend is still simplified.

## Practical rules

- Use `STL` with bounded operators like `G[0,49]`, `F[0,49]`, `U[0,10]` for finite-horizon tasks.
- Use `LTL` with unbounded `G(...)`, `F(...)`, `G(F ...)`, `F(G ...)`, and `G(p -> F q)` for infinite-horizon tasks.
- Predicate names in the formula must match `predicates[].name`.
- Predicate `source` values must match keys present in your trajectory state dictionaries.
- The `threshold` is always task-dependent. Change it per task, not globally.
