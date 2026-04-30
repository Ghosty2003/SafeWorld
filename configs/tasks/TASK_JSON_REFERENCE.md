Task JSON files define temporal-logic tasks only. They should not be used as
runtime configuration for models, environments, rollout counts, devices, or
calibration settings. Runtime configuration lives in `configs/settings/`.

Use task JSON when you need predicates that are not already represented by a
built-in spec in `specs/`.

## Required Shape

```json
{
  "id": "obstacle_avoidance",
  "task_name": "Obstacle avoidance",
  "description": "Human-readable task description.",
  "metadata": {
    "intended_environment": "SafetyPointGoal2Gymnasium-v0",
    "intended_model": "simple_pointgoal2",
    "note": "Metadata is documentation only. SAFEWORLD does not load runtime settings from this block."
  },
  "predicates": [
    {
      "name": "safe",
      "type": "distance",
      "source": "hazard_dist",
      "threshold": 0.0,
      "operator": ">"
    }
  ],
  "specification": {
    "type": "STL",
    "formula": "G[0,49] safe"
  }
}
```

## Fields

- `id`: stable task identifier.
- `task_name`: display name.
- `description`: human-readable task summary.
- `metadata`: optional documentation for contributors. This can mention the
  environment/model the task was designed around, but runtime code must not
  read model or environment settings from here.
- `predicates`: named scalar predicates computed from wrapper trajectory
  state dictionaries.
- `predicates[].source`: key already emitted by a wrapper, such as
  `hazard_dist`, `velocity`, or `goal_dist`.
- `specification.type`: `STL` or `LTL`.
- `specification.formula`: formula string over predicate names.

## Contributor Workflow

When adding a new model/environment pair:

1. Add a runtime JSON file under `configs/settings/<model_name>.json`.
2. Add or update the corresponding wrapper in `wrappers/`.
3. Ensure the wrapper emits AP keys used by task predicates.
4. Add task JSON files only for task formulas and predicate definitions.
5. Document any model-specific AP semantics in the wrapper or settings README.
