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
  `hazard_dist`, `velocity`, `goal_dist`, or `carrying`.
- `specification.type`: `STL` or `LTL`.
- `specification.formula`: formula string over predicate names.

## AP Source Reference

The following AP keys are emitted by `safety_point_goal_adapter`:

| Key | Typical range | Notes |
|---|---|---|
| `hazard_dist` | `[-0.5, 1.0]` | Positive = clear of hazards. From `cost_hazards` or direct lidar. |
| `goal_dist` | `[0, ∞)` | Euclidean distance to goal. Lower is closer. |
| `velocity` | `[0, ∞)` | Speed magnitude in m/s. |
| `near_obstacle` | `[-0.5, 1.0]` | Signed clearance to nearest obstacle. |
| `carrying` | `0.0` or `1.0` | **Stateful** — requires `CarryingTracker`. See below. |
| `zone_a`, `zone_b`, `zone_c` | `0.0` or `1.0` | Zone membership; `0.0` when environment does not expose position. |
| `near_human` | `0.0` or `1.0` | Human proximity; `0.0` when not supported. |

### Using `carrying` in a task

`carrying` is `0.0` by default because SafetyPointGoal environments do not
emit it.  To make it live, the environment config must have a `button_zones`
block and the wrapper must create a `CarryingTracker`:

```python
from environment.adapters import CarryingTracker

zones = env_config.get("button_zones", {})
tracker = CarryingTracker.from_config(zones) if zones else None

# Inside rollout loop:
state = safety_point_goal_adapter(obs, info=info, tracker=tracker)
# state["carrying"] == 1.0 after entering button1, 0.0 after button2
```

Without a tracker, tasks that reference `carrying` will silently evaluate
`carrying = 0.0` every step, making `carrying`-triggered conditions
permanently false (degenerate verification).

## Contributor Workflow

When adding a new model/environment pair:

1. Add a runtime JSON file under `configs/settings/<model_name>.json`.
2. Add or update the corresponding wrapper in `wrappers/`.
3. Ensure the wrapper emits AP keys used by task predicates.
4. Add task JSON files only for task formulas and predicate definitions.
5. If a task uses `carrying`, add a `button_zones` block to the relevant
   `configs/environments/*.json` and wire up `CarryingTracker` in the wrapper.
6. Document any model-specific AP semantics in the wrapper or settings README.
