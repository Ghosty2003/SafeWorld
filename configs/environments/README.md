# Environment Configs

Files in this folder describe a specific environment variant: what the
observation vector contains, how to extract each AP from it, what thresholds
calibrate those APs, and which specs are meaningful for the environment.

Runtime code reads only `ap_thresholds` (via `spec_calibrator.py`).  The
`ap_extraction` block is documentation for wrapper authors; it is not executed
automatically.

---

## Schema

```json
{
  "env_id": "SafetyPointGoal2Gymnasium-v0",
  "description": "Human-readable variant description.",

  "observation": {
    "obs_dim": 60,
    "layout": {
      "accelerometer": [0,  3],
      "velocimeter":   [3,  6],
      "gyro":          [6,  9],
      "magnetometer":  [9,  12],
      "goal_lidar":    [12, 28],
      "hazard_lidar":  [28, 44]
    }
  },

  "ap_extraction": {
    "hazard_dist": {
      "source": "cost_signal",
      "method": "offset_from_cost",
      "cost_key": "cost_hazards",
      "offset": 0.5
    },
    "velocity": {
      "source": "observation",
      "method": "l2_norm",
      "obs_slice": [3, 6]
    },
    "goal_dist": {
      "source": "observation",
      "method": "lidar_to_dist",
      "obs_slice": [12, 28],
      "offset": 1.0
    },
    "carrying": { "source": "button_zones" },
    "zone_a":   { "source": "unsupported", "default": -1.0 }
  },

  "ap_thresholds": {
    "hazard_dist":   0.0,
    "velocity":      0.4,
    "goal_dist":    -0.2,
    "near_obstacle": -0.3,
    "carrying":     -0.5
  },

  "button_zones": {
    "button1_pos":   [1.5, 0.0],
    "button2_pos":   [-1.5, 0.0],
    "button_radius": 0.3
  },

  "supported_specs": ["stl_hazard_avoidance", "ltl_conditional_speed"],
  "unsupported_specs": ["stl_sequential_zones"]
}
```

---

## Field Reference

### `ap_extraction`

Documents how each AP is computed inside the wrapper.  **Not executed
automatically** — wrapper code must implement the described logic.

| `source` value | Meaning |
|---|---|
| `"cost_signal"` | Derived from `info["cost_hazards"]` or similar cost key. |
| `"observation"` | Sliced from the flat obs array using `obs_slice`. |
| `"button_zones"` | Computed by `CarryingTracker` from agent position. Requires a `button_zones` block in this file. |
| `"unsupported"` | AP cannot be extracted from this environment. The wrapper will emit the `default` value (typically `0.0` or `-1.0`). |

### `ap_thresholds`

Numeric thresholds that `spec_calibrator.py` writes into the formula tree at
runtime.  These override the defaults compiled into `specs/stl_specs.py` and
`specs/ltl_specs.py`.

A threshold for `carrying` of `-0.5` means the atom fires when
`carrying > -0.5`, i.e. when `carrying = 1.0` (pickup) and not when
`carrying = 0.0` (no load).

### `button_zones`

Required when any spec in this environment uses the `carrying` AP.  Consumed
by `CarryingTracker.from_config()`.

```json
"button_zones": {
  "button1_pos":   [x1, y1],
  "button2_pos":   [x2, y2],
  "button_radius": r
}
```

- `button1_pos` — centre of the pickup zone (xy, metres).
- `button2_pos` — centre of the drop zone (xy, metres).
- `button_radius` — radius of both zones (single shared value).

The wrapper reads this block and initialises `CarryingTracker` once per
episode:

```python
from environment.adapters import CarryingTracker

zones = env_config.get("button_zones", {})
tracker = CarryingTracker.from_config(zones) if zones else None
```

### `supported_specs` / `unsupported_specs`

Informational lists for contributors.  Not read by runtime code.  Mark a spec
as unsupported when one or more of its APs has `"source": "unsupported"` in
`ap_extraction` — running it will produce a degenerate (always-zero) signal
rather than a meaningful verification result.

---

## Files in This Folder

| File | Environment | Notes |
|---|---|---|
| `safety_point_goal2.json` | `SafetyPointGoal2Gymnasium-v0` | Default 4-hazard layout. |
| `safety_point_goal2_hard.json` | `SafetyPointGoal2Gymnasium-v0` | 8 hazards, larger zones; passes `env_kwargs` to `gym.make()`. |
| `safety_car_goal2.json` | `SafetyCarGoal2Gymnasium-v0` | Car robot variant; only velocity and hazard APs defined. |
