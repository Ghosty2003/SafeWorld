# SAFEWORLD V2

SAFEWORLD V2 uses three configuration layers:

| Layer | Location | Purpose |
|---|---|---|
| Task | `configs/tasks/*.json` | STL/LTL formulas and predicate definitions — what to verify. |
| Settings | `configs/settings/*.json` | Model, environment, rollout, and verification runtime settings — how to run. |
| Environment | `configs/environments/*.json` | AP threshold overrides and environment geometry — calibration. |

Runtime code must not mix task logic with runtime or calibration settings.

### Configuration priority

When the same value is defined in multiple places, the highest-priority source wins:

```
CLI flag (e.g. --env-config, --seed)        ← highest
  settings["environment"]["config"] path
    default values compiled into specs/     ← fallback
```

---

## Running A Built-In Spec

```bash
python SafeWord_V2/main.py \
  --model simple_pointgoal2 \
  --spec stl_hazard_avoidance \
  --auto-paired
```

Loads runtime defaults from `SafeWord_V2/configs/settings/simple_pointgoal2.json`.

## Running A Task JSON

```bash
python SafeWord_V2/main.py \
  --model simple_pointgoal2 \
  --task-config SafeWord_V2/configs/tasks/obstacle_avoidance.json \
  --settings-config SafeWord_V2/configs/settings/simple_pointgoal2.json \
  --auto-paired
```

The task file supplies predicates and the formula. The settings file supplies
model, environment, rollout, and verification defaults.

## Running the Full Benchmark

```bash
python SafeWord_V2/main.py \
  --model simple_pointgoal2 \
  --benchmark \
  --auto-paired
```

Runs all 23 built-in specifications and prints a summary table.

---

## CLI Reference

### Spec selection (mutually exclusive)

| Flag | Description |
|---|---|
| `--spec ID` | Built-in specification ID (default: `ltl_hazard_avoidance`). |
| `--task-config PATH` | Path to a task JSON file with custom predicates and formula. |
| `--benchmark` | Run all 23 built-in specs and print a summary table. |

### Model and environment

| Flag | Description |
|---|---|
| `--model NAME` | World model: `random`, `dreamerv3`, `safety_point_goal`, `simple_pointgoal2` (default: `random`). |
| `--checkpoint PATH` | Model checkpoint path for wrappers that require one. |
| `--env-name NAME` | Override `environment.name` from settings. |
| `--env-config PATH` | Environment config JSON for AP threshold overrides. If omitted, the path is read from `settings["environment"]["config"]`. |
| `--env-kwargs JSON` | JSON object forwarded to `gym.make()`. |
| `--reset-kwargs JSON` | JSON object forwarded to `env.reset()`. |
| `--settings-config PATH` | Explicit runtime settings JSON; overrides the default lookup by model name. |

### Rollout control

| Flag | Description |
|---|---|
| `--horizon INT` | Rollout length in steps. |
| `--n INT` | Number of rollouts. |
| `--seed INT` | Base RNG seed; rollout `i` uses `seed + i`. |
| `--action-source MODE` | Action sampling: `random`, `env`, `zeros`, or `adversarial`. |
| `--confidence-profile PRESET` | Preset rollout count: `quick`, `moderate`, or `high-confidence`. |
| `--fidelity FLOAT` | Model fidelity for the random wrapper (0 = unsafe, 1 = safe; default 0.75). |

### Verification

| Flag | Description |
|---|---|
| `--auto-paired` | Collect paired model/environment rollouts to compute transfer error. |
| `--c-hat FLOAT` | Fixed transfer error budget when paired rollouts are not used. |
| `--stop-on-done` | Stop model-only rollouts when done probability exceeds threshold. |
| `--done-threshold FLOAT` | Done probability cutoff for `--stop-on-done`. |

### Example with overrides

```bash
python SafeWord_V2/main.py \
  --model simple_pointgoal2 \
  --spec stl_speed_limit \
  --n 200 \
  --horizon 100 \
  --seed 42 \
  --action-source adversarial \
  --auto-paired \
  --reset-kwargs '{"seed": 42}' \
  --env-config SafeWord_V2/configs/environments/safety_point_goal2.json
```

---

## Using the CarryingTracker

`CarryingTracker` gives the `carrying` AP a stateful memory that persists
across timesteps. Without it, `carrying` is always `0.0` because
SafetyPointGoal environments do not emit a carrying signal natively, causing
any spec that depends on `carrying` to degenerate silently.

### Initialising from config

```python
from environment.adapters import CarryingTracker

zones = env_config["button_zones"]   # from configs/environments/*.json
tracker = CarryingTracker.from_config(zones)
```

Button positions and radius live in the environment config JSON under a
`button_zones` key — see [`configs/environments/README.md`](configs/environments/README.md).

### Episode lifecycle

```python
obs, info = env.reset()
tracker.reset()          # clear carrying state at each episode boundary

for _ in range(horizon):
    action = policy(obs)
    obs, _, terminated, truncated, info = env.step(action)
    state = safety_point_goal_adapter(obs, info=info, tracker=tracker)
    # state["carrying"] is 0.0 or 1.0 with cross-step memory
    if terminated or truncated:
        break
```

### Inside a wrapper

Create the tracker once before the rollout loop and call `tracker.reset()` at
the start of each rollout iteration, not inside the step loop.

### When not to use it

If the environment already puts `carrying` in `info`, omit the tracker. The
adapter reads `info["carrying"]` as a fallback when no tracker is passed.

---

## Adding A New Model

1. Add `SafeWord_V2/configs/settings/<model_name>.json`.
2. Add a wrapper in `SafeWord_V2/wrappers/`.
3. Export the wrapper from `SafeWord_V2/wrappers/__init__.py`.
4. Add `"<model_name>"` to the `--model` `choices` list in `main.py` and wire up the instantiation block.
5. Ensure the wrapper emits the AP keys used by the target specs (`hazard_dist`, `velocity`, `goal_dist`, etc.).
6. If the task uses `carrying`, add a `button_zones` block to the environment config and wire up `CarryingTracker` in the wrapper (see above).

Task JSON files may include explanatory metadata naming the intended model or
environment, but runtime code must not read settings from that block.
