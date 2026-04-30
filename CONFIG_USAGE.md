# SAFEWORLD V2 Configuration Usage

SAFEWORLD V2 uses two separate configuration layers:

- `configs/tasks/*.json`: task formulas and predicates only.
- `configs/settings/*.json`: model, environment, rollout, and verification runtime settings.

The code should not read model or environment settings from task JSON files.

## Running A Built-In Spec

```bash
python SafeWord_V2/main.py \
  --model simple_pointgoal2 \
  --spec stl_hazard_avoidance \
  --auto-paired
```

This loads runtime defaults from:

```text
SafeWord_V2/configs/settings/simple_pointgoal2.json
```

## Running A Task JSON

```bash
python SafeWord_V2/main.py \
  --model simple_pointgoal2 \
  --task-config SafeWord_V2/configs/tasks/obstacle_avoidance.json \
  --settings-config SafeWord_V2/configs/settings/simple_pointgoal2.json \
  --auto-paired
```

The task file supplies predicates and the formula. The settings file supplies
the model, environment, rollout, and verification defaults.

## CLI Overrides

CLI flags override settings JSON values:

```bash
python SafeWord_V2/main.py \
  --model simple_pointgoal2 \
  --spec stl_speed_limit \
  --n 200 \
  --horizon 100 \
  --seed 42 \
  --action-source adversarial \
  --auto-paired \
  --reset-kwargs '{"seed": 42}'
```

Accepted runtime override flags:

- `--settings-config PATH`: explicit runtime settings JSON.
- `--env-name NAME`: override `environment.name`.
- `--env-kwargs JSON`: JSON object passed to environment construction.
- `--reset-kwargs JSON`: JSON object passed to `env.reset()`.
- `--horizon INT`: rollout length.
- `--n INT`: number of rollouts.
- `--confidence-profile quick|moderate|high-confidence`: preset rollout count.
- `--seed INT`: rollout seed.
- `--action-source random|env|zeros|adversarial`: action sampling mode.
- `--auto-paired`: collect paired model/environment rollouts when supported.
- `--c-hat FLOAT`: fixed transfer error budget when paired rollouts are not used.
- `--stop-on-done`: stop model-only rollouts when done probability is high.
- `--done-threshold FLOAT`: threshold used by `--stop-on-done`.
- `--checkpoint PATH`: model checkpoint path for wrappers that need one.

## Adding A New Model

1. Add `SafeWord_V2/configs/settings/<model_name>.json`.
2. Add a wrapper in `SafeWord_V2/wrappers/`.
3. Export the wrapper in `SafeWord_V2/wrappers/__init__.py`.
4. Update `main.py` if the model needs a new CLI `--model` choice.
5. Ensure the wrapper emits AP keys used by specs/tasks, such as
   `hazard_dist`, `velocity`, and `goal_dist`.

Task JSON files may include explanatory metadata naming the intended model or
environment, but runtime code should not consume that metadata.
