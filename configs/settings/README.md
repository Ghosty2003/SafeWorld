# Runtime Settings

Settings JSON files are the source of runtime configuration for SAFEWORLD V2.
Task JSON files define formulas; settings JSON files define how to run them.

Default lookup:

```text
--model simple_pointgoal2  -> configs/settings/simple_pointgoal2.json
--model random             -> configs/settings/random.json
--model dreamerv3          -> configs/settings/dreamerv3.json
--model safety_point_goal  -> configs/settings/safety_point_goal.json
```

Use `--settings-config path/to/file.json` to override the default.

## Schema

```json
{
  "model": {
    "type": "simple_pointgoal2",
    "checkpoint_path": null
  },
  "environment": {
    "name": "SafetyPointGoal2Gymnasium-v0",
    "wrapper": "SimplePointGoal2WorldModelWrapper",
    "config": "SafeWord_V2/configs/environments/safety_point_goal2.json",
    "kwargs": {},
    "reset_kwargs": {"seed": 42}
  },
  "rollout": {
    "horizon": 50,
    "n_rollouts": 20,
    "action_source": "random",
    "seed": 42,
    "device": "cpu"
  },
  "verification": {
    "auto_collect_paired_rollouts": false,
    "model_error_budget": 0.08
  },
  "extra": {
    "stop_on_done": false,
    "done_threshold": 0.5
  }
}
```

### Field reference

| Field | Purpose |
|---|---|
| `model.type` | Must match the `--model` CLI choice. |
| `model.checkpoint_path` | Path to a saved checkpoint; `null` for wrappers that self-contain their model. |
| `environment.name` | Gymnasium environment ID passed to `EnvWrapper`. |
| `environment.wrapper` | Class name used for display only; actual instantiation is in `main.py`. |
| `environment.config` | Path to a `configs/environments/*.json` file with AP extraction rules and thresholds. Also consumed by `CarryingTracker` when the task uses `carrying`. |
| `environment.kwargs` | Keyword arguments forwarded to `gym.make()`. |
| `environment.reset_kwargs` | Keyword arguments forwarded to `env.reset()`. Always set `"seed"` here for reproducible paired rollouts. |
| `rollout.horizon` | Number of steps per rollout. |
| `rollout.n_rollouts` | Number of rollouts collected for the latent monitor. |
| `rollout.action_source` | `random`, `env`, `zeros`, or `adversarial`. |
| `rollout.seed` | Base RNG seed; rollout `i` uses `seed + i`. |
| `verification.model_error_budget` | Fixed `c_hat` when paired rollouts are not collected. |
| `verification.auto_collect_paired_rollouts` | Set to `true` to always run paired rollouts without `--auto-paired`. |
| `extra.*` | Wrapper-specific keys passed through `roll_cfg.extra`. |

## Contributor Notes

For a new model/environment pair:

1. Add a JSON file here named after the CLI model type.
2. Add a matching wrapper in `SafeWord_V2/wrappers/`.
3. Export the wrapper from `wrappers/__init__.py`.
4. Make sure the wrapper returns state dictionaries with AP keys used by tasks.
5. Set `environment.config` to the relevant `configs/environments/*.json` path.
6. Always set `environment.reset_kwargs.seed` and `rollout.seed` for reproducibility.
7. If the task uses the `carrying` AP, the environment config must include a
   `button_zones` block (see `configs/environments/README.md`).
8. Keep model/environment settings out of task JSON files.
