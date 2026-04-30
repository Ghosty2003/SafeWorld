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
    "kwargs": {},
    "reset_kwargs": {}
  },
  "rollout": {
    "horizon": 50,
    "n_rollouts": 20,
    "action_source": "random",
    "seed": 0,
    "device": "cpu"
  },
  "verification": {
    "auto_collect_paired_rollouts": false,
    "model_error_budget": 0.08
  },
  "extra": {}
}
```

## Contributor Notes

For a new model/environment pair:

1. Add a JSON file here named after the CLI model type.
2. Add a matching wrapper in `SafeWord_V2/wrappers/`.
3. Export the wrapper from `wrappers/__init__.py`.
4. Make sure the wrapper returns state dictionaries with AP keys used by tasks.
5. Keep model/environment settings out of task JSON files.
