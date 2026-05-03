# Environment Notes

## SafetyPointGoal2

The `simple_pointgoal2` world model checkpoint was trained on the Gymnasium
environment `SafetyPointGoal2Gymnasium-v0`, with `OfflinePointGoal2Gymnasium-v0`
used as the DSRL offline dataset source when available.

Expected dimensions:

- Observation: 60
- Action: 2
- Checkpoint: `~/.cache/huggingface/hub/models--helenant--simple_pointgoal2_worldmodel/snapshots/d9158d06d2eea9940a354c02eee63bf175e08d21/simple_pointgoal2_worldmodel/checkpoints/simple_world_model.pt`

### Required packages

For paired environment rollouts:

```
gymnasium
safety-gymnasium
mujoco
```

For reading the offline DSRL dataset directly:

```
dsrl
h5py
```

As of 2026-04-28, `gymnasium` and `torch` were installed in the active
environment, but `safety_gymnasium`, `dsrl`, and `h5py` were not. No
SafetyPointGoal environments were registered, so `gym.make("SafetyPointGoal2Gymnasium-v0")`
failed with `NameNotFound`. Install `safety-gymnasium` and `mujoco` to enable
paired rollouts.

### Transfer calibration

Once the environment stack is installed, instantiate
`SimplePointGoal2WorldModelWrapper` and call `sample_paired_rollouts()`:

```python
from wrappers import SimplePointGoal2WorldModelWrapper
from configs.settings import RolloutConfig

wrapper = SimplePointGoal2WorldModelWrapper(RolloutConfig())
wrapper.load(env_name="SafetyPointGoal2Gymnasium-v0")
pairs = wrapper.sample_paired_rollouts()
```

## Adapter

`safety_point_goal_adapter` in `adapters.py` converts a raw observation and
`info` dict into the semantic state dict used by the SAFEWORLD verifier. See
[`../configs/environments/README.md`](../configs/environments/README.md) for
the full AP key reference and environment config schema.
