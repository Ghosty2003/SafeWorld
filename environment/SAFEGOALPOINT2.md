# SAFEGOALPOINT2 Environment Notes

The simple PointGoal2 checkpoint was trained on the Gymnasium environment
`SafetyPointGoal2Gymnasium-v0`, with `OfflinePointGoal2Gymnasium-v0` used as
the DSRL offline dataset source when available.

Expected model dimensions:

- Observation: 60
- Action: 2
- World model checkpoint:
  `/home/chenmg93@netid.washington.edu/.cache/huggingface/hub/models--helenant--simple_pointgoal2_worldmodel/snapshots/d9158d06d2eea9940a354c02eee63bf175e08d21/simple_pointgoal2_worldmodel/checkpoints/simple_world_model.pt`

Packages needed to make paired environment rollouts:

- `gymnasium`
- `safety-gymnasium`
- `mujoco`

Packages needed to read the offline DSRL dataset directly:

- `dsrl`
- `h5py`

In the active environment checked on 2026-04-28, `gymnasium` and `torch` were
installed, but `safety_gymnasium`, `dsrl`, and `h5py` were not. No
SafetyPointGoal environments were registered, so `gym.make("SafetyPointGoal2Gymnasium-v0")`
failed with `NameNotFound`.

After installing the missing environment stack, use:

```bash
python - <<'PY'
from SafeWord_V2.environment.safegoalpoint2_info import check_safegoalpoint2_env
print(check_safegoalpoint2_env())
PY
```

For transfer calibration, instantiate
`SimplePointGoal2WorldModelWrapper` and call `sample_paired_rollouts()`. That
method requires `SafetyPointGoal2Gymnasium-v0` to be registered.
