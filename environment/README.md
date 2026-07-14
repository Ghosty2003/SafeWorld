# Environment Notes

## SafetyPointGoal2

`DreamerV3Wrapper.decode_and_replay()` defaults to replaying against the
Gymnasium environment `SafetyPointGoal2Gymnasium-v0` (override via
`RolloutConfig.extra["env_name"]`). That environment requires:

```
gymnasium
safety-gymnasium
mujoco
```

As of 2026-04-28, `gymnasium` and `torch` were installed in the active
environment, but `safety_gymnasium` was not. No SafetyPointGoal environments
were registered, so `gym.make("SafetyPointGoal2Gymnasium-v0")` failed with
`NameNotFound`. Install `safety-gymnasium` and `mujoco` to enable paired
rollouts against this environment.

## Adapter

`safety_point_goal_adapter` in `adapters.py` converts a raw observation and
`info` dict into the semantic state dict used by the SAFEWORLD verifier. See
[`../configs/environments/README.md`](../configs/environments/README.md) for
the full AP key reference and environment config schema.
