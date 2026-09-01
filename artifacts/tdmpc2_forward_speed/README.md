# TD-MPC2 walker-walk forward-speed probe

Target: signed centre-of-mass horizontal velocity in metres/second from
`dm_control.suite.walker.Physics.horizontal_velocity()`. This is the physical
signal used by the walker-walk reward. It is not `obs[15]`, which is the root
generalized velocity.

Scope: `walker-walk-3.pt`, checkpoint seed 3, upstream TD-MPC2 commit
`e9f59321933cbc8e11a002b842adc7d4ffae8ff1`.

## Result

- C0, nested episode-grouped CV: **passed**.
  - 15 real episodes: 8 MPC + 7 random, 7,500 states.
  - R² 0.9867, MAE 0.0617 m/s, overall 1 m/s threshold agreement 99.43%.
  - MPC-only p90 absolute error 0.0647 m/s, threshold agreement 99.65%.
- C1, 200 exact anchors from 8 independent MPC episodes: **100-step transfer
  inconclusive**.
  - Depth 1/5/10 p90 absolute error: 0.0774/0.1279/0.2280 m/s.
  - Fixed C1 criteria hold through depth 15.
  - Depth 100 p90 absolute error 2.3679 m/s and 1 m/s threshold agreement 1%.

The saved sklearn probe is validated for posterior latent decoding. The C1
failure is a combined probe + learned-dynamics result: the imagined model keeps
predicting walking speed while the same model-generated actions, replayed open
loop from the exact state in MuJoCo, lead the real walker to slow or fall. Do not
use this artifact to support a 100-step L2 warrant.

Files:

- `forward_speed_probe.joblib`: fitted StandardScaler + Ridge (`alpha=10`,
  numerically stable `lsqr` solver).
- `validation.json`: complete protocol, metrics, hashes, and fixed criteria.
- `c0_real_posterior.npz`: ignored by Git; durable local raw C0 data.
- `c1_exact_mpc_anchors.npz`: ignored by Git; exact physics states.
- `c1_paired_predictions.npz`: ignored by Git; paired C1 outputs.

Reproduce after cloning TD-MPC2 at the commit above to `/tmp/tdmpc2_src`:

```bash
conda run --no-capture-output -n dyno python \
  tdmpc2/validate_forward_speed_probe.py --resume
```
