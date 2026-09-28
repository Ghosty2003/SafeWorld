# SafeDreamer finite-window recurrence pilot

This is a new fit-only reset-budget entry, not paper Theorem5.6, not the W/U
conditional-expectation experiments and not a TD-MPC2 experiment.

Run from SafeWorld, using an unused output directory:

```bash
env PATH=/home/sunyhg/miniconda3/envs/safedreamer/bin:$PATH \
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  MPLCONFIGDIR=/tmp/safeworld-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  /home/sunyhg/miniconda3/envs/safedreamer/bin/python -u \
  experiments/safedreamer_finite_recurrence_pilot.py \
  --output artifacts/NEW_FIT_PILOT
```

The fixed seed map makes this a replay of the same experiment, not a new
independent statistical replication. New studies must reserve new nonoverlapping
reset and imagination seeds and record that change before collection.

`--role cal_delta`, `--role cal_CP` and `--role test` deliberately raise an error.
M_MIN, final gate definitions and Delta must be frozen in the proper sequence
before a future calibration/test entry is enabled. The pilot does not issue CP
bounds. Do not treat fit_max_residual as an independently calibrated Delta.

The active prototype region gate combines a fit-state envelope with soundness
on observed low-budget sources. It is not a globally validated/invariant set.
CV and sliding constraints are explicitly disabled in this pilot.

Audit a completed run (writes new audit.json and fit_rows.csv, refuses overwrite):

```bash
/home/sunyhg/miniconda3/envs/safedreamer/bin/python \
  experiments/safedreamer_finite_recurrence_audit.py artifacts/NEW_FIT_PILOT
```

Tests:

```bash
/home/sunyhg/miniconda3/envs/safedreamer/bin/python -m unittest discover \
  -s tests -p 'test_finite_recurrence_pilot.py'
```

Future sequence: freeze candidate/spec/rules on fit; fresh cal_delta estimates
one whole-path score quantile; freeze Delta; fresh cal_CP estimates the fixed
binary certificate event; only then fresh1000 Test1, with exact binomial testing.
Do not use calibration or test to tune the candidate or detector.
