# SafeDreamer: transfer and reproduction checklist

Snapshot: 2026-09-28. This accompanies the `safedreamer-l2` branch update.
Git contains code, tests, documentation and selected lightweight reports,
not a complete executable experiment bundle. No new evaluation was run for
this documentation/upload update.

## 1. Clone the source

```bash
git clone --branch safedreamer-l2 https://github.com/Ghosty2003/SafeWorld.git
cd SafeWorld
git rev-parse HEAD
```

Record the commit when reproducing. The separate SafeDreamer installation used
by these experiments was at commit
`529221f02b54d5093e304d877eb2aaf6da63a9ff`. Its only untracked file at the time
of this inventory was `TrainingCommand.md`. Copy that installation or obtain
the same upstream revision, including its configuration files and dependencies.

## 2. Transfer non-Git experiment inputs separately

For complete historical replay, copy the original `SafeWorld/artifacts/`
directory (approximately 2.3 GB at inventory time), including ignored `.pt`,
`.npz` and raw metadata files, using external storage or a private file transfer.
Do not replace the original raw data with newly generated paths and call it
replay. Transferring all artifacts is recommended because the scripts cross-load
earlier fit plans, frozen candidates, regions and prior sample identities.

Important directories include:

- `safedreamer_l2_clear_window_holdout/`
- `safedreamer_l2_clear_window_v_linear6000/`
- `safedreamer_l2_clear_window_revalidation_v1/`
- `safedreamer_b_fit100_v1/`
- `safedreamer_l2_spec_search_v2/`
- `safedreamer_l2_motion_v_refine_v1/`
- `safedreamer_l2_motion_v_ensemble_v1/`
- `safedreamer_l2_motion_v_wide_v1/`
- `safedreamer_l2_motion_final_v1/`

This is an orientation list, not an exhaustive minimal dependency closure.
In particular, keep `selected_bundle.pt`, selected V weights, `region.npz`,
policy configuration, plans, identity records, raw arrays and report/audit files.

Also transfer the external world-model checkpoint:

```text
SafeDreamer/checkpoint/20240307-010600_osrp_vector_safetygymcoor_SafetyPointGoal1-v0_0.ckpt
SHA256: 09d9f094417a29e2a37dd7cc6fdafebca927416dc2c89da8d7de6939d1255bfa
```

Verify its hash with `sha256sum` after copying. For a full transfer, create a
checksum manifest of the source files and verify the destination against it.
Check checkpoint/data redistribution permissions before publishing them.

## 3. Match the runtime

The local `safedreamer` environment used Python 3.8 and these versions, queried
at packaging time (not a claim that every historical run used an identical
environment):

| Package | Version |
|---|---|
| jax | 0.3.25 |
| jaxlib | 0.3.25+cuda11.cudnn82 |
| torch | 2.0.1+cpu |
| numpy | 1.23.5 |
| scipy | 1.10.1 |
| gymnasium | 0.28.1 |
| safety-gymnasium | 1.2.1.dev15+gae966e5 |
| mujoco | 2.3.3 |
| tensorflow | 2.12.0 |
| chex | 0.1.5 |
| optax | 0.1.4 |
| ruamel.yaml | 0.17.21 |
| protobuf | 4.25.9 |
| cloudpickle | 3.1.2 |
| matplotlib | 3.7.5 |

This is a key-package inventory, not a complete lockfile. Preserve an export of
the original Conda environment and a full pip inventory for exact migration;
inspect exports for private URLs/credentials before sharing. The GPU JAX wheel
and development Safety-Gymnasium revision may require their original wheel or
source, not just installation by a public PyPI version string. Check CUDA/driver
and headless MuJoCo/EGL availability on the destination. PyTorch being CPU-only
does not mean the JAX world model ran on CPU.

## 4. Resolve paths without invalidating historical evidence

The original layout was `/home/sunyhg/Documents/{SafeWorld,SafeDreamer}`.
Some scripts and saved plans use absolute paths, and audits check code and file
hashes. A clone into a different path is therefore **not** automatically portable.

Before running, inspect checkpoint paths, SafeDreamer import paths and all
paths in the frozen plan. For exact replay, reproduce the original layout in
an appropriate container or filesystem mapping. If changing code or manifests
for portability, retain untouched originals, record the mapping and changes,
and treat the adapted run as a new run. Never silently remove hash checks or
rewrite historical audit claims to make an audit pass.

## 5. Tests, replay and new sampling are different operations

From the repository root, with the matching environment activated:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -m unittest \
  tests.test_l2_motion_final tests.test_clear_window_revalidation
python experiments/l2_motion_final.py --help
python experiments/l2_clear_window_revalidation.py --help
```

Both final entry points expose `prepare`, `collect` and `audit` stages with
`--output`. Inspect the script and frozen plan before choosing a stage. Existing
audit outputs may be write-once; work on a copy of the complete artifact bundle
for replay and preserve the originals. `collect` runs expensive new sampling;
it is not needed just to read the published results.

For a new confirmatory run, predeclare fresh, unused calibration and Test1 seed
ranges and a new output directory. Keep model/specification/V/gates fixed and
the two datasets separate. Do not reuse final outcomes for model selection.
Matching software and seeds does not by itself promise bitwise equality on
different GPU hardware.

## 6. Published scope

See [the experiment guide](README.md#final-l2-results-2026-09-28) for full counts:
the 48-state window run has calibration certificate CP lower bound 0.9739
and verdict SAFE (finite model-only); motion has 0.8664 and ABSTAIN.
Neither has 100% certificate coverage. L3 pilot and finite-recurrence development
reports are not completed infinite-horizon proofs.
