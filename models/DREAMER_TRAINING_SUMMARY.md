# DreamerV3 PointButton Training Notes

## Environment

- Conda env: `sw_bench_mini`
- Project root: `F:\codex\worldmodel\sw_bench`
- Environment/task: `SafetyPointButton1-v0`
- Robot: Point robot
- Observation dim: `76`
- Action dim: `2`
- Replay format: Dreamer `train_eps/*.npz`

## Offline Replay

Main offline replay dataset:

```text
checkpoints/dreamer/offline_run_1000_v2/train_eps
```

Summary:

```text
episodes: 1000
transitions: 300000
button_pressed_rate: 0.282
goal_reached_rate: 0.280
success_rate: 0.255
hazard_episode_rate: 0.321
cost_hazard_episode_rate: 0.414
fast_episode_rate: 0.470
l7_violation_rate: 0.217
```

Important metric distinction:

```text
hazard AP = geometry-based hazard proposition
cost_hazard = simulator cost > 0
```

These are related but not identical, so reports should keep them separate.

## Large Model Run

Logdir:

```text
checkpoints/dreamer/offline_run_1000_v2
```

Model size used:

```text
dyn_deter=1024
dyn_hidden=1024
units=1024
batch_size=32
```

Command, from `F:\codex\worldmodel\sw_bench`:

```powershell
C:\ProgramData\anaconda3\Scripts\conda.exe run -n sw_bench_mini --no-capture-output python -X utf8 scripts/train_dreamer.py --logdir checkpoints/dreamer/offline_run_1000_v2 --steps 400000 --batch_size 32 --dyn_deter 1024 --dyn_hidden 1024 --units 1024 --device cuda:0
```

Note: `--steps` is total replay/environment steps. Since offline replay starts at `300000`, `--steps 400000` means about `100000` additional online steps.

## Smaller 512 Model Run

Logdir:

```text
checkpoints/dreamer/run_offline
```

Model size:

```text
dyn_deter=512
dyn_hidden=512
units=512
batch_size=32
batch_length=64
```

Replay was copied from:

```text
checkpoints/dreamer/offline_run_1000_v2/train_eps
```

to:

```text
checkpoints/dreamer/run_offline/train_eps
```

Copy command, from `F:\codex\worldmodel\sw_bench`:

```powershell
New-Item -ItemType Directory -Force checkpoints\dreamer\run_offline\train_eps
Copy-Item -Recurse -Force checkpoints\dreamer\offline_run_1000_v2\train_eps\* checkpoints\dreamer\run_offline\train_eps\
```

Training command, from `F:\codex\worldmodel\sw_bench`:

```powershell
C:\ProgramData\anaconda3\Scripts\conda.exe run -n sw_bench_mini --no-capture-output python -X utf8 scripts/train_dreamer.py --logdir checkpoints/dreamer/run_offline --steps 340000 --batch_size 32 --batch_length 64 --dyn_deter 512 --dyn_hidden 512 --units 512 --device cuda:0
```

If `run_offline/train_eps` has already been appended to by online training, check its current step count first. For example, if it starts at `305000`, then `--steps 340000` means about `35000` additional online steps.

## Progress Checks

Large model:

```powershell
Get-Content checkpoints\dreamer\offline_run_1000_v2\metrics.jsonl -Tail 200 | Select-String '"step"' | Select-Object -Last 1
```

512 model:

```powershell
Get-Content checkpoints\dreamer\run_offline\metrics.jsonl -Tail 200 | Select-String '"step"' | Select-Object -Last 1
```

GPU check:

```powershell
nvidia-smi
```

## Notes

- Do not pass `--traindir` to this wrapper unless the underlying Dreamer config parsing is fixed; the current run path works by putting `train_eps` under `--logdir`.
- Dreamer reads existing `logdir/train_eps` and appends new online episodes to the same folder.
- For strict comparisons, copy or regenerate a clean offline replay directory for each experiment.
- Current Dreamer training does not consume AP labels directly. AP issues affect reports and future spec/planner evaluation, not the Dreamer replay training itself.
