# CarDreamer (DreamerV3 / CARLA) verification suite

Everything CarDreamer-specific lives in this folder. The wrapper itself is
`wrappers/cardreamer_wrapper.py` (it must stay in `wrappers/` — that is the
extension point shared by all models; see `wrappers/README.md`).

Supported tasks: `carla_four_lane`, `carla_roundabout` (Town04 / Town03).

## Prerequisites

```bash
conda activate cardreamer          # JAX, dreamerv3, car_dreamer, carla, cv2, sklearn
# CARLA server (only needed for the scripts marked [CARLA] below):
/path/to/CARLA_0.9.15/CarlaUE4.sh -RenderOffScreen -carla-port=2000 -benchmark -fps=10
```

Default paths (checkpoint, CARLA root, probe data dirs) are constants at the
top of `probe_common.py` and CLI flags on each script — adapt to your machine.

## Operational rules (each learned the hard way)

1. **Never share a CARLA server between verification and training.** CARLA
   binds ports N..N+2, so put the verification instance on a distant port
   (scripts default to 2100). Check `ps -o cmd -p <PID>` before killing
   anything that holds a port.
2. **Use a fresh CARLA server for measurements.** Servers that survived
   `kill -9` accumulate ghost actors → fake collision spikes, then a C++ abort.
3. **Freeze the checkpoint you verify.** Training overwrites the logdir
   checkpoint in place; copy it aside and reference the copy.
4. **Terminal events censor observation windows.** Collisions / arrivals end
   episodes, so they fall in partial windows — naive non-overlapping slicing
   silently deletes exactly the events you care about (fixed in
   `validate_real.py::slice_windows` and the goal-probe C2 check).

## Pipeline (order matters — A validates everything downstream)

| Stage | Script | Needs CARLA | Purpose |
|---|---|---|---|
| A. CV validity | `diag_cardreamer.py`, `diag_frame_inspect.py` | no | channel order, detection completeness, sentinel audit |
| B. Progress probe | `collect_probe_data.py` | yes | four_lane cum-waypoint probe + 6-criteria transfer check |
| B'. Position probe | `collect_goal_probe.py` | yes | roundabout (x,y) probe; C0 accuracy / C1 prior-vs-real / C2 goal confusion |
| C. Model-side verdicts | `../main.py --model cardreamer --spec <id>` | no | imagination → CV APs → STL robustness |
| D. Seed stability | `witness_step_dist.py` | no | verdict reproducibility × violation-step distribution (artifact check) |
| E. Real-env cross-check | `validate_real.py` | yes | same actor/extractor/monitor on real episodes |
| F. cerr calibration | `calibrate_cerr.py` | yes | honest conformal error budget via same-posterior forking |
| G. Probe-based verdicts | `eval_l2_roundabout.py`, `eval_l3_live.py` | yes | L2 / L3 with coordinate APs |
| H. Offline analyses | `offline_checks.py {l3,l5,l5sem}` | no | reruns on saved probe data |

## Methodology notes for probe-based specs

- Probe **step-level reliability is probe-specific — measure it** (C1
  alignment against real futures); never extrapolate across probes or tasks.
- Three-layer validity check before trusting any probe-based verdict:
  (1) probe accuracy, (2) verdict agreement vs real, (3) **property liveness**
  — non-trivial real base rate AND decision scale above probe resolution.
  A spec can pass (1)(2) and still verify a meaningless property.
- Bounded STL F over a window is *windowed* reachability; report conditional
  (physically-reachable stratum) claims alongside unconditional verdicts.
