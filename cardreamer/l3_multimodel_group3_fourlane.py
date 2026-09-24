"""L3 four-gate screen, Group 3: carla_four_lane lane-departure recurrence
candidate. Requires launching CARLA (explicitly authorized for this item
only) to collect 20 real episodes with posterior latent + privileged
lateral lane offset + longitudinal progress, then closes CARLA. Everything
after data collection (structure/eyes/anti-fraud gates) uses only the
collected data, no further CARLA needed.

Lateral offset: signed perpendicular distance from the ego vehicle's
location to the nearest lane-centerline waypoint (`world.carla_map.
get_waypoint(vehicle_location)`), projected onto that waypoint's right
vector -- the same `carla_map.get_waypoint` access pattern already used
elsewhere in this exact environment file (carla_four_lane_env.py L71).

No frozen artifact modified. No sealed calibration data touched.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

OUT = pathlib.Path("/home/bot/SafeWorld/artifacts/l3_multimodel_screen/group3_fourlane")
N_EPISODES = 20
HORIZON = 500


def lateral_offset(world, vehicle):
    loc = vehicle.get_location()
    wp = world.carla_map.get_waypoint(loc)
    wp_loc = wp.transform.location
    right = wp.transform.get_right_vector()
    dx, dy = loc.x - wp_loc.x, loc.y - wp_loc.y
    return float(dx * right.x + dy * right.y)


def collect():
    import jax
    from wrappers.cardreamer_wrapper import CarDreamerWrapper
    from configs.settings import RolloutConfig

    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/3] loading CarDreamerWrapper (four_lane) ...", flush=True)
    cfg = RolloutConfig(n_rollouts=1, horizon=HORIZON, seed=0, action_source="actor")
    w = CarDreamerWrapper(cfg)
    w.load()
    jax_agent = w._jax_agent

    print("[2/3] connecting to CARLA ...", flush=True)
    import car_dreamer
    env, _ = car_dreamer.create_task("carla_four_lane")
    print("  CARLA connected.", flush=True)

    all_h, all_lat, all_prog = [], [], []
    ep_summary = []
    try:
        for ep in range(N_EPISODES):
            raw_obs = env.reset()
            done = False
            step = 0
            cum_wpt = 0
            pol_state = None
            ep_h, ep_lat, ep_prog = [], [], []
            while not done and step < HORIZON:
                is_first = (step == 0)
                obs_dict = {
                    **{k: np.asarray(v)[None] for k, v in raw_obs.items()},
                    "is_first": np.array([is_first], dtype=bool),
                    "is_last": np.array([False], dtype=bool),
                    "is_terminal": np.array([False], dtype=bool),
                    "reward": np.array([0.0], dtype=np.float32),
                }
                outs, pol_state = jax_agent.policy(obs_dict, pol_state, mode="eval")
                latent = pol_state[0][0]
                h_np = np.asarray(latent["deter"])[0]
                ep_h.append(h_np)
                ep_lat.append(lateral_offset(env._world, env.ego))
                ep_prog.append(cum_wpt)

                act_oh = np.asarray(outs["action"])[0]
                act_idx = int(np.argmax(act_oh))
                raw_obs, reward, done, info = env.step(act_idx)
                cum_wpt += int(info.get("num_completed", 0))
                step += 1
            all_h.append(np.array(ep_h))
            all_lat.append(np.array(ep_lat))
            all_prog.append(np.array(ep_prog))
            ep_summary.append(step)
            print(f"  ep {ep + 1}/{N_EPISODES} len={step} lat_range=[{min(ep_lat):.2f},{max(ep_lat):.2f}] final_prog={cum_wpt}", flush=True)
    finally:
        env.close()
        print("  CARLA closed.", flush=True)

    np.savez(OUT / "fourlane_20episodes.npz",
             h=np.array(all_h, dtype=object), lat=np.array(all_lat, dtype=object),
             prog=np.array(all_prog, dtype=object), ep_len=np.array(ep_summary))
    return all_h, all_lat, all_prog


def analyze(all_h, all_lat, all_prog):
    from scipy.signal import find_peaks
    from scipy.stats import beta
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    n = len(all_lat)
    min_len = min(len(x) for x in all_lat)
    print(f"[3/3] analysis: {n} episodes, min_len={min_len}", flush=True)

    lat_arr = np.stack([x[:min_len] for x in all_lat])
    h_arr = np.stack([x[:min_len] for x in all_h])

    # structure gate: zero-crossing (center-line crossing) count + oscillation amplitude
    crossings_per_traj = []
    amp_per_traj = []
    for i in range(n):
        sign = np.sign(lat_arr[i])
        cross = int((np.diff(sign) != 0).sum())
        crossings_per_traj.append(cross)
        amp_per_traj.append(float(lat_arr[i].std()))
    mean_crossings = float(np.mean(crossings_per_traj))
    mean_amp = float(np.mean(amp_per_traj))

    # probe error for lateral offset, from posterior deter
    Hflat = h_arr.reshape(-1, h_arr.shape[-1])
    Lflat = lat_arr.reshape(-1)
    Htr, Hte, Ltr, Lte = train_test_split(Hflat, Lflat, test_size=0.3, random_state=0)
    probe = Ridge(alpha=10.0).fit(Htr, Ltr)
    err = np.abs(probe.predict(Hte) - Lte)
    probe_err = dict(mae=float(err.mean()), p90=float(np.percentile(err, 90)), p95=float(np.percentile(err, 95)))

    structure_pass = mean_crossings >= 15  # per-episode-length window (matched to whatever min_len is, reported explicitly)
    ratio = mean_amp / probe_err["p95"] if probe_err["p95"] > 0 else None

    report = dict(
        n_episodes=n, min_episode_len=min_len,
        lateral_offset_range=[float(lat_arr.min()), float(lat_arr.max())],
        mean_centerline_crossings_per_episode=mean_crossings,
        std_crossings=float(np.std(crossings_per_traj)),
        mean_oscillation_amplitude_std=mean_amp,
        structure_gate_pass=bool(structure_pass),
        probe_error=probe_err,
        ratio_amp_vs_probe_p95=ratio,
    )
    print(json.dumps(report, indent=2), flush=True)
    (OUT / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run():
    all_h, all_lat, all_prog = collect()
    return analyze(all_h, all_lat, all_prog)


if __name__ == "__main__":
    run()
