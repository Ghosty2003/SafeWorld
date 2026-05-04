所有关键数字都在这里了：

Button（按钮）
参数
数值
含义
buttons.num
4
共4个按钮
buttons.size
0.1 m
按钮球体半径
button_radius（AP触发）
0.35 m
进入这个距离才触发 at_button
buttons.keepout
0.2 m
摆放时与其他物体的最小距离
buttons.reward_goal
1.0
按到目标按钮的奖励
buttons.cost
1.0
按到非目标按钮的惩罚
buttons.resampling_delay
10 steps
按完后等10步才换新goal button

Hazard（危险区）
参数
数值
含义
hazards.num
4
共4个
hazards.size（实际geom）
0.2 m
危险区半径
hazard_radius（AP）
0.20 m
进入触发 hazard
near_hazard范围
0.20+0.25 = 0.45 m
触发 near_hazard
hazards.cost
1.0
每步在内部cost+1

Gremlin（会移动的障碍物）
参数
数值
含义
gremlins.num
4
共4个
gremlins.size
0.1 m
球体半径
gremlins.travel
0.35 m
每次移动距离
gremlins.dist_threshold
0.2 m
接触距离（触发cost）
gremlins.contact_cost
1.0
接触cost

速度 / fast AP
参数
数值
含义
speed_threshold
0.8 m/s
超过这个速度 → fast=True

速度没有物理限速，0.8 m/s 只是AP标签的判断阈值，不是硬性上限。
Button位置（seed=0，每集随机）
button0: (-0.352,  0.303)
button1: (-0.672, -0.112)
button2: (-1.052,  0.037)  ← goal_button
button3: (-0.311, -1.034)

全部找到了，直接告诉你每个数字在哪个文件里：

你自己项目里的文件：
sw_bench/src/safeworld/envs/pointbutton_ap_wrapper.py 第48–54行
"button_radius":      0.35   # at_button触发距离
"goal_radius":        0.30
"hazard_radius":      0.20
"near_hazard_margin": 0.25   # near_hazard = 0.20+0.25 = 0.45m
"speed_threshold":    0.80   # fast AP阈值 (m/s)


safety-gymnasium 安装包（只读，不要改）：
button_level1.py 第31–34行
self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]  # 放置范围±1.5m
self._add_geoms(Hazards(num=4, keepout=0.18))
self._add_mocaps(Gremlins(num=4, travel=0.35, keepout=0.4))

button_level0.py 第32–35行
self.placements_conf.extents = [-1, -1, 1, 1]  # level0是±1m（level1覆盖为±1.5m）
self._add_geoms(Buttons(num=4, is_constrained=False))
self._add_geoms(Goal(size=self.buttons.size * 2, alpha=0.1))  # goal disc = size 0.2

assets/geoms/buttons.py 第31–46行
size: float = 0.1          # 球体半径
keepout: float = 0.2       # 摆放时与其他物体最小距离
resampling_delay: float = 10  # 按完后等10步换新goal
cost: float = 1.0          # 按错按钮的惩罚
reward_goal: float = 1.0   # 按对的奖励
reward_distance: float = 1.0  # 趋近奖励系数


总结：你能改的只有 pointbutton_ap_wrapper.py 里的 thresholds；环境本身的参数在 safety-gymnasium 安装包里，要改的话需要用 _add_geoms(Buttons(num=4, size=0.2, ...)) 这种方式继承后覆盖。


smalldim_only_RL_data: 没有用预训练数据，dim更小（512）

{
  "env_id": "SafetyPointButton1-v0",
  "description": "PointButton1 DreamerV3 config — aux-decoder APs from latent RSSM.",

  "latent_mode": "feat",
  "_latent_mode_note": "feat: aux decoder receives cat(h, z). h deterministic (512), z categorical (32×32). Use 'h_only' for paper-faithful mode.",

  "model_arch": {
    "obs_dim":       76,
    "act_dim":       2,
    "deter_dim":     512,
    "stoch_dim":     32,
    "stoch_classes": 32,
    "enc_hidden":    [512, 512, 512],
    "dec_hidden":    [512, 512, 512],
    "aux_keys": [
      "cost", "speed", "goal_button_distance",
      "nearest_hazard_distance", "nearest_gremlin_distance"
    ]
  },

  "ap_extraction": {
    "hazard_dist": {
      "source":  "aux_decoder",
      "head":    "nearest_hazard_distance",
      "method":  "threshold_subtract",
      "comment": "signed dist = decoder_output - hazard_safe_dist  (>0 clear, <0 inside hazard)"
    },
    "near_hazard": {
      "source":  "aux_decoder",
      "head":    "nearest_hazard_distance",
      "method":  "threshold_subtract",
      "comment": "signed dist = decoder_output - near_hazard_dist  (>0 clear, <0 near hazard)"
    },
    "at_button": {
      "source":  "aux_decoder",
      "head":    "goal_button_distance",
      "method":  "threshold_subtract",
      "comment": "signed dist = decoder_output - button_radius  (<0 within button reach zone)"
    },
    "near_gremlin": {
      "source":  "aux_decoder",
      "head":    "nearest_gremlin_distance",
      "method":  "threshold_subtract",
      "comment": "signed dist = decoder_output - gremlin_safe_dist  (>0 clear, <0 too close to moving obstacle)"
    },
    "velocity": {
      "source":  "aux_decoder",
      "head":    "speed",
      "method":  "direct"
    },
    "button_pressed": {
      "source":  "unsupported",
      "default": 0.0,
      "comment": "stateful sticky AP — cannot be decoded from instantaneous latent alone"
    },
    "carrying": {
      "source":  "unsupported",
      "default": 0.0,
      "comment": "button_pressed AND NOT goal — stateful, deferred"
    },
    "goal": {
      "source":  "unsupported",
      "default": 0.0,
      "comment": "contact-based + known premature-fire bug in PointButtonAPWrapper — deferred"
    },
    "zone_a": { "source": "unsupported", "default": -1.0 },
    "zone_b": { "source": "unsupported", "default": -1.0 },
    "zone_c": { "source": "unsupported", "default": -1.0 }
  },

  "ap_thresholds": {
    "_sources": "pointbutton_ap_wrapper.py DEFAULT_THRESHOLDS + safety_gymnasium button_level1.py",

    "hazard_safe_dist":    0.20,
    "_hazard_note":        "hazards.size=0.20 (buttons.py); matches wrapper hazard_radius",

    "near_hazard_dist":    0.45,
    "_near_hazard_note":   "hazard_radius(0.20) + near_hazard_margin(0.25) from DEFAULT_THRESHOLDS",

    "button_radius":       0.35,
    "_button_note":        "at_button trigger distance from DEFAULT_THRESHOLDS; buttons.size=0.10 (物理大小), 0.35是AP触发半径",

    "gremlin_safe_dist":   0.20,
    "_gremlin_note":       "gremlins.dist_threshold=0.20 from button_level1.py — cost triggered within this distance",

    "hazard_dist":    0.0,
    "near_hazard":    0.0,
    "at_button":      0.0,
    "near_gremlin":   0.0,
    "velocity":       0.80,
    "_velocity_note": "speed_threshold=0.80 m/s from DEFAULT_THRESHOLDS in pointbutton_ap_wrapper.py"
  },

  "obs_normalisation": {
    "enabled":   false,
    "load_from": "checkpoint",
    "comment":   "latest.pt has no obs_mean/obs_std keys — disabled."
  },

  "button_zones": {
    "enabled": false,
    "comment": "Button positions are randomised per episode — no fixed coordinates. Use goal_button_distance AP head instead of fixed zone positions.",
    "arena_extents": [-1.5, -1.5, 1.5, 1.5],
    "_arena_note":   "placements_conf.extents from button_level1.py line 31"
  },

  "supported_specs": [
    "stl_hazard_avoidance",
    "ltl_hazard_avoidance",
    "stl_safe_goal_reach",
    "ltl_safe_goal",
    "stl_velocity_limit",
    "stl_gremlin_avoidance"
  ],

  "unsupported_specs": [
    "ltl_carrying_task",
    "ltl_zone_sequence",
    "stl_zone_coverage",
    "ltl_button_sequence"
  ]
}






data情况
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


largedim_usingdata Model size used:

```text
dyn_deter=1024
dyn_hidden=1024
units=1024
batch_size=32
```

Command, from `F:\codex\worldmodel\sw_bench`:

```powershell
run -n sw_bench_mini --no-capture-output python -X utf8 scripts/train_dreamer.py --logdir checkpoints/dreamer/offline_run_1000_v2 --steps 400000 --batch_size 32 --dyn_deter 1024 --dyn_hidden 1024 --units 1024 --device cuda:0
```

Note: `--steps` is total replay/environment steps. Since offline replay starts at `300000`, `--steps 400000` means about `100000` additional online steps.

## Smaller 512 Model Run

Model size:

```text
dyn_deter=512
dyn_hidden=512
units=512
batch_size=32
batch_length=64
```

```powershell
run -n sw_bench_mini --no-capture-output python -X utf8 scripts/train_dreamer.py --logdir checkpoints/dreamer/run_offline --steps 340000 --batch_size 32 --batch_length 64 --dyn_deter 512 --dyn_hidden 512 --units 512 --device cuda:0
```

If `run_offline/train_eps` has already been appended to by online training, check its current step count first. For example, if it starts at `305000`, then `--steps 340000` means about `35000` additional online steps.


