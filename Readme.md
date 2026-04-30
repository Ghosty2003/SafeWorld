# SAFEWORLD V2 Configuration Usage (Updated)

SAFEWORLD V2 uses three separated configuration layers:

configs/tasks/*.json: task formulas and predicates only (what to verify).
configs/settings/*.json: model, environment, rollout, and verification runtime settings (how to run).
--env-config or settings["environment"]["config"]: environment calibration overrides (threshold tuning layer).

The code must not mix task logic with runtime or calibration settings.

🔧 Configuration Architecture
1. Task Config (Logic Layer)

Located in:

configs/tasks/*.json

Defines:

STL/LTL formulas
predicates
safety properties

❌ Must NOT include:

model selection
environment name
rollout parameters
2. Settings Config (Runtime Layer)

Located in:

configs/settings/*.json

Defines:

model selection
environment name
rollout configuration
verifier parameters
execution defaults

Example:

{
  "model": "simple_pointgoal2",
  "environment": {
    "name": "SafetyPointGoal2Gymnasium-v0",
    "config": "configs/environments/safety_pointgoal2.json"
  },
  "rollout": {
    "n": 20,
    "horizon": 50
  }
}
3. Environment Config (Calibration Layer) ✨ NEW

Used via:

CLI: --env-config PATH
OR settings: environment.config

Defines:

AP threshold overrides
safety margin calibration
environment-specific correction factors

Example:

{
  "ap_thresholds": {
    "hazard_dist": 0.25,
    "goal_dist": 0.15
  }
}

This layer is applied after spec loading and before verification.

🔁 Configuration Priority Order

When multiple sources define environment calibration:

CLI --env-config                        (highest priority)
        ↓
settings["environment"]["config"]
        ↓
default values in spec definitions     (fallback)
🚀 Running A Built-In Spec
python SafeWord_V2/main.py \
  --model simple_pointgoal2 \
  --spec stl_hazard_avoidance \
  --auto-paired

This loads runtime defaults from:

SafeWord_V2/configs/settings/simple_pointgoal2.json
📦 Running A Task JSON
python SafeWord_V2/main.py \
  --model simple_pointgoal2 \
  --task-config SafeWord_V2/configs/tasks/obstacle_avoidance.json \
  --settings-config SafeWord_V2/configs/settings/simple_pointgoal2.json \
  --auto-paired
Task file provides: predicates + formulas
Settings file provides: runtime execution config
⚙️ CLI Overrides

CLI flags override all settings JSON values:

python SafeWord_V2/main.py \
  --model simple_pointgoal2 \
  --spec stl_speed_limit \
  --n 200 \
  --horizon 100 \
  --seed 42 \
  --action-source adversarial \
  --auto-paired \
  --reset-kwargs '{"seed": 42}' \
  --env-config configs/environments/safety_pointgoal2.json
🧩 Accepted Runtime Override Flags
--settings-config PATH
Explicit runtime settings JSON.
--env-config PATH
Environment calibration overrides (AP thresholds, safety margins).
--env-name NAME
Override environment name.
--env-kwargs JSON
JSON passed to environment constructor.
--reset-kwargs JSON
JSON passed to env.reset().
--horizon INT
rollout length.
--n INT
number of rollouts.
--confidence-profile quick|moderate|high-confidence
preset rollout counts.
--seed INT
randomness seed.
--action-source random|env|zeros|adversarial
action sampling mode.
--auto-paired
enable paired model/environment rollouts.
--c-hat FLOAT
fixed transfer error budget (when pairing unavailable).
--stop-on-done
stop early when termination probability is high.
--done-threshold FLOAT
threshold for early stopping.
--checkpoint PATH
model checkpoint path.
➕ Adding A New Model

Add:

SafeWord_V2/configs/settings/<model_name>.json

Implement wrapper in:

SafeWord_V2/wrappers/

Export wrapper in:

SafeWord_V2/wrappers/__init__.py
Register model in main.py if needed.
Ensure wrapper emits required AP keys:
hazard_dist
velocity
goal_dist
⚠️ Design Rule

Task configs may include metadata for documentation only.

👉 Runtime code must NOT consume task metadata for execution decisions.

All execution-critical configuration must come from:

settings layer
env-config layer
CLI overrides