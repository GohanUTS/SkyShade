# SkyShade: Autonomous Drone Umbrella (Simulation)

> An AI-driven quadcopter simulation that perceives, tracks, and shades a moving user, autonomously deploying its umbrella canopy in response to weather conditions.

**Course:** AI for Robotics — Engineering Project
**Institution:** University of Technology Sydney (UTS)
**Team Leader:** Gohan Idrisoglu (24736668)
**Crew:** Dinesh Saravanan · Aaron · Saaranj

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Running the Simulation](#running-the-simulation)
- [Subsystems](#subsystems)
  - [Sub-1: Perception](#sub-1-perception)
  - [Sub-2: Flight Control](#sub-2-flight-control)
  - [Sub-3: Environmental Decision](#sub-3-environmental-decision)
  - [Sub-4: Navigation and Safety](#sub-4-navigation-and-safety)
- [Training the Models](#training-the-models)
- [Validation](#validation)
- [Troubleshooting](#troubleshooting)

---

## Overview

SkyShade is a fully simulated autonomous drone system. A quadcopter hovers above a moving user, tracks their position using a virtual downward-facing camera, and decides in real time whether to deploy or stow an umbrella canopy based on simulated weather sensor readings.

The simulation runs entirely in **PyBullet** for physics and flight, with **ROS2** nodes handling communication between the four AI subsystems.

```
Simulated sensors (camera, IMU, env. sensors)
            |
     -------+--------+----------+----------
     |               |          |          |
  [SUB-1]        [SUB-2]    [SUB-3]    [SUB-4]
 Perception      Flight     Env.       Nav +
 CV + HSV        PID + PPO  Decision   Safety
                            SVM+PCA    MDP + SAC
     |               |          |          |
     +---------------+          |          |
             |                  |          |
      Flight Controller    Umbrella    Override
       (PyBullet)           Servo       / RTH
```

**Key design rule:** Sub-4 safety wins. Return-home and land-now override normal flight at all times.

---

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.8+ | |
| ROS2 | Humble or later | [Install guide](https://docs.ros.org/en/humble/Installation.html) |
| PyBullet | latest | Physics simulator |
| Git LFS | latest | For pulling model checkpoints |

> All simulation runs on CPU. No GPU required.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-team/skyshade.git
cd skyshade
git lfs pull
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy
opencv-python
scikit-learn
scipy
matplotlib
pybullet
stable-baselines3
tensorboard
rclpy
```

### 4. Build the ROS2 workspace

```bash
cd ros2_ws
colcon build
source install/setup.bash
```

---

## Project Structure

```
skyshade/
├── ros2_ws/                    # ROS2 workspace
│   └── src/
│       └── skyshade/
│           ├── launch/
│           │   └── skyshade_sim.launch.py
│           └── skyshade/
│               ├── perception_node.py
│               ├── flight_node.py
│               ├── env_decision_node.py
│               └── nav_safety_node.py
│
├── sub1_perception/
│   ├── tracker.py              # HSV marker tracking + EMA smoothing
│   ├── distance_estimator.py   # Pinhole focal-length distance estimation
│   └── test_perception.py
│
├── sub2_flight/
│   ├── env/
│   │   └── hover_env.py        # PyBullet hover environment (staged curriculum)
│   ├── train_ppo.py            # PPO training script
│   ├── runtime_control.py      # PID controller + PPO nudge at runtime
│   └── test_flight.py
│
├── sub3_env/
│   ├── train_svm.py            # SVM training script
│   ├── feature_engineering.py  # 9-D feature vector builder
│   ├── classifier.py           # Runtime SVM wrapper with confirmation window
│   └── test_env_decision.py
│
├── sub4_nav/
│   ├── mdp.py                  # MDP state/action/reward definition (battery safety)
│   ├── solve_mdp.py            # Offline value iteration solver
│   ├── policy_table.py         # Runtime policy table lookup
│   ├── train_sac.py            # SAC obstacle-navigation training
│   ├── nav_policy.py           # Runtime SAC policy (lidar-based avoidance)
│   └── test_nav_safety.py
│
├── data/
│   ├── env_sensor_log.csv      # Labelled lux/rain/wind samples
│   └── mdp_scenarios/          # 50 scripted battery/obstacle episodes
│
├── models/                     # Trained model artefacts (Git LFS)
│   ├── ppo_flight_v1.zip
│   ├── svm_v1.pkl
│   ├── policy_table_v1.npy
│   └── sac_nav_v1.zip
│
├── requirements.txt
└── README.md
```

---

## Running the Simulation

### Standalone simulator

```bash
python3 run_sim.py
python3 run_sim.py --scenario forest
```

The default `park` scenario shows the drone following the user in an open scene. The `forest` scenario sends the user along a wooded trail while the drone adds local tree-avoidance forces.

### Full pipeline (all four nodes at once)

```bash
# From the ros2_ws directory, with the workspace sourced
ros2 launch skyshade skyshade_sim.launch.py
```

This starts the PyBullet environment and all four subsystem nodes. A visualisation window opens showing the drone, the simulated user, and the umbrella state.

### Run nodes individually (for development)

Open a separate terminal for each node:

```bash
# Sub-1: Perception
ros2 run skyshade perception_node

# Sub-2: Flight control
ros2 run skyshade flight_node

# Sub-3: Environmental decision
ros2 run skyshade env_decision_node

# Sub-4: Navigation and safety
ros2 run skyshade nav_safety_node
```

### ROS2 topics reference

| Topic | Type | Published by | Consumed by |
|---|---|---|---|
| `/skyshade/user_position` | `geometry_msgs/Point` | Sub-1 | Sub-2 |
| `/skyshade/tracking_confidence` | `std_msgs/Float32` | Sub-1 | Sub-4 |
| `/skyshade/flight_cmd` | `geometry_msgs/Twist` | Sub-2 | Flight Controller |
| `/skyshade/umbrella_cmd` | `std_msgs/String` | Sub-3 | Umbrella Servo |
| `/skyshade/nav_override` | `std_msgs/String` | Sub-4 | Sub-2 |
| `/skyshade/battery_level` | `std_msgs/Float32` | Simulator | Sub-4 |

### Node rates

| Node | Rate | Why |
|---|---|---|
| `perception_node` | 30 Hz | Camera tracking must update quickly |
| `flight_node` | 20 Hz | Flight control needs regular correction |
| `env_decision_node` | 1 Hz | Weather changes slowly vs. drone motion |
| `nav_safety_node` | 5 Hz | Fast enough to react to battery / confidence changes |

---

## Subsystems

### Sub-1: Perception

Identifies the user in the simulated camera feed and outputs a relative position estimate in the drone's body frame.

**How it works:**

1. Receives an RGB frame from the PyBullet virtual camera
2. Applies HSV colour segmentation to isolate the user's red cap marker, using **two red hue ranges** because red wraps around the hue scale
3. Finds the largest contour and takes its centre as the marker location in image space
4. Estimates distance using a pinhole focal-length model based on the known marker width (smaller marker = further away)
5. Smooths the (Δx, Δy, Δz) output with EMA to suppress per-frame jitter, and holds the last known position through short occlusions
6. Publishes position and tracking confidence to `/skyshade/user_position`

**Key parameters** (set in `sub1_perception/tracker.py`):

```python
EMA_ALPHA             = 0.3    # Smoothing factor; higher = less smoothing
CONFIDENCE_THRESH     = 0.7    # Minimum confidence to publish a position
MAX_OCCLUSION_FRAMES  = 20     # Hold last known position for this many frames
MARKER_WIDTH_M        = 0.1    # Known physical width of the user's marker (metres)
```

**Simulated camera spec:**

```python
CAMERA_FOV    = 60         # Degrees
CAMERA_RES    = (640, 480) # Pixels
CAMERA_FPS    = 30
```

---

### Sub-2: Flight Control

Stable PID underneath, learned motion on top. Once the drone knows where the user is, Sub-2 keeps it positioned overhead by combining a reliable **PID controller** with a **PPO** reinforcement-learning policy.

**How it works:**

- The **PID controller** computes the error between the drone's current position and the target, then applies correction forces. This is the reliability backbone.
- The **PPO policy** adds learned behaviour on top. Its output is **clamped so it can only nudge** the drone — it cannot overpower the PID, which keeps the system stable.
- The controller works in **continuous velocity setpoints** rather than discrete actions. An earlier discrete-action version caused the drone to buzz around the hover point; continuous setpoints made the motion smooth.
- A **repulsive obstacle force** pushes the drone away from trees, buildings, and other objects when it gets too close.
- Sub-2 always listens to Sub-4: an `RTH` or `LAND_NOW` override immediately changes the flight target before PID runs.

**Runtime control** (`sub2_flight/runtime_control.py`):

```python
pid_target  = [user_pos.x, user_pos.y, TARGET_ALTITUDE]
nudge       = ppo_policy.act(obs)          # bounded velocity nudge
force       = pid.compute_force(drone_pos, drone_vel, pid_target, dt)
force[:2]  += clamp(nudge, -NUDGE_MAX, NUDGE_MAX)
force[2]   += HOVER_FORCE_N                # gravity compensation
force      += repulsive_obstacle_force(drone_pos, obstacles)
```

Sub-4 can override the target before PID runs:

| `nav_override` | PID target |
|---|---|
| `CONTINUE` | User XY + target altitude |
| `RTH` | Origin (0, 0) + target altitude |
| `LAND_NOW` | Current XY + 0.3 m (descend) |

**PPO training** (`sub2_flight/train_ppo.py`) runs through three curriculum stages:

```python
# 1. Stationary user, no wind
# 2. Stationary user, gusty wind
# 3. Walking user, gusty wind
```

---

### Sub-3: Environmental Decision

A binary SVM classifier that reads simulated weather sensors and decides whether the umbrella should be deployed or stowed.

**Feature vector (9-D):**

```python
features = [
    lux,            # Current light level
    rain_raw,       # Raw rain sensor reading
    wind_speed,     # Wind speed estimate
    lux_delta,      # Change in lux since last reading
    rain_delta,     # Change in rain since last reading
    wind_delta,     # Change in wind since last reading
    prev_action_t1, # Decision at t-1 (gives the model short-term memory)
    prev_action_t2, # Decision at t-2
    prev_action_t3, # Decision at t-3
]
```

The previous-decision features plus a runtime **confirmation window** stop the umbrella flickering open/closed when the weather sits on the decision boundary — the decision must stay consistent for several readings before the umbrella actually changes state.

**Classifier config** (set in `sub3_env/train_svm.py`):

```python
SVM_KERNEL        = 'rbf'
SVM_C             = 1.0
SVM_GAMMA         = 'scale'
PCA_COMPONENTS    = 3       # Compressed for visualisation only; SVM trains on full 9-D
CV_FOLDS          = 10
TARGET_ACC        = 0.90
HYSTERESIS_WINDOW = 3       # Frames before flipping deploy/stow decision
```

---

### Sub-4: Navigation and Safety

Explainable safety plus learned obstacle navigation. Sub-4 sits above normal flight as an override layer and is split into two parts that solve different problems.

**Part 1 — Battery safety (MDP).** An MDP solved offline with value iteration. It is used because it is *interpretable*: you can read the table and understand exactly why the drone continues, returns home, or lands.

```python
battery_buckets  = [HIGH, MEDIUM, LOW, CRITICAL]   # 4 levels
distance_buckets = [NEAR, MID, FAR]                # Distance to home
# Total states: 4 x 3 = 12
```

**Actions:**

```
CONTINUE    — pass control to Sub-2
RTH         — override Sub-2; navigate home
LAND_NOW    — override Sub-2; land immediately
```

**Reward structure:**

```python
REWARD_SAFE_COMPLETION = +100
REWARD_PER_STEP        =   -1
REWARD_CRASH           = -100
REWARD_BATTERY_EMPTY   = -100
```

**Solver config** (set in `sub4_nav/solve_mdp.py`):

```python
GAMMA             = 0.95
CONVERGENCE_DELTA = 1e-6   # Stop when max Bellman update is below this
```

**Part 2 — Obstacle navigation (SAC).** A Soft Actor-Critic policy handles obstacle avoidance, which is too open-ended to write by hand. The agent reads **lidar rays** — a shorter ray means an obstacle is close in that direction — and learns to steer around obstacles toward the goal instead of crashing.

**SAC config** (set in `sub4_nav/train_sac.py`):

```python
GAMMA          = 0.99
TAU            = 0.005   # Target network soft-update rate
LR             = 3e-4
BUFFER_SIZE    = 1_000_000
BATCH_SIZE     = 256
LIDAR_RAYS     = 16      # Range readings fed to the policy
```

---

## Training the Models

Train each model independently before running the full simulation. Pre-trained checkpoints are available via `git lfs pull`.

### Sub-2: PPO flight agent

```bash
python sub2_flight/train_ppo.py \
  --output models/ppo_flight_v1.zip

# Monitor training in TensorBoard
tensorboard --logdir runs/sub2
```

Training runs through three curriculum stages automatically:

1. Stationary user, no wind
2. Stationary user, gusty wind
3. Walking user, gusty wind

> We look for an upward reward trend that climbs and then settles — that means the policy found a smoother control strategy. The continuous-velocity action space replaced an earlier discrete one that caused buzzing at the hover point.

### Sub-3: SVM classifier

```bash
python sub3_env/train_svm.py \
  --data data/env_sensor_log.csv \
  --output models/svm_v1.pkl

# Outputs: confusion matrix, CV accuracy, PCA 3-D plot
```

### Sub-4: MDP value iteration (battery safety)

```bash
python sub4_nav/solve_mdp.py \
  --gamma 0.95 \
  --output models/policy_table_v1.npy

# Outputs: policy table, convergence curve, monotone safety check
```

### Sub-4: SAC obstacle navigation

```bash
python sub4_nav/train_sac.py \
  --output models/sac_nav_v1.zip

# Monitor training in TensorBoard
tensorboard --logdir runs/sub4
```

---

## Validation

Run the test suite for each subsystem individually:

```bash
# Sub-1: Perception accuracy on held-out frames
python sub1_perception/test_perception.py

# Sub-2: Evaluate the PPO+PID controller across 10 episodes per scenario
python sub2_flight/test_flight.py

# Sub-3: Run classifier on 5-minute synthetic weather trajectory
python sub3_env/test_env_decision.py

# Sub-4: Run all 50 scripted battery/obstacle scenarios
python sub4_nav/test_nav_safety.py
```

### Expected results (passing bar)

| Subsystem | Metric | Target |
|---|---|---|
| Sub-1 | Tracking continuity | > 70% of frames above confidence threshold |
| Sub-1 | Position MAE | < 0.15 m in simulation |
| Sub-2 | Mean episode reward | > +150 across 10 eval episodes per scenario |
| Sub-2 | Hover error | Within 0.5 m in 9 of 10 episodes |
| Sub-3 | 10-fold CV accuracy | >= 90% |
| Sub-3 | False deploy rate | Minimised per confusion matrix |
| Sub-4 (MDP) | Scripted scenarios | Pass all 50 |
| Sub-4 (MDP) | RTH trigger | Fires within one decision tick of low-battery injection |
| Sub-4 (SAC) | Goal reach rate | Reward trends upward; reaches goal while avoiding obstacles |

---

## Troubleshooting

**PyBullet window does not open**
Make sure you are not running in a headless environment. If you are on a remote server, use a virtual display:
```bash
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1
```

**ROS2 nodes not finding each other**
Make sure all terminals have the workspace sourced:
```bash
source ros2_ws/install/setup.bash
```

**PPO reward not improving**
Confirm the curriculum is advancing through its three stages and the action space is continuous velocity setpoints — discrete actions cause buzzing around the hover point and a flat reward. Check TensorBoard for a rising trend; if it stays flat, lower the learning rate or extend the timesteps.

**Drone buzzes / oscillates around the hover point**
The PPO nudge is likely too large or unbounded. Confirm the nudge is clamped (`NUDGE_MAX`) so it cannot overpower the PID, and verify the controller uses continuous velocity setpoints rather than discrete actions.

**SVM accuracy below 90%**
Check the class balance in `env_sensor_log.csv`. If heavily skewed, use `class_weight='balanced'` in the SVM config. Also confirm the feature vector is being standardised with `StandardScaler` before fitting.

**Umbrella flickers open/closed near the weather boundary**
Increase `HYSTERESIS_WINDOW` so the deploy/stow decision must stay consistent for more readings before the umbrella changes state.

**Sub-4 always outputs LAND_NOW**
The policy table may not have converged. Re-run `solve_mdp.py` and confirm the convergence delta reaches below `1e-6`. Check that `REWARD_SAFE_COMPLETION` is set to `+100` (a wrong sign here makes the policy treat finishing as bad and become too eager to land) and not overridden by the per-step cost.

**SAC agent keeps crashing into obstacles**
Confirm the lidar rays are being fed to the policy and normalised. Extend training timesteps and check that the reward penalises collisions; an under-trained SAC policy will take bad paths until the reward curve climbs.

---

*SkyShade v0.1 — AI for Robotics, UTS, May 2026*