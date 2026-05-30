# SkyShade# SkyShade: Autonomous Drone Umbrella (Simulation)

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
 CV + HSV        PID        Decision   Safety
                            SVM+PCA    MDP
     |               |          |          |
     +---------------+          |          |
             |                  |          |
      Flight Controller    Umbrella    Override
       (PyBullet)           Servo       / RTH
```

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
│   │   └── hover_env.py        # PyBullet hover environment
│   ├── train_qlearning.py      # Q-learning training script
│   ├── policy.py               # Q-table lookup at runtime
│   └── test_flight.py
│
├── sub3_env/
│   ├── train_svm.py            # SVM training script
│   ├── feature_engineering.py  # 9-D feature vector builder
│   ├── classifier.py           # Runtime SVM wrapper with hysteresis
│   └── test_env_decision.py
│
├── sub4_nav/
│   ├── mdp.py                  # MDP state/action/reward definition
│   ├── solve_mdp.py            # Offline value iteration solver
│   ├── policy_table.py         # Runtime policy table lookup
│   └── test_nav_safety.py
│
├── data/
│   ├── env_sensor_log.csv      # Labelled lux/rain/wind samples
│   └── mdp_scenarios/          # 50 scripted battery/obstacle episodes
│
├── models/                     # Trained model artefacts (Git LFS)
│   ├── qtable_v1.npy
│   ├── svm_v1.pkl
│   └── policy_table_v1.npy
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

---

## Subsystems

### Sub-1: Perception

Identifies the user in the simulated camera feed and outputs a relative position estimate in the drone's body frame.

**How it works:**

1. Receives an RGB frame from the PyBullet virtual camera
2. Applies HSV colour segmentation to isolate the user's coloured marker
3. Estimates distance using a pinhole focal-length model based on the known marker width
4. Smooths the (Δx, Δy, Δz) output with EMA to suppress per-frame jitter
5. Publishes position and tracking confidence to `/skyshade/user_position`

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

A PID flight controller that tracks the user position published by Sub-1 and applies a 3-D force vector to the drone at 30 Hz. A constant hover thrust offset compensates for gravity every physics step.

**Runtime controller** (`sub2_flight/policy.py`):

```python
pid_target = [user_pos.x, user_pos.y, TARGET_ALTITUDE]
force      = pid.compute_force(drone_pos, drone_vel, pid_target, dt)
force[2]  += HOVER_FORCE_N   # gravity compensation
```

Sub-4 can override the target before PID runs:

| `nav_override` | PID target |
|---|---|
| `CONTINUE` | User XY + target altitude |
| `RTH` | Origin (0, 0) + target altitude |
| `LAND_NOW` | Current XY + 0.3 m (descend) |

**Q-learning (archived — used for training, not runtime)**

A tabular Q-agent (225 states, 7 actions) was trained for 50 k episodes through three curriculum stages. Discretisation caused chattering near the hover target at runtime, so the PID controller replaced it for the integrated simulation. The trained Q-table is kept in `models/qtable_v1.npy` for reference.

```python
# Archived hyperparameters (sub2_flight/train_qlearning.py)
ALPHA         = 0.1
GAMMA         = 0.95
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.9995
EPISODES      = 50_000
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
    prev_action_t1, # Decision at t-1 (hysteresis)
    prev_action_t2, # Decision at t-2
    prev_action_t3, # Decision at t-3
]
```

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

An MDP solved offline with value iteration. At runtime it reads global state and either lets Sub-2 fly freely or issues an override command.

**State space:**

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

---

## Training the Models

Train each model independently before running the full simulation. Pre-trained checkpoints are available via `git lfs pull`.

### Sub-2: Q-learning flight agent

```bash
python sub2_flight/train_qlearning.py \
  --episodes 50000 \
  --output models/qtable_v1.npy

# Monitor training in TensorBoard
tensorboard --logdir runs/sub2
```

Training runs through three curriculum stages automatically:

1. Stationary user, no wind
2. Stationary user, gusty wind
3. Walking user, gusty wind

### Sub-3: SVM classifier

```bash
python sub3_env/train_svm.py \
  --data data/env_sensor_log.csv \
  --output models/svm_v1.pkl

# Outputs: confusion matrix, CV accuracy, PCA 3-D plot
```

### Sub-4: MDP value iteration

```bash
python sub4_nav/solve_mdp.py \
  --gamma 0.95 \
  --output models/policy_table_v1.npy

# Outputs: policy table, convergence curve, monotone safety check
```

---

## Validation

Run the test suite for each subsystem individually:

```bash
# Sub-1: Perception accuracy on held-out frames
python sub1_perception/test_perception.py

# Sub-2: Evaluate Q-agent across 10 episodes per scenario
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
| Sub-4 | Scripted scenarios | Pass all 50 |
| Sub-4 | RTH trigger | Fires within one decision tick of low-battery injection |

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

**Q-learning not converging**  
Reduce the state space or increase training episodes. Check TensorBoard to confirm epsilon is decaying and reward is trending upward. If the reward is flat after 10,000 episodes, lower `ALPHA` to `0.05`.

**SVM accuracy below 90%**  
Check the class balance in `env_sensor_log.csv`. If heavily skewed, use `class_weight='balanced'` in the SVM config. Also confirm the feature vector is being standardised with `StandardScaler` before fitting.

**Sub-4 always outputs LAND_NOW**  
The policy table may not have converged. Re-run `solve_mdp.py` and confirm the convergence delta reaches below `1e-6`. Check that `REWARD_SAFE_COMPLETION` is set correctly and not overridden by the per-step cost.

---

*SkyShade v0.1 — AI for Robotics, UTS, May 2026*
