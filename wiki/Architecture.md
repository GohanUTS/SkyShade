# Architecture

## System diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    SkyShade Launcher                        │
│  Training Grounds hub  ──►  train Sub-2 PPO + Sub-4 MDP    │
│  before launching the integrated simulation                 │
└───────────────────────────┬─────────────────────────────────┘
                            │ launch
                            ▼
         Simulated sensors (camera, IMU, env. sensors)
                    │
     ───────────────┼──────────────────────────────
     │              │              │               │
  [SUB-1]       [SUB-2]       [SUB-3]         [SUB-4]
 Perception     Flight        Env.            Nav +
 HSV tracker    PPO policy    Decision        Safety
 + distance     + PID         SVM + PCA       MDP +
   estimator    fallback                      PPO nav
     │              │              │               │
     └──────┬────── ┘              │               │
            │                     │               │
     PyBullet Flight          Umbrella        Override
     Controller               Servo           / RTH
```

---

## Subsystem responsibilities

| Subsystem | Algorithm | Runtime model | Trains in |
|---|---|---|---|
| Sub-1 Perception | HSV + EMA | No model (deterministic) | Calibration room (live check) |
| Sub-2 Flight | PPO → PID fallback | `models/ppo_flight_v1.zip` | 3D hover arena (~8 min) |
| Sub-3 Environment | SVM | `models/svm_v1.pkl` | Pre-trained |
| Sub-4 Battery Safety | MDP value iteration | `models/policy_table_v1.npy` | <1 sec |
| Sub-4 Nav (obstacle) | PPO | `models/ppo_nav_v1.zip` | 3D obstacle room (~3 min) |

---

## ROS 2 topics

| Topic | Message type | Published by | Consumed by |
|---|---|---|---|
| `/skyshade/user_position` | `geometry_msgs/Point` | Sub-1 | Sub-2 |
| `/skyshade/tracking_confidence` | `std_msgs/Float32` | Sub-1 | Sub-4 |
| `/skyshade/flight_cmd` | `geometry_msgs/Twist` | Sub-2 | Flight Controller |
| `/skyshade/umbrella_cmd` | `std_msgs/String` | Sub-3 | Umbrella Servo |
| `/skyshade/nav_override` | `std_msgs/String` | Sub-4 | Sub-2 |
| `/skyshade/battery_level` | `std_msgs/Float32` | Simulator | Sub-4 |

---

## Control flow

1. Every frame (30 Hz), **Sub-1** renders a virtual downward camera image from PyBullet, runs HSV segmentation, and publishes the user's relative position and tracking confidence.
2. **Sub-2** reads the user position, checks the current `nav_override` from Sub-4, builds a 7-D observation `[dx, dy, dz, vx, vy, vz, wind_norm]`, passes it through the trained PPO network to get a velocity setpoint, then converts via inner P-controller to a thrust force applied in PyBullet. Falls back to PID if no PPO model is found.
3. Every second (1 Hz), **Sub-3** reads simulated weather sensors and publishes `DEPLOY` or `STOW` to the umbrella servo.
4. Every 200 ms (5 Hz), **Sub-4 battery** reads battery level and distance from home, looks up the MDP policy table, and publishes the resulting override (`CONTINUE`, `RTH`, or `LAND_NOW`).

---

## Override priority

Sub-4 has hard override authority. When it publishes `RTH` or `LAND_NOW`, Sub-2 replaces its PPO/PID target immediately regardless of where the user is. `CONTINUE` restores normal user-following behaviour.

---

## Training Grounds hub

Before the sim can be launched, Sub-2 PPO and Sub-4 MDP must be trained. The launcher opens the **Training Grounds** hub (`TrainingGroundsHub` class in `run_sim.py`) which provides three independent training environments:

| Tab | Environment | What it shows |
|---|---|---|
| Sub-1 Perception | PyBullet calibration room | Live camera feed, tracker bounding box, confidence chart |
| Sub-2 Flight PPO | 3D hover arena (auto-rotating) | Quadcopter drone, wind arrows, reward curve with ETA |
| Sub-4 Nav Safety | 3D obstacle room (auto-rotating) | Red cylinders, 8 lidar rays, live drone trail, reward curve |

Each tab has a **Evaluate Model** button that runs 5 deterministic test episodes and draws the best episode's path as a green trail in the 3D view.

---

## Controller priority chain (Sub-2)

```
RuntimeFlightController.compute_force():
  1. PPO policy     (if models/ppo_flight_v1.zip exists)
  2. Q-table        (legacy fallback, models/qtable_v1.npy)
  3. PID            (always available — no model file needed)
```

---

## File layout

```
sub1_perception/
├── tracker.py                  HSV segmentation + EMA
├── distance_estimator.py       Pinhole distance model
├── training.py                 Camera calibration helpers
└── training_env.py             PyBullet calibration room

sub2_flight/
├── env/
│   ├── hover_env.py            Base PyBullet physics env
│   └── ppo_hover_env.py        Gymnasium PPO training env (7D obs, 3D action)
├── policy.py                   PPOFlightPolicy / FlightPolicy / PIDFlightPolicy
├── runtime_control.py          RuntimeFlightController (PPO→Q→PID chain)
├── training_worker.py          Background PPO training thread (warm-start)
├── eval_worker.py              5-episode evaluation worker
└── train_ppo.py                CLI training script

sub3_env/
├── classifier.py               SVM inference wrapper
└── train_svm.py                Training script

sub4_nav/
├── mdp.py                      MDP state/action/reward definitions
├── solve_mdp.py                CLI value-iteration solver
├── training_worker.py          Background MDP solver thread
├── policy_table.py             Runtime battery-safety lookup
├── obstacle_env.py             10×8m room + 8 lidar rays + PPO gymnasium env
├── nav_training_worker.py      Background PPO nav training thread (warm-start)
└── eval_worker.py              5-episode nav evaluation worker

ros2_ws/src/skyshade/skyshade/
├── perception_node.py          Sub-1 ROS 2 node
├── flight_node.py              Sub-2 ROS 2 node
├── env_decision_node.py        Sub-3 ROS 2 node
└── nav_safety_node.py          Sub-4 ROS 2 node

run_sim.py                      Launcher + integrated simulation + Training Grounds hub
```
