# Architecture

## System diagram

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
2. **Sub-2** reads the user position, checks the current `nav_override` from Sub-4, computes a PID force, and applies it to the drone in PyBullet.
3. Every second (1 Hz), **Sub-3** reads simulated weather sensors (lux, rain, wind) and publishes `DEPLOY` or `STOW` to the umbrella servo.
4. Every 200 ms (5 Hz), **Sub-4** reads battery level and distance from home, looks up the MDP policy, and publishes the resulting override (`CONTINUE`, `RTH`, or `LAND_NOW`).

---

## Override priority

Sub-4 has hard override authority. When it publishes `RTH` or `LAND_NOW`, Sub-2 replaces its PID target immediately regardless of where the user is. `CONTINUE` restores normal user-following behaviour.

In the integrated simulation, `RTH` is additionally suppressed while battery > 50% to allow normal following during healthy flight — this guard is in `run_sim.py`, not in the MDP itself.

---

## File layout

```
ros2_ws/src/skyshade/skyshade/
├── perception_node.py     # Sub-1 ROS 2 node
├── flight_node.py         # Sub-2 ROS 2 node
├── env_decision_node.py   # Sub-3 ROS 2 node
└── nav_safety_node.py     # Sub-4 ROS 2 node
```

Pure-Python subsystem logic lives in `sub1_perception/`, `sub2_flight/`, `sub3_env/`, and `sub4_nav/` — the ROS 2 nodes are thin wrappers around those modules.
