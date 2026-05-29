# Running the Simulation

## Prerequisites

| Dependency | Version |
|---|---|
| Python | 3.8+ |
| ROS 2 | Humble or later |
| PyBullet | latest |

Install Python dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Build the ROS 2 workspace:

```bash
cd ros2_ws
colcon build
source install/setup.bash
cd ..
```

---

## Full integrated simulation

```bash
python run_sim.py
```

Runs for 120 s by default with a PyBullet GUI window. A HUD overlay shows time, battery, nav override, umbrella state, tracking confidence, hover error, and drone altitude.

**Options:**

```bash
python run_sim.py --duration 60     # run for 60 s
python run_sim.py --no-gui          # headless (no window)
```

---

## Via ROS 2 launch

```bash
# From ros2_ws with workspace sourced
ros2 launch skyshade skyshade_sim.launch.py
```

---

## Individual nodes (development)

Open a separate terminal for each:

```bash
ros2 run skyshade perception_node
ros2 run skyshade flight_node
ros2 run skyshade env_decision_node
ros2 run skyshade nav_safety_node
```

---

## What to expect

- The blue cube (drone) hovers above the red sphere (user)
- The user walks a figure-8 path; the drone follows
- The umbrella disc above the drone turns green when `DEPLOY`ed, grey when `STOW`ed
- Weather cycles clear → rainy → clear every 60 s
- Battery drains at 0.5% per second; RTH triggers when it falls below 50%
- Console logs every 10 s show battery, nav override, umbrella state, confidence, and hover error

---

## Troubleshooting

**PyBullet window does not open**

```bash
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1
python run_sim.py
```

**ROS 2 nodes not finding each other**

```bash
source ros2_ws/install/setup.bash
```

**`pybullet` not found**

```bash
pip install pybullet
```
