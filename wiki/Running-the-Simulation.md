# Running the Simulation

## Prerequisites

| Dependency | Version |
|---|---|
| Python | 3.10+ |
| ROS 2 | Humble or later |
| PyBullet | latest |
| PyTorch | 2.x (CPU build) |
| Stable-Baselines3 | 2.x |
| Gymnasium | 1.x |

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

## Step 1 — Train the models (required before first launch)

The launcher checks for trained models and blocks the sim if they are missing. Open the launcher first:

```bash
python run_sim.py
```

Click each training card and train before launching:

| Card button | Opens | Time |
|---|---|---|
| **Calibrate** (Sub-1) | Live calibration check — no training needed | Instant |
| **Train PPO** (Sub-2) | PPO hover training — 3D room with quadcopter drone | ~8 min first run |
| **Solve MDP** (Sub-4) | Battery-safety value iteration | < 1 sec |
| **Train Navigation** (Sub-4 tab) | PPO obstacle navigation | ~3 min first run |

**Warm-start:** Every subsequent training run automatically loads the existing model and fine-tunes it — shorter LR, continued curriculum. You do not need to train from scratch again. Each repeated run improves the model.

After training completes the status bar shows:
```
✓ Model trained — predicted hover efficiency ~73%  (fine-tuned from previous model)
```

To validate the model before launching, click **Evaluate Model** in the Sub-2 or Sub-4 tab — it runs 5 test episodes and draws the best path as a green trail in the 3D view.

---

## Step 2 — Launch the simulation

### Full integrated simulation (GUI)

```bash
# If you activated the venv (recommended):
python run_sim.py

# Or use python3 directly (no venv activation needed):
python3 run_sim.py
```

> **Note:** `python` may not be found on some systems — use `python3`. If your venv is active (`source venv/bin/activate`), both `python` and `python3` work.

A PyBullet window opens with the drone, user, and environment. A HUD overlay shows battery, nav override, umbrella state, tracking confidence, hover error, and altitude.

**Options:**

```bash
python run_sim.py --duration 60            # run for 60 s
python run_sim.py --no-gui                 # headless (no PyBullet window)
python run_sim.py --scenario forest        # forest path with tree avoidance
python run_sim.py --scenario buildings     # urban block with obstacles
python run_sim.py --flight pid             # force PID controller (ignore PPO)
python run_sim.py --demo-low-battery       # trigger RTH early for testing
```

### Via ROS 2 launch

```bash
# From ros2_ws with workspace sourced
ros2 launch skyshade skyshade_sim.launch.py
```

### Individual nodes (development)

```bash
ros2 run skyshade perception_node
ros2 run skyshade flight_node
ros2 run skyshade env_decision_node
ros2 run skyshade nav_safety_node
```

---

## What to expect

- The blue quadcopter hovers above the red sphere (user) at 2.5 m altitude
- The user walks a figure-8 path; the drone follows using the PPO policy
- The telemetry dashboard shows live AI evidence: marker tracking, avoidance force, battery, umbrella decision
- Battery drains at 0.5% per second; the Sub-4 MDP triggers RTH when battery falls below the safe threshold
- The umbrella disc turns green on `DEPLOY`, grey on `STOW`
- The terminal logs a row every 5 seconds showing: `Time  Bat  Nav  Flight  Umbrella  Conf  ErrXY  Z`
- **After the simulation ends**, the terminal prints a full **Efficiency Report** with grades for each subsystem — see [Validation and Results](Validation-and-Results) for how to read it

### Weather cycle and visuals

Weather cycles automatically through four phases over ~60 s:

| Phase | Lux | Rain | Wind | Visual effects in PyBullet |
|---|---|---|---|---|
| **Clear ☀** | High | 0 | Low | No clouds, no rain |
| **Cloudy ⛅** | Medium | Trace | Medium | 7 grey clouds across the horizon |
| **Rainy 🌧** | Low | 0.3–0.7 | Moderate | 80–250 animated rain drops across 14×14 m, 9 dark storm clouds, wind streaks |
| **Storm ⛈** | Very low | > 0.7 | High | Dense rain, ground-level mist wisps near the floor, strong wind streamers |

**Wind streamers** appear whenever wind ≥ 1.5 m/s — horizontal grey-blue streaks showing wind direction and speed. They slowly rotate so the wind direction changes over time.

The Sub-3 SVM reads the weather sensors every second and decides `DEPLOY`/`STOW` for the umbrella canopy.

### Via ROS 2 launch (advanced)

The ROS 2 launch approach starts separate processes for each subsystem node. It requires the subsystem packages to be on the Python path:

```bash
cd /home/slal/ros2_ws
colcon build
source install/setup.bash
export PYTHONPATH=$PYTHONPATH:/home/slal/ros2_ws/src/SkyShade
ros2 launch skyshade skyshade_sim.launch.py
```

> For most users, `python3 run_sim.py` is simpler and fully equivalent.

---

## CLI training (without the GUI hub)

```bash
# Sub-2 PPO hover
python sub2_flight/train_ppo.py --steps 500000 --output models/ppo_flight_v1

# Sub-4 battery MDP
python sub4_nav/solve_mdp.py --gamma 0.95 --output models/policy_table_v1.npy
```

---

## Troubleshooting

**PyBullet window does not open**

```bash
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1
python run_sim.py
```

**"Models not trained" dialog on launch**

Open the Training Grounds hub and complete the training steps above. The dialog offers to launch with fallback controllers if you want to proceed without training (flight will use PID).

**`stable_baselines3` or `gymnasium` not found**

```bash
pip install stable-baselines3 gymnasium torch --index-url https://download.pytorch.org/whl/cpu
```

**ROS 2 nodes not finding each other**

```bash
source ros2_ws/install/setup.bash
```

**`pybullet` not found**

```bash
pip install pybullet
```
