# README — Running SkyShade

Quick reference for launching the simulation, the validation tests, and
retraining the learning models. All commands run from the project root:

```bash
cd /home/gohan/ros2_ws/src/SkyShade
```

The project uses the bundled virtualenv. Either activate it once:

```bash
source venv/bin/activate            # then use `python ...`
```

or call it directly per command (used below): `venv/bin/python ...`.

---

## 1. Run the simulation

```bash
venv/bin/python run_sim.py                         # default: patrol + PID
```

A PyBullet window opens (the city + drone + user) plus the **SkyShade Dashboard**.

Flags:

| Flag | Values | What it does |
|---|---|---|
| `--scenario` | `patrol` (default), `downpour` | `downpour` forces sustained heavy rain so the umbrella deploys. |
| `--flight` | `pid` (default), `q` | `pid` = PID controller (smooth, production). `q` = the **learned Q‑policy**. |
| `--duration` | seconds (default 120) | How long the run lasts. |
| `--demo-low-battery` | – | Starts near empty so Sub‑4 fires RTH / LAND_NOW quickly. |
| `--no-gui` | – | Headless (no window) — for quick smoke tests. |

Examples:

```bash
venv/bin/python run_sim.py --scenario downpour          # rain + umbrella demo
venv/bin/python run_sim.py --flight q                   # fly with the learned policy
venv/bin/python run_sim.py --demo-low-battery           # safety RTH/LAND demo
venv/bin/python run_sim.py --no-gui --duration 10       # fast headless check
```

### Dashboard scenario buttons (click while running)
- **Random Weather** – normal random dry/rain cycling.
- **Force Rain** – it starts raining → watch the umbrella **deploy**.
- **Force Clear** – rain stops → watch the umbrella **stow**.

The dashboard also shows the live **Sub‑4 safety reward** (the MDP reward the
nav policy is optimising) and, in `--flight q`, the per‑tick flight reward.

---

## 2. Run the validation tests (one per subsystem)

```bash
export PYTHONPATH=$PWD
venv/bin/python sub1_perception/test_perception.py     # HSV tracker
venv/bin/python sub2_flight/test_flight.py             # PID + learned Q-policy
venv/bin/python sub3_env/test_env_decision.py          # SVM umbrella classifier
venv/bin/python sub4_nav/test_nav_safety.py            # MDP safety (50 scenarios)
```

You can also click the **Validation Tests** buttons inside the dashboard.

---

## 3. Rebuild / retrain the learning models

The trained artifacts live in `models/`. Regenerate them with:

```bash
export PYTHONPATH=$PWD
venv/bin/python sub4_nav/solve_mdp.py                  # -> models/policy_table_v1.npy (+ convergence_curve.png)
venv/bin/python sub3_env/train_svm.py                  # -> models/svm_v1.pkl (+ confusion_matrix.png, pca_3d.png)
venv/bin/python sub2_flight/train_qlearning.py --episodes 50000   # -> models/qtable_v1.npy  (~12 min)
```

The Q‑learning trainer prints the reward curve every 500 episodes. After it
finishes, `sub2_flight/test_flight.py` validates the new table.

---

## 4. ROS2 launch (optional, full node graph)

```bash
# from the ros2_ws directory, with the workspace sourced
ros2 launch skyshade skyshade_sim.launch.py
```

---

## Notes on the controllers
- **PID** is the production flight controller (smooth, handles wind well).
- **Q‑learning** is the learned policy: it reliably flies in and hovers/follows
  in wind. In perfectly still air a coarse discrete controller is more
  sensitive, which is why PID remains the default. Wind level is part of the
  Q‑agent's state, so it is trained to react to gusts.
