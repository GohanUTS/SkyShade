# SkyShade — Autonomous Drone Umbrella Simulation

> A multi-subsystem AI drone that perceives a moving person, hovers above them, and autonomously deploys a shade/rain canopy based on real-time weather. Everything runs in a single self-contained PyBullet simulation.

**Course:** AI for Robotics — Engineering Project  
**Institution:** University of Technology Sydney (UTS)  
**Team Leader:** Gohan Idrisoglu (24736668)  
**Crew:** Dinesh Saravanan · Aaron · Saaranj

---

## Quick Start

```bash
git clone https://github.com/GohanUTS/SkyShade.git
cd SkyShade
pip install -r requirements.txt
python3 run_sim.py            # opens the launcher GUI
```

All trained models are included — no retraining required to run.

---

## Scenarios

Eleven hand-crafted simulation worlds, each stressing a different part of the AI stack:

| Key | Name | Environment | Challenge |
|---|---|---|---|
| `park` | Park | Grass loop, trees, path tiles | Baseline hover and tracking |
| `forest` | Forest Trail | Dense canopy, winding dirt trail | Tree occlusion drops tracker confidence |
| `buildings` | Building District | City plaza, tall buildings, vehicles, pedestrians | Nav must avoid building near-misses |
| `trail` | Urban Trail | Bridge/underpass, crowd pedestrians | Crowd distractors + shadow confuse tracker |
| `night` | Night Park | Fountain, benches, flower beds, streetlamps | Low-light shadow HSV band engaged |
| `rooftop` | Rooftop | Parapet walls, HVAC, solar panels, water tower | Constant 3–8 m/s wind; confined edges |
| `beach` | Coastal Beach | Ocean, lifeguard tower, palm trees, volleyball | Steady lateral sea-breeze crosswind |
| `parking` | Parking Lot | Detailed cars, store building, lamp posts | Rectangular box obstacles, aisle navigation |
| `vineyard` | Vineyard | Vine trellis rows, grape clusters, farmhouse | Rhythmic post occlusion; tight nav corridors |
| `snow` | Snowy Field | Frozen pond, pine trees, snowmen, wooden fence | Omnidirectional gusting wind; cold overcast |
| `stadium` | Stadium | Oval track, tiered stands, floodlights, scoreboard | Circular target path tests the curved-walk curriculum |

---

## How to Run

### Launcher (GUI)

```bash
python3 run_sim.py
```

Opens the SkyShade Launcher. The right panel shows all ten scenarios in a scrollable list — pick one, set the duration, and click **Select → Launch**. After each run a **Post-Run Efficiency Report** shows per-subsystem grades (A–D) and a trend chart over the last 20 runs.

### Command line

```bash
python3 run_sim.py --scenario snow               # specific scene, GUI
python3 run_sim.py --scenario forest --no-gui    # headless (no 3-D window)
python3 run_sim.py --scenario buildings --duration 180
python3 run_sim.py --demo-low-battery --scenario trail   # stress-test battery
```

### Per-scenario performance report

```bash
python3 scenario_report.py           # coloured terminal breakdown
python3 scenario_report.py --json    # machine-readable JSON
```

---

## System Architecture

Four AI subsystems run together inside a single PyBullet physics simulation:

```
  Downward camera (PyBullet virtual)
           │
    ┌──────▼──────┐
    │   SUB-1     │  HSV colour tracker → user position (dx, dy, dz)
    │ Perception  │  + tracking confidence [0–1]
    └──────┬──────┘
           │
    ┌──────▼──────┐   ← wind force from weather controller
    │   SUB-2     │
    │   Flight    │  PPO hover policy — learns to stay above the user
    │  (PPO+PID)  │  despite wind, movement, and position noise
    └──────┬──────┘
           │ corrective force
    ┌──────▼──────┐
    │  PyBullet   │  Physics engine (drone body, obstacles, terrain)
    └──────┬──────┘
           │
   ┌───────┴──────────────┐
   │                      │
 ┌─▼──────────┐    ┌──────▼──────────────────┐
 │   SUB-3    │    │        SUB-4             │
 │  Weather   │    │  Navigation & Safety     │
 │  SVM+hyst  │    │  SAC obstacle-avoidance  │
 │            │    │  + Battery MDP (RTH/Land) │
 └────────────┘    └──────────────────────────┘
 DEPLOY / STOW      avoidance force + override cmd
```

---

## Subsystems

### Sub-1 — Perception

**File:** `sub1_perception/tracker.py`

A stateful HSV colour tracker that isolates the user's red marker in every camera frame and outputs a 3-D body-frame position offset plus a confidence score [0–1].

Key improvements made during development:

| Feature | Detail |
|---|---|
| Shadow fallback HSV band | Value 30–80 catches the marker under bridges and forest canopy |
| EMA confidence smoothing | α = 0.35 damps single-frame dips from partial occlusion |
| Extended occlusion hold | 35 frames before declaring lock lost (up from 20) |
| Reduced saturation area | 250 px² (down from 500) — distant/partial blobs score higher |
| Velocity-biased gimbal search | On lock-loss, 65% of the search sweeps toward the target's last velocity |

These improvements took Urban Trail tracker lock from **58% → 84%** over the course of development.

### Sub-2 — Flight Control

**Files:** `sub2_flight/`

A **PPO (Proximal Policy Optimisation)** policy that outputs 3-axis velocity setpoints; an inner proportional controller converts these to forces applied in PyBullet.

#### Curriculum

| Stage | Target | Wind | When |
|---|---|---|---|
| 1 | Stationary | None | Steps 0 – 500k |
| 2 | Stationary | Gusty (0–4.5 m/s) | Steps 500k – 1M |
| 3 | Moving | Gusty | Steps 1M+ |

#### Stage 3 walk diversity

Each episode randomly samples one of three movement patterns so the policy generalises across all ten scenario walk styles:

- **Random** (40%) — ±0.2 m random jumps every 20 steps
- **Linear** (35%) — constant-velocity directional walk, 0.08–0.18 m/s
- **Circular** (25%) — slow orbit at randomised radius and angular speed

#### Current model: 12.8M steps

> **Important for retraining:** observation noise must remain **zero** (`_OBS_NOISE = {1: 0.0, 2: 0.0, 3: 0.0}` in `sub2_flight/env/ppo_hover_env.py`). Any noise added during warm-start training causes the policy to overfit to the noisy environment and perform poorly on the clean real simulator. Keep warm-start LR = 3e-5, clip = 0.08, and max 1.5M steps per run.

### Sub-3 — Environmental Decision

**Files:** `sub3_env/`

An **RBF SVM** classifier (scikit-learn Pipeline: StandardScaler → SVC) that decides every second whether to DEPLOY or STOW the umbrella canopy.

| Parameter | Value |
|---|---|
| Input features | 9-D: lux, rain, wind + their frame-to-frame deltas + 3 prior actions |
| Training samples | 189 labelled readings covering sun, rain, grey-zone, high-wind, shadow |
| Cross-validation accuracy | 92.1% (10-fold stratified) |
| Hyperparameters | C = 5.0, γ = 0.1, class_weight = balanced |
| Hysteresis | Decision only flips after 3 consecutive matching predictions |

To retrain:
```bash
python3 sub3_env/train_svm.py --data data/env_sensor_log.csv --output models/svm_v1.pkl
```

### Sub-4 — Navigation and Safety

**Files:** `sub4_nav/`

Two independent components that run in parallel:

**Obstacle-avoidance SAC** — a Soft Actor-Critic policy trained in a 10×8 m obstacle room with 8 lidar rays. Produces a lateral avoidance force that steers the drone around obstacles in real time. The obstacle layout used during training is scenario-specific.

- Current model: **~512k steps, 72% efficiency**
- Algorithm: SAC (off-policy, sample-efficient; no curriculum needed)

**Battery-safety MDP** — a pre-computed value-iteration policy over a discrete (battery level × distance-to-home) state space. Returns one of: CONTINUE / RTH (Return to Home) / LAND_NOW. RTH is suppressed while battery > 50% so the drone keeps following; it activates only when battery drops low.

To retrain SAC:
```bash
python3 train_sub4.py --steps 200000 --scenario buildings
```

---

## Training Scripts

Headless scripts that warm-start from the existing saved models:

```bash
# Sub-2 PPO  — do NOT exceed 1.5M steps per warm-start run
python3 train_sub2.py --steps 1500000

# Sub-4 Nav SAC
python3 train_sub4.py --steps 200000 --scenario parking
```

---

## Trained Models

All models are committed to the repository and load automatically on launch:

| File | Algorithm | Size | Description |
|---|---|---|---|
| `models/ppo_flight_v1.zip` | PPO | 12.8M steps | Sub-2 hover policy |
| `models/ppo_nav_v1.zip` | SAC | ~512k steps | Sub-4 obstacle-avoidance nav |
| `models/svm_v1.pkl` | RBF SVM | 189 samples | Sub-3 umbrella deploy/stow |
| `models/policy_table_v1.npy` | MDP value iteration | — | Sub-4 battery safety |

---

## Project Structure

```
SkyShade/
├── run_sim.py              ← main simulation: launcher + physics loop + all 11 scenarios
├── scenario_report.py      ← per-scenario performance analysis tool
├── train_sub2.py           ← headless PPO training (Sub-2)
├── train_sub4.py           ← headless SAC training (Sub-4)
│
├── sub1_perception/
│   ├── tracker.py              HSV tracker (shadow band, EMA, velocity-biased gimbal)
│   ├── distance_estimator.py   Pinhole focal-length 3-D position estimator
│   └── training.py             Live calibration helper used by the launcher GUI
│
├── sub2_flight/
│   ├── env/
│   │   ├── hover_env.py        PyBullet hover physics environment
│   │   └── ppo_hover_env.py    Gymnasium wrapper with multi-walk curriculum
│   ├── training_worker.py      Background PPO training thread (used by GUI)
│   └── runtime_control.py      Flight controller wrapper (PPO velocity → force)
│
├── sub3_env/
│   ├── train_svm.py            SVM training script
│   ├── feature_engineering.py  9-D feature builder (deltas + action history)
│   └── classifier.py           Runtime SVM with hysteresis filter
│
├── sub4_nav/
│   ├── mdp.py                  MDP state/action/reward definitions
│   ├── solve_mdp.py            Offline value-iteration solver
│   ├── nav_training_worker.py  Background SAC training thread (used by GUI)
│   ├── obstacle_env.py         SAC lidar-based obstacle environment
│   └── policy_table.py         Runtime battery-safety MDP lookup
│
├── data/
│   └── env_sensor_log.csv      189 labelled lux/rain/wind samples for SVM
│
├── models/                     ← trained artefacts, committed to git
│   ├── ppo_flight_v1.zip
│   ├── ppo_nav_v1.zip
│   ├── svm_v1.pkl
│   └── policy_table_v1.npy
│
└── reports/
    └── run_metrics_history.json   last 20 run results (auto-generated)
```

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `pybullet`, `stable-baselines3`, `torch`, `scikit-learn`, `numpy`, `opencv-python`, `matplotlib`

No GPU required — all training and inference runs on CPU.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: pybullet` | Run with `/usr/bin/python3` instead of the venv — system Python has pybullet installed |
| Forest hover drops below 30% after training | Observation noise or LR was too high; run `python3 train_sub2.py --steps 1500000` to recover |
| Launcher scenario list doesn't scroll | Use the mouse scroll wheel — the list is a scrollable canvas |
| Rain / wind not visible in 3-D window | They render as PyBullet debug lines which only appear in the GUI window, not the dashboard camera feed |
| Slow simulation | The `--no-gui` flag disables the PyBullet 3-D renderer and is significantly faster |
