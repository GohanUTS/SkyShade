# SkyShade — Autonomous Drone Umbrella Simulation

> A multi-subsystem AI drone that perceives a moving person, hovers above them, and autonomously deploys a shade/rain canopy based on real-time weather. Everything runs in a single self-contained PyBullet simulation.

**Course:** AI for Robotics — Engineering Project  
**Institution:** University of Technology Sydney (UTS)  
**Team Leader:** Gohan Idrisoglu (24736668)  
**Crew:** Dinesh Saravanan · Aaron · Saaranj

---

## Quick Start (no setup needed — models are included)

```bash
git clone https://github.com/GohanUTS/SkyShade.git
cd SkyShade
pip install -r requirements.txt
python3 run_sim.py                        # launcher GUI
python3 run_sim.py --scenario snow        # jump straight to a scene
python3 run_sim.py --no-gui --scenario park --duration 120   # headless
```

> All trained models ship in `models/`. No retraining is required to run.

---

## Scenarios

Ten simulation worlds, each with detailed hand-crafted environments and a different challenge for the AI stack:

| Key | Name | What makes it hard |
|---|---|---|
| `park` | Park | Baseline — open flat loop, figure-8 walk |
| `forest` | Forest Trail | Tree canopy occludes tracker; hover under noisy position signal |
| `buildings` | Building District | City blocks, cars, pedestrians; nav avoids building near-misses |
| `trail` | Urban Trail | Crowd distractors + bridge shadow drop tracker confidence |
| `night` | Night Park | Low-light HSV tracking; fountain, benches, flower beds, streetlamps |
| `rooftop` | Rooftop | Constant 3–8 m/s wind; HVAC units, solar panels, water tower, satellite dish |
| `beach` | Coastal Beach | Lateral sea-breeze; lifeguard tower, palm trees, ocean, volleyball net |
| `parking` | Parking Lot | Detailed cars with cabins/wheels/lights; store building; constant-speed aisle walk |
| `vineyard` | Vineyard | Rhythmic vine-post occlusion; farmhouse, grape clusters, irrigation pipes |
| `snow` | Snowy Field | Omnidirectional gusting wind; frozen pond, pine trees, snowmen, wooden fence |

---

## How to Run

### GUI launcher
```bash
python3 run_sim.py
```
Opens the SkyShade Launcher. Pick a scenario, configure duration, and click **Select → Launch**. The scrollable scenario list shows all nine worlds. After each run a Post-Run Efficiency Report pops up with per-subsystem grades and a trend chart over the last 20 runs.

### CLI (headless or windowed)
```bash
# All scenarios, all options
python3 run_sim.py --scenario <key> --duration 120 --flight ppo
python3 run_sim.py --no-gui   --scenario forest --duration 120

# Stress-test low battery
python3 run_sim.py --demo-low-battery --scenario buildings
```

### Performance report
```bash
python3 scenario_report.py           # coloured terminal report
python3 scenario_report.py --json    # machine-readable JSON
```

---

## System Architecture

```
Camera (PyBullet virtual, downward-facing)
           │
    ┌──────▼──────┐
    │   Sub-1     │  HSV marker tracker → position + confidence
    │  Perception │  Shadow-tolerant band · EMA smoothing
    └──────┬──────┘
           │ user position (dx, dy, dz)
    ┌──────▼──────┐     wind force
    │   Sub-2     │◄────────────────── Weather controller
    │   Flight    │  PPO hover policy (12.8M steps)
    │  (PPO+PID)  │  Curriculum: stationary → wind → multi-walk
    └──────┬──────┘
           │ force setpoint
    ┌──────▼──────┐
    │  PyBullet   │  Physics (drone + scene)
    └──────┬──────┘
           │
    ┌──────▼──────┐     ┌─────────────┐     ┌─────────────┐
    │   Sub-3     │     │   Sub-4     │     │   Sub-4     │
    │  Weather    │     │  Nav SAC    │     │  Battery    │
    │  SVM + hyst │     │ obstacle av │     │  MDP table  │
    └─────────────┘     └─────────────┘     └─────────────┘
    umbrella cmd        avoidance force      CONTINUE/RTH/LAND
```

---

## Subsystems

### Sub-1 — Perception (`sub1_perception/tracker.py`)

HSV colour tracker with multiple improvements for challenging environments:

- **Primary band** — well-lit conditions (Value ≥ 55)
- **Shadow fallback band** — bridge / forest canopy shadow (Value 30–80, Sat ≥ 80); fires only if primary misses, avoiding false positives
- **EMA confidence smoothing** (α = 0.35) — damps single-frame dips from partial occlusion
- **Extended occlusion hold** — 35 frames before declaring lock lost (was 20)
- **Velocity-biased gimbal search** — when lock is lost, 65% of the search sweep biases toward the target's last velocity direction; recovers faster on linear-path scenarios
- **Reduced confidence saturation area** — 250 px² (was 500); distant/partial blobs score higher

Result: Urban Trail lock improved 58% → 84% from session start.

### Sub-2 — Flight Control (`sub2_flight/`)

PPO (Proximal Policy Optimisation) hover policy with a 3-stage curriculum:

| Stage | Condition | Steps |
|---|---|---|
| 1 | Stationary target, no wind | 0 – 500k |
| 2 | Stationary target, gusty wind (0–4.5 m/s) | 500k – 1M |
| 3 | Moving target, gusty wind | 1M+ |

**Stage 3 walk diversity** — three modes sampled each episode so the policy generalises across all scenario walk patterns:
- *Random* (40%) — ±0.2 m random steps (legacy)
- *Linear* (35%) — constant-velocity directional walk (0.08–0.18 m/s) — matches forest/trail/beach
- *Circular* (25%) — orbit at randomised radius and speed — matches park figure-8

> **Training note:** `_OBS_NOISE = {1: 0.0, 2: 0.0, 3: 0.0}` — observation noise is permanently disabled. Any noise in warm-start training causes catastrophic forgetting of clean hovering. Warm-start params: LR = 3e-5, clip = 0.08.

Current model: **12.8M steps**

### Sub-3 — Environmental Decision (`sub3_env/`)

RBF SVM classifier (sklearn Pipeline: StandardScaler + SVC) that outputs DEPLOY/STOW for the umbrella canopy.

- **Input:** 9-D feature vector — lux, rain, wind + their deltas + 3 prior actions
- **Training data:** 189 labelled samples (`data/env_sensor_log.csv`) covering sun, rain, grey-zone, high-wind, and shadow conditions
- **CV accuracy:** 92.1% (10-fold stratified)
- **Hyperparameters:** C = 5.0, γ = 0.1, class_weight = balanced
- **Hysteresis filter:** decision only flips after 3 consecutive matching predictions

Retrain:
```bash
python3 sub3_env/train_svm.py --data data/env_sensor_log.csv --output models/svm_v1.pkl
```

### Sub-4 — Navigation and Safety (`sub4_nav/`)

Two independent components:

**Battery MDP** (`policy_table_v1.npy`) — offline value-iteration over a (battery × distance) state space. Returns CONTINUE / RTH / LAND_NOW. RTH is suppressed while battery > 50% so the drone keeps following.

**Obstacle-avoidance SAC** (`ppo_nav_v1.zip`, filename legacy) — Soft Actor-Critic learns to navigate a 10×8 m obstacle room from lidar rays. Obstacle layouts are scenario-specific (park, forest, buildings, trail, parking, beach/rooftop/night).

Current SAC model: **~512k steps, 72% efficiency**

Retrain SAC:
```bash
python3 train_sub4.py --steps 200000 --scenario buildings
```

---

## Headless Training Scripts

Both scripts warm-start from the existing saved model and use conservative hyperparameters.

```bash
# Sub-2 PPO — do NOT add --steps beyond 1.5M per run
python3 train_sub2.py --steps 1500000

# Sub-4 Nav SAC
python3 train_sub4.py --steps 200000 --scenario parking
```

---

## Models (`models/`)

All models are committed to git and ship ready-to-use:

| File | Algorithm | Steps / Samples | Notes |
|---|---|---|---|
| `ppo_flight_v1.zip` | PPO | 12.8M | Sub-2 hover; zero obs noise |
| `ppo_nav_v1.zip` | SAC | ~512k | Sub-4 nav (legacy filename) |
| `svm_v1.pkl` | RBF SVM | 189 samples | Sub-3 umbrella, CV 92.1% |
| `policy_table_v1.npy` | MDP (value iter.) | — | Sub-4 battery safety |
| `qtable_v1.npy` | Q-table | — | Sub-2 legacy fallback |

---

## Project Structure

```
SkyShade/
├── run_sim.py              # Main integrated simulation (launcher + sim loop)
├── scenario_report.py      # Per-scenario performance analysis
├── train_sub2.py           # Headless PPO training
├── train_sub4.py           # Headless SAC training
│
├── sub1_perception/
│   ├── tracker.py          # HSV tracker with shadow band + EMA
│   ├── distance_estimator.py
│   └── training.py         # Calibration helper for the launcher
│
├── sub2_flight/
│   ├── env/
│   │   ├── hover_env.py        # PyBullet hover physics
│   │   └── ppo_hover_env.py    # Gymnasium PPO env (multi-walk curriculum)
│   ├── training_worker.py      # Background PPO thread (used by GUI)
│   ├── runtime_control.py      # PPO/Q/PID flight controller wrappers
│   └── policy.py
│
├── sub3_env/
│   ├── train_svm.py        # SVM training script
│   ├── classifier.py       # Runtime SVM with hysteresis
│   └── feature_engineering.py
│
├── sub4_nav/
│   ├── mdp.py              # MDP definition
│   ├── solve_mdp.py        # Value iteration solver
│   ├── nav_training_worker.py  # SAC background thread
│   ├── obstacle_env.py     # SAC gymnasium environment
│   └── policy_table.py     # Runtime MDP lookup
│
├── data/
│   └── env_sensor_log.csv  # 189 labelled weather samples for SVM
│
├── models/                 # Trained artefacts (committed to git)
│   ├── ppo_flight_v1.zip
│   ├── ppo_nav_v1.zip
│   ├── svm_v1.pkl
│   ├── policy_table_v1.npy
│   └── qtable_v1.npy
│
└── reports/                # Auto-generated per-run JSON metrics
    └── run_metrics_history.json
```

---

## Prerequisites

```
Python 3.10+
pybullet
stable-baselines3
torch
scikit-learn
numpy
opencv-python
matplotlib
tensorboard
```

Install:
```bash
pip install -r requirements.txt
```

> No GPU required. All training and inference runs on CPU.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: pybullet` | Run with `/usr/bin/python3` (system Python has pybullet), not the venv |
| Forest hover very low (<30%) | Run `python3 train_sub2.py --steps 1500000` to recover the policy |
| Launcher scenario list cut off | Scroll with mouse wheel — the list is scrollable |
| Weather / rain not visible | Open the PyBullet 3D window — rain/wind draw there, not in the dashboard |
