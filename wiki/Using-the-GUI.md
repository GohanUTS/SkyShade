# Using the GUI

SkyShade has two main GUI windows: the **Launcher** (where you train and configure the sim) and the **Simulation** (where the drone flies). This page covers how to use both.

---

## The Launcher

```bash
python run_sim.py
```

The launcher opens automatically. It has two columns:

```
┌─────────────────────────────┬────────────────────────────────────────┐
│  Training Ground (left)     │  Scenario + Mission Setup (right)      │
│                             │                                        │
│  Sub-1 Perception  [Calib]  │  [Park]  [Forest Trail]  [Buildings]  │
│  Sub-2 Flight      [Train]  │                                        │
│  Sub-4 Nav Safety  [Solve]  │  Duration: [120]  seconds              │
│                             │                                        │
│  Status message             │  Flight Control: PPO active            │
│                             │                                        │
│                             │  [ Show PyBullet GUI ]                 │
│                             │                                        │
│                             │  [ Launch ]   [ Cancel ]               │
└─────────────────────────────┴────────────────────────────────────────┘
```

### Left panel — Training Ground

The three cards show each trainable subsystem. Click the coloured button on the right of each card to open the **Training Grounds hub** on the relevant tab.

| Card | Button | Colour | Opens |
|---|---|---|---|
| Sub-1 Perception | **Calibrate** | Green | Calibration room — live tracker check |
| Sub-2 Flight | **Train PPO** | Blue | 3D hover arena — PPO training + evaluate |
| Sub-4 Nav Safety | **Solve MDP** | Purple | 3D obstacle room — MDP + nav training |

The status message below the cards updates to explain what each subsystem does.

> **Train these before launching.** The Launch button will warn you if models are missing.

### Right panel — Scenario

Click one of the scenario cards to select the environment the drone will fly in:

| Scenario | Description |
|---|---|
| **Park** | Open park loop with light obstacles and a figure-8 walking path. Best for first runs. |
| **Forest Trail** | Long wooded trail with tree avoidance and path reset. Tests obstacle avoidance. |
| **Building District** | City block with buildings, roads, vehicles, and pedestrians. Most complex scenario. |

The selected card is highlighted in blue.

### Mission Setup

| Field | Default | Description |
|---|---|---|
| **Duration** | 120 s | How long the simulation runs before stopping automatically |
| **Flight Control** | PPO active | Shows which flight controller will be used (PPO / Q-RL / PID) |
| **Show PyBullet GUI** | ✓ checked | Whether to open the 3D PyBullet window. Uncheck for headless mode. |

### Launch button

Clicking **Launch** checks for trained models. If any are missing:

```
The following subsystems are not trained:
  • Sub-2 Flight PPO (ppo_flight_v1.zip)

Open Training Grounds to train them, or launch anyway?
```

- **No** → returns to the launcher and opens the Training Grounds hub
- **Yes** → launches with fallback controllers (PID for flight; no obstacle nav)

When all models are ready, **Launch** immediately starts the simulation.

### Open Camera button

The **Open Camera** button (top right of the training panel) opens the **Sub-1 Perception calibration window** directly — a quick shortcut to check the tracker is locked before launching.

---

## The Simulation Window

After clicking Launch, two windows open:

1. **PyBullet 3D view** — the rendered physics world
2. **SkyShade Drone POV and AI Evidence** — the telemetry dashboard

### PyBullet 3D view

The physics world renders in real time. You can:
- **Left-click + drag** to orbit the camera
- **Right-click + drag** to pan
- **Scroll wheel** to zoom
- The blue quadcopter is the drone; the red sphere is the simulated user

### Drone POV and AI Evidence dashboard

```
┌──────────────────────────────────────────────────────────────┐
│  Drone POV: camera feed with red-marker detection            │
│                                                              │
│   ┌─────────────────────────────────────┐                   │
│   │  [Camera image with bounding box,   │                   │
│   │   crosshair, and detection overlay] │                   │
│   └─────────────────────────────────────┘                   │
│                                                              │
│  AI gimbal: right + back + down (predictive track)           │
│  Runtime AI evidence: live tracking, control, avoidance,     │
│  and decisions.                                              │
│                                                              │
│  Marker tracking  ━━━━━━━━━━━━                               │
│  Tree avoidance   ━━━━━━━━━━━━                               │
│  Umbrella dec.    ━━━━━━━━━━━━                               │
│  Coverage error   ━━━━━━━━━━━━                               │
│  Battery          ━━━━━━━━━━━━                               │
│                                                              │
│  Sub-1 Camera ● live check                                   │
│  Sub-2 Flight  Q-learning                                    │
│  Sub-3 Weather  SVM                                          │
│  Sub-4 Safety  MDP                                           │
│  Environment  scenarios                                      │
└──────────────────────────────────────────────────────────────┘
```

#### Camera feed (top section)

Shows what the drone's downward camera sees. The Sub-1 perception system draws:
- **Green bounding box** — detected user marker
- **White crosshair** — centre of detection
- Confidence and marker area shown in overlay text
- The gimbal direction text (e.g. `AI gimbal: right + back + down`) shows where the camera is pointing

#### Live AI telemetry charts (middle section)

Five scrolling line charts update in real time:

| Chart | What it shows |
|---|---|
| **Marker tracking** | Sub-1 tracker confidence over time. Stays high when the drone has a clear view. |
| **Coverage error** | Distance between drone and hover target (m). Lower = drone is on-target. |
| **Battery** | Remaining battery percentage. Drains at 0.5% per second. |
| **Tree avoidance** | Magnitude of obstacle-avoidance force (N). Spikes near trees or buildings. |
| **Umbrella decision** | Sub-3 umbrella state — toggles between DEPLOY and STOW as weather changes. |

Each chart has a "What these show" explanation panel on the right.

#### Subsystem status row (bottom)

Shows which AI model is actively running for each subsystem:
- `live check` — Sub-1 tracker actively running
- `Q-learning` or `PPO` — Sub-2 flight controller in use
- `SVM` — Sub-3 classifier running
- `MDP` — Sub-4 battery policy running
- `scenarios` — current environment

#### Training Ground section (bottom of dashboard)

A compact preview of the Sub-1 tracker calibration, including a mini camera view and status text, is shown here during the simulation so you can monitor tracker health while the drone is flying.

---

## What to watch during the simulation

| What you see | What it means |
|---|---|
| Drone following the red sphere | Sub-2 PPO / PID working correctly |
| Drone drifting sideways | Wind gust — should self-correct within 1–2 seconds if PPO is trained |
| Umbrella disc turns green | Sub-3 detected rain or cloud → `DEPLOY` |
| Console: `Nav override: RTH` | Sub-4 MDP triggered return-to-home (low battery or far from home) |
| Console: `Nav override: LAND_NOW` | Battery critical — drone descending |
| Marker tracking chart drops | User occluded or confidence low — drone holds last known position |
| Coverage error spikes | Drone temporarily off-target (wind, turn, obstacle) |

---

## Stopping the simulation

The simulation stops automatically when the duration expires. You can also:
- Close the PyBullet window to stop immediately
- Press `Ctrl+C` in the terminal

After stopping, the launcher window returns and you can re-launch with different settings.

---

## Keyboard shortcuts (PyBullet window)

| Key | Action |
|---|---|
| `r` | Reset camera to default view |
| `w` | Toggle wireframe mode |
| `g` | Toggle grid |
| `Esc` | Close the PyBullet window |
