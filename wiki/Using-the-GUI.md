# Using the GUI

SkyShade has two main GUI windows: the **Launcher** (where you train and configure the sim) and the **Simulation** (where the drone flies). This page covers how to use both.

---

## The Launcher

```bash
python3 run_sim.py
```

The launcher opens automatically. It's split into two columns:

| Left column — **Training Ground** | Right column — **Scenario & Mission** |
|---|---|
| 🎥 Sub-1 Perception → **Calibrate** | Scenario: **Park · Forest Trail · Buildings · Urban Trail** |
| 🚁 Sub-2 Flight → **Train PPO** | Duration: text box (default `120` s) |
| 🌧️ Sub-3 Weather → **Train SVM** | Flight control: PPO (falls back to PID) |
| 🧭 Sub-4 Nav Safety → **Solve MDP** | **Launch** / **Cancel** buttons |
| 🚀 **Auto-Train All** · 🔄 **Auto-Retrain from Scratch** | |

### Fastest way — Auto-Train buttons

Two large buttons sit below the subsystem cards:

| Button | Colour | What happens | Time |
|---|---|---|---|
| **🚀 Auto-Train All** | Green | Opens the hub, trains all 4 subsystems sequentially with no button pressing | ~3 min |
| **🔄 Auto-Retrain from Scratch** | Amber | Deletes all model files first, then trains everything from random weights | ~3 min |

The hub opens, the progress banner shows `"🚀 Auto-Train Step 2 / 4 — Sub-3 Weather SVM (~5 sec)…"`, tabs switch automatically, and when complete the banner turns green: `"✅ Training complete! → Select scenario in the Launcher and click Launch."`

### Left panel — Individual subsystem cards

To train a specific subsystem only, click its coloured button:

| Card | Button | Colour | Opens |
|---|---|---|---|
| Sub-1 Perception | **Calibrate** | Green | Calibration room — live tracker check |
| Sub-2 Flight | **Train PPO** | Blue | 3D hover arena — PPO training + evaluate |
| Sub-3 Weather | **Train SVM** | Pink | 3D weather scene — SVM training + live demo |
| Sub-4 Nav Safety | **Solve MDP** | Purple | 3D obstacle room — MDP + SAC nav training |

> **Train all before launching.** The Launch button warns if any model is missing and offers to open the Training Grounds or launch anyway with fallback controllers.

### Right panel — Scenario

Click one of the scenario cards to select the environment the drone will fly in. The list is scrollable because the launcher now exposes all 11 worlds:

| Scenario | Description |
|---|---|
| **Park** | Open park loop with light obstacles and a figure-8 walking path. Best for first runs. |
| **Forest Trail** | Long wooded trail with tree avoidance and path reset. Tests obstacle avoidance. |
| **Building District** | City block with buildings, roads, vehicles, and pedestrians. Most complex scenario. |
| **Urban Trail** | Bridge/underpass with crowd pedestrians and tracker distractors. |
| **Night Park** | Low-light streetlamp scene testing the shadow HSV band. |
| **Rooftop** | Confined rooftop platform with parapets, clutter, and strong wind. |
| **Coastal Beach** | Open beach with steady lateral sea breeze. |
| **Parking Lot** | Vehicle rows and aisle navigation around box obstacles. |
| **Vineyard** | Trellis rows with repeated post occlusion and tight corridors. |
| **Snowy Field** | Snow, pines, frozen pond, snowmen, and gusty overcast conditions. |
| **Stadium** | Oval track and a faster circular target in an athletics stadium. |

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

The POV window is what the drone's downward camera actually sees — the user's red cap marker dead in the frame, with the detection box and crosshair drawn on top.

<p align="center">
  <img src="images/scene_pov.png" alt="Drone downward POV camera view" width="640">
</p>

<p align="center"><sub><em>The drone's-eye view: it looks straight down and keeps the red marker centred as the user walks the city streets.</em></sub></p>

Below the camera feed, the window stacks five live traces and a row of subsystem status pills:

| Section | What it shows |
|---|---|
| 📷 **Camera feed** | Live downward image + green detection box, crosshair, and gimbal direction |
| 📈 **Marker tracking** | Sub-1 confidence over time (higher = locked on) |
| 🌲 **Avoidance force** | Sub-2 repulsion spikes when near trees / buildings / crowds |
| ☂️ **Umbrella decision** | Sub-3 stepping between STOW (0) and DEPLOY (1) |
| 🔋 **Coverage error & battery** | Distance from the user, and battery draining over the run |
| 🟢 **Status pills** | Sub-1 Camera · Sub-2 Flight · Sub-3 Weather · Sub-4 Safety · Environment |

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
| Building District user walks slowly toward buildings | The 120 s city visit path is active; this gives avoidance time to steer around walls |
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
