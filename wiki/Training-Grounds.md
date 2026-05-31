# Training Grounds Hub

The Training Grounds hub is a dedicated window for training, visualising, and evaluating each subsystem's AI model before launching the main simulation. It opens from the SkyShade Launcher when you click any training card button.

---

## Opening the hub

```
python3 run_sim.py
```

### Option A — One-click Auto-Train (recommended)

Two large buttons sit below the subsystem cards on the launcher:

| Button | Colour | What it does | Time |
|---|---|---|---|
| **🚀 Auto-Train All** | Green | Warm-starts from existing models, trains all subsystems sequentially, no button pressing | ~3 min |
| **🔄 Auto-Retrain from Scratch** | Amber | Deletes all model files first, then trains everything fresh | ~3 min |

The hub opens automatically, switches tabs, and advances through all stages without any user input. A green progress banner shows which step is running. When done it turns green: `✅ Training complete! → Select scenario in the Launcher and click Launch.`

### Option B — Train individually

Click any subsystem card button to open the hub on that specific tab:

| Button | Colour | Opens hub on tab |
|---|---|---|
| **Calibrate** | Green | Sub-1 Perception |
| **Train PPO** | Blue | Sub-2 Flight PPO |
| **Train SVM** | Pink | Sub-3 Weather SVM |
| **Solve MDP** | Purple | Sub-4 Nav Safety |

The hub window is titled **SkyShade — Training Grounds** and has four tabs along the top.

### Auto-Train sequence (what happens automatically)

| Stage | Subsystem | Steps | Time |
|---|---|---|---|
| 1 / 4 | Sub-2 Flight PPO | 100 000 | ~60 sec |
| 2 / 4 | Sub-3 Weather SVM | 99 samples | ~3 sec |
| 3 / 4 | Sub-4 Battery MDP | value iteration | ~1 sec |
| 4 / 4 | Sub-4 Nav SAC | 80 000 | ~50 sec |

**Total: ~2 min 15 sec** — well under 5 minutes.

---

## Layout overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  SkyShade — Training Grounds   Train each subsystem before launch.  │
├────────────────────┬──────────────────────┬─────────────────────────┤
│  Sub-1 Perception  │  Sub-2  Flight PPO   │  Sub-4  Nav Safety      │
└────────────────────┴──────────────────────┴─────────────────────────┘
│                                                                      │
│   LEFT PANEL (60%)                RIGHT PANEL (40%)                  │
│   ─────────────                   ────────────────                   │
│   3D animated scene               Reward curve / chart               │
│   (auto-rotates)                  (updates live during training)     │
│                                                                      │
│   [ Controls / buttons / status bar ]                                │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  Sub-1 Perception ●   Sub-2 Flight PPO ○   Sub-4 Battery MDP ●     │
│  Sub-4 Nav SAC ●   Launch the main sim from the SkyShade Launcher.  │
└──────────────────────────────────────────────────────────────────────┘
```

The footer shows a `●` (trained/ready) or `○` (not yet trained) badge for each model. The sim can be launched from the launcher window once all required models are ready.

---

## Sub-1 Perception tab

### Purpose

Sub-1 uses a deterministic HSV colour tracker — no neural network. This tab lets you verify the tracker works correctly against a synthetic 3D scene before running the full simulation.

### Left panel — Live Camera View

A PyBullet room renders in real time. A **red sphere** moves along a Lissajous path (a smooth 3D figure-eight curve). The camera gently pans to simulate a drone gimbal tracking the sphere.

| Overlay | What it means |
|---|---|
| **Green bounding box** | Where the HSV tracker detected the sphere in this frame |
| **White crosshair (+)** | Centre of the detected region |
| **Yellow dot** | Where the sphere actually is (projected from its 3D world position using camera matrices) |
| **Blue ring** | Acceptance zone — 18 px radius. The crosshair must land inside this ring for the frame to count as a pass |
| **"err Xpx" label** | Pixel error — distance in pixels between the crosshair and the yellow dot |
| Box colour **green** | Detection found AND within the acceptance zone (LOCKED) |
| Box colour **orange** | Detection found but outside the acceptance zone (off target) |

### Right panel — Tracking Accuracy chart

- **Blue line** — confidence score [0–1] over the last 200 frames (left axis)
- **Blue dashed** — confidence threshold (0.7); frames above this count as locked
- **Orange line** — pixel error in pixels (right axis)
- **Orange dashed** — 18 px pass threshold
- **Green shaded region** — frames where confidence is above threshold

### Status bar

```
LOCKED ✓   confidence 1.00   pixel error 0.0px
SEARCHING   confidence 0.23   no detection
FOUND – off target   confidence 0.81   pixel error 31px
```

### Controls

| Button | Action |
|---|---|
| **Start Live View** | Initialises the PyBullet room and begins streaming frames |
| **Stop Live View** | Pauses the calibration check and closes the PyBullet session |

Sub-1 is always shown as ready (●) in the footer — no model file is required.

---

## Sub-2 Flight PPO tab

### Purpose

Train and evaluate the PPO agent that keeps the drone hovering above the user. The 3D hover arena visualises what the drone is learning at each stage of the curriculum.

### Left panel — 3D Hover Arena

The scene auto-rotates slowly (0.35°/tick) so you can see it from all angles without clicking.

| Element | What it represents |
|---|---|
| **Grey floor with grid** | Room floor (7 × 7 m arena) |
| **Semi-transparent walls** | Four room walls — you're looking inside a training space |
| **Green dashed circle on floor** | Landing pad — directly below the hover target |
| **Green ring at 2.5 m altitude** | The hover zone — drone must stay inside this at `TARGET_ALTITUDE` |
| **Quadcopter drone (X-frame + rotors)** | The agent being trained. In Stage 1 it barely drifts; in Stage 2 it sways to show wind resistance; in Stage 3 it tracks a moving person |
| **Orange arrows** (stage 2+) | Gusts — 6 arrows in Stage 2, 10 in Stage 3, rotating inward to show force direction |
| **Purple figure on floor** (stage 3) | Walking user — orbits the arena. The drone must follow them |
| **Dashed line drone → target** | Shows the current gap the drone is trying to close |
| **Green trail** (after Evaluate) | Best evaluation episode — the actual path the trained drone flew |

#### Stage indicator (top-left of 3D pane)

```
Stage 1 / 3  —  Calm air
Drone learns to fly up and
hold position above the target.
No wind, stationary user.
```

Stages advance automatically based on timestep count:

| Stage | Timesteps | Conditions |
|---|---|---|
| 1 | 0 – 200 k | No wind, stationary user — learn basic hover |
| 2 | 200 k – 400 k | Random gusts 0–4.5 m/s — learn to resist wind |
| 3 | 400 k + | Gusty wind + walking user — learn to track |

### Right panel — PPO Reward Curve

- Coloured line segments: **blue** = stage 1, **pink** = stage 2, **purple** = stage 3
- Coloured background bands show which stage each region belongs to
- **White moving-average line** — smoothed trend; easier to see if learning is progressing
- **`↑ improving` / `→ flat`** — bottom-right indicator showing whether the last 20 rollouts improved

#### Title bar shows live training stats:
```
PPO Reward  ·  38% done  ·  ~14m left  ·  1024 steps/s
```

### Controls

| Control | Colour | Description |
|---|---|---|
| **Timesteps** entry | — | Steps to train. Default 500k. For a quick 1-min run use 100k. |
| **Start Training** / **Fine-tune (Run N)** | Blue | Warm-starts from existing model; adds incremental improvement |
| **🔄 Retrain from Scratch** | Amber | Deletes model file + chart history; trains from random weights |
| **Stop** | Red (while running) | Stops training and saves the partial model |
| **Evaluate Model (N/5 ✓)** | Teal | Runs 5 test episodes; draws best path in the arena |

### Multi-run reward chart

Each training session appears as a **different coloured line** on the reward chart:
- Run 1: blue, Run 2: green, Run 3: orange, Run 4: purple, etc.
- Completed runs remain as faded lines so you can compare improvement across sessions
- Current active run draws on top with full brightness
- A white moving-average line shows the smoothed trend
- `↑ improving` / `→ flat` label shows whether the last 20 rollouts improved
- **Initial dip is normal:** When fine-tuning, reward temporarily drops before rising — the optimizer is adjusting weights before settling on a better configuration

### Speed: 4 parallel environments

Training uses **4 parallel PyBullet physics servers** (one per CPU core):

| Environments | Steps/sec | 100k steps |
|---|---|---|
| 1 (old) | ~572 | ~3 min |
| 4 (current) | ~1 800–2 400 | **~50 sec** |

### Status bar messages

| Message | Meaning |
|---|---|
| `Run 2 — fine-tuning… 4 parallel envs ~1 min` | Warm-starting from Run 1's model |
| `Run 1 — training from scratch… 4 parallel envs ~2 min` | First training session |
| `Run 2 · 38% · ~2m left · 1924 steps/s` | Live ETA during training |
| `✓ Run 2 done — predicted hover efficiency ~73% (fine-tuned)` | Training completed |
| `⏹ Run 2 stopped at 180k steps — partial save` | User pressed Stop |

### Evaluate Model

Click **Evaluate Model** to run 5 deterministic test episodes on stage-2 (gusty wind) conditions:

- Each episode: drone spawned at a random offset, must fly in and hover
- Per-episode status: `Eval ep 3/5 ✓  reward +184  hover 67%`
- Best episode path drawn as a **bright green trail** in the 3D arena
- Final verdict: `PASS ✓ 4/5 episodes hovered` (pass = ≥3/5 episodes spent ≥30% of time in the hover zone)
- Button relabels to show score: `Evaluate Model  (4/5 ✓)`

---

## Sub-3 Weather SVM tab

### Purpose

Train and validate the SVM that decides whether to deploy or stow the umbrella canopy based on weather sensor readings. This is the fastest training in the hub — completes in under 2 seconds.

### Left panel — 3D Weather Scene

The scene auto-rotates and animates a 30-second weather cycle continuously.

| Element | What it shows |
|---|---|
| **Cloud blob** | Grows larger and darker as rain intensity rises |
| **Blue rain particles** | Vertical lines falling from cloud — more lines = heavier rain |
| **Orange wind arrows** | At drone altitude — count increases with wind speed |
| **Quadcopter drone** | At 2.5 m, same X-frame as Sub-2 |
| **Green umbrella disc** | Canopy open — SVM predicted DEPLOY |
| **Grey folded line** | Canopy closed — SVM predicted STOW |
| **`☂ DEPLOY`** banner | Green = umbrella is open |
| **`✕ STOW`** banner | Grey = umbrella is closed |

The weather cycle runs automatically: **Clear ☀ → Cloudy ⛅ → Rainy 🌧 → Storm ⛈** (30 s per cycle). The umbrella should open during the rainy/storm phases and close during clear/cloudy.

### Right panel — SVM Decision & Weather

The right panel has three sections, all updated every 200ms:

**Weather gauges (top):** Three horizontal bars showing current sensor values with units:
- **Lux** (yellow) — e.g. `17k lux`
- **Rain** (blue) — e.g. `0.74`
- **Wind** (cyan) — e.g. `4.0 m/s`

**Decision banner (centre):** Large coloured banner with explanation:
- `☂ DEPLOY` (green box) + `"Umbrella open — heavy rain detected"`
- `✕ STOW` (dark box) + `"Umbrella closed — conditions clear"`

**Confusion matrix (bottom, appears after training):** 2×2 grid with large count numbers and labels:

| Cell | Colour | Meaning |
|---|---|---|
| Top-left | Green | **Correct stow** — weather was clear, correctly stayed closed |
| Top-right | Red | **Wrong deploy** — weather was clear but umbrella opened (false positive) |
| Bottom-left | Red | **Wrong stow** — it was raining but umbrella stayed closed (false negative) |
| Bottom-right | Green | **Correct deploy** — it was raining, umbrella correctly opened |

Below the matrix: `Accuracy: 96.0% ✓ passes 90% target` in green.

### Controls

| Control | Colour | Description |
|---|---|---|
| **🔄 Train / Retrain SVM** | Pink | Trains on `data/env_sensor_log.csv` every time — always from scratch, < 2 sec |

> The SVM doesn't "improve" with more training runs — it always trains on the same 99-sample CSV and reaches the same ~96% accuracy. To improve accuracy, add more samples to `data/env_sensor_log.csv`.

### Status bar messages

| Message | Meaning |
|---|---|
| `Training SVM on weather sensor data…` | Training in progress |
| `✓ SVM trained — CV accuracy 96.0%  (≥90% target met)` | Training succeeded, model is good |
| `○ NOT TRAINED — click Train / Retrain SVM` | Model file missing |

### There is no Evaluate button — validation is the live demo

The 30-second weather cycle IS the evaluation. Watch the umbrella in the 3D scene:
- **Clear ☀ / Cloudy ⛅** → umbrella folded (grey line) = correct
- **Rainy 🌧 / Storm ⛈** → umbrella open (green disc) = correct
If the umbrella doesn't respond correctly to the phase transitions, retrain.

---

## Sub-4 Nav Safety tab

This tab controls two independent Sub-4 policies.

### Left panel — 3D Obstacle Navigation Room

A 10 × 8 × 3.5 m room with 7 cylindrical pillars. The auto-rotating scene shows:

| Element | What it represents |
|---|---|
| **Grey wireframe** | Room outline — floor, 4 walls, ceiling frame |
| **Red cylinders** | 7 obstacle pillars (bottom + top circle + vertical lines) |
| **Green star + ring** (east end) | Goal zone — drone must reach this to complete the episode |
| **Blue arrow** (west end) | Start position |
| **Blue sphere** | Drone (live position during training) |
| **8 orange lines** from drone | Lidar rays — each points in a 45° increment. The ray shortens when it hits a pillar, showing the drone actively sensing nearby obstacles |
| **Blue trail** | Recent drone positions — shows the path taken |
| **Green trail** (after Evaluate) | Best evaluation episode path through the obstacle course |

#### Bottom text

```
✓ Model trained — predicted navigation success ~61%  (fine-tuned from previous model)
```

### Right panel — Nav SAC Reward Curve

The chart has two reference lines that tell you immediately whether the drone is reaching the goal:

| Line | Colour | Meaning |
|---|---|---|
| Upper dashed | Green | **Goal reached zone** — reward > 0 means the +200 arrival bonus is dominating |
| Lower dashed | Red | **Stuck/crashing zone** — reward near −150 = drone hitting pillars or timing out |

**The three phases of a training run:**

| Phase | Steps | What's happening |
|---|---|---|
| Random | 0 – 10k | Filling replay buffer; occasional goal stumbles |
| The dip | 10k – 25k | Q-function inaccurate — tries "smart" moves that fail, crashes more |
| Learning | 25k – 80k | Q-function converges, routes around pillars |

**How to read the current reward:**
- **Reward < 0**: still learning, drone hasn't cracked consistent navigation yet
- **Reward > 0**: drone reaching goal in most episodes ✓
- **Reward ≈ +150–200**: goal reached in nearly every episode — excellent

A trend label at the bottom-right updates automatically:
- `✓ Goal being reached consistently!` (green) — reward > 50
- `↑ Approaching goal zone` — reward > 0 but < 50
- `↑ Learning — reward rising` — negative but improving
- `Still exploring` — flat or falling

### Controls — Navigation training (teal row)

| Control | Colour | Description |
|---|---|---|
| **Steps** entry | — | Default 150k ≈ 3 min first run; 30k ≈ ~50 sec repeat |
| **Train Navigation** | Teal | SAC training — warm-starts from existing model if compatible |
| **🔄 Retrain from Scratch** | Amber | Deletes model file; trains fresh SAC from random weights |
| **Stop** | Red (active) | Stops training; saves partial model |
| **Evaluate Model** | Teal | Runs 5 test episodes; draws best path as green trail |

> **⚠ Old PPO model incompatibility:** If `ppo_nav_v1.zip` exists from old PPO training, clicking "Train Navigation" will detect the mismatch, **delete the incompatible file automatically**, and train a fresh SAC model. Status shows: `"⚠ Old PPO model was incompatible with SAC — deleted. Training from scratch…"`

### Controls — Battery Safety MDP (purple row, compact)

| Control | Description |
|---|---|
| **Solve Battery-Safety MDP** | Runs value iteration (~18 iterations, < 1 second) |
| **Stop** | Cancels mid-solve (rarely needed given the speed) |

The convergence curve was shown in an earlier version and is saved to `convergence_curve.png`. The policy heatmap result is:

| Battery | NEAR | MID | FAR |
|---|---|---|---|
| HIGH | RTH | RTH | RTH |
| MEDIUM | RTH | RTH | RTH |
| LOW | RTH | RTH | RTH |
| CRITICAL | LAND | LAND | LAND |

### Evaluate Model (nav)

Runs 5 deterministic episodes through the obstacle room:
- Per-episode: `✓ GOAL` or `✗ miss`, reward, steps taken
- Best path drawn as **green trail** between red pillars
- Verdict: `PASS ✓ 3/5 episodes reached goal`

---

## Footer — Training status

```
Sub-1 ●   Sub-2 PPO ○   Sub-3 SVM ●   Sub-4 MDP ●   Sub-4 Nav ●
```

- `●` = model file exists on disk (trained/ready)
- `○` = model file missing (needs training)
- The main sim launcher checks these before allowing launch

---

## Warm-starting explained

Every "Start Training" click checks for an existing `.zip` model:

| Scenario | Behaviour | Learning rate |
|---|---|---|
| **No model exists** | Creates a fresh network, trains from scratch | `3e-4` |
| **Model exists** | Loads weights, continues training | `1e-4` (hover) / `5e-5` (nav) |

`reset_num_timesteps=False` is passed on warm runs so the curriculum stage counter continues — a 200k warm-start picks up mid-stage-2 if the previous run ended there.

**You never lose prior learning.** Each training session builds on the last.

---

## Closing the hub

Click the window's ✕ button. All background training threads are stopped and any partially trained models are saved before the window closes.
