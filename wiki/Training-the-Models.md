# Training the Models

All four trainable subsystems have a **Training Grounds** hub built into the launcher. You must train each one before the simulation will launch — the launcher warns you if any model is missing.

---

## Quick-start checklist

```
python run_sim.py        ← open the launcher
```

Click each card on the left panel in order:

| Step | Card | Button | Time | Result |
|---|---|---|---|---|
| 1 | Sub-1 Perception | **Calibrate** | Instant | Verify tracker — no model needed |
| 2 | Sub-2 Flight | **Train PPO** | ~8 min first / ~5 min repeat | `models/ppo_flight_v1.zip` |
| 3 | Sub-3 Weather | **Train SVM** | < 2 seconds | `models/svm_v1.pkl` |
| 4 | Sub-4 Nav Safety | **Solve MDP** | < 1 second | `models/policy_table_v1.npy` |
| 4b | Sub-4 Nav Safety | **Train Navigation** | ~3 min first / ~2 min repeat | `models/ppo_nav_v1.zip` (SAC weights, legacy filename) |

Once all five model files exist, the footer badges turn `●` and the **Launch** button works.

---

## Sub-1: Perception — calibration check

Sub-1 uses a deterministic HSV colour tracker. There is no model to train. The calibration tab verifies the algorithm works correctly in a synthetic 3D scene before you commit to the full sim.

### How to calibrate

1. Click **Calibrate** on the Sub-1 card → Training Grounds hub opens on the Perception tab
2. Click **Start Live View**
3. A PyBullet room opens: a red sphere moves in a Lissajous path; the camera pans gently to track it
4. Watch the overlays on the camera view:
   - **Green box** = where the tracker detected the sphere
   - **Yellow dot** = where the sphere actually is (truth, projected from 3D)
   - **Blue ring** = 18 px acceptance zone — crosshair must land inside
5. Status bar should read: `LOCKED ✓  confidence ≥ 0.70  pixel error < 18px`

### How to verify it is working

| Indicator | Passing | Failing |
|---|---|---|
| Status bar | `LOCKED ✓` | `SEARCHING` or `FOUND – off target` |
| Confidence chart (blue) | Line above the dashed threshold | Dips below threshold repeatedly |
| Pixel error chart (orange) | Line below the dashed 18 px line | Frequent spikes above 18 px |
| Box colour | Green | Orange (detected but off-centre) |

If it consistently shows `SEARCHING`, the red sphere HSV range may not match. Check `sub1_perception/tracker.py` HSV thresholds.

### No model file produced — Sub-1 is always ●

---

## Sub-2: Flight PPO — hover agent

The PPO agent learns to keep the drone hovering within 0.5 m of the user at 2.5 m altitude. It progresses through three curriculum stages automatically.

### What the training does

| Stage | Timesteps | Conditions | Goal |
|---|---|---|---|
| 1 | 0 – 200 k | No wind, stationary user | Learn to fly up and hold position |
| 2 | 200 k – 400 k | Random gusts 0–4.5 m/s | Learn to resist wind disturbance |
| 3 | 400 k + | Walking user + gusts | Learn to track a moving target |

### How to train

1. Click **Train PPO** on the Sub-2 card
2. (Optional) Adjust timesteps — default 500 000 ≈ 8 min on CPU
3. Click **Start Training**
4. Watch the 3D hover arena:
   - Stage 1: drone barely drifts, no wind arrows
   - Stage 2: 6 orange wind arrows appear, drone sways more
   - Stage 3: walking user (purple figure) appears, drone tracks it
5. Watch the reward curve:
   - Early: very negative (−500 to −1 500) — drone crashing and drifting
   - After ~100 k steps: reward rises (−200 to 0) — basic hover emerging
   - After ~400 k steps: reward positive — stable hover in wind
6. Check the title bar: `PPO Reward · 38% done · ~5m left · 1024 steps/s`
7. Check the trend indicator: `↑ improving` means learning is progressing

### Warm-start (second run onwards)

Every subsequent "Start Training" run automatically **loads the existing model and fine-tunes it** at a lower learning rate. You never lose prior learning. Each session adds improvement. Run 2–3 sessions of 500 k steps each to build a robust policy.

### How to verify it is working

Click **Evaluate Model** in the Sub-2 tab after training:

| Result | Meaning |
|---|---|
| `PASS ✓ 4/5 episodes hovered` | Good — ready to launch |
| `PASS ✓ 3/5 episodes hovered` | Acceptable — can launch, consider one more training run |
| `FAIL ✗ 1/5` | Train for another 500 k steps |
| Efficiency `≥ 60%` in status bar | Model is performing well |
| Green trail in 3D arena | Best eval episode path — should stay near the hover ring |

**Passing threshold:** ≥ 3 of 5 episodes spend ≥ 30% of their time inside the 0.5 m hover zone.

### CLI alternative

```bash
python sub2_flight/train_ppo.py --steps 500000 --output models/ppo_flight_v1
```

---

## Sub-3: Weather SVM — umbrella decision

The SVM classifier decides whether to `DEPLOY` or `STOW` the umbrella canopy based on three weather sensors: light level (lux), rain intensity, and wind speed.

### What the training does

The SVM trains on `data/env_sensor_log.csv` — 99 labelled samples of (lux, rain, wind) paired with the correct umbrella action. It builds a 9-D feature vector:

```
[lux, rain, wind, Δlux, Δrain, Δwind, prev_action_t-1, prev_action_t-2, prev_action_t-3]
```

The `prev_action` features (deployment history) give the classifier memory, preventing it from toggling the umbrella every frame.

### How to train

1. Click **Train SVM** on the Sub-3 card
2. The hub opens on the Sub-3 Weather tab — click **Train SVM** button (pink)
3. Training completes in under 2 seconds
4. Status bar shows: `✓ SVM trained — CV accuracy 96.0%  (≥90% target met)`

### How to verify it is working

**Automatic during training:**
- The status bar shows 10-fold cross-validation accuracy (target ≥ 90%)
- The 2×2 confusion matrix appears in the right panel after training:
  - Green cells (TP/TN) = correct predictions
  - Red cells (FP/FN) = mistakes
  - A good model has no red cells, or only 1–2 FN (missed deploys are safer than false deploys)

**Live demo on the Sub-3 tab:**
- The weather scene cycles automatically: **Clear ☀ → Cloudy ⛅ → Rainy 🌧 → Storm ⛈** every 30 seconds
- Watch the umbrella canopy above the drone:
  - Clear conditions → `✕ STOW` (grey folded line)
  - Rain/storm conditions → `☂ DEPLOY` (green disc opens)
- The gauges on the right show lux/rain/wind values in real time
- If the umbrella opens and closes at the right times, the SVM is working correctly

**Passing threshold:** 10-fold CV accuracy ≥ 90%. A confusion matrix with fewer than 3 errors in 99 samples is excellent.

**Runtime verification** (during the main sim):
- The Umbrella decision chart in the telemetry dashboard flips between 0 (stow) and 1 (deploy) as weather cycles
- The HUD shows `umbrella: deployed` / `umbrella: stowed`

### CLI alternative

```bash
python sub3_env/train_svm.py --data data/env_sensor_log.csv --output models/svm_v1.pkl
```

---

## Sub-4: Battery Safety MDP — value iteration

The MDP policy decides when the drone should CONTINUE flying, Return To Home (RTH), or LAND immediately, based on battery level and distance from home.

### What the training does

Value iteration solves the 12-state Markov Decision Process (4 battery levels × 3 distance zones) offline. It finds the policy that maximises expected future reward given transition probabilities for battery drain and return-flight progress.

### How to solve

1. Click **Solve MDP** on the Sub-4 card (or the compact purple button in the Sub-4 tab)
2. Completes in under 1 second (~18 value-iteration sweeps)
3. Status bar: `Solved — 18 iterations. ● READY — policy_table_v1.npy found`

### How to verify it is working

The convergence curve appears in the left panel during solving — max Bellman Δ should drop exponentially to below 1e-6.

**Expected policy (correct output):**

| Battery | Any Distance | Action |
|---|---|---|
| HIGH / MEDIUM / LOW | — | RTH |
| CRITICAL | — | LAND_NOW |

This is the optimal policy given the reward structure. The drone always prefers to return home safely rather than continue.

**Runtime verification:**
- In the main sim, the nav override telemetry chart shows `RTH` when battery falls below 50%
- `LAND_NOW` triggers when battery reaches CRITICAL level

**Never needs repeating** — the MDP state space is tiny and value iteration always converges to the same optimal policy.

---

## Sub-4: Obstacle Navigation SAC

The SAC nav agent learns to fly the drone from one side of a 10 × 8 m room to the other, using 8 lidar rays to detect and avoid 7 cylindrical obstacles.

### What the training does

The agent receives 12 observations per step (8 lidar ray fractions + 2D goal direction + 2D velocity) and outputs a 2D velocity setpoint. Reward is given for progress toward the goal, penalised for collisions and time. A +200 bonus on arrival reinforces goal-seeking.

### How to train

1. In the Sub-4 Nav Safety tab, set steps (default 150 000 ≈ 3 min)
2. Click **Train Navigation** (teal button)
3. Watch the 3D obstacle room:
   - Blue drone sphere moves through the room
   - Orange lidar rays extend from the drone, shortening near red pillars
   - Blue trail traces the recent path
4. Watch the reward curve: starts negative (collisions), should trend upward as the agent learns to navigate
5. Status bar shows time remaining and steps/sec

### Warm-start (same as Sub-2)

Each subsequent training run fine-tunes the existing model. Run 2–3 sessions to build a robust policy.

### How to verify it is working

Click **Evaluate Model** in the Sub-4 nav section:

| Result | Meaning |
|---|---|
| `PASS ✓ 3/5 episodes reached goal` | Ready to launch |
| `PASS ✓ 4–5/5` | Excellent |
| `FAIL ✗ 1/5` | Train for another 150 k steps |
| Green trail in 3D room | Shows the actual path between pillars — should weave around all obstacles and reach the green star |

**Passing threshold:** ≥ 3 of 5 evaluation episodes reach the east-side goal zone within 1 200 steps.

### CLI alternative

```bash
# There is no separate CLI script — use the hub or call NavTrainingWorker directly
python sub4_nav/solve_mdp.py --output models/policy_table_v1.npy  # MDP only
```

---

## Summary: what "trained" means for each subsystem

| Subsystem | "Trained" means | File produced |
|---|---|---|
| Sub-1 Perception | Tracker shows LOCKED ≥ 70% of frames | None — deterministic |
| Sub-2 Flight PPO | Evaluate shows ≥ 3/5 episodes hovering; efficiency ≥ 60% | `models/ppo_flight_v1.zip` |
| Sub-3 Weather SVM | CV accuracy ≥ 90%; umbrella deploys in rain, stows in clear | `models/svm_v1.pkl` |
| Sub-4 Battery MDP | Policy table solved; CRITICAL → LAND, LOW/MED/HIGH → RTH | `models/policy_table_v1.npy` |
| Sub-4 Nav SAC | Evaluate shows ≥ 3/5 episodes reaching the goal | `models/ppo_nav_v1.zip` (SAC weights, legacy filename) |
