# Training the Models

All three trainable subsystems have a **Training Grounds** hub built into the launcher. Click any training card button to open it. The hub must be used to train Sub-2 (PPO) and Sub-4 (MDP + Nav PPO) before the first sim launch — the launcher will warn you if models are missing.

---

## Training Grounds hub

Open the launcher and click one of the training buttons (Calibrate / Train PPO / Solve MDP). The **SkyShade — Training Grounds** window opens with three tabs:

| Tab | Subsystem | What trains |
|---|---|---|
| Sub-1 Perception | HSV tracker calibration | No ML training — live calibration check in a PyBullet room |
| Sub-2 Flight PPO | PPO hover agent | Neural network learning velocity setpoints |
| Sub-4 Nav Safety | MDP + PPO nav agent | Value-iteration safety policy + obstacle navigation |

---

## Sub-1: Perception calibration

Sub-1 uses a deterministic HSV tracker — no model file is needed. The calibration tab verifies the tracker works correctly.

**Click "Start Live View"** to open a PyBullet calibration room:
- A red sphere moves along a Lissajous path
- The camera renders a live view (PyBullet `getCameraImage`)
- The HSV tracker runs on each rendered frame
- The camera view shows:
  - **Green bounding box** — where the tracker detected the sphere
  - **White crosshair** — centre of detection
  - **Yellow dot** — where the sphere actually is (projected from 3D position)
  - **Blue acceptance ring** (18 px radius) — detection must land inside this to pass
- The right chart shows **confidence** (blue, left axis) and **pixel error** (orange, right axis) over time
- Status shows: `LOCKED ✓ confidence 1.00 pixel error 0.0px` when tracking correctly

Sub-1 is always marked ready in the launcher footer (no saved model required).

---

## Sub-2: PPO flight agent

### What it learns

The PPO agent learns to output **3D velocity setpoints** `[vx, vy, vz]` that keep the drone hovering within 0.5 m of the user at 2.5 m altitude. An inner P-controller converts velocity setpoints to thrust forces.

### 3-stage curriculum

Training automatically advances through three stages based on timestep count:

| Stage | Steps | Condition | What the agent must learn |
|---|---|---|---|
| 1 | 0 – 500 k | No wind, stationary user | Fly up, reach the hover zone, stay still |
| 2 | 500 k – 1 M | Random gusts 0–4.5 m/s | Resist wind disturbance while hovering |
| 3 | 1 M – 1.5 M | Walking user + gusts | Track a moving target against wind |

### Training via hub

1. Open Training Grounds → **Sub-2 Flight PPO** tab
2. Set timesteps (default 1 500 000)
3. Click **Start Training**
4. Watch the 3D room:
   - The quadcopter drone drifts more in higher stages (showing harder conditions)
   - Wind arrows appear and grow in stage 2/3
   - A walking user appears in stage 3
5. Watch the reward curve — it starts very negative (drone crashing, not hovering) and should trend upward over time
6. The title bar shows: `38% done · ~14m left · 1024 steps/s`
7. A `↑ improving` / `→ flat` indicator shows whether learning is progressing

**Expected training time:** 20–60 min depending on hardware (CPU-bound, no GPU needed).

### Is it working correctly?

- Reward in stage 1 should start around −500 and trend toward 0/+200 over ~200 k steps
- If reward stays below −800 after 100 k steps, the environment may not be initialising correctly
- The reward curve colour changes blue → pink → purple as curriculum advances — this is normal

### CLI training

```bash
python sub2_flight/train_ppo.py --steps 1500000 --output models/ppo_flight_v1
```

### Evaluating before launch

Click **Evaluate Model** (teal button, below Start Training):
- Runs 5 episodes with the trained model (deterministic, stage 2 conditions)
- Best episode path shown as a **green trail** in the 3D room
- Pass: `≥3/5 episodes spent ≥30% of time within the 0.5 m hover radius`

---

## Sub-4: Battery safety MDP

### What it learns

Value iteration solves the optimal policy for: given battery level + distance from home → should the drone CONTINUE, RTH (return to home), or LAND_NOW?

### Solving via hub

1. Open Training Grounds → **Sub-4 Nav Safety** tab
2. Click **Solve Battery-Safety MDP** (purple button, compact row at the bottom)
3. Completes in under 1 second (~18 iterations)
4. The convergence curve shows max Bellman Δ dropping exponentially to below 1e-6

### CLI

```bash
python sub4_nav/solve_mdp.py --gamma 0.95 --output models/policy_table_v1.npy
```

---

## Sub-4: Obstacle navigation PPO

### What it learns

The PPO agent learns to navigate a 10 × 8 m room from west to east, avoiding 7 cylindrical pillars using 8 lidar rays as the only sensor input.

### Training via hub

1. Open Training Grounds → **Sub-4 Nav Safety** tab
2. Set steps (default 500 000)
3. Click **Train Navigation** (teal)
4. Watch the 3D room:
   - The blue drone sphere moves through the room
   - **8 orange lidar rays** extend from the drone, shortening when they hit pillars
   - A **blue trail** traces where the drone has been
   - The reward curve on the right builds up over training
5. A successful policy will navigate the room without hitting pillars

**Expected training time:** 15–40 min.

### Evaluating before launch

Click **Evaluate Model** in the Sub-4 tab:
- Runs 5 episodes with the trained policy
- Shows each episode outcome: `✓ GOAL` or `✗ miss`
- The best episode's path is drawn as a **green trail** through the obstacle room — you can see the exact route the drone chose between the red pillars
- Pass: `≥3/5 episodes reached the goal zone`

---

## Launch gate

The launcher checks for `models/ppo_flight_v1.zip` and `models/policy_table_v1.npy` before launching the simulation. If either is missing, a dialog appears:

> "The following subsystems are not trained... Launch with fallback controllers, or stay and train?"

Choosing **No** returns to the Training Grounds hub. Choosing **Yes** launches with PID fallback for flight and no obstacle avoidance.

---

## Sub-3: SVM classifier (pre-trained)

```bash
python sub3_env/train_svm.py \
  --data data/env_sensor_log.csv \
  --output models/svm_v1.pkl
```

The SVM classifier is pre-trained and does not need re-training unless the sensor feature set changes. See `sub3_env/` for details.
