# Validation and Results

This page covers how to verify each subsystem is working — through the Training Grounds hub, via test scripts, and via the automatic post-run efficiency report printed in the terminal after every simulation.

---

## Post-Run Efficiency Report

After every simulation run the terminal prints a graded report. This is the fastest way to see if training is working.

```
════════════════════════════════════════════════════════════
  SkyShade — Post-Run Efficiency Report
════════════════════════════════════════════════════════════
  Sub-2  Hover accuracy   :  73.4%  ✓ good
         Mean hover error  :  0.38 m  ✓ within 0.5m
  Sub-1  Tracker lock     :  98.2%  ✓ reliable
  Sub-3  Umbrella correct :  91.7%  ✓ accurate
  Sub-4  Battery at end   :  42.0%  ✓ safe
════════════════════════════════════════════════════════════
  Overall system score: 87%  Grade: A
════════════════════════════════════════════════════════════
```

### What each metric means

| Metric | What is measured | Pass (✓) | OK (~) | Fail (✗) |
|---|---|---|---|---|
| **Hover accuracy** | % of sim time the drone was within 0.5 m of the user | ≥ 70% | ≥ 40% | < 40% |
| **Mean hover error** | Average XY distance from user throughout the sim | ≤ 0.5 m | ≤ 1.0 m | > 1.0 m |
| **Tracker lock** | % of frames where Sub-1 confidence ≥ 0.7 | ≥ 80% | ≥ 60% | < 60% |
| **Umbrella correct** | % of time umbrella state matched rain conditions | ≥ 85% | ≥ 65% | < 65% |
| **Battery at end** | Remaining battery when sim ends | ≥ 30% | ≥ 10% | < 10% |

### Overall grade

The overall score is `(hover accuracy + tracker lock + umbrella accuracy) / 3`.

| Grade | Score | Meaning |
|---|---|---|
| A | ≥ 80% | All subsystems performing well — good to demo |
| B | ≥ 65% | Solid — minor training improvements would help |
| C | ≥ 50% | Some subsystems need more training |
| D | < 50% | One or more subsystems critically underperforming |

### If the score is low — what to do

The report prints specific advice, e.g.:
- `→ Train Sub-2 PPO more (hover accuracy is the bottleneck)` — click Fine-tune in Sub-2 tab
- `→ Check Sub-1 calibration (tracker confidence is low)` — open Sub-1 Calibrate tab
- `→ Retrain Sub-3 SVM (umbrella decisions are inaccurate)` — click 🔄 Train/Retrain SVM

### Console log (every 5 seconds)

During the sim, the terminal prints a row every 5 seconds:

```
  Time     Bat  Nav         Flight   Umbrella   Conf   ErrXY      Z
---------------------------------------------------------------------------
   0.0s  100.0%  CONTINUE    PPO      STOW      1.00    0.00m  2.50m
   5.0s   97.5%  CONTINUE    PPO      STOW      1.00    0.12m  2.48m
  10.0s   95.0%  CONTINUE    PPO      STOW      0.98    0.34m  2.51m
  45.9s   77.0%  CONTINUE    PPO      DEPLOY    1.00    0.37m  2.41m
```

- **ErrXY** is the most important live metric — how far the drone is from the user
- **Conf** — Sub-1 tracker confidence (should stay ≥ 0.7)
- **Umbrella DEPLOY** appearing means Sub-3 detected rain

---

---

## Sub-1: Perception — how to verify

### In the Training Grounds hub

1. Open hub → **Sub-1 Perception** tab → click **Start Live View**
2. A PyBullet calibration room opens with a moving red sphere

**Check these indicators:**

| Indicator | Pass | Fail |
|---|---|---|
| Status bar | `LOCKED ✓  confidence 1.00  pixel error 0.0px` | `SEARCHING` or `FOUND – off target` |
| Confidence (blue chart) | Line stays above the dashed threshold (0.7) | Frequent dips below threshold |
| Pixel error (orange chart) | Line stays below 18 px dashed line | Spikes regularly above 18 px |
| Bounding box colour | Green | Orange = found but off-centre |
| Yellow dot vs crosshair | Yellow dot is inside the blue acceptance ring | Crosshair is outside the ring |

### Via test script

```bash
python sub1_perception/test_perception.py
```

Reports tracking continuity (% frames above threshold) and position MAE.

**Pass targets:**
- Tracking continuity > 70% of frames
- Position MAE < 0.15 m

---

## Sub-2: Flight PPO — how to verify

### In the Training Grounds hub

After training, click **Evaluate Model** (teal button) in the Sub-2 tab.

The evaluator runs **5 deterministic episodes** in stage-2 conditions (stationary user + gusty wind).

**What you see:**
- Per-episode status: `Eval ep 3/5 ✓  reward +184  hover 67%`
- Best episode path drawn as a **bright green trail** in the 3D hover arena — it should stay close to the green hover ring
- Final verdict in the status bar:
  - `PASS ✓ 4/5 episodes hovered  mean reward +161` ← ready to launch
  - `FAIL ✗ 1/5 episodes hovered` ← train more
- Button relabels: `Evaluate Model  (4/5 ✓)`

**Pass thresholds:**
- ≥ 3 of 5 episodes spend ≥ 30% of their time inside the 0.5 m hover radius
- Mean episode reward > −200

**Efficiency estimate** shown after training:
```
✓ Model trained — predicted hover efficiency ~73%  (fine-tuned from previous model)
```
This is computed from the final training buffer. ≥ 60% means the model is performing well.

### Via test script

```bash
python sub2_flight/test_flight.py
```

Evaluates both PID and the trained PPO policy across 3 scenarios (calm, gusty, walking). Reports per-scenario mean reward and hover success rate.

**Pass targets:**
- Eval pass rate ≥ 3/5 episodes
- Mean eval reward > −200 on stage-2 wind

### Signs the model needs more training

- Green trail in the arena barely moves — drone is not reaching the hover zone
- Efficiency < 40% after training
- Reward curve is flat after 300k+ steps — try restarting with `Stop` + `Start Training` again (fresh warm-start at a different point in the curriculum)

---

## Sub-3: Weather SVM — how to verify

### In the Training Grounds hub

After clicking **Train SVM**, two things appear immediately:

**1. CV accuracy in the status bar:**
```
✓ SVM trained — CV accuracy 96.0%  (≥90% target met)
```

**2. Confusion matrix in the right panel:**

```
┌─────────────────┬─────────────────┐
│ TN  stow=48  ✓  │ FP  deploy=0 ✗  │  ← predicted stow
├─────────────────┼─────────────────┤
│ FN  stow=4   ✗  │ TP  deploy=47 ✓ │  ← predicted deploy
└─────────────────┴─────────────────┘
     actual stow        actual deploy
```

Green cells (TN/TP) = correct; Red cells (FP/FN) = errors.

- **FP (false deploy)** — umbrella opens in clear weather. Annoying but harmless.
- **FN (false stow)** — umbrella stays closed in rain. This is the more important error to minimise.
- A good model has 0 FP and ≤ 2 FN.

**3. Live weather demo** (runs automatically without clicking anything):
- Weather cycles Clear → Cloudy → Rainy → Storm every 30 seconds
- Watch the umbrella: **green disc opens in rain/storm, grey line in clear**
- If the umbrella opens during the rainy phase and closes during clear, the SVM is correct
- The decision banner shows `☂ DEPLOY` (green) or `✕ STOW` (grey) in real time

### Via test script

```bash
python sub3_env/test_env_decision.py
```

Runs a 5-minute synthetic weather trajectory and checks that:
- DEPLOY rate is high during rainy conditions
- STOW rate is high during clear conditions
- No rapid flickering between states (hysteresis is working)

**Pass targets:**
- 10-fold CV accuracy ≥ 90%
- Confusion matrix: 0 FP, ≤ 3 FN (out of 99 samples)

### Signs the model needs retraining

- Accuracy < 90% — add more diverse training samples to `data/env_sensor_log.csv`
- High FP rate — the decision boundary is too aggressive; raise the SVM C value in `sub3_env/train_svm.py`
- Umbrella stays closed in rain during the live demo — high FN rate; check CSV labels

---

## Sub-4: Battery Safety MDP — how to verify

### In the Training Grounds hub

The solver completes in < 1 second. The status shows:
```
Solved — 18 iterations. ● READY — policy_table_v1.npy found
```

**Verify the policy is correct** by checking it matches the expected table:

| Battery | NEAR | MID | FAR | Expected |
|---|---|---|---|---|
| HIGH | RTH | RTH | RTH | ✓ |
| MEDIUM | RTH | RTH | RTH | ✓ |
| LOW | RTH | RTH | RTH | ✓ |
| CRITICAL | LAND | LAND | LAND | ✓ |

### Via test script

```bash
python sub4_nav/test_nav_safety.py
```

Runs all 50 scripted battery scenarios (e.g. battery drops to CRITICAL while FAR from home → LAND_NOW must trigger within 200 ms).

**Pass target:** All 50 scenarios pass. RTH trigger latency ≤ 1 decision tick (200 ms).

### Runtime verification (in the main sim)

The nav override telemetry chart in the dashboard shows the current action:
- Stays at CONTINUE during normal flight (battery > 50%)
- Switches to RTH when battery falls below ~50%
- Switches to LAND_NOW when battery reaches CRITICAL

If you never see RTH, use `--demo-low-battery` to force an early battery drain:
```bash
python run_sim.py --demo-low-battery
```

---

## Sub-4: Nav SAC — how to verify

### In the Training Grounds hub

Click **Evaluate Model** in the Sub-4 nav section. The evaluator runs **5 episodes** through the obstacle room.

**What you see:**
- Per-episode: `Eval ep 2/5 ✓ GOAL  reward +392  steps 647`
- Best episode path drawn as **bright green trail** through the red pillars — it should weave through the room and end at the green star (goal zone)
- Final verdict:
  - `PASS ✓ 3/5 episodes reached goal` ← ready
  - `FAIL ✗ 1/5` ← train more

**Pass threshold:** ≥ 3 of 5 episodes reach the east-side goal zone within 1 200 steps.

**Signs of a good model:**
- Green trail clearly avoids all red pillars
- Episodes that fail tend to get close before failing, not crashing immediately
- Reward is consistently above 0 (goal reached gives +200 bonus)

**Signs the model needs more training:**
- Trail goes straight into a pillar
- All 5 episodes fail — train another 150k steps
- Trail reaches the goal sometimes but not consistently — one more warm-start run

---

## Full test suite

Run all subsystem tests at once:

```bash
# Sub-1 perception
python sub1_perception/test_perception.py

# Sub-2 PPO flight evaluation
python sub2_flight/test_flight.py

# Sub-3 weather classifier
python sub3_env/test_env_decision.py

# Sub-4 battery safety
python sub4_nav/test_nav_safety.py
```

---

## Pass targets summary

| Subsystem | Metric | Target |
|---|---|---|
| Sub-1 | Tracking continuity | > 70% of frames above threshold |
| Sub-1 | Position MAE | < 0.15 m |
| Sub-1 | Calibration room | LOCKED, pixel error < 18 px |
| Sub-2 PPO | Evaluation pass rate | ≥ 3 / 5 episodes in hover zone |
| Sub-2 PPO | Efficiency estimate | ≥ 60% |
| Sub-2 PPO | Mean eval reward | > −200 on stage-2 wind |
| Sub-3 SVM | CV accuracy | ≥ 90% |
| Sub-3 SVM | Confusion matrix FP | 0 (never deploy in clear) |
| Sub-3 SVM | Live demo | Umbrella opens in rain, closes in clear |
| Sub-4 MDP | Scripted scenarios | All 50 pass |
| Sub-4 MDP | RTH latency | ≤ 200 ms |
| Sub-4 Nav | Evaluation pass rate | ≥ 3 / 5 episodes reach goal |

---

## Validation artefacts on disk

| File | Description |
|---|---|
| `models/ppo_flight_v1.zip` | Sub-2 trained PPO hover policy |
| `models/svm_v1.pkl` | Sub-3 trained SVM umbrella classifier |
| `models/policy_table_v1.npy` | Sub-4 solved MDP policy (12 integers) |
| `models/ppo_nav_v1.zip` | Sub-4 trained **SAC** obstacle navigation policy (legacy filename) |
| `confusion_matrix.png` | Sub-3 SVM confusion matrix (training set) |
| `pca_3d.png` | Sub-3 3-D PCA of the 9-D weather feature space |
| `convergence_curve.png` | Sub-4 MDP Bellman delta per iteration |
