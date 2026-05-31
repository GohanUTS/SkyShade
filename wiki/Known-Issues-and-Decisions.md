# Known Issues and Decisions

A log of significant design decisions, bugs fixed, and current limitations — so future contributors understand the "why" behind the current state.

---

## Design decisions

### Sub-2: Q-learning replaced by PPO

**Decision:** The runtime flight controller now uses PPO (Proximal Policy Optimisation) with a continuous velocity-setpoint action space. The Q-table is kept in `models/qtable_v1.npy` as an archived reference.

**Why Q-learning was replaced:**
- The 6075-state tabular grid caused oscillation at bucket boundaries — the drone couldn't brake smoothly because all states near the target mapped to the same discrete action
- Discrete bang-bang actions (MOVE_NORTH / MOVE_SOUTH etc.) made gentle hovering impossible in wind
- The state space couldn't generalise: training at one wind speed didn't help at a slightly different speed

**Why PPO works better:**
- Continuous 3D velocity setpoint output → smooth, fine-grained control
- The PPO network learns to generalise across wind speeds (the `wind_norm` input)
- Hierarchical design: PPO decides *where to move*, an inner P-controller handles *how to thrust* — this reuses the existing PID infrastructure at the execution layer

**Impact:** The Q-table and `train_qlearning.py` are retained for academic completeness. `FlightPolicy` (Q-table lookup) is still in `policy.py` as a legacy fallback in `RuntimeFlightController`.

---

### Sub-2: Velocity setpoint as PPO action space

**Decision:** The PPO action space is `[vx_desired, vy_desired, vz_desired]` (m/s), not raw thrust.

**Why:** Raw thrust outputs require the network to learn the drone's full dynamics (mass, drag, gravity compensation). Velocity setpoints abstract this away — the network only needs to learn navigation geometry, while the existing P-controller (`force = 8 × (v_des − v_cur)`) handles the physics. This halved training time and improved stability.

---

### Sub-4: Two separate Sub-4 policies

**Decision:** Sub-4 now has two distinct trained policies running in parallel:
1. **Battery Safety MDP** — decides CONTINUE / RTH / LAND\_NOW based on battery + distance
2. **Obstacle Navigation PPO** — navigates a room with 7 obstacles using 8 lidar rays

**Why separate:** The battery-safety problem is small (12 states, solved in <1 sec via value iteration) and interpretable. The obstacle navigation problem is continuous and too complex for a tabular approach. Keeping them separate makes each independently testable and explainable.

---

### Sub-4: 50% battery guard in `run_sim.py`

**Decision:** `RTH` from the MDP is suppressed while battery > 50%.

**Why:** Value iteration assigns maximum value to `RTH` across almost all states because `CONTINUE` never earns `REWARD_SAFE_COMPLETION` in the current reward model. Without the guard, the drone would immediately return home at the start of every run.

**Impact:** The MDP correctly triggers RTH when battery is genuinely low. Restructuring the reward function to reward user-following during `CONTINUE` is listed as future work.

---

### Training Grounds hub — warm-start by default

**Decision:** Every "Start Training" run in the Training Grounds hub loads the existing model and fine-tunes it rather than training from scratch.

**Why:** Starting from scratch each time discards learned weights. Warm-starting with a reduced learning rate (`lr = 1e-4` for hover, `5e-5` for nav) refines the existing policy. The first run takes ~8 min; subsequent runs add incremental improvement in ~5 min each.

---

### Sub-3: Hysteresis window added post-training

**Decision:** A 3-frame hysteresis window was added to the runtime classifier wrapper, not baked into training.

**Why:** Without it, the SVM flipped `DEPLOY`/`STOW` every 1–2 frames during borderline weather, causing rapid oscillation.

---

## Bugs fixed

### Sub-4 always outputting `LAND_NOW`

**Symptom:** After re-running `solve_mdp.py`, every state mapped to `LAND_NOW`.

**Root cause:** `REWARD_SAFE_COMPLETION` was accidentally set to `-100` during a refactor.

**Fix:** Restored `REWARD_SAFE_COMPLETION = +100`.

---

### Drone drifting on simulation start

**Symptom:** On startup the drone drifted before stabilising.

**Root cause:** PyBullet accumulates velocity during URDF loading before the control loop starts.

**Fix:** Added a 30-step settle loop applying hover force, followed by `resetBaseVelocity` to zero.

---

### Training Grounds 3D axes accumulating on every tick

**Symptom:** After ~60 seconds the reward chart showed garbled, inverted y-axis.

**Root cause:** `ax3 = ax2.twinx()` was called inside `_sub1_step()` which fires every 200 ms — each call created a new overlapping right axis, stacking hundreds on top of each other.

**Fix:** The twinx axis is now created once in `_build_sub1_tab()` and stored as `self._sub1_ax_err`. Each tick calls `ax3.cla()` and redraws into the stored axis.

### Sub-2: 4 parallel environments for training speedup

**Decision:** `FlightTrainingWorker` now uses `SubprocVecEnv(n_envs=4)` — four independent PyBullet DIRECT processes collect experience in parallel.

**Why:** Single-env throughput was ~572 steps/sec (14 min for 500k steps). With 4 envs on a 4-core CPU, effective throughput reaches ~1800–2400 steps/sec (100k steps ≈ 50 sec). PyBullet DIRECT mode creates isolated physics servers per process with no shared state.

**Impact:** Default training is now 100k steps ≈ ~1 min rather than 500k steps ≈ 14 min. Smaller rollout per env (n_steps=512) with larger batch (256) keeps update frequency the same.

---

### Sub-2: Multi-run training history chart

**Decision:** Each completed training session is saved to `_sub2_runs` and displayed as a permanent coloured line on the reward chart. The current active run draws on top in a brighter colour.

**Why:** With warm-start training, users couldn't see whether repeated sessions were actually improving the model. Each session being a different colour makes improvement (or lack of it) visually obvious.

**Impact:** Chart shows up to 7 past runs. New sessions get a new colour from the palette. The chart history resets when "Retrain from Scratch" is clicked.

---

### Sub-4 Nav: SAC replaces PPO

**Decision:** Obstacle navigation uses SAC (Soft Actor-Critic) instead of PPO.

**Why:** SAC is off-policy — it stores all past experience in a replay buffer and relearns from it. PPO discards experience after each update. For a static obstacle environment, SAC reaches comparable performance in ~3× fewer steps. The automatic entropy tuning means no manual curriculum stages are needed.

**SAC/PPO incompatibility:** SAC and PPO network architectures differ (SAC has separate actor + two Q-networks; PPO has a single actor-critic). If an old PPO model file exists, `SAC.load()` fails. The worker now catches this, deletes the incompatible file, and trains a fresh SAC model automatically.

---

### Training Grounds: only render the visible tab

**Decision:** The `_tick()` loop now checks `self._nb.index(self._nb.select())` and only redraws the currently visible tab's matplotlib scene.

**Why:** All three 3D matplotlib scenes (Sub-2 hover arena, Sub-3 weather scene, Sub-4 obstacle room) were being fully redrawn every 200ms regardless of which tab was visible. Each redraw calls `ax.cla()` + dozens of 3D draw calls. Combined with background SAC/PPO training consuming CPU cores, this caused severe UI lag.

**Impact:** Rendering went from 3 × full 3D render per 200ms (≈15 renders/sec) to 1 × render per 350ms (≈2.9 renders/sec) — a ~5× reduction in rendering work.

---

### ETA percentage bug with warm-start training

**Symptom:** Training progress showed "869%" instead of a percentage ≤ 100.

**Root cause:** With `reset_num_timesteps=False` (warm-start), SB3's `num_timesteps` counter continues from the previous training session (e.g. step 260,000). `_sub2_total` held only the current-run budget (e.g. 30,000). So 260,000/30,000 = 869%.

**Fix:** `_sub2_run_start_step` records the cumulative step count from the first progress message of each training run. Percentage is computed as `(current - start) / total`, clamped to 100%.

---

## Bugs fixed

### Sub-4 always outputting `LAND_NOW`

**Symptom:** After re-running `solve_mdp.py`, every state mapped to `LAND_NOW`.

**Root cause:** `REWARD_SAFE_COMPLETION` was accidentally set to `-100` during a refactor.

**Fix:** Restored `REWARD_SAFE_COMPLETION = +100`.

---

### Drone drifting on simulation start

**Symptom:** On startup the drone drifted before stabilising.

**Root cause:** PyBullet accumulates velocity during URDF loading before the control loop starts.

**Fix:** Added a 30-step settle loop applying hover force, followed by `resetBaseVelocity` to zero.

---

### Training Grounds 3D axes accumulating on every tick

**Symptom:** After ~60 seconds the reward chart showed garbled, inverted y-axis.

**Root cause:** `ax3 = ax2.twinx()` was called inside `_sub1_step()` which fires every 200 ms.

**Fix:** The twinx axis is created once in `_build_sub1_tab()` and stored as `self._sub1_ax_err`.

---

## Open issues / future work

| # | Issue | Suggested fix |
|---|---|---|
| 1 | Sub-4 MDP biases toward RTH | Restructure reward so CONTINUE earns positive reward for successful user-following |
| 2 | HSV tracker sensitivity to lighting | Add adaptive threshold or replace with a lightweight CNN detector |
| 3 | No sim-to-real validation | Domain randomisation + hardware-in-the-loop testing |
| 4 | Single-user only | Extend Sub-1 to track multiple targets; add collision avoidance to Sub-4 nav |
| 5 | Obstacle nav generalisation | Fixed 7-pillar layout — policy doesn't generalise to different room configurations |
| 6 | SAC nav training single-process | Could paralllelise SAC collection (SubprocVecEnv) for further speedup |
</content>
