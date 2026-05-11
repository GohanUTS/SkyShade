# Known Issues and Decisions

A log of significant design decisions, bugs fixed, and current limitations — so future contributors understand the "why" behind the current state.

---

## Decisions

### Sub-2: PID replaced Q-learning at runtime

**Decision:** The integrated simulation uses a PID controller, not the trained Q-table.

**Why:** The tabular Q-agent discretised position into 5 × 5 × 3 buckets. At bucket boundaries the drone oscillated between adjacent states, producing unstable hover. The Q-table is retained in `models/qtable_v1.npy` as a training artefact and academic reference.

**Impact:** The Q-learning reward function and curriculum training remain as demonstrable AI work; only the runtime inference path changed.

---

### Sub-4: 50% battery guard added in `run_sim.py`

**Decision:** `RTH` from the MDP is suppressed in the simulation runner while battery > 50%.

**Why:** Value iteration assigns maximum value to `RTH` across almost all states because `CONTINUE` never earns `REWARD_SAFE_COMPLETION` in the current reward model. Without the guard, the drone immediately returns home at the start of every run.

**Impact:** The MDP correctly triggers RTH when battery is genuinely low. The reward model is a known limitation for future work.

---

### Sub-3: Hysteresis window added post-training

**Decision:** A 3-frame hysteresis window was added to the runtime classifier wrapper, not baked into training.

**Why:** Without it, the SVM flipped `DEPLOY`/`STOW` every 1–2 frames during borderline weather, causing the umbrella servo to oscillate rapidly.

---

## Bugs fixed

### Sub-4 always outputting `LAND_NOW`

**Symptom:** After re-running `solve_mdp.py`, the policy table mapped every state to `LAND_NOW`.

**Root cause:** `REWARD_SAFE_COMPLETION` had been accidentally set to `-100` (matching `REWARD_CRASH`) during a refactor.

**Fix:** Restored `REWARD_SAFE_COMPLETION = +100`.

### Drone drifting on simulation start

**Symptom:** On startup the drone drifted before stabilising.

**Root cause:** PyBullet accumulates velocity during the URDF loading phase before the control loop starts.

**Fix:** Added a 30-step settle loop applying hover force before the main loop begins, followed by `resetBaseVelocity` to zero.

---

## Open issues / future work

| # | Issue | Suggested fix |
|---|---|---|
| 1 | Sub-4 MDP always prefers RTH in unconstrained solve | Restructure reward so CONTINUE accumulates positive reward for successful user-following |
| 2 | Sub-2 Q-learning chattering | Replace with continuous-action Deep RL (PPO/SAC) |
| 3 | HSV tracker sensitivity to lighting | Add adaptive threshold or replace with a learned detector |
| 4 | No sim-to-real validation | Domain randomisation + hardware-in-the-loop testing |
| 5 | Single-user only | Extend Sub-1 to track multiple targets; add collision avoidance to Sub-4 |
