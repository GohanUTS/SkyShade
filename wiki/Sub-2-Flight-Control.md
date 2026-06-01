# Sub-2: Flight Control

## Responsibility

Keep the drone hovering directly above the user at `TARGET_ALTITUDE`, following them as they walk. Yield control to Sub-4 when a safety override is active.

---

## Runtime controller — PID

The integrated simulation uses a PID controller (`sub2_flight/policy.py`). At 30 Hz it computes a 3-D force vector from the error between the drone's current position/velocity and the target, then applies a constant hover-thrust offset to compensate for gravity.

```python
pid_target = [user_pos.x, user_pos.y, TARGET_ALTITUDE]
force      = pid.compute_force(drone_pos, drone_vel, pid_target, dt)
force[2]  += HOVER_FORCE_N
```

### Override targets

| `nav_override` from Sub-4 | PID target |
|---|---|
| `CONTINUE` | User XY + `TARGET_ALTITUDE` |
| `RTH` | (0, 0) + `TARGET_ALTITUDE` |
| `LAND_NOW` | Current XY + 0.3 m |

---

## Archived — Q-learning agent

A tabular Q-learning agent was trained before the PID controller was adopted. It is kept in `models/qtable_v1.npy` for reference and academic completeness.

### Why it was replaced

The discretised state space (225 states) caused the agent to oscillate at bucket boundaries when the drone was close to the target, producing unstable hover. The PID controller resolves this by operating in continuous state.

### Training details

**Curriculum stages:**

1. Stationary user, no wind
2. Stationary user, gusty wind
3. Walking user, gusty wind

**State space (225 total):**

```python
dx_buckets   = [-2, -1, 0, 1, 2]   # metres
dy_buckets   = [-2, -1, 0, 1, 2]
dz_buckets   = [-1,  0,  1]
wind_buckets = [LOW, MEDIUM, HIGH]
```

**Action space:**

```
0: MOVE_NORTH  1: MOVE_SOUTH  2: MOVE_EAST
3: MOVE_WEST   4: MOVE_UP     5: MOVE_DOWN  6: HOLD
```

**Reward function:**

```python
reward = progress_to_target   # getting closer to user
       - step_penalty          # cost per timestep
       - attitude_penalty      # penalise excessive tilt
       + hover_bonus           # bonus for staying within 0.5 m
       - crash_penalty         # large negative on collision
```

**Hyperparameters:**

```python
ALPHA         = 0.1
GAMMA         = 0.95
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.9995
EPISODES      = 50_000
```

---

## Validation target

| Metric | Target |
|---|---|
| Mean episode reward | > +150 across 10 eval episodes per scenario |
| Hover error | Within 0.5 m in 9 of 10 episodes |

---

## Relevant files

- `sub2_flight/policy.py` — PID flight controller (runtime)
- `sub2_flight/train_qlearning.py` — Q-learning training script (archived)
- `sub2_flight/env/hover_env.py` — PyBullet hover environment
- `sub2_flight/test_flight.py` — evaluation script
- `models/qtable_v1.npy` — trained Q-table (archived)
- `ros2_ws/src/skyshade/skyshade/flight_node.py` — ROS 2 wrapper
