# Sub-4: Navigation and Safety

## Responsibility

Monitor battery level and distance from home. Issue override commands to Sub-2 when continuing normal flight would be unsafe.

---

## How it works

1. Every 200 ms (5 Hz), battery percentage and distance from the home origin are read
2. Both are discretised into buckets to produce a state index
3. The pre-computed MDP policy table is looked up for that state
4. The resulting action (`CONTINUE`, `RTH`, or `LAND_NOW`) is published to `/skyshade/nav_override`
5. Sub-2 checks this topic before each PID calculation and redirects its target accordingly

---

## State space (12 states)

```python
battery_buckets  = [HIGH, MEDIUM, LOW, CRITICAL]   # 4 levels
distance_buckets = [NEAR, MID, FAR]                # Distance to home origin
# 4 × 3 = 12 total states
```

---

## Actions

| Action | Behaviour |
|---|---|
| `CONTINUE` | Pass control to Sub-2; drone follows user |
| `RTH` | Sub-2 navigates to home origin at target altitude |
| `LAND_NOW` | Sub-2 descends to 0.3 m and holds |

---

## Reward structure

```python
REWARD_SAFE_COMPLETION = +100
REWARD_PER_STEP        =   -1
REWARD_CRASH           = -100
REWARD_BATTERY_EMPTY   = -100
```

---

## Solver configuration

```python
GAMMA             = 0.95
CONVERGENCE_DELTA = 1e-6   # Stop when max Bellman update < this
```

The policy table is solved offline before the simulation runs. Runtime lookup is O(1).

---

## Known design issue — RTH bias

Because `CONTINUE` never earns `REWARD_SAFE_COMPLETION`, value iteration assigns the highest value to `RTH` in almost all states. In the integrated simulation this is patched by suppressing `RTH` while battery > 50%, so the drone only triggers the override when battery is genuinely low.

A future fix would restructure the reward function so `CONTINUE` accumulates positive reward for successful user-following, making the MDP balance safety against mission performance rather than always preferring RTH.

---

## Validation target

| Metric | Target |
|---|---|
| Scripted scenarios passed | 50 / 50 |
| RTH trigger latency | Within 1 decision tick of low-battery injection |

---

## Relevant files

- `sub4_nav/mdp.py` — MDP state/action/reward definition
- `sub4_nav/solve_mdp.py` — offline value iteration solver
- `sub4_nav/policy_table.py` — runtime policy lookup
- `sub4_nav/test_nav_safety.py` — 50-scenario test suite
- `models/policy_table_v1.npy` — solved policy table
- `convergence_curve.png` — value iteration convergence plot
- `ros2_ws/src/skyshade/skyshade/nav_safety_node.py` — ROS 2 wrapper
