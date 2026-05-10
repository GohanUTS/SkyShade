"""
Sub-4 Navigation and Safety — MDP state/action/reward definition.

State space  : 12 states  (4 battery levels × 3 distance-to-home levels)
Action space : 3 actions  (CONTINUE, RTH, LAND_NOW)

States are represented as integer tuples (battery_idx, distance_idx) or
flattened to a single integer via state_to_idx / idx_to_state.

Transition model (T[state, action] → [(prob, next_state), ...]):
  CONTINUE  — drone stays airborne; battery drops by one level each step
               with probability P_BATTERY_DROP; distance stays the same.
  RTH       — drone flies home; distance decreases by one bucket each tick
               with probability P_RTH_PROGRESS; battery also drops.
  LAND_NOW  — drone lands immediately from wherever it is;
               terminal transition.

All terminal states absorb: (BATTERY_CRITICAL, any) and the LAND_NOW
transition lead to an absorbing "done" state encoded as state index N_STATES.
"""

import numpy as np

# ── State buckets ─────────────────────────────────────────────────────────────
BATTERY_HIGH     = 0   # > 75 %
BATTERY_MEDIUM   = 1   # 50–75 %
BATTERY_LOW      = 2   # 25–50 %
BATTERY_CRITICAL = 3   # < 25 %
N_BATTERY = 4

DISTANCE_NEAR = 0   # < 20 m from home
DISTANCE_MID  = 1   # 20–50 m
DISTANCE_FAR  = 2   # > 50 m
N_DISTANCE = 3

N_STATES = N_BATTERY * N_DISTANCE   # 12 states (+ 1 absorbing terminal)
TERMINAL_STATE = N_STATES           # Index 12 — absorbing

# ── Actions ───────────────────────────────────────────────────────────────────
ACTION_CONTINUE = 0
ACTION_RTH      = 1
ACTION_LAND_NOW = 2
N_ACTIONS = 3

ACTION_NAMES = ["CONTINUE", "RTH", "LAND_NOW"]

# ── Reward structure ──────────────────────────────────────────────────────────
REWARD_SAFE_COMPLETION =  100.0
REWARD_PER_STEP        =   -1.0
REWARD_CRASH           = -100.0
REWARD_BATTERY_EMPTY   = -100.0

# ── Transition probabilities ──────────────────────────────────────────────────
P_BATTERY_DROP  = 0.3   # Probability battery degrades one level per step
P_RTH_PROGRESS  = 0.7   # Probability distance-to-home decreases during RTH
# ─────────────────────────────────────────────────────────────────────────────


def state_to_idx(battery: int, distance: int) -> int:
    return battery * N_DISTANCE + distance


def idx_to_state(idx: int) -> tuple:
    return divmod(idx, N_DISTANCE)


def build_transitions():
    """
    Build the full transition model.

    Returns T, R where:
      T[s, a] = list of (probability, next_state_idx) tuples
      R[s, a] = scalar immediate reward
    """
    T = [[[] for _ in range(N_ACTIONS)] for _ in range(N_STATES)]
    R = np.zeros((N_STATES, N_ACTIONS), dtype=float)

    for s in range(N_STATES):
        bat, dist = idx_to_state(s)

        # ── CONTINUE ─────────────────────────────────────────────────────────
        a = ACTION_CONTINUE
        R[s, a] = REWARD_PER_STEP

        if bat == BATTERY_CRITICAL:
            # Force crash — battery empty
            R[s, a] = REWARD_BATTERY_EMPTY
            T[s][a] = [(1.0, TERMINAL_STATE)]
        else:
            # Battery may drop one level
            bat_drop = min(bat + 1, BATTERY_CRITICAL)
            T[s][a] = [
                (P_BATTERY_DROP,       state_to_idx(bat_drop, dist)),
                (1 - P_BATTERY_DROP,   state_to_idx(bat, dist)),
            ]

        # ── RTH ──────────────────────────────────────────────────────────────
        a = ACTION_RTH
        R[s, a] = REWARD_PER_STEP

        if bat == BATTERY_CRITICAL:
            R[s, a] = REWARD_BATTERY_EMPTY
            T[s][a] = [(1.0, TERMINAL_STATE)]
        elif dist == DISTANCE_NEAR:
            # Arrived home safely
            R[s, a] = REWARD_SAFE_COMPLETION
            T[s][a] = [(1.0, TERMINAL_STATE)]
        else:
            bat_drop = min(bat + 1, BATTERY_CRITICAL)
            dist_close = max(dist - 1, DISTANCE_NEAR)

            # Combined: battery may drop AND distance may decrease
            outcomes = {}
            for p_bat, new_bat in [(P_BATTERY_DROP, bat_drop), (1 - P_BATTERY_DROP, bat)]:
                for p_dist, new_dist in [(P_RTH_PROGRESS, dist_close), (1 - P_RTH_PROGRESS, dist)]:
                    ns = state_to_idx(new_bat, new_dist)
                    outcomes[ns] = outcomes.get(ns, 0.0) + p_bat * p_dist
            T[s][a] = [(prob, ns) for ns, prob in outcomes.items()]

        # ── LAND_NOW ─────────────────────────────────────────────────────────
        a = ACTION_LAND_NOW
        if dist == DISTANCE_NEAR:
            R[s, a] = REWARD_SAFE_COMPLETION
        else:
            # Landing away from home is sub-optimal but not a crash
            R[s, a] = REWARD_PER_STEP * 10  # penalty for abandoned mission
        T[s][a] = [(1.0, TERMINAL_STATE)]

    return T, R
