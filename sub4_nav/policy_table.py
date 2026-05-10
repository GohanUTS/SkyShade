"""
Sub-4 Navigation and Safety — runtime policy table lookup.

Loads the pre-computed policy array and maps (battery_level, distance) to one
of three override actions.  The nav_safety_node.py ROS2 node imports this and
calls `decide` on every tick.

Battery and distance readings from the simulator are continuous; this module
converts them to the discrete bucket indices the MDP was solved over.
"""

import os
import numpy as np

from sub4_nav.mdp import (
    N_STATES, state_to_idx, idx_to_state, ACTION_NAMES,
    BATTERY_HIGH, BATTERY_MEDIUM, BATTERY_LOW, BATTERY_CRITICAL,
    DISTANCE_NEAR, DISTANCE_MID, DISTANCE_FAR,
    ACTION_CONTINUE, ACTION_RTH, ACTION_LAND_NOW,
)

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "policy_table_v1.npy"
)

# ── Continuous → bucket thresholds ───────────────────────────────────────────
# Battery percentage thresholds (0–100 %)
BAT_THRESHOLDS = [25.0, 50.0, 75.0]   # below 25 → CRITICAL, ..., above 75 → HIGH

# Distance to home thresholds (metres)
DIST_THRESHOLDS = [20.0, 50.0]        # below 20 → NEAR, 20–50 → MID, above 50 → FAR
# ─────────────────────────────────────────────────────────────────────────────


def _battery_to_idx(battery_pct: float) -> int:
    if battery_pct < BAT_THRESHOLDS[0]:
        return BATTERY_CRITICAL
    if battery_pct < BAT_THRESHOLDS[1]:
        return BATTERY_LOW
    if battery_pct < BAT_THRESHOLDS[2]:
        return BATTERY_MEDIUM
    return BATTERY_HIGH


def _distance_to_idx(distance_m: float) -> int:
    if distance_m < DIST_THRESHOLDS[0]:
        return DISTANCE_NEAR
    if distance_m < DIST_THRESHOLDS[1]:
        return DISTANCE_MID
    return DISTANCE_FAR


class NavSafetyPolicy:
    """Runtime wrapper around the MDP policy table."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Policy table not found at {model_path}. "
                "Run sub4_nav/solve_mdp.py first."
            )
        self._policy = np.load(model_path)
        if len(self._policy) != N_STATES:
            raise ValueError(
                f"Policy table length mismatch: expected {N_STATES}, "
                f"got {len(self._policy)}"
            )

    def decide(self, battery_pct: float, distance_m: float) -> int:
        """
        Return the policy action for the given continuous readings.

        Returns one of: ACTION_CONTINUE (0), ACTION_RTH (1), ACTION_LAND_NOW (2)
        """
        bat_idx  = _battery_to_idx(battery_pct)
        dist_idx = _distance_to_idx(distance_m)
        s = state_to_idx(bat_idx, dist_idx)
        return int(self._policy[s])

    def decide_name(self, battery_pct: float, distance_m: float) -> str:
        return ACTION_NAMES[self.decide(battery_pct, distance_m)]
