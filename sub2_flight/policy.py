"""
Sub-2 Flight — Q-table policy wrapper for runtime use.

Loads a pre-trained Q-table from disk and exposes a single `select_action`
method.  The flight_node.py ROS2 node imports this module and calls
`select_action` on every control tick.
"""

import os
import numpy as np

from sub2_flight.env.hover_env import N_STATES, N_ACTIONS

ACTION_NAMES = ["MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST",
                "MOVE_UP", "MOVE_DOWN", "HOLD"]

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "qtable_v1.npy"
)


class FlightPolicy:
    """Greedy policy backed by a pre-trained Q-table."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Q-table not found at {model_path}. "
                "Run sub2_flight/train_qlearning.py first."
            )
        self._q_table = np.load(model_path)
        if self._q_table.shape != (N_STATES, N_ACTIONS):
            raise ValueError(
                f"Q-table shape mismatch: expected ({N_STATES}, {N_ACTIONS}), "
                f"got {self._q_table.shape}"
            )

    def select_action(self, state: int) -> int:
        """Return the greedy action for the given state index."""
        return int(np.argmax(self._q_table[state]))

    def action_name(self, action: int) -> str:
        return ACTION_NAMES[action]
