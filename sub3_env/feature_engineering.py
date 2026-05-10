"""
Sub-3 Environmental Decision — 9-D feature vector builder.

Maintains a rolling window of raw sensor readings and constructs the feature
vector used by both the SVM trainer and the runtime classifier.

Feature vector layout:
  [0] lux            — current light level (lux)
  [1] rain_raw       — raw rain sensor voltage (0–1 normalised)
  [2] wind_speed     — wind speed estimate (m/s)
  [3] lux_delta      — change in lux since last reading
  [4] rain_delta     — change in rain since last reading
  [5] wind_delta     — change in wind since last reading
  [6] prev_action_t1 — decision at t-1  (1=deploy, 0=stow)
  [7] prev_action_t2 — decision at t-2
  [8] prev_action_t3 — decision at t-3
"""

from collections import deque
import numpy as np

FEATURE_DIM = 9
HISTORY_LEN = 3  # How many past actions are tracked


class FeatureBuilder:
    """Stateful feature builder — one instance per session."""

    def __init__(self):
        self._prev_lux = 0.0
        self._prev_rain = 0.0
        self._prev_wind = 0.0
        # Ring buffer of past action decisions (0 or 1)
        self._action_history: deque = deque([0] * HISTORY_LEN, maxlen=HISTORY_LEN)

    def push_action(self, action: int):
        """Call after each decision tick to record the chosen action."""
        self._action_history.appendleft(int(action))

    def build(self, lux: float, rain_raw: float, wind_speed: float) -> np.ndarray:
        """
        Build and return the 9-D feature vector.

        Parameters
        ----------
        lux        : illuminance (lux)
        rain_raw   : rain sensor reading in [0, 1]
        wind_speed : wind speed in m/s

        Returns
        -------
        np.ndarray shape (9,), dtype float32
        """
        lux_delta = lux - self._prev_lux
        rain_delta = rain_raw - self._prev_rain
        wind_delta = wind_speed - self._prev_wind

        self._prev_lux = lux
        self._prev_rain = rain_raw
        self._prev_wind = wind_speed

        # history is newest-first: [t-1, t-2, t-3]
        hist = list(self._action_history)

        return np.array([
            lux,
            rain_raw,
            wind_speed,
            lux_delta,
            rain_delta,
            wind_delta,
            hist[0],
            hist[1],
            hist[2],
        ], dtype=np.float32)

    def reset(self):
        """Reset state between episodes."""
        self._prev_lux = 0.0
        self._prev_rain = 0.0
        self._prev_wind = 0.0
        self._action_history = deque([0] * HISTORY_LEN, maxlen=HISTORY_LEN)
