"""
Sub-3 Environmental Decision — runtime SVM classifier with hysteresis.

Wraps the trained sklearn Pipeline and applies a hysteresis filter so the
umbrella does not toggle on every noisy reading.  The decision only flips when
the same new action has been recommended for HYSTERESIS_WINDOW consecutive
frames.

Usage (from env_decision_node.py):
    clf = UmbrellaClassifier()
    action = clf.predict(lux, rain_raw, wind_speed)  # Returns 0 (stow) or 1 (deploy)
"""

import os
import pickle
import numpy as np
from sub3_env.feature_engineering import FeatureBuilder

HYSTERESIS_WINDOW = 3   # Frames before flipping deploy/stow decision

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "svm_v1.pkl"
)


class UmbrellaClassifier:
    """Stateful SVM wrapper with built-in hysteresis and feature accumulation."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"SVM model not found at {model_path}. "
                "Run sub3_env/train_svm.py first."
            )
        with open(model_path, "rb") as f:
            self._pipeline = pickle.load(f)

        self._feature_builder = FeatureBuilder()
        self._current_action = 0          # Committed umbrella state
        self._candidate_action = 0        # Proposed new action
        self._candidate_count = 0         # Consecutive frames recommending candidate

    def predict(self, lux: float, rain_raw: float, wind_speed: float) -> int:
        """
        Return the umbrella action (0=stow, 1=deploy) after applying hysteresis.

        Also updates the feature builder's action history with the committed
        decision so future predictions are context-aware.
        """
        feat = self._feature_builder.build(lux, rain_raw, wind_speed)
        raw = int(self._pipeline.predict(feat.reshape(1, -1))[0])

        if raw == self._current_action:
            # Prediction agrees with current state — reset candidate tracker
            self._candidate_action = raw
            self._candidate_count = 0
        elif raw == self._candidate_action:
            # Another vote for the same new action
            self._candidate_count += 1
            if self._candidate_count >= HYSTERESIS_WINDOW:
                # Hysteresis threshold reached — commit the flip
                self._current_action = raw
                self._candidate_count = 0
        else:
            # New proposal — start fresh candidate window
            self._candidate_action = raw
            self._candidate_count = 1

        self._feature_builder.push_action(self._current_action)
        return self._current_action

    @property
    def umbrella_state(self) -> str:
        return "DEPLOY" if self._current_action == 1 else "STOW"

    def reset(self):
        self._feature_builder.reset()
        self._current_action = 0
        self._candidate_action = 0
        self._candidate_count = 0
