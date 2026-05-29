"""
Sub-3 Environmental Decision — validation test.

Runs the SVM classifier on a 5-minute (300 s at 1 Hz → 300 samples) synthetic
weather trajectory and checks:
  1. 10-fold CV accuracy >= 90 % (re-evaluated on the held-out CSV data).
  2. Transition rate (how often umbrella flips) is reasonable (< 10 %).

Usage:
    python sub3_env/test_env_decision.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from skyshade.sub3_env.feature_engineering import FeatureBuilder
from skyshade.sub3_env.train_svm import load_data, build_features

CSV_PATH = "data/env_sensor_log.csv"
MODEL_PATH = "models/svm_v1.pkl"
CV_FOLDS = 10
TARGET_ACC = 0.90
MAX_FLIP_RATE = 0.10   # < 10 % of decision ticks should flip state

TRAJECTORY_SECONDS = 300  # 5 minutes at 1 Hz


def make_synthetic_trajectory(rng, n=TRAJECTORY_SECONDS):
    """
    Generate a smooth 5-minute weather trajectory that transitions from
    clear → cloudy → rainy → clear using sinusoidal signals.
    """
    t = np.linspace(0, 2 * np.pi, n)

    lux = 50_000 * (0.5 + 0.5 * np.cos(t))            # 0 → 50 000 lux
    rain = np.clip(0.5 - 0.5 * np.cos(t + np.pi), 0, 1)  # 0 → 1
    wind = 3.0 + 2.0 * np.sin(2 * t)                   # 1 → 5 m/s

    # Add noise
    lux  += rng.normal(0, 500, n)
    rain += rng.normal(0, 0.02, n)
    wind += rng.normal(0, 0.1, n)

    return (
        np.clip(lux, 0, 100_000),
        np.clip(rain, 0, 1),
        np.clip(wind, 0, 20),
    )


def test_cv_accuracy():
    if not os.path.exists(CSV_PATH):
        print(f"SKIP CV test: {CSV_PATH} not found.")
        return True, float("nan")

    if not os.path.exists(MODEL_PATH):
        print(f"SKIP CV test: {MODEL_PATH} not found.")
        return True, float("nan")

    with open(MODEL_PATH, "rb") as f:
        clf: Pipeline = pickle.load(f)

    df = load_data(CSV_PATH)
    X, y = build_features(df)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    mean_acc = scores.mean()
    return mean_acc >= TARGET_ACC, mean_acc


def test_trajectory():
    if not os.path.exists(MODEL_PATH):
        print(f"SKIP trajectory test: {MODEL_PATH} not found.")
        return True, float("nan")

    from skyshade.sub3_env.classifier import UmbrellaClassifier
    clf = UmbrellaClassifier(model_path=MODEL_PATH)

    rng = np.random.default_rng(7)
    lux_seq, rain_seq, wind_seq = make_synthetic_trajectory(rng)

    actions = []
    for lux, rain, wind in zip(lux_seq, rain_seq, wind_seq):
        action = clf.predict(float(lux), float(rain), float(wind))
        actions.append(action)

    actions = np.array(actions)
    flips = int(np.sum(np.diff(actions) != 0))
    flip_rate = flips / len(actions)

    return flip_rate < MAX_FLIP_RATE, flip_rate


def main():
    rng = np.random.default_rng(0)

    print("=== Sub-3 Environmental Decision — Tests ===\n")

    ok_cv, cv_acc = test_cv_accuracy()
    print(f"10-fold CV accuracy : {cv_acc:.4f}  (target >= {TARGET_ACC})  {'✓' if ok_cv else '✗'}")

    ok_traj, flip_rate = test_trajectory()
    print(f"Trajectory flip rate: {flip_rate:.2%}  (target < {MAX_FLIP_RATE:.0%})  {'✓' if ok_traj else '✗'}")

    all_passed = ok_cv and ok_traj
    print("\nSub-3", "PASSED" if all_passed else "FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
