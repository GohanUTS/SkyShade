"""
Sub-3 Environmental Decision — SVM training worker.

Trains the umbrella-deploy SVM on data/env_sensor_log.csv in a background
thread so the Training Grounds hub stays responsive.  Training completes in
< 2 seconds on 100 samples.

Queue message format
────────────────────
("progress", msg: str)
("done",  model_path: str, cv_accuracy: float, confusion_matrix: np.ndarray)
("error", traceback_tail: str)
"""

import os
import json
import pickle
import queue
import threading
import time
import traceback

import numpy as np

_MODELS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
_DATA_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                             "env_sensor_log.csv")


class SVMTrainingWorker(threading.Thread):
    """Train the umbrella-decision SVM and report accuracy + confusion matrix."""

    OUTPUT_PATH = os.path.join(_MODELS_DIR, "svm_v1.pkl")

    def __init__(self, progress_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self._q    = progress_queue
        self._stop_event = stop_event

    def run(self):
        try:
            import csv as _csv
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            from sklearn.metrics import confusion_matrix
            from sub3_env.feature_engineering import FeatureBuilder

            self._q.put(("progress", "Loading training data…"))
            if not os.path.exists(_DATA_PATH):
                self._q.put(("error", f"Data file not found: {_DATA_PATH}"))
                return

            # ── Load CSV ──────────────────────────────────────────────────────
            lux_arr, rain_arr, wind_arr, labels = [], [], [], []
            with open(_DATA_PATH, newline="") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    lux_arr.append(float(row["lux"]))
                    rain_arr.append(float(row["rain_raw"]))
                    wind_arr.append(float(row["wind_speed"]))
                    labels.append(int(row["label"]))

            lux   = np.array(lux_arr);  rain = np.array(rain_arr)
            wind  = np.array(wind_arr); y    = np.array(labels)

            self._q.put(("progress",
                         f"Loaded {len(y)} samples  "
                         f"(deploy={y.sum()}, stow={(y==0).sum()})"))

            # ── Build 9-D feature matrix ──────────────────────────────────────
            lux_d  = np.diff(lux,  prepend=lux[0])
            rain_d = np.diff(rain, prepend=rain[0])
            wind_d = np.diff(wind, prepend=wind[0])
            prev1  = np.roll(y, 1); prev1[0] = 0
            prev2  = np.roll(y, 2); prev2[:2] = 0
            prev3  = np.roll(y, 3); prev3[:3] = 0

            X = np.column_stack([lux, rain, wind,
                                  lux_d, rain_d, wind_d,
                                  prev1, prev2, prev3]).astype(np.float32)

            # ── Train SVM ─────────────────────────────────────────────────────
            self._q.put(("progress", "Running 10-fold cross-validation…"))
            clf = Pipeline([("scaler", StandardScaler()),
                            ("svm",    SVC(kernel="rbf", C=1.0, gamma="scale"))])

            cv     = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            acc    = float(scores.mean())

            self._q.put(("progress",
                         f"10-fold CV accuracy: {acc:.1%} ± {scores.std():.1%}"))

            # Fit on full dataset
            clf.fit(X, y)
            cm = confusion_matrix(y, clf.predict(X))

            # ── Save model ────────────────────────────────────────────────────
            os.makedirs(_MODELS_DIR, exist_ok=True)
            with open(self.OUTPUT_PATH, "wb") as f:
                pickle.dump(clf, f)
            with open(self.OUTPUT_PATH + ".meta.json", "w", encoding="utf-8") as f:
                json.dump({
                    "subsystem": "Sub-3 Weather",
                    "algorithm": "SVM RBF",
                    "model": os.path.basename(self.OUTPUT_PATH),
                    "samples": int(len(y)),
                    "cv_accuracy": float(acc),
                    "trained_at": time.time(),
                }, f, indent=2)

            self._q.put(("done", self.OUTPUT_PATH, acc, cm))

        except Exception:
            self._q.put(("error", traceback.format_exc()[-400:]))
