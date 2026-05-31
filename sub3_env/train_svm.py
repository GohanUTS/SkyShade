"""
Sub-3 Environmental Decision — SVM training script.

Reads labelled sensor data from a CSV, builds 9-D feature vectors, trains an
RBF SVM classifier, and saves the fitted model artefact.

Outputs:
  • <output>               — pickled sklearn Pipeline (StandardScaler + SVC)
  • confusion matrix plot  — displayed via matplotlib
  • 10-fold CV accuracy    — printed to stdout
  • PCA 3-D scatter plot   — displayed via matplotlib (visualisation only)

Usage:
    python sub3_env/train_svm.py \
        --data data/env_sensor_log.csv \
        --output models/svm_v1.pkl

CSV format (columns in any order, must include header row):
    lux, rain_raw, wind_speed, label
where label is 1 (deploy umbrella) or 0 (stow umbrella).

The feature builder converts raw readings into the 9-D vector by computing
deltas and appending three prior action labels from training history.  During
training, prev_action features are synthesised from the ground-truth labels to
approximate deployment history.
"""

import argparse
import csv
import os
import sys
import pickle
import tempfile
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "skyshade_matplotlib"))
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — swap to "TkAgg" for pop-ups
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── Classifier config ─────────────────────────────────────────────────────────
SVM_KERNEL = "rbf"
SVM_C = 1.0
SVM_GAMMA = "scale"
PCA_COMPONENTS = 3       # Used for 3-D visualisation only; SVM trains on full 9-D
CV_FOLDS = 10
TARGET_ACC = 0.90
HYSTERESIS_WINDOW = 3    # Kept here for reference; runtime use is in classifier.py
# ─────────────────────────────────────────────────────────────────────────────


def load_data(csv_path: str):
    required = {"lux", "rain_raw", "wind_speed", "label"}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        missing = required - columns
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        data = {name: [] for name in required}
        for row in reader:
            data["lux"].append(float(row["lux"]))
            data["rain_raw"].append(float(row["rain_raw"]))
            data["wind_speed"].append(float(row["wind_speed"]))
            data["label"].append(int(row["label"]))

    return {name: np.asarray(values) for name, values in data.items()}


def build_features(df: dict) -> tuple:
    """
    Construct the 9-D feature matrix from the CSV.

    Deltas are computed row-to-row (first row has zero deltas).
    prev_action features are approximated from the shifted ground-truth label.
    """
    lux = np.asarray(df["lux"], dtype=float)
    rain = np.asarray(df["rain_raw"], dtype=float)
    wind = np.asarray(df["wind_speed"], dtype=float)
    labels = np.asarray(df["label"], dtype=int)

    lux_delta = np.diff(lux, prepend=lux[0])
    rain_delta = np.diff(rain, prepend=rain[0])
    wind_delta = np.diff(wind, prepend=wind[0])

    # Approximate historical actions from ground-truth labels (training only)
    prev1 = np.roll(labels, 1); prev1[0] = 0
    prev2 = np.roll(labels, 2); prev2[:2] = 0
    prev3 = np.roll(labels, 3); prev3[:3] = 0

    X = np.column_stack([
        lux, rain, wind,
        lux_delta, rain_delta, wind_delta,
        prev1, prev2, prev3,
    ]).astype(np.float32)

    return X, labels


def plot_confusion(clf, X, y):
    cm = confusion_matrix(y, clf.predict(X))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stow", "Deploy"])
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    ax.set_title("SVM Confusion Matrix (training set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100)
    print("Confusion matrix saved → confusion_matrix.png")


def plot_pca(X, y):
    pca = PCA(n_components=PCA_COMPONENTS)
    Xr = pca.fit_transform(X)
    series = [(0, "Stow", "steelblue"), (1, "Deploy", "tomato")]

    try:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for label, name, colour in series:
            mask = y == label
            ax.scatter(Xr[mask, 0], Xr[mask, 1], Xr[mask, 2],
                       label=name, c=colour, s=10, alpha=0.6)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.set_title("PCA 3-D — weather sensor feature space")
    except Exception as exc:
        plt.close("all")
        print(f"3-D PCA plot unavailable ({exc}); falling back to 2-D.")
        fig, ax = plt.subplots()
        for label, name, colour in series:
            mask = y == label
            ax.scatter(Xr[mask, 0], Xr[mask, 1],
                       label=name, c=colour, s=10, alpha=0.6)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title("PCA 2-D — weather sensor feature space")

    ax.legend()
    plt.tight_layout()
    plt.savefig("pca_3d.png", dpi=100)
    print("PCA plot saved → pca_3d.png")


def train(csv_path: str, output_path: str):
    df = load_data(csv_path)
    X, y = build_features(df)
    print(f"Loaded {len(X)} samples  (deploy={y.sum()}, stow={(y==0).sum()})")

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA)),
    ])

    # 10-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    mean_acc = scores.mean()
    print(f"\n{CV_FOLDS}-fold CV accuracy: {mean_acc:.4f} ± {scores.std():.4f}")

    if mean_acc < TARGET_ACC:
        print(f"WARNING: accuracy {mean_acc:.2%} < target {TARGET_ACC:.0%}. "
              "Consider tuning C / class_weight='balanced'.")
    else:
        print(f"Accuracy target met ({mean_acc:.2%} >= {TARGET_ACC:.0%}).")

    # Fit on full dataset for deployment
    clf.fit(X, y)

    try:
        plot_confusion(clf, X, y)
        plot_pca(X, y)
    except Exception as exc:
        print(f"WARNING: plot generation failed ({exc}); continuing to save model.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nSVM pipeline saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Sub-3 SVM classifier")
    parser.add_argument("--data",   default="data/env_sensor_log.csv")
    parser.add_argument("--output", default="models/svm_v1.pkl")
    args = parser.parse_args()
    train(args.data, args.output)


if __name__ == "__main__":
    main()
