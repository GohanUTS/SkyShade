# Sub-3: Environmental Decision

## Responsibility

Read simulated weather sensors once per second and decide whether the umbrella canopy should be `DEPLOY`ed or `STOW`ed.

---

## How it works

1. Three raw sensor readings are sampled: lux (light level), rain intensity, wind speed
2. Per-reading deltas (change since last reading) are computed for each sensor
3. The last three umbrella decisions are appended as hysteresis features
4. The resulting 9-D feature vector is standardised and passed to the trained SVM
5. A 3-frame hysteresis window prevents rapid flicker — the decision only flips after three consecutive opposing predictions
6. The result (`DEPLOY` or `STOW`) is published to `/skyshade/umbrella_cmd`

---

## Feature vector (9-D)

```python
features = [
    lux,            # Current light level
    rain_raw,       # Raw rain sensor reading
    wind_speed,     # Wind speed estimate
    lux_delta,      # Change in lux since last reading
    rain_delta,     # Change in rain since last reading
    wind_delta,     # Change in wind since last reading
    prev_action_t1, # Decision at t-1
    prev_action_t2, # Decision at t-2
    prev_action_t3, # Decision at t-3
]
```

PCA to 3 components is used for visualisation only — the SVM trains and infers on the full 9-D space.

---

## Classifier configuration

| Parameter | Value |
|---|---|
| Kernel | RBF |
| C | 1.0 |
| Gamma | `scale` |
| Cross-validation | 10-fold |
| Target accuracy | ≥ 90% |
| Hysteresis window | 3 frames |
| Update rate | 1 Hz |

---

## Simulated weather cycle

The integrated simulation cycles weather over a 60 s period:

```
clear (high lux, no rain, low wind)
  → cloudy (dropping lux, rising wind)
    → rainy (low lux, rain > 0, gusty wind)
      → clear
```

---

## Validation target

| Metric | Target |
|---|---|
| 10-fold CV accuracy | ≥ 90% |
| False deploy rate | Minimised (see confusion matrix) |

---

## Relevant files

- `sub3_env/train_svm.py` — training script; outputs `models/svm_v1.pkl`
- `sub3_env/feature_engineering.py` — 9-D feature vector builder
- `sub3_env/classifier.py` — runtime SVM wrapper with hysteresis
- `sub3_env/test_env_decision.py` — 5-minute synthetic weather trajectory test
- `data/env_sensor_log.csv` — labelled lux/rain/wind training samples
- `models/svm_v1.pkl` — trained SVM model
- `confusion_matrix.png` — validation output
- `pca_3d.png` — 3-D PCA visualisation
- `ros2_ws/src/skyshade/skyshade/env_decision_node.py` — ROS 2 wrapper
