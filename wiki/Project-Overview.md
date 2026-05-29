# Project Overview

## What is SkyShade?

SkyShade is a fully simulated autonomous quadcopter that:

1. **Tracks** a moving user using a downward-facing virtual camera
2. **Hovers** above the user at a fixed altitude while compensating for wind
3. **Decides** whether to deploy or stow an umbrella canopy based on live weather sensor readings
4. **Overrides** normal flight and returns home or lands immediately when battery is critically low

The simulation runs entirely in **PyBullet** for physics, with **ROS 2** nodes handling communication between the four AI subsystems.

---

## Problem being solved

Traditional umbrellas require constant manual attention. SkyShade removes that burden by giving the user hands-free weather protection — the drone follows them and makes its own shade/no-shade decisions.

---

## Design requirements

| # | Requirement | Target |
|---|---|---|
| 1 | Track user from downward-facing camera only (no GPS) | Position MAE < 0.15 m |
| 2 | Maintain hover above user despite wind | Lateral error ≤ 0.5 m |
| 3 | Classify weather → DEPLOY / STOW umbrella | ≥ 90% CV accuracy |
| 4 | Return home or land safely on low battery | Trigger within 1 decision tick |

---

## Scope

- **In scope:** simulation only, four AI subsystems, ROS 2 integration, offline model training
- **Out of scope:** real hardware, GPS, multi-user tracking, outdoor deployment

---

## Technology stack

| Component | Technology |
|---|---|
| Physics simulation | PyBullet |
| Robot middleware | ROS 2 Humble |
| Language | Python 3.10 |
| Flight control | PID (Q-learning archived) |
| Weather classification | SVM with RBF kernel |
| Safety policy | MDP + Bellman value iteration |
| Computer vision | OpenCV HSV segmentation |
