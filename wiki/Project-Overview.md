# Project Overview

## What is SkyShade?

Picture a personal umbrella that flies itself. SkyShade is a simulated quadcopter that:

1. **Tracks** you with a downward-facing camera (no GPS — just vision)
2. **Hovers** overhead at a fixed height, holding station through wind gusts
3. **Decides** when to open or close its umbrella from live light, rain and wind readings
4. **Plays it safe** — when the battery runs low it stops following and heads home (or lands on the spot)

Everything runs in **PyBullet** for the physics, with **ROS 2** nodes wiring the four AI subsystems together.

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
| Flight control | PPO policy + PID fallback (Q-learning archived) |
| Obstacle navigation | Soft Actor-Critic (SAC) |
| Weather classification | SVM with RBF kernel |
| Safety policy | MDP + Bellman value iteration |
| Computer vision | OpenCV HSV segmentation |
