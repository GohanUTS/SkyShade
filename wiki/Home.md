# SkyShade Wiki

Welcome to the SkyShade project wiki. SkyShade is a fully simulated autonomous drone system that perceives, tracks, and shades a moving user by deploying an umbrella canopy in response to real-time weather conditions.

**Course:** AI for Robotics — UTS, May 2026  
**Team:** Gohan Idrisoglu (Lead) · Dinesh Saravanan · Aaron · Saaranj

---

## Pages

| Page | Description |
|---|---|
| [Project Overview](Project-Overview) | Goals, scope, and system summary |
| [Architecture](Architecture) | How the four subsystems connect, file layout, control flow |
| [Using the GUI](Using-the-GUI) | **Step-by-step guide to the Launcher, Training Grounds hub, and Simulation dashboard** |
| [Training Grounds](Training-Grounds) | **Full guide to every tab, chart, button, and visual overlay in the hub** |
| [Training the Models](Training-the-Models) | How to train each subsystem, expected times, verify results |
| [Running the Simulation](Running-the-Simulation) | Prerequisites, launch options, post-run efficiency report |
| [Sub-1: Perception](Sub-1-Perception) | HSV tracking, distance estimation, EMA, calibration room |
| [Sub-2: Flight Control](Sub-2-Flight-Control) | PPO hover agent (4 parallel envs), multi-run training chart |
| [Sub-3: Environmental Decision](Sub-3-Environmental-Decision) | SVM umbrella classifier, weather visualisation |
| [Sub-4: Navigation and Safety](Sub-4-Navigation-Safety) | MDP battery-safety + SAC obstacle navigation |
| [Validation and Results](Validation-and-Results) | Metrics, evaluation workflow, post-run efficiency report |
| [Known Issues and Decisions](Known-Issues-and-Decisions) | Design decisions, bugs fixed, future work |

---

## Quick start

```bash
# Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Build ROS 2 workspace
cd ros2_ws && colcon build && source install/setup.bash && cd ..

# Open the launcher — train each subsystem first, then launch
python run_sim.py
```

## Before first launch — train the models

The launcher blocks the sim if trained models are missing:

| Step | Card | Button | Time (first run) | Time (repeat) |
|---|---|---|---|---|
| 1 | Sub-1 Perception | **Calibrate** | Instant — verify tracker only | Instant |
| 2 | Sub-2 Flight | **Train PPO** | ~2 min (4 parallel envs) | ~1 min fine-tune |
| 3 | Sub-3 Weather | **Train SVM** | < 2 sec | < 2 sec |
| 4 | Sub-4 Safety | **Solve MDP** | < 1 sec | < 1 sec |
| 5 | Sub-4 Safety | **Train Navigation** | ~3 min (SAC) | ~1 min fine-tune |

See [Training Grounds](Training-Grounds) for a full guide to every tab and visual, and [Using the GUI](Using-the-GUI) for step-by-step instructions.

## After running the simulation

The terminal prints a full **Efficiency Report** with a grade:

```
════════════════════════════════════════════════════════════
  SkyShade — Post-Run Efficiency Report
════════════════════════════════════════════════════════════
  Sub-2  Hover accuracy   :  73.4%  ✓ good
         Mean hover error  :  0.38 m  ✓ within 0.5m
  Sub-1  Tracker lock     :  98.2%  ✓ reliable
  Sub-3  Umbrella correct :  91.7%  ✓ accurate
  Sub-4  Battery at end   :  42.0%  ✓ safe
════════════════════════════════════════════════════════════
  Overall system score: 87%  Grade: A
════════════════════════════════════════════════════════════
```

See [Validation and Results](Validation-and-Results) for what each metric means and how to improve it.
