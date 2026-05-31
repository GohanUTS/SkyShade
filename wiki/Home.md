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
| [Using the GUI](Using-the-GUI) | **How to use the Launcher, Training Grounds hub, and Simulation dashboard** |
| [Training Grounds](Training-Grounds) | **Full guide to every tab, chart, button, and visual overlay in the hub** |
| [Training the Models](Training-the-Models) | CLI commands and hyperparameter reference for re-training |
| [Running the Simulation](Running-the-Simulation) | Prerequisites, launch options, troubleshooting |
| [Sub-1: Perception](Sub-1-Perception) | HSV tracking, distance estimation, EMA, calibration room |
| [Sub-2: Flight Control](Sub-2-Flight-Control) | PPO hover agent, training curriculum, PID fallback |
| [Sub-3: Environmental Decision](Sub-3-Environmental-Decision) | SVM umbrella classifier |
| [Sub-4: Navigation and Safety](Sub-4-Navigation-Safety) | MDP battery-safety + PPO obstacle navigation |
| [Validation and Results](Validation-and-Results) | Metrics, targets, evaluation workflow |
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

| Card | Button | Time |
|---|---|---|
| Sub-1 Perception | Calibrate | Instant — no training needed |
| Sub-2 Flight | Train PPO | ~8 min first run; fine-tuned on repeat |
| Sub-4 Nav Safety | Solve MDP | < 1 sec (battery MDP) + ~3 min (nav PPO) |

See [Training Grounds](Training-Grounds) for a full guide to every control and visual in the hub, and [Using the GUI](Using-the-GUI) for step-by-step instructions.
