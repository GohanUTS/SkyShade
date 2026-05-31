# SkyShade Wiki

Welcome to the SkyShade project wiki. SkyShade is a fully simulated autonomous drone system that perceives, tracks, and shades a moving user by deploying an umbrella canopy in response to real-time weather conditions.

**Course:** AI for Robotics — UTS, May 2026  
**Team:** Gohan Idrisoglu (Lead) · Dinesh Saravanan · Aaron · Saaranj

---

## Pages

| Page | Description |
|---|---|
| [Project Overview](Project-Overview) | Goals, scope, and system summary |
| [Architecture](Architecture) | How the four subsystems connect over ROS 2 |
| [Sub-1: Perception](Sub-1-Perception) | HSV tracking, distance estimation, EMA smoothing |
| [Sub-2: Flight Control](Sub-2-Flight-Control) | PPO hover agent + PID fallback (Q-learning archived) |
| [Sub-3: Environmental Decision](Sub-3-Environmental-Decision) | SVM umbrella classifier |
| [Sub-4: Navigation and Safety](Sub-4-Navigation-Safety) | MDP battery-safety policy + PPO obstacle navigation |
| [Running the Simulation](Running-the-Simulation) | How to run the full integrated sim |
| [Training the Models](Training-the-Models) | Re-training each subsystem model |
| [Validation and Results](Validation-and-Results) | Metrics, targets, and test outcomes |
| [Known Issues and Decisions](Known-Issues-and-Decisions) | Design decisions, bugs fixed, and current limitations |

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

## Training before first launch

The launcher blocks the sim if trained models are missing. Open it and click:

| Card | Button | What it opens |
|---|---|---|
| Sub-1 Perception | Calibrate | Live PyBullet calibration room (no training needed) |
| Sub-2 Flight | Train PPO | PPO hover training — 3D room with drone + wind, ~30 min |
| Sub-4 Nav Safety | Solve MDP | Battery MDP solver (< 1 sec) + PPO obstacle nav trainer |

Click **Evaluate Model** in the Sub-2 and Sub-4 tabs to validate each model before launching the full simulation.
