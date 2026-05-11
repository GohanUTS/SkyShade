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
| [Sub-2: Flight Control](Sub-2-Flight-Control) | PID controller and archived Q-learning agent |
| [Sub-3: Environmental Decision](Sub-3-Environmental-Decision) | SVM umbrella classifier |
| [Sub-4: Navigation and Safety](Sub-4-Navigation-Safety) | MDP safety policy |
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

# Run the full simulation (120 s, GUI)
python run_sim.py
```
