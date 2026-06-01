# SkyShade Assessment README

## Project Title

**SkyShade: Autonomous Drone Umbrella Simulation**

SkyShade is a simulated intelligent robot project where a drone follows a moving user and deploys an umbrella canopy when rain conditions are detected. The project is built in PyBullet and Python, with a live dashboard that shows the drone state, weather state, umbrella state, battery level, tracking confidence, and subsystem test controls.

This README is written for the final in-class presentation and can be used as source material for PowerPoint slides.

---

## Team

| Role | Team Member | Notes |
|---|---|---|
| Team lead / integration | Gohan Idrisoglu | Main simulation runner, GUI integration, debugging, system assembly |
| Team member | Dinesh Saravanan | Subsystem development and testing |
| Team member | Aaron | Must join the correct Canvas team so marking can identify the right Aaron |
| Team member | Saaranj | Subsystem development and testing |

### Canvas Admin Note

There are two students named Aaron in the subject. Aaron needs to join the correct team on Canvas, or the team needs to tell the tutor which Aaron should be marked. The safest detail to provide is Aaron's full name and student ID/email in the team submission.

---

## Assessment Task 2 Focus

For Task 2, the team presents a final in-class demonstration with slides covering:

1. What the project is and what the team is aiming to solve.
2. What the team has tried so far.
3. What has been successful at this stage.
4. What challenges remain.

The demonstration is simulation-only, which is acceptable for the project. The goal is to show an AI-controlled robot operating in a simulated environment and explain the design decisions, technical work, and lessons learned.

---

## 1. Project Presentation: What Are We Aiming To Solve?

### Problem

The problem SkyShade addresses is personal weather protection using an autonomous robot. In real life, a person may need shade or rain cover while walking outdoors, but carrying an umbrella can be inconvenient. SkyShade explores the idea of a drone that follows a user and automatically deploys an umbrella canopy when rain is detected.

The project is not trying to build a physical flying umbrella. Instead, it uses simulation to test the AI pipeline safely and quickly. Simulation lets the team test perception, flight control, weather decision-making, and safety logic without damaging hardware or needing outdoor flight approval.

### Project Aim

The team aims to demonstrate a simulated drone that can:

- Track a moving user represented by a red ball.
- Follow the user smoothly in PyBullet.
- Detect random rain events in the simulated environment.
- Deploy or stow an umbrella canopy based on weather readings.
- Show the current robot and environment state through a GUI dashboard.
- Apply safety logic such as battery monitoring and return-to-home behaviour.

### Final Demonstration Goal

The final demonstration should show:

- A PyBullet simulation window with the drone, user ball, umbrella canopy, and rain.
- A dashboard window showing mission state, weather state, umbrella state, tracking confidence, battery level, and validation buttons.
- Random rain occurring from time to time across the map.
- The dashboard reporting when the user ball is inside the rain.
- The umbrella changing state when the weather classifier decides rain protection is needed.

---

## 2. System Overview

SkyShade is divided into four main subsystems.

| Subsystem | Purpose | Current Approach |
|---|---|---|
| Sub-1 Perception | Detect and track the user | RGB camera frame from PyBullet, HSV red-marker tracking, distance estimation |
| Sub-2 Flight Control | Move the drone to follow the user | PID controller for stable runtime control; Q-learning kept as a training/learning experiment |
| Sub-3 Environment Decision | Decide whether to deploy the umbrella | SVM classifier using light, rain, and wind readings with hysteresis |
| Sub-4 Navigation Safety | Decide whether to continue, return home, or land | MDP policy table solved with value iteration |

### Runtime Architecture

```text
PyBullet simulation
  |
  |-- RGB camera image
  |-- Drone state
  |-- User position
  |-- Random rain / weather readings
  |
  v
Sub-1 Perception ---> user offset + confidence
  |
  v
Sub-2 Flight Control ---> drone force command

Sub-3 Environment Decision ---> DEPLOY / STOW umbrella

Sub-4 Navigation Safety ---> CONTINUE / RTH / LAND_NOW

Telemetry GUI ---> live dashboard + subsystem test buttons
```

---

## 3. What The Team Has Tried So Far

### Simulation Environment

The team built a complete simulation in PyBullet. The environment includes:

- A drone represented by a small blue body.
- A red user ball that moves along a figure-eight path.
- A flat umbrella canopy attached above the drone.
- Random weather readings for light, rain, and wind.
- Random rain events drawn in the PyBullet GUI.
- Battery drain over time.
- A separate Tkinter dashboard for telemetry.

This simulation-only approach became the main project direction because it is reliable, safe, and acceptable for the assessment.

### Perception

The team originally considered more complex perception methods, but the current simulation uses a simpler RGB camera pipeline:

1. PyBullet renders a camera image from the drone.
2. The user is represented by a red marker/ball.
3. HSV colour thresholding isolates the red marker.
4. The tracker finds the marker centroid.
5. A distance estimator converts image position into an approximate user offset.
6. Exponential moving average smoothing reduces jitter.

This is simpler and more appropriate than using heavy deep-learning perception for a controlled simulated environment.

### Flight Control

The team tried Q-learning for flight control. The Q-learning script trains a tabular policy in curriculum stages:

1. Stationary user with no wind.
2. Stationary user with gusty wind.
3. Moving user with gusty wind.

However, the tabular Q-learning approach was difficult for smooth drone control because the drone has momentum in PyBullet. A discrete position-only state table does not represent velocity well enough, so the drone can overshoot or oscillate.

For the final runtime demonstration, the team uses a PID controller because it is more stable and easier to explain. The Q-learning component remains useful as evidence of experimentation and learning, but the final demo prioritises a working intelligent robot over unnecessary complexity.

### Environment Decision

The team trained an SVM classifier for umbrella decisions. The classifier reads simulated weather values:

- Light level.
- Rain sensor value.
- Wind speed.
- Recent changes in those readings.
- Previous umbrella actions for hysteresis.

The output is:

- `STOW` when rain protection is not needed.
- `DEPLOY` when rain protection is needed.

Hysteresis is included so the umbrella does not rapidly flicker between deploy and stow when the readings are noisy.

### Navigation And Safety

The team built a simple MDP safety layer. It considers:

- Battery level.
- Distance from home.

It chooses one of:

- `CONTINUE`: keep following the user.
- `RTH`: return to home.
- `LAND_NOW`: land immediately.

The MDP policy is solved offline with value iteration and used at runtime as a safety override.

### GUI And Demonstration Tools

The team added a dashboard to make the demo easier to understand. The dashboard shows:

- Mission state.
- Battery percentage.
- Umbrla state.
- Runtime.
- Drone action.
- Distance from user.
- Camera confidence.
- Weather reading.
- Whether the user ball is inside rain.
- Position summaries.
- Subsystem status.
- Buttons to run subsystem validation tests.

This improves communication because the audience does not need to interpret raw terminal logs.

---

## 4. Current Successes

### Working End-To-End Simulation

The main success is that the project now runs as one integrated simulation. The drone, user, weather logic, umbrella logic, safety logic, and dashboard all operate together.

### Stable Drone Following

The PID controller gives smoother behaviour than the early Q-learning runtime controller. This makes the final demo more reliable because the drone can follow the moving user without needing perfect reinforcement-learning convergence.

### Simplified Perception

Using an RGB camera and colour tracking made the perception pipeline easier to debug and explain. This responds directly to feedback that an RGB camera could simplify the perception methods.

### Random Rain Simulation

Rain now occurs randomly from time to time. It is drawn across the whole map, starts higher in the sky, and falls for longer so it is more visible. The GUI reports when the user ball is inside the rain.

### Clear Dashboard

The GUI dashboard makes the project presentation stronger because it shows the audience what each subsystem is doing:

- The drone state is visible.
- The weather state is visible.
- The umbrella decision is visible.
- The rain/user interaction is visible.
- Test buttons show that the system is being validated.

### Honest Simplification

The team responded to feedback by simplifying the methodology. The project no longer tries to present too many advanced AI components as equally central. The final system focuses on a small number of working methods:

- HSV camera tracking for perception.
- PID control for runtime flight.
- SVM classification for umbrella deployment.
- MDP safety policy for battery/navigation decisions.

This is easier to defend than claiming the system depends on linear regression, MLP, DQN, PPO, PCA, SVM, value iteration, and Q-learning all at once.

---

## 5. Current Challenges

### Drone Control Is Still Hard

Drone control is challenging because PyBullet physics includes gravity, inertia, damping, and momentum. The team learned that a discrete Q-table can be hard to use for smooth continuous movement unless the state includes enough information such as velocity.

The final demo uses PID for reliability, while Q-learning remains a training experiment. A future improvement would be to either:

- Add velocity to the Q-learning state space.
- Use a continuous-control RL method.
- Continue using PID as the main low-level controller and reserve AI for higher-level decisions.

### Rain Visual Performance

Rain must be visible but not too laggy. Earlier visual effects used cloud outlines and many debug lines, which slowed down the PyBullet GUI. The team removed the cloud effect and kept only rain streaks.

Current rain is spread across the map and drawn with a controlled number of debug lines. If performance drops, the team can reduce:

- Drop count.
- Line lifetime.
- Draw frequency.

### Methodology Scope

The project originally had too many AI methods. Feedback suggested simplifying the methodology, and this was correct. The current challenge is to explain the project clearly without overclaiming.

The final presentation should say:

- We explored several methods.
- We selected the methods that worked best for the final integrated demo.
- We simplified the final system so it is understandable and reliable.

### Team And Submission Admin

Aaron must join the correct Canvas team. Because there are two Aarons in the subject, the team should identify Aaron clearly using full name and student ID/email.

### Presentation Audio

Feedback said the background music made the teaser video difficult to understand. For the final presentation/video, the team should remove background music or keep it extremely quiet so speech is clear.

---

## 6. Response To Feedback From The Teaser Video

### Feedback: "Can you tell Aaron to join your team on Canvas?"

Response:

Aaron needs to join the correct Canvas team before marking. Since there are two Aarons in the subject, the team should provide Aaron's full name and student ID/email to avoid marking confusion.

Action:

- Ask Aaron to join the Canvas group.
- Confirm with the tutor which Aaron is in the team.

### Feedback: "Remove the music in the background."

Response:

The team will remove background music from future videos. The final presentation/video should prioritise clear speech and clear explanation.

Action:

- Use no music in the final video.
- If music is used at all, keep it very low and only during non-speaking sections.

### Feedback: "Simulation only is totally fine for the project."

Response:

The team will focus on a strong simulation-only project. This is appropriate because drone hardware is risky, expensive, and difficult to test safely in the available timeframe.

Action:

- Present the PyBullet simulation as the main project deliverable.
- Explain why simulation is suitable for AI robotics testing.

### Feedback: "If you are planning to use a RGB camera, you could simplify a bunch of the perception methods."

Response:

The team simplified perception to RGB camera tracking with HSV colour thresholding. The red user marker is easy to detect in simulation, so deep-learning object detection is unnecessary for the current scope.

Action:

- Explain HSV tracking clearly in the slides.
- Avoid overcomplicating perception with methods that are not needed for the demo.

### Feedback: "The methodology is a bit too complex."

Response:

The team has simplified the final methodology. Earlier exploration included more AI ideas, but the final demo focuses on fewer components that work together:

- HSV tracking.
- PID flight control.
- SVM weather decision.
- MDP safety policy.

The team should mention Q-learning as something tested, not as the main runtime controller.

Action:

- Do not claim the final system uses every method equally.
- Make the final architecture simple and defensible.

### Feedback: "If you are struggling with drone control, consider curriculum learning."

Response:

The Q-learning training script uses curriculum stages: easier conditions first, then wind, then a moving user. This helped structure training, but the team still chose PID for runtime stability.

Action:

- Mention curriculum learning as an experiment used during training.
- Explain that PID was chosen for final runtime control because it was more stable in PyBullet.

---

## 7. Final Methodology Summary

The final methodology is intentionally simplified.

### Perception: RGB + HSV Tracking

Input:

- RGB camera frame from the simulated drone camera.

Processing:

- Convert RGB to HSV.
- Threshold red hue range.
- Find largest contour.
- Estimate marker position.
- Smooth with exponential moving average.

Output:

- User offset.
- Tracking confidence.

Reason for choice:

- It is simple, fast, explainable, and suitable for a controlled simulation.

### Flight: PID Runtime Control

Input:

- Drone position.
- Drone velocity.
- Target user position.

Processing:

- Compute position error.
- Apply proportional, integral, and derivative terms.
- Clamp force values to keep movement stable.

Output:

- Force applied to the drone in PyBullet.

Reason for choice:

- PID gives smooth and stable control in a continuous physics environment.

### Environment Decision: SVM

Input:

- Light.
- Rain.
- Wind.
- Recent changes in readings.
- Previous umbrella decisions.

Processing:

- Build a feature vector.
- Run SVM classifier.
- Apply hysteresis to avoid rapid toggling.

Output:

- `DEPLOY` or `STOW`.

Reason for choice:

- SVM is suitable for a small structured sensor dataset and is easy to explain.

### Navigation Safety: MDP

Input:

- Battery bucket.
- Distance-from-home bucket.

Processing:

- Runtime lookup from an offline value-iteration policy table.

Output:

- `CONTINUE`, `RTH`, or `LAND_NOW`.

Reason for choice:

- MDPs are suitable for discrete safety decisions under uncertainty.

---

## 8. How To Run The Demo

From the project root:

```bash
source venv/bin/activate
python run_sim.py
```

Or without activating the virtual environment:

```bash
venv/bin/python run_sim.py
```

To run without GUI for quick testing:

```bash
venv/bin/python run_sim.py --duration 10 --no-gui
```

Expected result:

- A PyBullet window opens.
- A SkyShade Dashboard window opens.
- The drone follows the red user ball.
- Rain appears randomly from time to time.
- The umbrella deploys/stows based on weather decisions.
- The dashboard reports whether the user ball is inside rain.

---

## 9. Validation And Testing

The dashboard includes buttons for subsystem tests. Tests can also be run manually:

```bash
venv/bin/python sub1_perception/test_perception.py
venv/bin/python sub2_flight/test_flight.py
venv/bin/python sub3_env/test_env_decision.py
venv/bin/python sub4_nav/test_nav_safety.py
```

### What The Tests Show

| Test | Purpose |
|---|---|
| Perception test | Checks whether the red marker can be detected and tracked |
| Flight test | Checks whether the drone can hover/follow within acceptable error |
| Environment test | Checks whether weather readings produce sensible umbrella decisions |
| Navigation test | Checks safety decisions for battery and return-home cases |

---

## 10. Suggested Final Presentation Structure

### Slide 1: Title

**SkyShade: Autonomous Drone Umbrella Simulation**

Include:

- Team name and members.
- Course name.
- Final demonstration date.

Speaker point:

SkyShade is a simulation-only AI robotics project where a drone follows a user and deploys an umbrella when rain is detected.

### Slide 2: Problem And Aim

Include:

- Problem: hands-free personal rain/shade protection.
- Aim: demonstrate an intelligent robot that perceives, decides, and acts.
- Scope: simulation-only in PyBullet.

Speaker point:

The aim is not to build risky hardware, but to prove the AI pipeline in a controlled simulated robot.

### Slide 3: Final Demo Overview

Include:

- PyBullet simulation.
- Drone, red user ball, umbrella canopy, rain.
- Dashboard telemetry.

Speaker point:

The live demo shows the robot following the user, detecting weather, and reporting system state in real time.

### Slide 4: System Architecture

Include:

- Perception.
- Flight control.
- Weather decision.
- Navigation safety.
- Dashboard.

Speaker point:

Each subsystem has a clear role, and the final demo integrates them into one running system.

### Slide 5: Perception

Include:

- RGB camera.
- HSV red marker tracking.
- Distance estimation.
- Confidence score.

Speaker point:

The team simplified perception because the simulated environment has a controlled red marker, so HSV tracking is reliable and explainable.

### Slide 6: Flight Control

Include:

- Q-learning was tested.
- PID chosen for final runtime stability.
- Drone follows the moving user.

Speaker point:

Q-learning helped the team understand reinforcement learning, but PID produced the most stable final demo in PyBullet.

### Slide 7: Weather And Umbrella Decision

Include:

- Random rain events.
- Light/rain/wind sensor values.
- SVM classifier.
- Hysteresis.

Speaker point:

The umbrella decision is treated as a classification problem using simulated environmental sensor readings.

### Slide 8: Navigation Safety

Include:

- Battery monitoring.
- Return-home/land logic.
- MDP policy.

Speaker point:

The safety layer prevents the robot from blindly continuing when battery state becomes risky.

### Slide 9: What We Tried

Include:

- PyBullet simulation.
- RGB perception.
- Q-learning curriculum.
- PID flight.
- SVM classification.
- MDP safety.
- Dashboard.

Speaker point:

The team explored several methods, then simplified the final approach to the methods that were reliable and understandable.

### Slide 10: Successes

Include:

- Integrated simulation works.
- Drone follows user.
- Random rain visible.
- GUI shows rain/user status.
- Umbrella deploy/stow decisions work.
- Tests are available.

Speaker point:

The strongest success is the working end-to-end demo rather than any single isolated algorithm.

### Slide 11: Challenges

Include:

- Drone physics and momentum.
- Q-learning instability.
- Visual rain performance.
- Avoiding over-complex methodology.
- Canvas team admin issue.

Speaker point:

The project improved after the team reduced scope and focused on a reliable final demo.

### Slide 12: Response To Feedback

Include:

- Removed/avoid background music.
- Simulation-only accepted.
- Simplified RGB perception.
- Reduced algorithm complexity.
- Used curriculum learning as an experiment.
- Aaron must join the correct Canvas team.

Speaker point:

The team used the teaser feedback to simplify and improve the final project.

### Slide 13: Future Work

Include:

- Better drone model.
- More realistic camera noise.
- Improved rain/weather model.
- Hardware prototype as a long-term goal.
- Continuous-control RL if time allowed.

Speaker point:

The current project proves the concept in simulation, and future work could make the environment and control more realistic.

---

## 11. Short Q&A Answers

### What is the project trying to solve?

SkyShade is trying to solve hands-free personal weather protection by simulating a drone that follows a user and deploys an umbrella when rain is detected.

### Why simulation only?

Simulation is safer, cheaper, easier to test, and accepted for the assessment. It also lets the team test robot intelligence without hardware risk.

### What AI is used?

The final system uses HSV-based perception, an SVM weather classifier, and an MDP safety policy. Q-learning was explored for flight training, but PID is used in the final runtime for stable control.

### Why simplify the methods?

The project originally had too many components. Simplifying made the system easier to debug, easier to present, and more reliable for the final demo.

### Why not use deep learning for perception?

The simulated user is a red marker, so HSV tracking is enough. A deep model would add complexity without improving the controlled demo.

### Why PID instead of Q-learning for flight?

PyBullet drone physics includes momentum, and a simple Q-table did not handle continuous motion smoothly enough. PID uses position and velocity directly, making it more stable for the live demo.

### How does the rain work?

Rain randomly turns on and off. When rain is active, thick blue rain lines are drawn across the map from high in the sky. The GUI reports when the user ball is inside rain.

### How does the umbrella know when to deploy?

The environment classifier reads simulated light, rain, and wind values. It predicts whether the umbrella should deploy, then hysteresis prevents rapid flickering.

### What was the biggest challenge?

The biggest challenge was integrating multiple subsystems while keeping the drone stable and the methodology simple enough to explain clearly.

### What is the main success?

The main success is the integrated live simulation: the drone follows the user, weather changes randomly, the umbrella responds, and the dashboard explains the system state.

---

## 12. Final Project Summary

SkyShade demonstrates an AI robotics concept in simulation: a drone can track a user, react to weather, and make safety-aware decisions. The project began with a broad set of AI ideas, but the team improved it by simplifying the final methodology and focusing on a reliable demonstration.

The final system uses:

- PyBullet for simulation.
- RGB/HSV tracking for perception.
- PID control for stable drone following.
- SVM classification for umbrella deployment.
- MDP value iteration for navigation safety.
- A dashboard for communication and validation.

This makes the project suitable for the final Task 2 presentation because it shows a working intelligent robot, explains clear engineering decisions, and directly responds to feedback from the teaser video.

