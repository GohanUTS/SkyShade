# SkyShade — Video Presentation Script
**Target Duration: 10–15 minutes**

---

## SECTION 1 — INTRODUCTION (~ 2 min)

**[SCREEN: Show the PyBullet simulation running — drone hovering over a walking user, umbrella deployed in rain]**

> "What you're looking at is SkyShade — a fully autonomous drone simulation that follows a person, reads the weather, and decides when to open and close an umbrella canopy overhead. No human pilot. No remote control. Every decision is made by an AI system built from scratch."

**[SCREEN: Cut to architecture diagram]**

> "So why did we build this?
>
> The idea came from a simple question: could you train a drone to intelligently shade someone — not just fly near them, but actually perceive the environment, track the user, assess conditions, and act? It's a problem that sits at the intersection of computer vision, reinforcement learning, classical AI planning, and robotics middleware. That combination made it the perfect capstone project.
>
> More practically, it forced us to solve real engineering problems: how do you fuse sensor data from multiple sources into coherent decisions? How do you handle safety overrides that must work regardless of what the learning agent is doing? And how do you train four completely different AI models — using four completely different algorithms — and make them talk to each other cleanly?"

**[SCREEN: Bullet list of technologies]**

> "The system runs entirely on CPU — no GPU needed — inside a PyBullet physics simulation, with ROS2 Humble as the communication backbone. We used Python throughout, with Stable-Baselines3 for reinforcement learning and scikit-learn for the weather classifier.
>
> There are four subsystems. Let's walk through each one."

---

## SECTION 2 — SUBSYSTEM BREAKDOWN (~ 4 min)

---

### Sub-1: Perception (~1 min)

**[SCREEN: Show downward camera feed view, red marker being tracked, confidence score]**

> "Sub-1 is the drone's eyes. It takes the live camera feed from a downward-facing virtual camera and figures out exactly where the user is relative to the drone.
>
> We used HSV colour segmentation — the user wears a red marker, and the tracker isolates that colour using two HSV bands to handle the wraparound in hue at red. Once the contour is found, a pinhole camera model converts the pixel offset and apparent marker size into a real-world 3D position estimate in metres.
>
> To prevent jitter, we run an Exponential Moving Average filter over the output. And if the drone temporarily loses sight of the user, it holds the last known position for up to 20 frames before declaring a full tracking loss.
>
> The result is published at 30 Hz over the ROS2 topic `/skyshade/user_position` as a geometry_msgs Point, alongside a confidence score from 0 to 1."

**[SCREEN: Show tracker output — centroid pixel, 3D offset, confidence readout]**

> "In validation, we hit a 0.081 metre mean absolute error on position — well below our 0.15 metre target — with 100% tracking continuity across our test runs."

---

### Sub-2: Flight Control (~1 min)

**[SCREEN: Show drone following user, switching between PPO and PID, TensorBoard reward curve]**

> "Sub-2 is responsible for actually moving the drone. The controller operates a priority chain: a PPO reinforcement learning agent is the primary controller, with a traditional PID as a rock-solid fallback.
>
> PPO — Proximal Policy Optimisation — learns a continuous velocity setpoint policy: given the current position error, velocity, and wind strength, it outputs a desired velocity in X, Y, and Z. An inner P-controller then converts that into actual forces applied to the drone.
>
> We chose velocity setpoints instead of raw thrust because it abstracts away the drone's physics — the agent only needs to learn *where* to go, not how hard to push. This alone halved our training time.
>
> The flight controller also handles obstacle avoidance with a lightweight repulsive force system — the closer an obstacle, the stronger the push away from it.
>
> When Sub-4 issues a safety override, this controller immediately switches targets: from following the user, to returning home, or descending straight down."

---

### Sub-3: Environmental Decision (~1 min)

**[SCREEN: Show weather sensor gauges — lux, rain, wind — and umbrella deploying/stowing]**

> "Sub-3 is the drone's weather brain. It reads three simulated sensors — light level in lux, rain intensity from 0 to 1, and wind speed in metres per second — and decides every second whether the umbrella canopy should be deployed or stowed.
>
> We used a Support Vector Machine with an RBF kernel for this. The reason is that weather classification is a well-defined, low-dimensional problem. SVMs are interpretable, fast to train, and extremely reliable on small, labelled datasets — exactly what we had.
>
> But raw sensor values alone aren't enough. We engineered a 9-dimensional feature vector that adds the delta change in each reading since last tick, plus the last three deployment decisions as hysteresis terms. This gives the classifier *memory* — it doesn't flip the umbrella open and shut on borderline readings.
>
> We also apply a 3-frame confirmation window at runtime: the decision only changes if three consecutive predictions agree. This completely eliminates umbrella flutter."

---

### Sub-4: Navigation & Safety (~1 min)

**[SCREEN: Show battery level dropping, RTH triggering, then show obstacle room with SAC agent navigating]**

> "Sub-4 is actually two independent policies solving two different problems.
>
> The first is battery safety, solved with a Markov Decision Process and value iteration. The state space is just 12 states — four battery levels crossed with three distance-to-home buckets — and the three actions are: continue following the user, return to home, or land immediately. Value iteration converges in 18 iterations and under a millisecond. The result is a lookup table that gives us a mathematically guaranteed safe policy.
>
> The second policy handles obstacle navigation — flying through a room full of cylindrical pillars using 8 lidar rays. For this we used Soft Actor-Critic, an off-policy RL algorithm. We chose SAC over PPO here specifically because its replay buffer reuses all past experience, giving roughly three times the sample efficiency. And its automatic entropy tuning meant we didn't need to hand-design a curriculum.
>
> These two policies stay separate deliberately — the battery safety table is interpretable and auditable, while the obstacle nav policy is complex and continuous. Keeping them apart lets us test and explain each one independently."

---

## SECTION 3 — TRAINING DEEP DIVES (~ 4 min)

---

### Sub-2: PPO Flight Training (~1.5 min)

**[SCREEN: Open Training Grounds hub → Sub-2 tab, show the PyBullet training environment, live reward plot]**

> "Let's look at how we actually trained these models, starting with the PPO flight agent.
>
> Training happens inside a custom Gymnasium environment backed by a PyBullet simulation. The agent receives a 7-dimensional observation: the position error to the user, current velocity, and a normalised wind value. It outputs a 3D velocity setpoint.
>
> We run four parallel environments using SubprocVecEnv, which gives us roughly a four-times speedup — about 2,000 steps per second on CPU instead of 570. At 500k steps, training takes around 50 seconds for a quick run, or about four minutes for the full 1.5 million step curriculum.
>
> The curriculum has three stages. Stage one: stationary user, no wind. Stage two: stationary user, random gusts up to 4.5 metres per second. Stage three: walking user, gusty wind. The agent automatically advances through stages every 500k steps."

**[SCREEN: Show TensorBoard — reward climbing, ep_rew_mean, value_loss flattening]**

> "Here you can see the mean episode reward climbing through training. Our target was a mean reward above 150 — we consistently hit between 2,600 and 2,700, which reflects the agent learning to hover tightly and resist wind perturbations.
>
> One architectural decision worth highlighting: we originally tried tabular Q-learning with discrete actions. The drone oscillated badly around the hover point because discrete actions can't express the small corrections needed. Switching to PPO with continuous velocity setpoints eliminated that completely. The Q-table is still in the repo as a reference."

**[SCREEN: Show evaluation run — drone hovering within 0.5m circle, highlighted path]**

> "Evaluation runs 5 deterministic episodes and checks that the drone spends at least 30% of each episode within 0.5 metres of the user. We passed 10 out of 10 test episodes."

---

### Sub-3: SVM Weather Training (<1 min)

**[SCREEN: Show training data CSV, then the confusion matrix output, then PCA 3D scatter plot]**

> "The SVM trains in under two seconds on 99 labelled sensor samples covering the full range from clear skies — 85,000 lux, near-zero rain — to full storm conditions. We use 10-fold stratified cross validation to validate, and we log a confusion matrix and a 3D PCA scatter plot of the 9D feature space as visual outputs.
>
> The final model is a 3.4 kilobyte pickle file — a scikit-learn pipeline of a StandardScaler feeding into an SVC with RBF kernel. We hit 96% cross-validation accuracy, and in simulation we measured a 0% false deploy rate: the umbrella never opens in clear conditions."

---

### Sub-4A: MDP Value Iteration (<30 sec)

**[SCREEN: Show convergence curve — log-scale Bellman delta dropping to 1e-6]**

> "The MDP solver runs in well under a second. You can see here the Bellman delta — the maximum change across all state values in an iteration — dropping geometrically to below 10^-6 after just 18 iterations. The output is a 12-element numpy array: the optimal action for every battery-distance combination. The policy is then validated against 50 scripted scenarios including edge cases like CRITICAL battery at FAR distance — all 50 pass."

---

### Sub-4B: SAC Obstacle Navigation (~1 min)

**[SCREEN: Show the obstacle room — 7 pillars, blue drone, green goal zone — then show SAC agent navigating it]**

> "The SAC obstacle navigation environment is a 10 by 8 metre room with 7 cylindrical pillars. The agent gets 8 lidar rays at 45-degree intervals, a vector to the goal, and its current velocity — 12 values total. It outputs 2D velocity commands.
>
> The reward structure is deliberately shaped: 8 points per metre of progress toward the goal, minus a 0.15 step penalty to discourage hovering, plus 200 on arrival, minus 50 for hitting a pillar, and minus 30 for hitting a wall.
>
> SAC's off-policy replay buffer means it reuses every experience ever collected — at 150,000 steps this takes about 3 minutes and the agent reliably navigates the room."

**[SCREEN: Show evaluation — 3 out of 5 successful episodes, paths drawn]**

> "We pass the evaluation bar of 3 out of 5 successful runs. The remaining failures are typically attempts at a more aggressive line that clips a pillar — the agent hasn't fully converged to the most conservative safe path."

---

## SECTION 4 — SYSTEM INTEGRATION (~ 2 min)

**[SCREEN: Show ROS2 topic graph — four nodes, arrows between them]**

> "Now let's look at how all four subsystems talk to each other. The glue is ROS2's publish-subscribe messaging.
>
> There are four nodes running concurrently. Sub-1 — the perception node — publishes at 30 Hz. Sub-2 — flight — runs its control loop at 20 Hz. Sub-3 — weather decision — publishes at 1 Hz since weather changes slowly. Sub-4 — navigation and safety — runs at 5 Hz."

**[SCREEN: Show topic list with arrows]**

> "The data flow goes like this:
>
> Sub-1 publishes the user's 3D position and a confidence score. Sub-2 reads that position and uses it as its follow target. Sub-3 reads weather sensor values directly from the simulator and publishes `DEPLOY` or `STOW` to the umbrella servo. Sub-4 reads the confidence score and the battery level, and publishes one of three commands: `CONTINUE`, `RTH`, or `LAND_NOW`. Sub-2 listens to that override and changes its target accordingly.
>
> The critical design point is that Sub-4 has hard override authority. The moment it publishes `RTH`, Sub-2 drops user-following and targets the home position — no negotiation, no delay. `LAND_NOW` bypasses the learned policy entirely and drives the Z velocity directly to negative 0.5 m/s."

**[SCREEN: Show full simulation running — tracking + rain starts, umbrella deploys, battery drops, RTH triggers]**

> "Watching it all together: the drone follows the user, Sub-3 detects rain and deploys the canopy, and as the battery drains, Sub-4 issues RTH. The drone stops following, returns home, and lands — all without any human input."

**[SCREEN: Show the ROS2 launch file]**

> "The whole system starts with a single launch file — `skyshade_sim.launch.py` — which spins up all four nodes. You can also run the full integrated simulator with `python run_sim.py`, which handles the PyBullet physics, the GUI dashboard, and the scenario selection."

---

## SECTION 5 — SUMMARY, ISSUES & FUTURE WORK (~ 1.5 min)

**[SCREEN: Show dashboard with all systems running green]**

> "So to summarise: SkyShade is a four-subsystem autonomous drone simulation. Perception uses HSV vision and pinhole geometry. Flight uses PPO reinforcement learning with a PID safety net. Environmental decision uses a Support Vector Machine with hysteresis. Safety uses an MDP for battery management and SAC for obstacle navigation. All four are connected via ROS2 topics with a clear authority hierarchy."

---

**[SCREEN: Code or terminal with a highlighted issue]**

> "A few honest notes on issues we ran into and are still working on.
>
> The most subtle bug was in the MDP — the `REWARD_SAFE_COMPLETION` constant had accidentally been set to -100 instead of +100, which caused the solver to treat successful missions as failures and issue `LAND_NOW` constantly. Once we found it, one-character fix.
>
> We also had a drone drift issue on startup caused by PyBullet accumulating velocity during URDF loading before the simulation clock properly started. We fixed that with a 30-step settle loop and a velocity reset.
>
> The TensorBoard charts in the Training Grounds UI were accumulating axes because the axis object was being created every 200 milliseconds in the render tick instead of once. Rookie mistake with matplotlib — fixed by creating the axis once and storing it.
>
> One open issue we haven't fully resolved: the MDP policy has a bias toward RTH because the reward structure doesn't give enough credit for successful user-following during `CONTINUE`. We added a 50% battery guard to suppress premature RTH, but the real fix would be restructuring the reward to incentivise mission completion."

---

**[SCREEN: Future improvements slide or bulleted list]**

> "Looking ahead, the main improvements we'd want to make are:
>
> First — replace the HSV tracker in Sub-1 with a small CNN detector. HSV is brittle to lighting changes; a learned detector would be far more robust.
>
> Second — the obstacle navigation policy is trained on a fixed 7-pillar layout. It doesn't generalise to different room configurations. Adding domain randomisation during training would fix that.
>
> Third — there's been no sim-to-real validation. Moving to hardware would require domain randomisation, real sensor noise modelling, and hardware-in-the-loop testing.
>
> And fourth — the current system tracks a single user. Extending Sub-1 to track multiple targets and adding inter-drone collision avoidance would make it genuinely deployable."

---

**[SCREEN: Final shot of simulation — drone hovering over user, umbrella deployed, all systems nominal]**

> "Overall, SkyShade does what it set out to do: an end-to-end autonomous drone system, four different AI techniques working together, all validated and running. Thanks for watching."

---

## APPENDIX: SCREEN CUE CHECKLIST

Use this as a filming checklist. Capture each of these before recording narration.

| Timestamp | What to Show |
|---|---|
| 0:00 | Full simulation running (park/rain scenario, umbrella deployed) |
| 0:30 | Architecture diagram (from wiki) |
| 1:30 | Sub-1 downward camera feed, centroid tracking, confidence |
| 2:30 | Sub-2 drone following user, PPO in action |
| 3:30 | Sub-3 weather gauges, umbrella toggle |
| 4:30 | Sub-4 battery dropping → RTH, then obstacle room SAC nav |
| 5:30 | Training Grounds → Sub-2 tab, training loop + live reward chart |
| 7:00 | TensorBoard reward curve (ep_rew_mean climbing) |
| 7:45 | Sub-3 confusion matrix + PCA scatter |
| 8:15 | MDP convergence curve (log-scale Bellman delta) |
| 8:45 | SAC obstacle room training + evaluation paths |
| 9:45 | ROS2 topic graph (`ros2 topic list`, `ros2 run rqt_graph rqt_graph`) |
| 11:00 | Full integrated run: tracking → rain → deploy → battery → RTH → land |
| 12:30 | Terminal with the REWARD bug fix highlighted |
| 13:30 | Future improvements bullet list |
| 14:00 | Final simulation shot |
