# Validation and Results

## Running the test suite

```bash
# Sub-1: Perception accuracy on held-out frames
python sub1_perception/test_perception.py

# Sub-2: PPO policy evaluation (5 episodes, stage 2 conditions)
#   Uses FlightEvalWorker — or run the Evaluate Model button in the Training Grounds hub
python -c "
import queue, threading, numpy as np
from sub2_flight.eval_worker import FlightEvalWorker
q, stop = queue.Queue(), threading.Event()
w = FlightEvalWorker(q, stop); w.start(); w.join()
while not q.empty():
    print(q.get_nowait())
"

# Sub-3: Classifier on 5-minute synthetic weather trajectory
python sub3_env/test_env_decision.py

# Sub-4: Battery-safety scripted scenarios
python sub4_nav/test_nav_safety.py

# Sub-4: Nav PPO evaluation (5 episodes, obstacle room)
python -c "
import queue, threading
from sub4_nav.eval_worker import NavEvalWorker
q, stop = queue.Queue(), threading.Event()
w = NavEvalWorker(q, stop); w.start(); w.join()
while not q.empty():
    print(q.get_nowait())
"
```

---

## Pass targets

| Subsystem | Metric | Target |
|---|---|---|
| Sub-1 | Tracking continuity | > 70% of frames above `CONFIDENCE_THRESH` |
| Sub-1 | Position MAE | < 0.15 m |
| Sub-1 | Calibration room | LOCKED status with pixel error < 18 px |
| Sub-2 PPO | Hover efficiency (Training Grounds) | ≥ 60% (predicted from training buffer) |
| Sub-2 PPO | Evaluation pass rate | ≥ 3 / 5 episodes spend ≥ 30% in hover zone |
| Sub-2 PPO | Mean eval episode reward | > −200 on stage-2 wind conditions |
| Sub-3 | 10-fold CV accuracy | ≥ 90% |
| Sub-3 | False deploy rate | Minimised per confusion matrix |
| Sub-4 MDP | Scripted battery scenarios | Pass all 50 |
| Sub-4 MDP | RTH trigger latency | Within 1 decision tick (200 ms) of low-battery injection |
| Sub-4 Nav | Evaluation pass rate | ≥ 3 / 5 episodes reach the goal zone |

---

## How to read the Training Grounds evaluation

After clicking **Evaluate Model** in the hub:

- **Sub-2**: 5 episodes run on stage-2 conditions (stationary user + gusty wind). Each episode: the drone starts at a random offset from the target and must fly in and hover. Pass = spent ≥ 30% of the episode within 0.5 m radius. The best episode's 3D path is drawn in green in the hover arena.

- **Sub-4 Nav**: 5 episodes in the obstacle room (deterministic, seed varies per episode). Pass = drone reached the east-side goal zone within 1200 steps. The best episode's path is drawn in green through the red pillars.

---

## Validation artefacts

| File | Description |
|---|---|
| `confusion_matrix.png` | Sub-3 SVM per-class classification results |
| `pca_3d.png` | 3-D PCA projection of the 9-D weather feature space |
| `convergence_curve.png` | Sub-4 MDP value iteration convergence (max Bellman Δ per iteration) |
| `models/ppo_flight_v1.zip` | Trained Sub-2 PPO model (SB3 format) |
| `models/policy_table_v1.npy` | Solved Sub-4 MDP battery-safety policy |
| `models/ppo_nav_v1.zip` | Trained Sub-4 PPO obstacle-navigation model |

---

## Known limitations

| Subsystem | Limitation |
|---|---|
| Sub-2 | Q-table archived — PPO is the primary runtime policy; PID is the final fallback |
| Sub-2 | PPO trained in headless PyBullet (DIRECT mode); sim-to-real transfer unverified |
| Sub-4 | MDP reward structure biases toward RTH — patched with a 50% battery guard in `run_sim.py` |
| Sub-4 Nav | Obstacle positions are fixed; the policy does not generalise to different room layouts |
| All | Simulation only — no physical drone testing |
| Sub-1 | HSV thresholds tuned for the simulated marker; real-world lighting requires retuning |
