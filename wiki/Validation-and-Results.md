# Validation and Results

## Running the test suite

```bash
# Sub-1: Perception accuracy on held-out frames
python sub1_perception/test_perception.py

# Sub-2: Q-agent evaluation (archived — PID used at runtime)
python sub2_flight/test_flight.py

# Sub-3: Classifier on 5-minute synthetic weather trajectory
python sub3_env/test_env_decision.py

# Sub-4: All 50 scripted battery/obstacle scenarios
python sub4_nav/test_nav_safety.py
```

---

## Passing targets

| Subsystem | Metric | Target | Status |
|---|---|---|---|
| Sub-1 | Tracking continuity | > 70% of frames above confidence threshold | Pass |
| Sub-1 | Position MAE | < 0.15 m | Pass |
| Sub-2 | Mean episode reward | > +150 across 10 eval episodes | Pass (PID) |
| Sub-2 | Hover error | Within 0.5 m in 9 of 10 episodes | Pass |
| Sub-3 | 10-fold CV accuracy | ≥ 90% | Pass |
| Sub-3 | False deploy rate | Minimised per confusion matrix | Pass |
| Sub-4 | Scripted scenarios | Pass all 50 | Pass |
| Sub-4 | RTH trigger latency | Within 1 decision tick of low-battery injection | Pass |

---

## Validation artefacts

| File | Description |
|---|---|
| `confusion_matrix.png` | Sub-3 SVM per-class classification results |
| `pca_3d.png` | 3-D PCA projection of the 9-D weather feature space |
| `convergence_curve.png` | Sub-4 value iteration convergence (max Bellman delta per iteration) |

---

## Known limitations

| Subsystem | Limitation |
|---|---|
| Sub-2 | Q-table archived — PID used at runtime; Q-learning reward shaping is a learning outcome, not a deployed artefact |
| Sub-4 | MDP reward structure biases toward RTH; patched with a 50% battery guard in `run_sim.py` |
| All | Simulation only — no physical drone testing; sim-to-real performance is unverified |
| Sub-1 | HSV thresholds are tuned for the simulated marker; real-world lighting would require retuning |
