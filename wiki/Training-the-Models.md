# Training the Models

Pre-trained checkpoints are available in `models/` (via Git LFS). Only re-train if you change subsystem logic or hyperparameters.

---

## Sub-2: Q-learning flight agent (archived)

> The runtime simulation uses the PID controller, not this Q-table. Re-train only if you want to experiment with the Q-learning agent.

```bash
python sub2_flight/train_qlearning.py \
  --episodes 50000 \
  --output models/qtable_v1.npy
```

Monitor training:

```bash
tensorboard --logdir runs/sub2
```

Training runs through three automatic curriculum stages:

1. Stationary user, no wind
2. Stationary user, gusty wind
3. Walking user, gusty wind

**Convergence check:** epsilon should reach 0.05 and mean episode reward should trend upward. If reward is flat after 10 k episodes, lower `ALPHA` to `0.05` in `sub2_flight/train_qlearning.py`.

---

## Sub-3: SVM classifier

```bash
python sub3_env/train_svm.py \
  --data data/env_sensor_log.csv \
  --output models/svm_v1.pkl
```

Outputs:
- `models/svm_v1.pkl` — trained classifier
- `confusion_matrix.png` — per-class accuracy
- `pca_3d.png` — 3-D PCA visualisation of the feature space

**If CV accuracy is below 90%:**
- Check class balance in `env_sensor_log.csv`; add `class_weight='balanced'` if heavily skewed
- Confirm the feature vector is standardised with `StandardScaler` before fitting

---

## Sub-4: MDP value iteration

```bash
python sub4_nav/solve_mdp.py \
  --gamma 0.95 \
  --output models/policy_table_v1.npy
```

Outputs:
- `models/policy_table_v1.npy` — solved policy table
- `convergence_curve.png` — max Bellman update per iteration

**If Sub-4 always outputs `LAND_NOW`:** re-run the solver and confirm convergence delta reaches below `1e-6`. Check that `REWARD_SAFE_COMPLETION` is set correctly and not overridden by the per-step cost.
