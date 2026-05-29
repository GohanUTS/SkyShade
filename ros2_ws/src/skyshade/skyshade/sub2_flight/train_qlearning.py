"""
Sub-2 Flight — Q-learning training script.

Trains a tabular Q-learning agent to fly the drone to hover above a moving
user inside the PyBullet hover environment.  Training is structured as three
curriculum stages:

  Stage 1: Stationary user, no wind     (episodes 0 – 16 999)
  Stage 2: Stationary user, gusty wind  (episodes 17 000 – 33 999)
  Stage 3: Walking user, gusty wind     (episodes 34 000 – 49 999)

Usage:
    python sub2_flight/train_qlearning.py --episodes 50000 --output models/qtable_v1.npy

Outputs:
  • <output>         — Q-table as numpy array, shape (N_STATES, N_ACTIONS)
  • runs/sub2/       — TensorBoard event files (scalar: reward per episode)
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from skyshade.sub2_flight.env.hover_env import HoverEnv, N_STATES, N_ACTIONS

# ── Hyperparameters ───────────────────────────────────────────────────────────
ALPHA = 0.1           # Learning rate
GAMMA = 0.95          # Discount factor
EPSILON_START = 1.0   # Initial exploration rate
EPSILON_END = 0.05    # Final exploration rate
EPSILON_DECAY = 0.9995

# Curriculum thresholds (episode indices)
STAGE2_START = 17_000
STAGE3_START = 34_000

# Wind speed (m/s) per stage
STAGE1_WIND = 0.0
STAGE2_WIND = 3.0
STAGE3_WIND = 3.0

LOG_INTERVAL = 500   # Print progress every N episodes
TB_FLUSH_INTERVAL = 100
# ─────────────────────────────────────────────────────────────────────────────


def _make_user_pos(stage: int, rng: np.random.Generator) -> list:
    """Return a random user starting position for the given curriculum stage."""
    if stage < 3:
        return [rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0]
    # Stage 3: user starts up to 2 m away
    return [rng.uniform(-2, 2), rng.uniform(-2, 2), 0.0]


def _wind_for_episode(stage: int) -> float:
    if stage == 1:
        return STAGE1_WIND
    return STAGE2_WIND


def _curriculum_stage(episode: int) -> int:
    if episode < STAGE2_START:
        return 1
    if episode < STAGE3_START:
        return 2
    return 3


def train(episodes: int, output_path: str):
    # Attempt to import TensorBoard writer; fall back to no-op if unavailable
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir="runs/sub2")
    except ImportError:
        try:
            from tensorboard.summary.writer.event_file_writer import EventFileWriter
            writer = None
        except ImportError:
            writer = None
        try:
            # Try the standard TF-based writer
            from tensorflow.summary import create_file_writer
            writer = None  # Keep None; we'll use a custom lightweight logger
        except ImportError:
            pass

    # Lightweight fallback writer
    class _NoOpWriter:
        def add_scalar(self, *a, **kw): pass
        def flush(self): pass
        def close(self): pass

    if writer is None:
        writer = _NoOpWriter()

    q_table = np.zeros((N_STATES, N_ACTIONS), dtype=np.float32)
    env = HoverEnv(render=False)
    rng = np.random.default_rng(0)

    epsilon = EPSILON_START
    episode_rewards = []

    for ep in range(episodes):
        stage = _curriculum_stage(ep)
        wind = _wind_for_episode(stage)
        user_pos = _make_user_pos(stage, rng)

        # Stage 3: update user position mid-episode to simulate walking
        user_walk = stage == 3

        state = env.reset(wind_speed=wind, user_pos=user_pos)
        total_reward = 0.0
        done = False
        step = 0

        while not done:
            # Epsilon-greedy action selection
            if rng.random() < epsilon:
                action = rng.integers(0, N_ACTIONS)
            else:
                action = int(np.argmax(q_table[state]))

            # Simulate user walking: shift position every 20 steps
            if user_walk and step % 20 == 0 and step > 0:
                env._user_pos[0] += rng.uniform(-0.2, 0.2)
                env._user_pos[1] += rng.uniform(-0.2, 0.2)

            next_state, reward, done, _ = env.step(action)

            # Q-learning update
            best_next = np.max(q_table[next_state])
            q_table[state, action] += ALPHA * (
                reward + GAMMA * best_next - q_table[state, action]
            )

            state = next_state
            total_reward += reward
            step += 1

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)

        writer.add_scalar("reward/episode", total_reward, ep)
        writer.add_scalar("epsilon", epsilon, ep)

        if ep % LOG_INTERVAL == 0:
            mean_r = np.mean(episode_rewards[-LOG_INTERVAL:]) if ep > 0 else total_reward
            print(f"Ep {ep:>6}/{episodes}  stage={stage}  ε={epsilon:.4f}  "
                  f"mean_reward={mean_r:+.1f}")

        if ep % TB_FLUSH_INTERVAL == 0:
            writer.flush()

    env.close()
    writer.close()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, q_table)
    print(f"\nQ-table saved → {output_path}  shape={q_table.shape}")


def main():
    parser = argparse.ArgumentParser(description="Train Sub-2 Q-learning agent")
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--output", type=str, default="models/qtable_v1.npy")
    args = parser.parse_args()
    train(args.episodes, args.output)


if __name__ == "__main__":
    main()
