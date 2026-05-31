"""
Sub-2 Flight — PPO training script.

Trains a PPO agent using a continuous velocity-setpoint action space.
The inner P controller in PPOHoverEnv converts velocity setpoints to forces,
so the existing PID infrastructure is reused at the execution layer.

Curriculum (by timestep):
  Stage 1: Stationary user, no wind        (steps 0 – 499 999)
  Stage 2: Stationary user, gusty wind     (steps 500 000 – 999 999)
  Stage 3: Walking user, gusty wind        (steps 1 000 000 – 1 499 999)

Usage:
    python sub2_flight/train_ppo.py [--steps 1500000] [--output models/ppo_flight_v1]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from sub2_flight.env.ppo_hover_env import PPOHoverEnv, STAGE2_STEPS, STAGE3_STEPS

LOG_INTERVAL = 10_000

DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "models", "ppo_flight_v1")


class _CurriculumCallback(BaseCallback):
    """Advances curriculum stage and logs progress to stdout."""

    def __init__(self, log_interval: int = LOG_INTERVAL):
        super().__init__(verbose=0)
        self._log_interval = log_interval
        self._last_log = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        n = self.num_timesteps
        stage = 3 if n >= STAGE3_STEPS else (2 if n >= STAGE2_STEPS else 1)
        self.training_env.env_method("set_stage", stage)

        if n - self._last_log >= self._log_interval:
            buf = self.model.ep_info_buffer
            mean_r = float(np.mean([ep["r"] for ep in buf])) if buf else 0.0
            print(f"Step {n:>9}  stage={stage}  mean_ep_reward={mean_r:+.1f}")
            self._last_log = n


def train(total_steps: int, output_path: str):
    env = Monitor(PPOHoverEnv(stage=1))
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [256, 256]},
        seed=0,
        verbose=0,
    )

    cb = _CurriculumCallback()
    model.learn(total_timesteps=total_steps, callback=cb, progress_bar=False)
    env.close()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    model.save(output_path)
    print(f"\nPPO model saved → {output_path}.zip")


def main():
    parser = argparse.ArgumentParser(description="Train Sub-2 PPO flight agent")
    parser.add_argument("--steps",  type=int, default=1_500_000)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    train(args.steps, args.output)


if __name__ == "__main__":
    main()
