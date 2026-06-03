#!/usr/bin/env python3
"""
Sub-2 PPO hover policy — headless warm-start training script.

Warm-starts from models/ppo_flight_v1.zip (if present) using a fine-tuned
learning rate, then trains for the requested number of additional steps using
4 parallel PyBullet environments for ~4× throughput.

Usage:
    python3 train_sub2.py                     # 1 500 000 steps (full curriculum)
    python3 train_sub2.py --steps 500000      # lighter top-up run
"""

import argparse
import os
import sys
import time
import json

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from sub2_flight.env.ppo_hover_env import PPOHoverEnv

_MODEL_PATH  = os.path.join(_HERE, "models", "ppo_flight_v1")
_MODEL_ZIP   = _MODEL_PATH + ".zip"
_META_PATH   = _MODEL_PATH + ".meta.json"

# Curriculum thresholds (cumulative steps across all envs)
_STAGE2 = 500_000
_STAGE3 = 1_000_000

N_ENVS      = 4
LOG_EVERY   = 10_000


class _CurriculumCallback(BaseCallback):
    def __init__(self, log_every=LOG_EVERY):
        super().__init__(verbose=0)
        self._log_every  = log_every
        self._last_log   = 0
        self._t0         = time.time()

    def _on_rollout_end(self):
        n     = self.num_timesteps
        stage = 3 if n >= _STAGE3 else (2 if n >= _STAGE2 else 1)
        self.training_env.env_method("set_stage", stage)

        if n - self._last_log >= self._log_every:
            buf    = self.model.ep_info_buffer
            mean_r = float(np.mean([ep["r"] for ep in buf])) if buf else 0.0
            elapsed = time.time() - self._t0
            rate   = n / elapsed if elapsed > 0 else 0
            eta_s  = (self.locals.get("total_timesteps", n) - n) / rate if rate > 0 else 0
            eta_m  = eta_s / 60
            pct    = int((mean_r + 600) / 7.5)
            pct    = max(0, min(100, pct))
            print(f"  step {n:>9,}  stage={stage}  reward={mean_r:+7.1f}  "
                  f"efficiency≈{pct:3d}%  rate={rate:.0f} s/s  ETA {eta_m:.1f} min")
            self._last_log = n

    def _on_step(self):
        return True


def _efficiency(buf):
    if not buf:
        return 0
    mean_r = float(np.mean([ep["r"] for ep in buf]))
    return max(0, min(100, int((mean_r + 600) / 7.5)))


def train(total_steps: int):
    warm = os.path.exists(_MODEL_ZIP)

    def _make_env():
        return Monitor(PPOHoverEnv(stage=1))

    print(f"\n{'═' * 60}")
    print(f"  Sub-2 PPO Training  —  {'warm-start' if warm else 'from scratch'}")
    print(f"  Steps: {total_steps:,}   Envs: {N_ENVS}")
    print(f"{'═' * 60}\n")

    try:
        env = make_vec_env(_make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
        print(f"  Using {N_ENVS} parallel envs (SubprocVecEnv)")
    except Exception as e:
        print(f"  SubprocVecEnv failed ({e}), falling back to single env")
        env = make_vec_env(_make_env, n_envs=1, vec_env_cls=DummyVecEnv)

    if warm:
        model = PPO.load(_MODEL_PATH, env=env, learning_rate=3e-5, clip_range=0.08,
                         custom_objects={"tensorboard_log": None})
        model.tensorboard_log = None
        print(f"  Loaded existing model: {_MODEL_ZIP}")
    else:
        model = PPO(
            "MlpPolicy", env,
            n_steps=512, batch_size=256, n_epochs=10,
            gamma=0.99, gae_lambda=0.95,
            learning_rate=3e-4, clip_range=0.2, ent_coef=0.01,
            policy_kwargs={"net_arch": [256, 256]},
            seed=0, verbose=0,
        )
        print("  Initialising fresh model")

    print(f"\n  Curriculum:  stage 1 → 0–{_STAGE2//1000}k steps  "
          f"| stage 2 → {_STAGE2//1000}k–{_STAGE3//1000}k  "
          f"| stage 3 → {_STAGE3//1000}k+\n")

    cb = _CurriculumCallback()
    t0 = time.time()
    model.learn(
        total_timesteps=total_steps,
        callback=cb,
        progress_bar=False,
        reset_num_timesteps=not warm,
    )
    elapsed = time.time() - t0
    env.close()

    os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
    model.save(_MODEL_PATH)

    eff = _efficiency(model.ep_info_buffer)
    steps_done = cb.num_timesteps

    meta = {
        "subsystem": "Sub-2 Flight",
        "algorithm": "PPO",
        "model": "ppo_flight_v1.zip",
        "steps_done": int(steps_done),
        "efficiency_pct": int(eff),
        "warm_started": bool(warm),
        "trained_at": time.time(),
        "parallel_envs": N_ENVS,
    }
    with open(_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'═' * 60}")
    print(f"  Training complete in {elapsed/60:.1f} min")
    print(f"  Steps trained : {steps_done:,}")
    print(f"  Efficiency    : ~{eff}%")
    print(f"  Model saved   : {_MODEL_ZIP}")
    print(f"{'═' * 60}\n")
    return eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_500_000,
                    help="Training steps (default 1 500 000)")
    args = ap.parse_args()
    train(args.steps)


if __name__ == "__main__":
    main()
