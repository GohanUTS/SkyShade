#!/usr/bin/env python3
"""
Sub-4 Nav SAC — headless warm-start training script.

Trains the SAC obstacle-avoidance policy.  Warm-starts from the existing
models/ppo_nav_v1.zip (despite the filename it contains SAC weights).

Usage:
    python3 train_sub4.py                        # 200 000 steps (default)
    python3 train_sub4.py --steps 500000
    python3 train_sub4.py --steps 200000 --scenario buildings
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/skyshade_mpl")

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from sub4_nav.obstacle_env import ObstacleNavEnv, scenario_obstacles

_MODEL_PATH = os.path.join(_HERE, "models", "ppo_nav_v1")   # legacy filename
_MODEL_ZIP  = _MODEL_PATH + ".zip"
_META_PATH  = _MODEL_PATH + ".meta.json"

LOG_EVERY = 2048


class _Callback(BaseCallback):
    def __init__(self, log_every=LOG_EVERY):
        super().__init__(verbose=0)
        self._log_every = log_every
        self._last_log  = 0
        self._t0        = time.time()

    def _on_step(self):
        n = self.num_timesteps
        if n - self._last_log >= self._log_every:
            buf    = self.model.ep_info_buffer
            mean_r = float(np.mean([ep["r"] for ep in buf])) if buf else 0.0
            elapsed = time.time() - self._t0
            rate   = n / elapsed if elapsed > 0 else 0
            total  = self.locals.get("total_timesteps", n)
            eta_m  = ((total - n) / rate / 60) if rate > 0 else 0
            eff    = max(0, min(100, int((mean_r + 500) / 10)))
            print(f"  step {n:>8,}  reward={mean_r:+8.1f}  "
                  f"efficiency≈{eff:3d}%  rate={rate:.0f} s/s  ETA {eta_m:.1f} min")
            self._last_log = n
        return True


def _efficiency(buf):
    if not buf:
        return 0
    mean_r = float(np.mean([ep["r"] for ep in buf]))
    return max(0, min(100, int((mean_r + 500) / 10)))


def train(total_steps: int, scenario: str = None):
    warm = os.path.exists(_MODEL_ZIP)

    obstacles = scenario_obstacles(scenario)
    env = Monitor(ObstacleNavEnv(obstacles=obstacles))

    print(f"\n{'═' * 60}")
    print(f"  Sub-4 Nav SAC  —  {'warm-start' if warm else 'from scratch'}")
    print(f"  Steps: {total_steps:,}   Scenario layout: {scenario or 'default'}")
    print(f"{'═' * 60}\n")

    if warm:
        try:
            model = SAC.load(_MODEL_PATH, env=env,
                             learning_rate=5e-5,
                             custom_objects={"tensorboard_log": None})
            model.tensorboard_log = None
            print(f"  Loaded existing model: {_MODEL_ZIP}")
        except Exception as e:
            print(f"  Could not load existing model ({e}) — starting fresh")
            warm = False
            model = None
    else:
        model = None

    if model is None:
        model = SAC(
            "MlpPolicy", env,
            learning_rate   = 3e-4,
            buffer_size     = 100_000,
            batch_size      = 256,
            tau             = 0.005,
            gamma           = 0.99,
            ent_coef        = "auto",
            learning_starts = 1_000,
            train_freq      = 1,
            gradient_steps  = 1,
            policy_kwargs   = {"net_arch": [256, 256]},
            seed            = 42,
            verbose         = 0,
        )
        print("  Initialising fresh SAC model")

    cb = _Callback()
    t0 = time.time()
    model.learn(
        total_timesteps    = total_steps,
        callback           = cb,
        progress_bar       = False,
        reset_num_timesteps= not warm,
    )
    elapsed = time.time() - t0
    env.close()

    os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
    model.save(_MODEL_PATH)

    eff        = _efficiency(model.ep_info_buffer)
    steps_done = cb.num_timesteps

    meta = {
        "subsystem":      "Sub-4 Navigation",
        "algorithm":      "SAC",
        "model":          "ppo_nav_v1.zip",
        "steps_done":     int(steps_done),
        "efficiency_pct": int(eff),
        "warm_started":   bool(warm),
        "trained_at":     time.time(),
        "scenario":       scenario or "default",
        "legacy_filename": "ppo_nav_v1.zip",
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
    ap.add_argument("--steps",    type=int, default=200_000)
    ap.add_argument("--scenario", type=str, default=None,
                    help="Obstacle layout: park, forest, buildings, trail, night, rooftop")
    args = ap.parse_args()
    train(args.steps, args.scenario)


if __name__ == "__main__":
    main()
