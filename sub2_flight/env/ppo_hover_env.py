"""
Sub-2 Flight — PPO hover environment (gymnasium-compatible).

Observation space (7D float32):
    [dx, dy, dz, vx, vy, vz, wind_norm]
    dx/dy/dz : drone_pos − hover_target  (clipped ±4/±4/±2 m)
    vx/vy/vz : drone velocity (clipped ±5 m/s)
    wind_norm: wind_speed / 5.0  (∈ [0, 1])

Action space (3D float32, ∈ [-2, 2] m/s):
    [vx_des, vy_des, vz_des] — desired velocity setpoint.
    An inner proportional controller converts this to a corrective force:
        force = VELOCITY_GAIN × (v_des − v_current),  clipped to ±FORCE_CLIP N
    which is passed to HoverEnv.pid_step() alongside constant hover thrust.

Curriculum stages (set via set_stage()):
    1 — stationary user, no wind       (default)
    2 — stationary user, gusty wind    (randomised 0–4.5 m/s)
    3 — walking user, gusty wind
"""

import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sub2_flight.env.hover_env import HoverEnv, TARGET_ALTITUDE

VELOCITY_GAIN = 8.0     # N per m/s velocity error — inner P gain
ACTION_VEL_MAX = 2.0    # m/s — velocity setpoint clamp per axis
FORCE_CLIP = 12.0       # N  — max corrective force per axis

# Curriculum timestep thresholds
STAGE2_STEPS = 500_000
STAGE3_STEPS = 1_000_000

OBS_LOW  = np.array([-4., -4., -2., -5., -5., -5., 0.], dtype=np.float32)
OBS_HIGH = np.array([ 4.,  4.,  2.,  5.,  5.,  5., 1.], dtype=np.float32)


class PPOHoverEnv(gym.Env):
    """Gymnasium hover env for PPO — continuous velocity-setpoint action space."""

    metadata = {"render_modes": []}

    def __init__(self, render: bool = False, max_episode_steps: int = 500, stage: int = 1):
        super().__init__()
        self._render = render
        self._max_episode_steps = max_episode_steps
        self._stage = stage
        self._step_count = 0
        self._user_walk = False
        self._wind_speed = 0.0
        self._inner_env: HoverEnv | None = None

        self.observation_space = spaces.Box(OBS_LOW, OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-ACTION_VEL_MAX, high=ACTION_VEL_MAX, shape=(3,), dtype=np.float32
        )

    def set_stage(self, stage: int):
        """Update curriculum stage — called via SB3 env_method between rollouts."""
        self._stage = max(1, min(3, int(stage)))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        spread = 2.0 if self._stage == 3 else 1.0
        wind   = 0.0 if self._stage == 1 else float(rng.uniform(0.0, 4.5))
        user_pos = [
            float(rng.uniform(-spread, spread)),
            float(rng.uniform(-spread, spread)),
            0.0,
        ]
        drone_offset = [
            float(rng.uniform(-1.5, 1.5)),
            float(rng.uniform(-1.5, 1.5)),
            float(rng.uniform(-0.8, 0.8)),
        ]

        self._wind_speed = wind
        self._user_walk  = self._stage == 3
        self._step_count = 0

        if self._inner_env is None:
            # Large inner step limit — PPOHoverEnv controls all termination
            self._inner_env = HoverEnv(render=self._render, max_episode_steps=100_000)

        self._inner_env.reset(wind_speed=wind, user_pos=user_pos, drone_offset=drone_offset)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -ACTION_VEL_MAX, ACTION_VEL_MAX).astype(float)

        if self._user_walk and self._step_count > 0 and self._step_count % 20 == 0:
            rng = self.np_random
            self._inner_env._user_pos[0] += float(rng.uniform(-0.2, 0.2))
            self._inner_env._user_pos[1] += float(rng.uniform(-0.2, 0.2))
            self._inner_env.notify_target_moved()

        v_cur = self._inner_env.drone_vel
        force = VELOCITY_GAIN * (action - v_cur)
        force = np.clip(force, -FORCE_CLIP, FORCE_CLIP)

        _, reward, _, info = self._inner_env.pid_step(force)
        self._step_count += 1

        terminated = bool(self._inner_env.drone_pos[2] < 0.1)
        truncated  = self._step_count >= self._max_episode_steps

        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        pos  = self._inner_env.drone_pos
        vel  = self._inner_env.drone_vel
        user = self._inner_env._user_pos
        target = np.array([user[0], user[1], TARGET_ALTITUDE])
        delta  = pos - target
        obs = np.array([
            delta[0], delta[1], delta[2],
            vel[0],   vel[1],   vel[2],
            float(np.clip(self._wind_speed / 5.0, 0.0, 1.0)),
        ], dtype=np.float32)
        return np.clip(obs, OBS_LOW, OBS_HIGH)

    def close(self):
        if self._inner_env is not None:
            self._inner_env.close()
            self._inner_env = None
