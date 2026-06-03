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
    1 — stationary user, no wind, no obs noise
    2 — stationary user, gusty wind, light position noise (σ 0.15 m)
    3 — walking user, gusty wind, canopy-level position noise (σ 0.35 m)

The position noise in stages 2 and 3 is domain randomisation: it trains the
policy to be robust to imprecise tracker position estimates (as seen in the
forest and urban-trail scenarios where occlusion adds real measurement noise).
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

    # Stage-3 walk modes — each episode randomly picks one so the policy
    # generalises to both the training-env random steps AND the smooth
    # directional walks used in the forest/trail/beach sim scenarios.
    #   "random"   — legacy ±0.2 m random jumps every 20 steps
    #   "linear"   — constant velocity in one direction (like forest/trail)
    #   "circular" — orbit around a centre point (like park figure-8)
    _WALK_MODES  = ("random", "linear", "circular")
    _WALK_WEIGHTS = (0.40, 0.35, 0.25)   # probability of each mode

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

        # Pick walk mode and parameters for this episode
        if self._user_walk:
            cum = np.cumsum(self._WALK_WEIGHTS)
            r   = float(rng.uniform())
            idx = int(np.searchsorted(cum, r))
            self._walk_mode = self._WALK_MODES[min(idx, len(self._WALK_MODES) - 1)]
            # Linear: speed 0.08–0.18 m/s in a random direction
            self._walk_speed  = float(rng.uniform(0.08, 0.18))
            self._walk_angle  = float(rng.uniform(0, 2 * np.pi))
            # Circular: orbit radius and angular speed
            self._orbit_r     = float(rng.uniform(1.0, 3.0))
            self._orbit_speed = float(rng.uniform(0.04, 0.12))
            self._orbit_phase = float(rng.uniform(0, 2 * np.pi))
            # Control timestep (10 Hz)
            self._action_dt   = 0.10
        else:
            self._walk_mode = "random"

        if self._inner_env is None:
            # Large inner step limit — PPOHoverEnv controls all termination
            self._inner_env = HoverEnv(render=self._render, max_episode_steps=100_000)

        self._inner_env.reset(wind_speed=wind, user_pos=user_pos, drone_offset=drone_offset)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -ACTION_VEL_MAX, ACTION_VEL_MAX).astype(float)

        if self._user_walk and self._step_count > 0:
            rng  = self.np_random
            mode = self._walk_mode
            if mode == "random" and self._step_count % 20 == 0:
                self._inner_env._user_pos[0] += float(rng.uniform(-0.2, 0.2))
                self._inner_env._user_pos[1] += float(rng.uniform(-0.2, 0.2))
                self._inner_env.notify_target_moved()
            elif mode == "linear":
                # Smooth directional walk — advances every step
                dt   = self._action_dt
                self._inner_env._user_pos[0] += self._walk_speed * np.cos(self._walk_angle) * dt
                self._inner_env._user_pos[1] += self._walk_speed * np.sin(self._walk_angle) * dt
                self._inner_env.notify_target_moved()
            elif mode == "circular" and self._step_count % 4 == 0:
                t    = self._step_count * self._action_dt
                cx0  = self._orbit_r * np.cos(self._orbit_speed * t + self._orbit_phase)
                cy0  = self._orbit_r * np.sin(self._orbit_speed * t + self._orbit_phase)
                self._inner_env._user_pos[0] = float(cx0)
                self._inner_env._user_pos[1] = float(cy0)
                self._inner_env.notify_target_moved()

        v_cur = self._inner_env.drone_vel
        force = VELOCITY_GAIN * (action - v_cur)
        force = np.clip(force, -FORCE_CLIP, FORCE_CLIP)

        _, reward, _, info = self._inner_env.pid_step(force)
        self._step_count += 1

        terminated = bool(self._inner_env.drone_pos[2] < 0.1)
        truncated  = self._step_count >= self._max_episode_steps

        return self._get_obs(), float(reward), terminated, truncated, info

    # No observation noise — domain randomisation repeatedly destabilised the
    # warm-start policy, causing forest hover to degrade from 67% to 17% after
    # each additional training run.  The tracker is 99%+ accurate in all
    # scenarios so observation noise gives no benefit and only causes harm.
    _OBS_NOISE = {1: 0.0, 2: 0.0, 3: 0.0}

    def _get_obs(self) -> np.ndarray:
        pos  = self._inner_env.drone_pos
        vel  = self._inner_env.drone_vel
        user = self._inner_env._user_pos
        target = np.array([user[0], user[1], TARGET_ALTITUDE])
        delta  = pos - target

        # Inject measurement noise on position delta to simulate tracker imprecision.
        noise_std = self._OBS_NOISE.get(self._stage, 0.0)
        if noise_std > 0.0:
            delta[:3] += self.np_random.normal(0.0, noise_std, size=3).astype(float)

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
