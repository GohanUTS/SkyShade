"""
Sub-1 Perception — RL gimbal-tracking environment.

The HSV tracker (classic computer vision) *finds* the red marker in the frame.
This environment trains the part that classic CV does NOT do well: actively
*controlling the camera gimbal* to keep that marker centred while it moves —
i.e. learned visual servoing.

Task
────
A target moves around the image plane on a smooth, randomised path. The agent
commands gimbal pan/tilt rates to keep the target locked on the frame centre.

Observation (6-D, all normalised):
    [err_x, err_y, tgt_vx, tgt_vy, gimbal_x, gimbal_y]
        err   = where the target appears relative to centre (the pixel error)
        tgt_v = the target's apparent velocity in the frame
        gimbal= the current gimbal pointing offset

Action (2-D, continuous, [-1, 1]):
    [pan_rate, tilt_rate]  — angular velocity commands for the gimbal

Reward:
    -||err||²            keep the target centred
    -0.01·||action||²    discourage frantic/expensive gimbal motion
    +1.0 when locked     bonus for holding the target inside a tight centre zone

Solvable in a few hundred thousand SAC steps. Standalone smoke-test:
    python sub1_perception/gimbal_rl_env.py
"""

import math

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM = True
except ImportError:                      # allows import without gymnasium present
    gym = object                          # type: ignore
    _GYM = False

EPISODE_STEPS = 200
DT            = 0.05          # seconds per control step
GIMBAL_RATE   = 2.5          # max gimbal slew (frame-units / second)
GIMBAL_LIMIT  = 1.5          # gimbal cannot point further than this from centre
LOCK_RADIUS   = 0.10         # target is "locked" when within this of centre


class GimbalTrackingEnv(gym.Env if _GYM else object):
    """Visual-servoing gimbal that learns to keep a moving target centred."""

    metadata = {"render_modes": []}

    def __init__(self, episode_steps: int = EPISODE_STEPS, seed=None):
        if _GYM:
            super().__init__()
        self.episode_steps = episode_steps
        # obs bounds: error, target velocity, gimbal offset
        high = np.array([2.5, 2.5, 2.0, 2.0, GIMBAL_LIMIT, GIMBAL_LIMIT],
                        dtype=np.float32)
        if _GYM:
            self.observation_space = spaces.Box(-high, high, dtype=np.float32)
            self.action_space      = spaces.Box(-1.0, 1.0, shape=(2,),
                                                 dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._t   = 0
        self._gim = np.zeros(2, dtype=np.float32)
        self._locked = 0
        self._reset_target()

    # ── target motion (sum-of-sines Lissajous, randomised each episode) ──────
    def _reset_target(self):
        r = self._rng
        self._amp   = r.uniform(0.45, 0.95, size=2).astype(np.float32)
        self._freq  = r.uniform(0.4, 1.3, size=2).astype(np.float32)
        self._phase = r.uniform(0.0, 2 * math.pi, size=2).astype(np.float32)

    def _target_pos(self, t):
        ang = self._freq * (t * DT) + self._phase
        return (self._amp * np.sin(ang)).astype(np.float32)

    def _target_vel(self, t):
        ang = self._freq * (t * DT) + self._phase
        return (self._amp * self._freq * np.cos(ang)).astype(np.float32)

    def _obs(self):
        err = self._target_pos(self._t) - self._gim
        vel = self._target_vel(self._t)
        return np.concatenate([err, vel, self._gim]).astype(np.float32)

    # ── gymnasium API ────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        if _GYM:
            super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._locked = 0
        self._reset_target()
        # gimbal starts near (but off) the target so there's something to correct
        self._gim = np.clip(
            self._target_pos(0) + self._rng.uniform(-0.3, 0.3, size=2),
            -GIMBAL_LIMIT, GIMBAL_LIMIT).astype(np.float32)
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._gim = np.clip(self._gim + action * GIMBAL_RATE * DT,
                            -GIMBAL_LIMIT, GIMBAL_LIMIT).astype(np.float32)
        self._t += 1

        err   = self._target_pos(self._t) - self._gim
        dist2 = float(err[0] ** 2 + err[1] ** 2)
        reward = -dist2 - 0.01 * float(np.sum(action ** 2))
        locked = dist2 < LOCK_RADIUS ** 2
        if locked:
            reward += 1.0
            self._locked += 1

        truncated  = self._t >= self.episode_steps
        terminated = False
        info = {"dist": math.sqrt(dist2),
                "lock_rate": self._locked / max(1, self._t)}
        return self._obs(), float(reward), terminated, truncated, info

    def get_viz_state(self) -> dict:
        """Snapshot for the live UI (target + gimbal positions, lock rate)."""
        return {"target": self._target_pos(self._t).tolist(),
                "gimbal": self._gim.tolist(),
                "lock_rate": self._locked / max(1, self._t)}


if __name__ == "__main__":
    env = GimbalTrackingEnv()
    obs, _ = env.reset(seed=0)
    print("obs space :", env.observation_space)
    print("act space :", env.action_space)
    total = 0.0
    obs, _ = env.reset(seed=1)
    for _ in range(EPISODE_STEPS):
        # naive proportional controller as a sanity baseline (not the RL policy)
        act = np.clip(obs[:2] * 2.0, -1, 1)
        obs, r, term, trunc, info = env.step(act)
        total += r
        if term or trunc:
            break
    print(f"P-controller baseline episode reward {total:+.1f}  "
          f"final lock_rate {info['lock_rate']:.0%}")
