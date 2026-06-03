"""
Sub-4 Nav — obstacle avoidance navigation environment.

A 10×8 m rectangular room with 7 cylindrical pillars.  The drone starts at
the west end and must reach a goal zone at the east end while using 8 lidar
rays to sense and dodge obstacles.

Observation (12D float32)
──────────────────────────
lidar[0..7]  : 8 ray hit fractions ∈ [0, 1]  (1.0 = nothing within LIDAR_RANGE)
dx, dy       : vector from drone to goal (m), clipped ±12
vx, vy       : drone lateral velocity (m/s), clipped ±5

Action (2D float32, ±2 m/s)
─────────────────────────────
[vx_cmd, vy_cmd] — lateral velocity setpoint at fixed NAV_ALT.
Same inner P-controller as PPOHoverEnv: force = VEL_GAIN × (v_cmd − v_cur).

Reward
──────
+progress   : 8 × reduction in Euclidean goal distance each step
−step       : −0.15 per step
+arrival    : +200 when reaching GOAL_RADIUS
−collision  : −50 on obstacle hit; −30 on wall hit  (episode ends)
"""

import math
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import pybullet as p
    import pybullet_data
    _PB = True
except ImportError:
    _PB = False

# ── Room geometry ─────────────────────────────────────────────────────────────
ROOM_W    = 10.0   # m (X axis,  -5 … +5)
ROOM_D    =  8.0   # m (Y axis,  -4 … +4)
NAV_ALT   =  1.5   # m  fixed flight altitude
WALL_H    =  3.5   # m  height of room walls (visual only)

# Start / goal (x, y) in world frame
START_XY  = (-4.5, 0.0)
GOAL_XY   = ( 4.5, 0.0)
GOAL_R    =  0.9   # m  success radius

# Obstacle layout: (x, y, radius)
OBSTACLES = [
    (-2.5,  2.5, 0.35),
    (-2.5, -2.5, 0.35),
    ( 0.0,  1.5, 0.40),
    ( 0.0, -1.5, 0.40),
    ( 0.0,  0.0, 0.30),
    ( 2.5,  2.5, 0.35),
    ( 2.5, -2.5, 0.35),
]


def scenario_obstacles(scenario):
    """Return an obstacle layout (list of (x, y, radius)) shaped like the chosen
    launch scenario, mapped into this 10×8 nav room (start −4.5, goal +4.5).

    This is what makes Auto-Train "train on the selected scenario's world":
    the SAC nav policy learns to dodge a layout resembling the city / park /
    forest / trail it will actually fly in.  Unknown → the default pillar maze.
    """
    s = (scenario or "").lower()
    if s == "buildings":
        # City blocks — a denser 3×3 grid of larger pillars to weave through.
        layout = []
        for cx in (-2.6, 0.0, 2.6):
            for cy in (2.5, 0.0, -2.5):
                layout.append((cx, cy, 0.45))
        return layout
    if s == "park":
        # Open park — a few scattered small trees.
        return [(-1.6, 1.8, 0.30), (1.3, -1.7, 0.30), (0.2, 2.7, 0.28),
                (2.1, 1.3, 0.30), (-2.2, -1.5, 0.30)]
    if s == "forest":
        # Dense trail through trunks — many small obstacles in rows.
        layout = []
        for cx in (-3.0, -1.5, 0.0, 1.5, 3.0):
            layout.append((cx, 2.3, 0.24))
            layout.append((cx, -2.3, 0.24))
            layout.append((cx, 0.7 * math.sin(cx * 1.3), 0.22))
        return layout
    if s == "trail":
        # Urban trail — a central pinch (the bridge) plus a couple of bollards.
        return [(0.0, 1.7, 0.40), (0.0, -1.7, 0.40),
                (-2.3, 0.4, 0.28), (2.3, -0.4, 0.28)]
    if s == "parking":
        # Parking lot — two rows of wide rectangular obstacles (cars).
        # Mapped into the 10×8 nav room: cars at y≈±2.5 in two rows.
        layout = []
        for cx in (-3.0, -1.0, 1.0, 3.0):
            layout.append((cx,  2.5, 0.90))   # row A (y > 0)
            layout.append((cx, -2.5, 0.90))   # row B (y < 0)
        return layout
    if s in ("beach", "rooftop", "night"):
        # Open environments — minimal fixed obstacles; test pure navigation.
        return [(-3.5, 0.0, 0.25), (0.0, 2.5, 0.25), (3.5, 0.0, 0.25)]
    return list(OBSTACLES)

# ── Sensor / physics constants ─────────────────────────────────────────────────
N_LIDAR    = 8
LIDAR_R    = 5.0    # m  max ray range
DRONE_MASS = 1.5    # kg
HOVER_F    = DRONE_MASS * 9.81
VEL_GAIN   = 8.0
FORCE_CLIP = 12.0
SIM_DT     = 1 / 240
N_SUBSTEPS = 8

# ── Gym spaces ─────────────────────────────────────────────────────────────────
_OBS_LOW  = np.array([0.]*8 + [-12., -12., -5., -5.], dtype=np.float32)
_OBS_HIGH = np.array([1.]*8 + [ 12.,  12.,  5.,  5.], dtype=np.float32)


class ObstacleNavEnv(gym.Env):
    """PPO navigation env — fly from west to east through a pillar maze."""

    metadata = {"render_modes": []}

    def __init__(self, render: bool = False, max_steps: int = 1_200,
                 obstacles=None):
        super().__init__()
        self._render  = render
        self._max_steps = max_steps
        # Per-instance obstacle layout (scenario-aware); defaults to the maze.
        self.obstacles = [tuple(o) for o in (obstacles if obstacles else OBSTACLES)]
        self._phys_id  = None
        self._drone_id = None
        self._drone_pos = np.zeros(3)
        self._step_count = 0
        self._prev_dist  = 0.0

        self.observation_space = spaces.Box(_OBS_LOW, _OBS_HIGH, dtype=np.float32)
        self.action_space      = spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        sx = START_XY[0] + float(rng.uniform(-0.3, 0.3))
        sy = START_XY[1] + float(rng.uniform(-1.2, 1.2))

        if _PB:
            if self._phys_id is None:
                self._build_world()
            self._reset_drone(sx, sy)
        else:
            self._drone_pos = np.array([sx, sy, NAV_ALT])

        self._step_count = 0
        self._prev_dist  = self._goal_dist()
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -2.0, 2.0).astype(float)

        if _PB and self._phys_id is not None:
            vel = np.array(p.getBaseVelocity(self._drone_id,
                                              physicsClientId=self._phys_id)[0], dtype=float)
            fx = float(np.clip(VEL_GAIN * (action[0] - vel[0]), -FORCE_CLIP, FORCE_CLIP))
            fy = float(np.clip(VEL_GAIN * (action[1] - vel[1]), -FORCE_CLIP, FORCE_CLIP))
            total = [fx, fy, HOVER_F]
            for _ in range(N_SUBSTEPS):
                p.applyExternalForce(self._drone_id, -1, total, [0, 0, 0],
                                     p.WORLD_FRAME, physicsClientId=self._phys_id)
                p.stepSimulation(physicsClientId=self._phys_id)
            pos, _ = p.getBasePositionAndOrientation(self._drone_id,
                                                      physicsClientId=self._phys_id)
            self._drone_pos = np.array(pos)
        else:
            self._drone_pos[0] += action[0] * SIM_DT * N_SUBSTEPS
            self._drone_pos[1] += action[1] * SIM_DT * N_SUBSTEPS

        self._step_count += 1
        obs             = self._get_obs()
        reward, done    = self._compute_reward()
        truncated       = self._step_count >= self._max_steps

        return obs, reward, done, truncated, {}

    def get_viz_state(self) -> dict:
        """Return snapshot suitable for top-down visualisation."""
        return {
            "drone_xy": self._drone_pos[:2].copy(),
            "lidar":    self._cast_lidar(),
            "goal_xy":  np.array(GOAL_XY),
            "obstacles": self.obstacles,
            "room":     (ROOM_W, ROOM_D),
        }

    def close(self):
        if _PB and self._phys_id is not None:
            p.disconnect(self._phys_id)
            self._phys_id = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _goal_dist(self) -> float:
        return float(math.hypot(self._drone_pos[0] - GOAL_XY[0],
                                self._drone_pos[1] - GOAL_XY[1]))

    def _compute_reward(self):
        dist = self._goal_dist()
        reward = (self._prev_dist - dist) * 8.0 - 0.15
        self._prev_dist = dist

        if dist <= GOAL_R:
            return reward + 200.0, True

        # Wall collision
        if (self._drone_pos[0] < -ROOM_W/2 - 0.1 or self._drone_pos[0] > ROOM_W/2 + 0.1 or
                self._drone_pos[1] < -ROOM_D/2 - 0.1 or self._drone_pos[1] > ROOM_D/2 + 0.1):
            return reward - 30.0, True

        # Obstacle collision
        for ox, oy, r in self.obstacles:
            if math.hypot(self._drone_pos[0] - ox, self._drone_pos[1] - oy) < r + 0.18:
                return reward - 50.0, True

        return reward, False

    def _get_obs(self) -> np.ndarray:
        lidar = self._cast_lidar()
        dx    = float(np.clip(self._drone_pos[0] - GOAL_XY[0], -12, 12))
        dy    = float(np.clip(self._drone_pos[1] - GOAL_XY[1], -12, 12))

        if _PB and self._phys_id is not None:
            vel = np.array(p.getBaseVelocity(self._drone_id,
                                              physicsClientId=self._phys_id)[0], dtype=float)
            vx, vy = float(np.clip(vel[0], -5, 5)), float(np.clip(vel[1], -5, 5))
        else:
            vx, vy = 0.0, 0.0

        return np.array([*lidar, dx, dy, vx, vy], dtype=np.float32)

    def _cast_lidar(self) -> list:
        """Return 8 normalised lidar hit fractions (0 = obstacle right here, 1 = open)."""
        px, py = self._drone_pos[0], self._drone_pos[1]
        rays   = []
        angles = [i * (2 * math.pi / N_LIDAR) for i in range(N_LIDAR)]

        if _PB and self._phys_id is not None:
            z = NAV_ALT
            for a in angles:
                dx, dy = math.cos(a), math.sin(a)
                res = p.rayTest(
                    [px, py, z],
                    [px + dx * LIDAR_R, py + dy * LIDAR_R, z],
                    physicsClientId=self._phys_id,
                )
                hit = res[0][2] if res and res[0][0] != -1 else 1.0
                rays.append(float(hit))
        else:
            # Analytical stub — walls + circular obstacles
            for a in angles:
                dx, dy = math.cos(a), math.sin(a)
                t_min  = 1.0
                # Walls
                for (sign, axis) in [(1, 0), (-1, 0), (1, 1), (-1, 1)]:
                    limit = (ROOM_W / 2) * sign if axis == 0 else (ROOM_D / 2) * sign
                    comp  = dx if axis == 0 else dy
                    org   = px if axis == 0 else py
                    if abs(comp) > 1e-9:
                        t = (limit - org) / comp
                        if 0 < t <= LIDAR_R:
                            t_min = min(t_min, t / LIDAR_R)
                # Obstacles
                for ox, oy, r in self.obstacles:
                    fx, fy = ox - px, oy - py
                    b = fx * dx + fy * dy
                    c = fx*fx + fy*fy - (r + 0.18)**2
                    disc = b*b - c
                    if disc >= 0 and b > 0:
                        t = (b - math.sqrt(disc)) / LIDAR_R
                        if 0 < t < t_min:
                            t_min = t
                rays.append(float(np.clip(t_min, 0.0, 1.0)))

        return rays

    def _build_world(self):
        mode = p.GUI if self._render else p.DIRECT
        self._phys_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self._phys_id)
        p.setTimeStep(SIM_DT, physicsClientId=self._phys_id)
        p.loadURDF("plane.urdf", physicsClientId=self._phys_id)

        hw, hd = ROOM_W / 2, ROOM_D / 2
        for pos, he in [
            ([0, -hd, WALL_H/2], [hw, 0.08, WALL_H]),
            ([0,  hd, WALL_H/2], [hw, 0.08, WALL_H]),
            ([-hw, 0, WALL_H/2], [0.08, hd, WALL_H]),
            ([ hw, 0, WALL_H/2], [0.08, hd, WALL_H]),
        ]:
            c = p.createCollisionShape(p.GEOM_BOX, halfExtents=he, physicsClientId=self._phys_id)
            v = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=[0.4, 0.45, 0.55, 0.7],
                                    physicsClientId=self._phys_id)
            p.createMultiBody(0, c, v, pos, physicsClientId=self._phys_id)

        for ox, oy, r in self.obstacles:
            c = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=WALL_H,
                                       physicsClientId=self._phys_id)
            v = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=WALL_H,
                                    rgbaColor=[0.85, 0.35, 0.1, 1.0],
                                    physicsClientId=self._phys_id)
            p.createMultiBody(0, c, v, [ox, oy, WALL_H/2], physicsClientId=self._phys_id)

        # Goal zone (transparent green sphere)
        v = p.createVisualShape(p.GEOM_SPHERE, radius=GOAL_R, rgbaColor=[0.1, 0.9, 0.2, 0.35],
                                physicsClientId=self._phys_id)
        p.createMultiBody(0, -1, v, [GOAL_XY[0], GOAL_XY[1], NAV_ALT],
                          physicsClientId=self._phys_id)

        # Drone body
        c = p.createCollisionShape(p.GEOM_SPHERE, radius=0.20, physicsClientId=self._phys_id)
        v = p.createVisualShape(p.GEOM_SPHERE, radius=0.20, rgbaColor=[0.2, 0.5, 1.0, 1.0],
                                physicsClientId=self._phys_id)
        self._drone_id = p.createMultiBody(
            DRONE_MASS, c, v, [START_XY[0], START_XY[1], NAV_ALT],
            physicsClientId=self._phys_id,
        )
        p.changeDynamics(self._drone_id, -1, linearDamping=2.5, angularDamping=0.9,
                         physicsClientId=self._phys_id)

    def _reset_drone(self, sx, sy):
        p.resetBasePositionAndOrientation(
            self._drone_id, [sx, sy, NAV_ALT], [0, 0, 0, 1],
            physicsClientId=self._phys_id,
        )
        p.resetBaseVelocity(self._drone_id, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self._phys_id)
        for _ in range(10):
            p.stepSimulation(physicsClientId=self._phys_id)
        p.resetBaseVelocity(self._drone_id, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self._phys_id)
        pos, _ = p.getBasePositionAndOrientation(self._drone_id, physicsClientId=self._phys_id)
        self._drone_pos = np.array(pos)
