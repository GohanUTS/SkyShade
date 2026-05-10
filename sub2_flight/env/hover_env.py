"""
Sub-2 Flight — PyBullet hover environment.

The drone is represented as a rigid box (0.4 m × 0.4 m × 0.1 m) loaded from
a minimal URDF.  Because we focus on the Q-learning logic rather than
aerodynamics, thrust is applied directly as a force on the body — no rotor
model is included.

Coordinate convention (world frame):
    +x → East    +y → North    +z → Up

State returned to the agent is a tuple of four integers:
    (dx_idx, dy_idx, dz_idx, wind_idx)
where each index references a bucket defined below.

Actions:
    0 MOVE_NORTH  1 MOVE_SOUTH  2 MOVE_EAST  3 MOVE_WEST
    4 MOVE_UP     5 MOVE_DOWN   6 HOLD
"""

import math
import numpy as np

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("[hover_env] WARNING: pybullet not installed — using stub physics.")

# ── State discretisation ──────────────────────────────────────────────────────
DX_EDGES = [-2.0, -1.0, 0.0, 1.0, 2.0]   # 5 buckets for lateral x offset
DY_EDGES = [-2.0, -1.0, 0.0, 1.0, 2.0]   # 5 buckets for lateral y offset
DZ_EDGES = [-1.0, 0.0, 1.0]              # 3 buckets for altitude offset
WIND_LEVELS = [0, 1, 2]                   # 0=LOW, 1=MEDIUM, 2=HIGH

N_DX = len(DX_EDGES)   # 5
N_DY = len(DY_EDGES)   # 5
N_DZ = len(DZ_EDGES)   # 3
N_WIND = len(WIND_LEVELS)  # 3
N_STATES = N_DX * N_DY * N_DZ * N_WIND  # 225

N_ACTIONS = 7

# ── Drone physics constants ───────────────────────────────────────────────────
DRONE_MASS_KG = 1.5          # kg
HOVER_FORCE_N = DRONE_MASS_KG * 9.81  # Newtons to counteract gravity
STEP_FORCE_N = 5.0           # Force applied per action step
TARGET_ALTITUDE = 2.5        # Metres above ground (desired hover height)
HOVER_RADIUS_M = 0.5         # Within this radius counts as "hovering"
MAX_TILT_RAD = math.radians(30)  # Penalise tilts beyond 30°

# ── Reward shaping ────────────────────────────────────────────────────────────
STEP_PENALTY = -0.1
HOVER_BONUS = +5.0
ATTITUDE_PENALTY_SCALE = -2.0
CRASH_PENALTY = -100.0
PROGRESS_SCALE = 10.0        # Multiplier on approach progress

# ── Simulation constants ──────────────────────────────────────────────────────
SIM_TIMESTEP = 1.0 / 240.0
STEPS_PER_ACTION = 8         # Physics steps per agent action
MAX_EPISODE_STEPS = 500
# ─────────────────────────────────────────────────────────────────────────────


def _digitise(value: float, edges: list) -> int:
    """Map a continuous value to the nearest bucket index."""
    return int(np.digitize(value, sorted(edges)) - 1)


def _wind_speed_to_idx(wind_speed: float) -> int:
    if wind_speed < 2.0:
        return 0   # LOW
    elif wind_speed < 5.0:
        return 1   # MEDIUM
    return 2       # HIGH


class HoverEnv:
    """Quadcopter hover environment backed by PyBullet (or a lightweight stub)."""

    def __init__(self, render: bool = False):
        self._render = render
        self._physics_id = None
        self._drone_id = None
        self._user_id = None
        self._step_count = 0
        self._prev_dist = None
        self._user_pos = np.array([0.0, 0.0, 0.0])  # on the ground
        self._wind_speed = 0.0

    # ── Gym-style interface ───────────────────────────────────────────────────

    def reset(self, wind_speed: float = 0.0, user_pos=None):
        """Reset the episode.  Returns initial discrete state index."""
        self._wind_speed = wind_speed
        self._step_count = 0

        if user_pos is not None:
            self._user_pos = np.array(user_pos, dtype=float)
        else:
            self._user_pos = np.zeros(3)

        if PYBULLET_AVAILABLE:
            self._init_pybullet()
        else:
            self._drone_pos = np.array([0.0, 0.0, TARGET_ALTITUDE], dtype=float)

        target = np.array([self._user_pos[0], self._user_pos[1], TARGET_ALTITUDE])
        self._prev_dist = np.linalg.norm(self._drone_pos - target)

        return self._get_state()

    def step(self, action: int):
        """
        Apply action, advance physics, return (state, reward, done, info).

        action: integer in [0, 6]
        """
        self._apply_action(action)
        self._step_count += 1

        state = self._get_state()
        reward = self._compute_reward()
        done = self._is_done()
        info = {"wind_speed": self._wind_speed, "step": self._step_count}

        return state, reward, done, info

    def close(self):
        if PYBULLET_AVAILABLE and self._physics_id is not None:
            p.disconnect(self._physics_id)
            self._physics_id = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _init_pybullet(self):
        if self._physics_id is not None:
            p.disconnect(self._physics_id)

        mode = p.GUI if self._render else p.DIRECT
        self._physics_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self._physics_id)
        p.setTimeStep(SIM_TIMESTEP, physicsClientId=self._physics_id)

        # Ground plane
        p.loadURDF("plane.urdf", physicsClientId=self._physics_id)

        # Drone — simple box URDF loaded inline
        drone_start = [self._user_pos[0], self._user_pos[1], TARGET_ALTITUDE]
        self._drone_id = p.loadURDF(
            "cube_small.urdf",
            basePosition=drone_start,
            physicsClientId=self._physics_id,
        )
        p.changeDynamics(
            self._drone_id, -1, mass=DRONE_MASS_KG,
            physicsClientId=self._physics_id
        )

        # User marker — a small sphere on the ground
        user_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.15,
                                          physicsClientId=self._physics_id)
        self._user_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=user_col,
            basePosition=self._user_pos.tolist(),
            physicsClientId=self._physics_id,
        )

        # Step once to settle
        for _ in range(10):
            p.stepSimulation(physicsClientId=self._physics_id)

        pos, _ = p.getBasePositionAndOrientation(
            self._drone_id, physicsClientId=self._physics_id
        )
        self._drone_pos = np.array(pos)

    def _apply_action(self, action: int):
        """Translate discrete action to a force vector and step the sim."""
        force_map = {
            0: [0,  STEP_FORCE_N, 0],    # NORTH
            1: [0, -STEP_FORCE_N, 0],    # SOUTH
            2: [STEP_FORCE_N,  0, 0],    # EAST
            3: [-STEP_FORCE_N, 0, 0],    # WEST
            4: [0, 0,  STEP_FORCE_N],    # UP
            5: [0, 0, -STEP_FORCE_N],    # DOWN
            6: [0, 0, 0],               # HOLD
        }
        thrust = force_map.get(action, [0, 0, 0])

        # Wind disturbance — random lateral force
        wind_dir = np.random.uniform(-1, 1, 3)
        wind_dir[2] = 0   # horizontal only
        wind_force = wind_dir * self._wind_speed

        total_force = np.array(thrust) + wind_force
        total_force[2] += HOVER_FORCE_N  # constant anti-gravity thrust

        if PYBULLET_AVAILABLE and self._drone_id is not None:
            p.applyExternalForce(
                self._drone_id, -1,
                total_force.tolist(),
                [0, 0, 0],
                p.LINK_FRAME,
                physicsClientId=self._physics_id,
            )
            for _ in range(STEPS_PER_ACTION):
                p.stepSimulation(physicsClientId=self._physics_id)
            pos, _ = p.getBasePositionAndOrientation(
                self._drone_id, physicsClientId=self._physics_id
            )
            self._drone_pos = np.array(pos)
        else:
            # Stub: move drone analytically
            self._drone_pos += (np.array(thrust) * SIM_TIMESTEP * STEPS_PER_ACTION * 0.5)

    def _get_state(self) -> int:
        """Discretise current (dx, dy, dz, wind) into a flat state index."""
        target = np.array([self._user_pos[0], self._user_pos[1], TARGET_ALTITUDE])
        delta = self._drone_pos - target

        dx_idx = np.clip(_digitise(delta[0], DX_EDGES), 0, N_DX - 1)
        dy_idx = np.clip(_digitise(delta[1], DY_EDGES), 0, N_DY - 1)
        dz_idx = np.clip(_digitise(delta[2], DZ_EDGES), 0, N_DZ - 1)
        wind_idx = _wind_speed_to_idx(self._wind_speed)

        return int(dx_idx * N_DY * N_DZ * N_WIND
                   + dy_idx * N_DZ * N_WIND
                   + dz_idx * N_WIND
                   + wind_idx)

    def _compute_reward(self) -> float:
        target = np.array([self._user_pos[0], self._user_pos[1], TARGET_ALTITUDE])
        dist = np.linalg.norm(self._drone_pos - target)

        # Progress reward
        progress = (self._prev_dist - dist) * PROGRESS_SCALE
        self._prev_dist = dist

        reward = progress + STEP_PENALTY

        # Hover bonus
        if dist <= HOVER_RADIUS_M:
            reward += HOVER_BONUS

        # Attitude penalty (requires PyBullet orientation)
        if PYBULLET_AVAILABLE and self._drone_id is not None:
            _, orn = p.getBasePositionAndOrientation(
                self._drone_id, physicsClientId=self._physics_id
            )
            euler = p.getEulerFromQuaternion(orn)
            tilt = math.sqrt(euler[0] ** 2 + euler[1] ** 2)
            if tilt > MAX_TILT_RAD:
                reward += ATTITUDE_PENALTY_SCALE * (tilt - MAX_TILT_RAD)

        # Crash (hit the ground)
        if self._drone_pos[2] < 0.2:
            reward += CRASH_PENALTY

        return float(reward)

    def _is_done(self) -> bool:
        if self._step_count >= MAX_EPISODE_STEPS:
            return True
        if self._drone_pos[2] < 0.1:  # crashed
            return True
        return False

    @property
    def drone_pos(self) -> np.ndarray:
        return self._drone_pos.copy()
