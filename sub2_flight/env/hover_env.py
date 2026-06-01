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
# Position offset buckets (drone − target).
DX_EDGES = [-2.0, -1.0, 0.0, 1.0, 2.0]   # 5 buckets for lateral x offset
DY_EDGES = [-2.0, -1.0, 0.0, 1.0, 2.0]   # 5 buckets for lateral y offset
DZ_EDGES = [-1.0, 0.0, 1.0]              # 3 buckets for altitude offset
WIND_LEVELS = [0, 1, 2]                   # 0=LOW, 1=MEDIUM, 2=HIGH

# Velocity buckets — signed (moving negative / ≈still / moving positive).
# Adding velocity to the state is THE fix that lets the tabular agent learn to
# brake: without it the agent cannot tell a momentum-carrying fly-through from a
# stable hover, so it perpetually overshoots and crashes.
VEL_THRESH = 0.3   # m/s — below this magnitude the axis counts as "still"

N_DX = len(DX_EDGES)   # 5
N_DY = len(DY_EDGES)   # 5
N_DZ = len(DZ_EDGES)   # 3
N_VX = 3
N_VY = 3
N_VZ = 3
N_WIND = len(WIND_LEVELS)  # 3
N_STATES = N_DX * N_DY * N_DZ * N_VX * N_VY * N_VZ * N_WIND  # 6075

N_ACTIONS = 7

# ── Drone physics constants ───────────────────────────────────────────────────
DRONE_MASS_KG = 1.5          # kg
HOVER_FORCE_N = DRONE_MASS_KG * 9.81  # Newtons to counteract gravity
STEP_FORCE_N = 6.0           # Force per action step — gives the discrete agent
                             # clear authority over the gusty wind disturbance
                             # (which can reach ~4 N) while damping still arrests
                             # drift within a hover radius
TARGET_ALTITUDE = 2.5        # Metres above ground (desired hover height)
HOVER_RADIUS_M = 0.5         # Within this radius counts as "hovering"
MAX_TILT_RAD = math.radians(30)  # Penalise tilts beyond 30°

# ── Reward shaping ────────────────────────────────────────────────────────────
STEP_PENALTY = -0.1
HOVER_BONUS = +5.0
STABLE_HOVER_BONUS = +3.0    # Extra reward when inside the radius AND nearly still
STABLE_SPEED_THRESH = 0.3    # m/s lateral speed counted as "nearly still"
ATTITUDE_PENALTY_SCALE = -2.0
CRASH_PENALTY = -100.0
PROGRESS_SCALE = 10.0        # Multiplier on approach progress
SPEED_PENALTY_SCALE = -0.5   # Penalise lateral speed.  Crucially this also keeps
                             # the calm-air (deterministic, no-wind) policy stable:
                             # strongly preferring stillness stops the greedy agent
                             # drifting into a limit cycle when there's no wind to
                             # knock it out.  In gusty wind it makes the reward go
                             # negative from unavoidable buffeting — that's why the
                             # test scores the Q-agent on *convergence* (does it
                             # hold a hover) with reward only as a divergence guard.
LINEAR_DAMPING = 2.5         # PyBullet damping — must match what the PPO was trained
                             # with.  Changing this value invalidates the saved model.
                             # run_sim.py uses this same value so the learned policy
                             # transfers correctly.

# ── Simulation constants ──────────────────────────────────────────────────────
SIM_TIMESTEP = 1.0 / 240.0
STEPS_PER_ACTION = 8         # Physics steps per agent action
MAX_EPISODE_STEPS = 500
AMBIENT_TURBULENCE = 0.4     # Always-present micro-gust (m/s-equivalent).  Real
                             # air is never perfectly still; this small constant
                             # disturbance stops the greedy discrete policy getting
                             # stuck in a limit cycle in the "no wind" case, and is
                             # negligible at the higher wind levels.
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


def _vel_to_idx(v: float) -> int:
    """Signed velocity bucket: 0=moving negative, 1=≈still, 2=moving positive."""
    if v < -VEL_THRESH:
        return 0
    if v > VEL_THRESH:
        return 2
    return 1


# Discrete action → thrust vector (Newtons, world frame, excluding hover thrust).
FORCE_MAP = {
    0: [0,  STEP_FORCE_N, 0],    # NORTH
    1: [0, -STEP_FORCE_N, 0],    # SOUTH
    2: [STEP_FORCE_N,  0, 0],    # EAST
    3: [-STEP_FORCE_N, 0, 0],    # WEST
    4: [0, 0,  STEP_FORCE_N],    # UP
    5: [0, 0, -STEP_FORCE_N],    # DOWN
    6: [0, 0, 0],               # HOLD
}


def action_to_force(action: int) -> np.ndarray:
    """Map a discrete action to its thrust vector (no hover/wind component)."""
    return np.array(FORCE_MAP.get(action, [0, 0, 0]), dtype=float)


def discretize_state(drone_pos, drone_vel, user_pos, wind_speed: float) -> int:
    """
    Map continuous (pos, vel, user_pos, wind) to the flat discrete state index.

    Shared by HoverEnv (training/eval) and run_sim.py (live Q-flight) so both
    sides bucket identically.  Mixed-radix flatten over
    (dx, dy, dz, vx, vy, vz, wind).
    """
    drone_pos = np.asarray(drone_pos, dtype=float)
    drone_vel = np.asarray(drone_vel, dtype=float)
    target = np.array([user_pos[0], user_pos[1], TARGET_ALTITUDE])
    delta = drone_pos - target

    dx = int(np.clip(_digitise(delta[0], DX_EDGES), 0, N_DX - 1))
    dy = int(np.clip(_digitise(delta[1], DY_EDGES), 0, N_DY - 1))
    dz = int(np.clip(_digitise(delta[2], DZ_EDGES), 0, N_DZ - 1))
    vx = _vel_to_idx(drone_vel[0])
    vy = _vel_to_idx(drone_vel[1])
    vz = _vel_to_idx(drone_vel[2])
    w = _wind_speed_to_idx(wind_speed)

    idx = dx
    idx = idx * N_DY + dy
    idx = idx * N_DZ + dz
    idx = idx * N_VX + vx
    idx = idx * N_VY + vy
    idx = idx * N_VZ + vz
    idx = idx * N_WIND + w
    return int(idx)


class HoverEnv:
    """Quadcopter hover environment backed by PyBullet (or a lightweight stub)."""

    def __init__(self, render: bool = False, max_episode_steps: int = MAX_EPISODE_STEPS):
        self._render = render
        self._max_steps = max_episode_steps
        self._physics_id = None
        self._drone_id = None
        self._user_id = None
        self._step_count = 0
        self._prev_dist = None
        self._user_pos = np.array([0.0, 0.0, 0.0])  # on the ground
        self._drone_offset = np.zeros(3)             # drone start offset from target
        self._wind_speed = 0.0

    # ── Gym-style interface ───────────────────────────────────────────────────

    def reset(self, wind_speed: float = 0.0, user_pos=None, drone_offset=None):
        """Reset the episode.  Returns initial discrete state index.

        drone_offset : optional (dx, dy, dz) the drone starts at *relative to the
            hover target*.  Training samples this so the agent learns to approach
            and recover from off-target states (not just hold a perfect start) —
            without it the policy only ever sees the goal cell and drifts away the
            moment physics noise pushes it out.
        """
        self._wind_speed = wind_speed
        self._step_count = 0

        if user_pos is not None:
            self._user_pos = np.array(user_pos, dtype=float)
        else:
            self._user_pos = np.zeros(3)

        self._drone_offset = (np.array(drone_offset, dtype=float)
                              if drone_offset is not None else np.zeros(3))

        if PYBULLET_AVAILABLE:
            if self._physics_id is None:
                self._init_pybullet()   # connect + load world/bodies ONCE
            self._reset_bodies()        # cheap per-episode reposition
        else:
            self._drone_pos = (np.array([0.0, 0.0, TARGET_ALTITUDE], dtype=float)
                               + self._drone_offset)

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

    def pid_step(self, force_xyz: np.ndarray):
        """
        Apply a continuous 3-D force vector (world frame) for one action tick.
        Used by PIDFlightPolicy; bypasses the discrete action lookup.
        Returns (state, reward, done, info) like step().
        """
        wind_dir = np.random.uniform(-1, 1, 3)
        wind_dir[2] = 0
        wind_force = wind_dir * (self._wind_speed + AMBIENT_TURBULENCE)

        total = np.array(force_xyz) + wind_force
        total[2] += HOVER_FORCE_N  # always add anti-gravity

        if PYBULLET_AVAILABLE and self._drone_id is not None:
            for _ in range(STEPS_PER_ACTION):
                p.applyExternalForce(
                    self._drone_id, -1,
                    total.tolist(),
                    [0, 0, 0],
                    p.WORLD_FRAME,
                    physicsClientId=self._physics_id,
                )
                p.stepSimulation(physicsClientId=self._physics_id)
            pos, _ = p.getBasePositionAndOrientation(
                self._drone_id, physicsClientId=self._physics_id
            )
            self._drone_pos = np.array(pos)
        else:
            self._drone_pos += total * SIM_TIMESTEP * STEPS_PER_ACTION * 0.1

        self._step_count += 1
        state = self._get_state()
        reward = self._compute_reward()
        done = self._is_done()
        return state, reward, done, {"wind_speed": self._wind_speed, "step": self._step_count}

    def notify_target_moved(self):
        """Re-sync the progress baseline after the user (target) moves mid-episode.

        Call this whenever ``_user_pos`` is changed during an episode (the
        walking-user curriculum) so the next step's progress reward reflects
        only the drone's motion, not the target jump.
        """
        target = np.array([self._user_pos[0], self._user_pos[1], TARGET_ALTITUDE])
        self._prev_dist = float(np.linalg.norm(self._drone_pos - target))

    def close(self):
        if PYBULLET_AVAILABLE and self._physics_id is not None:
            p.disconnect(self._physics_id)
            self._physics_id = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _init_pybullet(self):
        """Connect and load the world + bodies once for the env's lifetime."""
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
            linearDamping=LINEAR_DAMPING, angularDamping=0.9,
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

    def _reset_bodies(self):
        """Reposition the existing drone/user to episode start (no reload)."""
        p.resetBasePositionAndOrientation(
            self._user_id, self._user_pos.tolist(), [0, 0, 0, 1],
            physicsClientId=self._physics_id,
        )
        p.resetBasePositionAndOrientation(
            self._drone_id,
            [self._user_pos[0] + self._drone_offset[0],
             self._user_pos[1] + self._drone_offset[1],
             TARGET_ALTITUDE + self._drone_offset[2]],
            [0, 0, 0, 1],
            physicsClientId=self._physics_id,
        )
        p.resetBaseVelocity(
            self._drone_id, [0, 0, 0], [0, 0, 0],
            physicsClientId=self._physics_id,
        )

        # Step briefly to let the physics world settle, then zero out the
        # velocity the drone gained during the free-fall settle steps — otherwise
        # that initial downward velocity persists throughout the episode because
        # the hover force exactly cancels gravity (zero net acceleration).
        for _ in range(10):
            p.stepSimulation(physicsClientId=self._physics_id)
        p.resetBaseVelocity(
            self._drone_id, [0, 0, 0], [0, 0, 0],
            physicsClientId=self._physics_id
        )

        pos, _ = p.getBasePositionAndOrientation(
            self._drone_id, physicsClientId=self._physics_id
        )
        self._drone_pos = np.array(pos)

    def _apply_action(self, action: int):
        """Translate discrete action to a force vector and step the sim."""
        thrust = action_to_force(action)

        # Wind disturbance — random lateral force (plus ambient micro-turbulence)
        wind_dir = np.random.uniform(-1, 1, 3)
        wind_dir[2] = 0   # horizontal only
        wind_force = wind_dir * (self._wind_speed + AMBIENT_TURBULENCE)

        # All forces in world frame so the hover thrust always points global-Z up,
        # regardless of drone tilt (LINK_FRAME would rotate the thrust with the body).
        # applyExternalForce only lasts ONE PyBullet step, so we re-apply it every
        # step inside the loop — otherwise gravity wins for 7/8 of the substeps.
        movement_and_wind = np.array(thrust) + wind_force
        movement_and_wind[2] += HOVER_FORCE_N  # constant anti-gravity thrust

        if PYBULLET_AVAILABLE and self._drone_id is not None:
            for _ in range(STEPS_PER_ACTION):
                p.applyExternalForce(
                    self._drone_id, -1,
                    movement_and_wind.tolist(),
                    [0, 0, 0],
                    p.WORLD_FRAME,
                    physicsClientId=self._physics_id,
                )
                p.stepSimulation(physicsClientId=self._physics_id)
            pos, _ = p.getBasePositionAndOrientation(
                self._drone_id, physicsClientId=self._physics_id
            )
            self._drone_pos = np.array(pos)
        else:
            # Stub: move drone analytically
            self._drone_pos += (np.array(thrust) * SIM_TIMESTEP * STEPS_PER_ACTION * 0.5)

    def _get_state(self) -> int:
        """Discretise current (dx, dy, dz, vx, vy, vz, wind) into a flat index."""
        return discretize_state(
            self._drone_pos, self.drone_vel, self._user_pos, self._wind_speed
        )

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

        # Attitude penalty + lateral speed penalty (requires PyBullet)
        if PYBULLET_AVAILABLE and self._drone_id is not None:
            _, orn = p.getBasePositionAndOrientation(
                self._drone_id, physicsClientId=self._physics_id
            )
            euler = p.getEulerFromQuaternion(orn)
            tilt = math.sqrt(euler[0] ** 2 + euler[1] ** 2)
            if tilt > MAX_TILT_RAD:
                reward += ATTITUDE_PENALTY_SCALE * (tilt - MAX_TILT_RAD)

            # Penalise lateral speed so the Q-agent prefers low-velocity hovering
            # over oscillation, and reward holding *still* on target so it learns
            # to brake rather than fly straight through the goal.
            lin_vel, _ = p.getBaseVelocity(self._drone_id, physicsClientId=self._physics_id)
            lateral_speed = math.sqrt(lin_vel[0] ** 2 + lin_vel[1] ** 2)
            reward += SPEED_PENALTY_SCALE * lateral_speed
            if dist <= HOVER_RADIUS_M and lateral_speed < STABLE_SPEED_THRESH:
                reward += STABLE_HOVER_BONUS

        # Crash (hit the ground)
        if self._drone_pos[2] < 0.2:
            reward += CRASH_PENALTY

        return float(reward)

    def _is_done(self) -> bool:
        if self._step_count >= self._max_steps:
            return True
        if self._drone_pos[2] < 0.1:  # crashed
            return True
        return False

    @property
    def drone_pos(self) -> np.ndarray:
        return self._drone_pos.copy()

    @property
    def drone_vel(self) -> np.ndarray:
        if PYBULLET_AVAILABLE and self._drone_id is not None:
            lin, _ = p.getBaseVelocity(self._drone_id, physicsClientId=self._physics_id)
            return np.array(lin)
        return np.zeros(3)

    @property
    def user_pos(self) -> np.ndarray:
        return self._user_pos.copy()
