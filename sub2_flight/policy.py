"""
Sub-2 Flight — flight policy implementations.

FlightPolicy    : greedy policy backed by a pre-trained Q-table.
PIDFlightPolicy : continuous PID hover controller (no Q-table required).

The Q-learning environment now includes velocity buckets in its state, so the
learned table can brake instead of flying through the hover target. PID remains
the default production controller because it is smoother for the live demo.
"""

import os
import numpy as np

from sub2_flight.env.hover_env import N_STATES, N_ACTIONS, TARGET_ALTITUDE

ACTION_NAMES = ["MOVE_NORTH", "MOVE_SOUTH", "MOVE_EAST", "MOVE_WEST",
                "MOVE_UP", "MOVE_DOWN", "HOLD"]

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "qtable_v1.npy"
)


class FlightPolicy:
    """Greedy policy backed by a pre-trained Q-table."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Q-table not found at {model_path}. "
                "Run sub2_flight/train_qlearning.py first."
            )
        self._q_table = np.load(model_path)
        if self._q_table.shape != (N_STATES, N_ACTIONS):
            raise ValueError(
                f"Q-table shape mismatch: expected ({N_STATES}, {N_ACTIONS}), "
                f"got {self._q_table.shape}"
            )

    def select_action(self, state: int) -> int:
        """Return the greedy action for the given state index."""
        return int(np.argmax(self._q_table[state]))

    def action_name(self, action: int) -> str:
        return ACTION_NAMES[action]


class PIDFlightPolicy:
    """
    PID hover controller.

    Reads continuous position + velocity from the environment each tick and
    outputs a 3-D force vector via env.pid_step().  It remains the smooth
    default runtime controller; the Q-table is available for learned-policy
    demos and validation.

    Gains are tuned for:
      DRONE_MASS_KG = 1.5 kg
      STEPS_PER_ACTION = 8 substeps at 1/240 s  (action dt ≈ 33 ms)
    """

    # PID gains — lateral (XY) and vertical (Z) tuned separately
    KP_XY = 8.0
    KI_XY = 0.1
    KD_XY = 6.0

    KP_Z  = 8.0
    KI_Z  = 0.2
    KD_Z  = 5.0

    # Anti-windup clamp on the integral term (Newtons)
    INTEGRAL_CLAMP_XY = 2.0
    INTEGRAL_CLAMP_Z  = 3.0

    # Maximum corrective force per axis (Newtons) — keeps behaviour realistic
    MAX_FORCE_XY = 8.0
    MAX_FORCE_Z  = 8.0

    def __init__(self):
        self._integral = np.zeros(3)
        self._prev_error = np.zeros(3)

    def reset(self):
        self._integral = np.zeros(3)
        self._prev_error = np.zeros(3)

    def compute_force(self, drone_pos: np.ndarray, drone_vel: np.ndarray,
                      target_pos: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute the corrective force vector (world frame, excluding hover thrust).

        Parameters
        ----------
        drone_pos  : current drone position [x, y, z]
        drone_vel  : current drone velocity [vx, vy, vz]
        target_pos : desired hover position [x, y, z]
        dt         : time elapsed since last call (seconds)

        Returns
        -------
        np.ndarray shape (3,) — corrective force in Newtons
        """
        error = target_pos - drone_pos

        # Integrate with anti-windup clamp
        self._integral += error * dt
        self._integral[:2] = np.clip(self._integral[:2],
                                     -self.INTEGRAL_CLAMP_XY, self.INTEGRAL_CLAMP_XY)
        self._integral[2]  = np.clip(self._integral[2],
                                     -self.INTEGRAL_CLAMP_Z, self.INTEGRAL_CLAMP_Z)

        # PD on XY (derivative from velocity, not finite-difference)
        fx = self.KP_XY * error[0] + self.KI_XY * self._integral[0] - self.KD_XY * drone_vel[0]
        fy = self.KP_XY * error[1] + self.KI_XY * self._integral[1] - self.KD_XY * drone_vel[1]
        fz = self.KP_Z  * error[2] + self.KI_Z  * self._integral[2] - self.KD_Z  * drone_vel[2]

        force = np.array([
            np.clip(fx, -self.MAX_FORCE_XY, self.MAX_FORCE_XY),
            np.clip(fy, -self.MAX_FORCE_XY, self.MAX_FORCE_XY),
            np.clip(fz, -self.MAX_FORCE_Z,  self.MAX_FORCE_Z),
        ])

        self._prev_error = error.copy()
        return force

    def run_episode(self, env, wind_speed: float = 0.0, user_pos=None,
                    walk: bool = False, rng=None):
        """
        Run one full episode using PID control.

        Parameters
        ----------
        env        : HoverEnv instance (must support pid_step and drone_vel)
        wind_speed : wind speed passed to env.reset
        user_pos   : starting user position
        walk       : if True, shift user position every 20 steps (stage 3)
        rng        : numpy Generator (for user walk noise)

        Returns
        -------
        (total_reward, reached_hover)
        """
        from sub2_flight.env.hover_env import STEPS_PER_ACTION, SIM_TIMESTEP
        dt = STEPS_PER_ACTION * SIM_TIMESTEP  # ≈ 0.0333 s per action tick

        self.reset()
        state = env.reset(wind_speed=wind_speed, user_pos=user_pos)

        total_reward = 0.0
        reached_hover = False
        step = 0

        while True:
            target = np.array([env.user_pos[0], env.user_pos[1], TARGET_ALTITUDE])
            force = self.compute_force(env.drone_pos, env.drone_vel, target, dt)

            state, reward, done, _ = env.pid_step(force)
            total_reward += reward

            dist_2d = np.linalg.norm(env.drone_pos[:2] - env.user_pos[:2])
            if dist_2d <= 0.5:
                reached_hover = True

            step += 1

            if walk and step % 20 == 0 and rng is not None:
                env._user_pos[0] += rng.uniform(-0.2, 0.2)
                env._user_pos[1] += rng.uniform(-0.2, 0.2)
                env.notify_target_moved()

            if done:
                break

        return total_reward, reached_hover
