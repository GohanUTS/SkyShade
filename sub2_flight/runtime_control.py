"""
Runtime flight-control helpers for the integrated SkyShade simulation.

This module keeps Sub-2's flight decisions out of run_sim.py.  The simulator
provides state (drone pose, velocity, user pose, nav override, wind) and receives
a corrective force.  PID and Q-learning can then evolve here without spreading
flight logic through the rest of the project.
"""

from dataclasses import dataclass

import numpy as np

from sub2_flight.env.hover_env import (
    HOVER_RADIUS_M,
    TARGET_ALTITUDE,
    action_to_force,
    discretize_state,
)
from sub2_flight.policy import ACTION_NAMES, FlightPolicy, PIDFlightPolicy


@dataclass
class FlightControlResult:
    """Output of one runtime flight-controller tick."""

    force: np.ndarray
    target: np.ndarray
    controller: str
    action: str
    reward: float | None
    using_q: bool


def lead_follow_target(
    user_pos,
    user_velocity,
    lead_seconds: float = 0.85,
    max_lead_m: float = 1.05,
    target_altitude: float = TARGET_ALTITUDE,
) -> np.ndarray:
    """Return a hover target slightly ahead of the walking user."""
    lead = np.array(user_velocity[:2], dtype=float) * lead_seconds
    lead_norm = float(np.linalg.norm(lead))
    if lead_norm > max_lead_m:
        lead *= max_lead_m / lead_norm

    return np.array(
        [user_pos[0] + lead[0], user_pos[1] + lead[1], target_altitude],
        dtype=float,
    )


def obstacle_avoidance_force(
    drone_pos,
    obstacles,
    radius: float = 0.75,
    gain: float = 4.5,
    max_force: float = 5.5,
) -> np.ndarray:
    """Repulsive XY force away from nearby circular obstacles."""
    if not obstacles:
        return np.zeros(3, dtype=float)

    force_xy = np.zeros(2, dtype=float)
    drone_xy = np.array(drone_pos[:2], dtype=float)
    nearby = []
    for obstacle in obstacles:
        away = drone_xy - obstacle["position"]
        distance = float(np.linalg.norm(away))
        clearance = distance - obstacle["radius"]
        nearby.append((clearance, distance, away))

    for clearance, distance, away in sorted(nearby, key=lambda item: item[0])[:6]:
        if distance < 1e-5 or clearance >= radius:
            continue

        direction = away / distance
        proximity = (radius - clearance) / radius
        force_xy += direction * gain * proximity * proximity

    magnitude = float(np.linalg.norm(force_xy))
    if magnitude > max_force:
        force_xy *= max_force / magnitude

    return np.array([force_xy[0], force_xy[1], 0.0], dtype=float)


class RuntimeFlightController:
    """Runtime adapter for PID or learned Q-flight control."""

    def __init__(self, mode: str = "pid"):
        self.requested_mode = "q" if str(mode).lower().startswith("q") else "pid"
        self.pid = PIDFlightPolicy()
        self.q_policy = None
        self.fallback_reason = None

        if self.requested_mode == "q":
            try:
                self.q_policy = FlightPolicy()
            except (FileNotFoundError, ValueError) as exc:
                self.fallback_reason = str(exc)

    @property
    def using_q(self) -> bool:
        return self.q_policy is not None

    @property
    def label(self) -> str:
        return "Q-RL" if self.using_q else "PID"

    def reset(self):
        self.pid.reset()

    def tune_pid_for_forest(self):
        """Use stronger lateral gains for the long forest trail."""
        self.pid.KP_XY = 13.0
        self.pid.KI_XY = 0.15
        self.pid.KD_XY = 7.0
        self.pid.MAX_FORCE_XY = 15.0

    def target_for_override(
        self,
        nav_override: str,
        drone_pos,
        user_pos,
        user_velocity,
    ) -> np.ndarray:
        if nav_override == "LAND_NOW":
            return np.array([drone_pos[0], drone_pos[1], 0.3], dtype=float)
        if nav_override == "RTH":
            return np.array([0.0, 0.0, TARGET_ALTITUDE], dtype=float)
        return lead_follow_target(user_pos, user_velocity)

    def compute_force(
        self,
        drone_pos,
        drone_vel,
        user_pos,
        user_velocity,
        nav_override: str,
        wind_speed: float,
        dt: float,
    ) -> FlightControlResult:
        """Compute corrective force, excluding constant hover thrust."""
        drone_pos = np.array(drone_pos, dtype=float)
        drone_vel = np.array(drone_vel, dtype=float)
        target = self.target_for_override(nav_override, drone_pos, user_pos, user_velocity)

        if self.using_q and nav_override != "LAND_NOW":
            state = discretize_state(
                drone_pos,
                drone_vel,
                [target[0], target[1], 0.0],
                wind_speed,
            )
            action = self.q_policy.select_action(state)
            force = action_to_force(action)

            target_hover = np.array([target[0], target[1], TARGET_ALTITUDE], dtype=float)
            dist3d = float(np.linalg.norm(drone_pos - target_hover))
            lat_speed = float(np.hypot(drone_vel[0], drone_vel[1]))
            reward = (5.0 if dist3d <= HOVER_RADIUS_M else max(-5.0, -dist3d))
            reward -= 0.5 * lat_speed

            return FlightControlResult(
                force=force,
                target=target_hover,
                controller="Q-RL",
                action=ACTION_NAMES[action],
                reward=reward,
                using_q=True,
            )

        force = self.pid.compute_force(drone_pos, drone_vel, target, dt)
        return FlightControlResult(
            force=force,
            target=target,
            controller="PID",
            action="PID_FORCE",
            reward=None,
            using_q=False,
        )
