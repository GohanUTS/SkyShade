"""
Runtime flight-control helpers for the integrated SkyShade simulation.

This module keeps Sub-2's flight decisions out of run_sim.py.  The simulator
provides state (drone pose, velocity, user pose, nav override, wind) and receives
a corrective force.  PPO, legacy Q-learning, and PID can then evolve here
without spreading flight logic through the rest of the project.
"""

from dataclasses import dataclass

import numpy as np

from sub2_flight.env.hover_env import (
    HOVER_RADIUS_M,
    TARGET_ALTITUDE,
    action_to_force,
    discretize_state,
)
from sub2_flight.policy import ACTION_NAMES, FlightPolicy, PIDFlightPolicy, PPOFlightPolicy


@dataclass
class FlightControlResult:
    """Output of one runtime flight-controller tick."""

    force: np.ndarray
    target: np.ndarray
    controller: str
    action: str
    reward: float | None
    using_q: bool
    using_learned: bool


def lead_follow_target(
    user_pos,
    user_velocity,
    lead_seconds: float = 0.5,
    max_lead_m: float = 0.7,
    target_altitude: float = TARGET_ALTITUDE,
) -> np.ndarray:
    """Return a hover target almost directly above the walking user."""
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
    """Runtime adapter — PPO primary, Q-table legacy fallback, PID final fallback."""

    def __init__(self, mode: str = "ppo"):
        raw = str(mode).lower()
        if raw.startswith("p") and raw != "pid":
            self.requested_mode = "ppo"
        elif raw.startswith("q"):
            self.requested_mode = "q"
        else:
            self.requested_mode = "pid"
        self.pid = PIDFlightPolicy()
        self.pid.KP_XY = 28.0
        self.pid.KI_XY = 0.08
        self.pid.KD_XY = 12.0
        self.pid.MAX_FORCE_XY = 34.0
        self.pid.KP_Z = 10.0
        self.pid.KD_Z = 6.0
        self.pid.MAX_FORCE_Z = 12.0
        self.ppo_policy = None
        self.q_policy = None
        self.fallback_reason = None
        # Lead-follow params — tuned per scenario via tune_pid_for_forest()
        self._lead_seconds = 0.5
        self._max_lead_m   = 0.7

        if self.requested_mode == "ppo":
            try:
                self.ppo_policy = PPOFlightPolicy()
            except Exception as ppo_exc:
                try:
                    self.q_policy = FlightPolicy()
                    self.fallback_reason = (
                        f"PPO unavailable ({ppo_exc}); using legacy Q-table"
                    )
                except (FileNotFoundError, ValueError) as q_exc:
                    self.fallback_reason = (
                        f"PPO unavailable ({ppo_exc}); Q-table unavailable ({q_exc})"
                    )
        elif self.requested_mode == "q":
            try:
                self.q_policy = FlightPolicy()
            except (FileNotFoundError, ValueError) as q_exc:
                self.fallback_reason = f"Q-table unavailable ({q_exc})"

    @property
    def using_q(self) -> bool:
        return self.q_policy is not None

    @property
    def using_learned(self) -> bool:
        return self.ppo_policy is not None or self.q_policy is not None

    @property
    def label(self) -> str:
        if self.ppo_policy is not None:
            return "PPO"
        if self.q_policy is not None:
            return "Q-RL"
        return "PID"

    def reset(self):
        self.pid.reset()

    def tune_pid_for_forest(self):
        """Conservative tuning for the winding forest trail.

        The trail changes direction constantly so a large look-ahead causes the
        drone to chase a phantom point that keeps shifting, creating oscillation.
        We zero the look-ahead (follow directly above the user) and use moderate
        gains to stay stable through tight turns.
        """
        self.pid.KP_XY = 20.0
        self.pid.KI_XY = 0.06
        self.pid.KD_XY = 11.0
        self.pid.MAX_FORCE_XY = 22.0
        self._lead_seconds = 0.0   # no look-ahead — trail direction changes too fast
        self._max_lead_m   = 0.0

    def target_for_override(
        self,
        nav_override: str,
        drone_pos,
        user_pos,
        user_velocity,
        altitude: float = TARGET_ALTITUDE,
    ) -> np.ndarray:
        if nav_override == "LAND_NOW":
            return np.array([drone_pos[0], drone_pos[1], 0.3], dtype=float)
        if nav_override == "RTH":
            return np.array([0.0, 0.0, altitude], dtype=float)
        return lead_follow_target(user_pos, user_velocity,
                                  lead_seconds=self._lead_seconds,
                                  max_lead_m=self._max_lead_m,
                                  target_altitude=altitude)

    def compute_force(
        self,
        drone_pos,
        drone_vel,
        user_pos,
        user_velocity,
        nav_override: str,
        wind_speed: float,
        dt: float,
        altitude_override: float = None,
    ) -> FlightControlResult:
        """Compute corrective force, excluding constant hover thrust."""
        from sub2_flight.env.ppo_hover_env import OBS_LOW, OBS_HIGH

        drone_pos = np.array(drone_pos, dtype=float)
        drone_vel = np.array(drone_vel, dtype=float)
        target_alt = altitude_override if altitude_override is not None else TARGET_ALTITUDE
        target = self.target_for_override(nav_override, drone_pos, user_pos, user_velocity,
                                          altitude=target_alt)

        # ── PPO path ──────────────────────────────────────────────────────────
        if self.ppo_policy is not None and nav_override != "LAND_NOW":
            target_3d = np.array([target[0], target[1], target_alt], dtype=float)
            delta = drone_pos - target_3d
            obs = np.array([
                delta[0], delta[1], delta[2],
                drone_vel[0], drone_vel[1], drone_vel[2],
                float(np.clip(wind_speed / 5.0, 0.0, 1.0)),
            ], dtype=np.float32)
            obs = np.clip(obs, OBS_LOW, OBS_HIGH)

            ppo_force = self.ppo_policy.compute_force(obs, drone_vel)
            pid_force = self.pid.compute_force(drone_pos, drone_vel, target_3d, dt)
            lateral_error = float(np.linalg.norm(delta[:2]))

            # Beyond 3 m the PPO policy is completely out of its training
            # distribution — use pure PID with boosted gain to recover fast.
            if lateral_error > 3.0:
                force = pid_force * min(2.0, lateral_error / 3.0)
            else:
                assist = float(np.clip((lateral_error - 0.10) / 0.45, 0.65, 1.0))
                force = (1.0 - assist) * ppo_force + assist * pid_force
            force = np.clip(force, -20.0, 20.0)

            dist3d = float(np.linalg.norm(drone_pos - target_3d))
            reward = 5.0 if dist3d <= HOVER_RADIUS_M else max(-5.0, -dist3d)

            return FlightControlResult(
                force=force,
                target=target_3d,
                controller="PPO+PID",
                action="PPO+STABLE",
                reward=reward,
                using_q=False,
                using_learned=True,
            )

        # ── Legacy Q-table path ───────────────────────────────────────────────
        if self.q_policy is not None and nav_override != "LAND_NOW":
            state = discretize_state(
                drone_pos, drone_vel, [target[0], target[1], 0.0], wind_speed,
            )
            action = self.q_policy.select_action(state)
            force = action_to_force(action)

            target_hover = np.array([target[0], target[1], target_alt], dtype=float)
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
                using_learned=True,
            )

        # ── PID fallback ──────────────────────────────────────────────────────
        force = self.pid.compute_force(drone_pos, drone_vel, target, dt)
        return FlightControlResult(
            force=force,
            target=target,
            controller="PID",
            action="PID_FORCE",
            reward=None,
            using_q=False,
            using_learned=False,
        )
