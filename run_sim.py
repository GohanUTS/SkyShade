"""
SkyShade — full integrated simulation runner.

Starts a PyBullet GUI window and runs all four subsystems together:
  Sub-1  Perception       HSV tracker + distance estimator (camera feed)
  Sub-2  Flight Control   PID hover controller
  Sub-3  Env Decision     SVM umbrella classifier
  Sub-4  Nav Safety       MDP policy table

A simulated user walks a figure-8 path below the drone.  Weather sensors
cycle through clear → cloudy → rainy → clear over 60 s.  Battery drains
and triggers the Sub-4 safety override.

Usage:
    python run_sim.py [--duration 120] [--no-gui]
"""

import argparse
import math
import sys
import time
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    sys.exit("pybullet not installed — run: pip install pybullet")

from sub1_perception.distance_estimator import DistanceEstimator
from sub1_perception.tracker import Tracker, CAMERA_RES, CAMERA_FOV, CONFIDENCE_THRESH
from sub2_flight.policy import PIDFlightPolicy
from sub2_flight.env.hover_env import (
    TARGET_ALTITUDE, HOVER_FORCE_N, DRONE_MASS_KG,
    SIM_TIMESTEP, STEPS_PER_ACTION, LINEAR_DAMPING,
)
from sub3_env.classifier import UmbrellaClassifier
from sub4_nav.policy_table import NavSafetyPolicy
from sub4_nav.mdp import ACTION_NAMES

CONTROL_HZ         = 30
BATTERY_DRAIN_RATE = 0.5    # % per second
USER_WALK_SPEED    = 0.3    # rad/s for figure-8
HOVER_RADIUS       = 0.5    # m
WEATHER_PERIOD     = 60.0   # s


def figure8(t, scale=1.5):
    d = 1 + math.sin(t) ** 2
    return np.array([scale * math.cos(t) / d,
                     scale * math.sin(t) * math.cos(t) / d,
                     0.0])


def weather_at(t):
    phase = (t % WEATHER_PERIOD) / WEATHER_PERIOD
    lux   = 80_000 * (0.5 + 0.5 * math.cos(2 * math.pi * phase))
    rain  = float(max(0.0, math.sin(2 * math.pi * phase)))
    wind  = 1.5 + 2.5 * abs(math.sin(math.pi * phase))
    return float(lux), rain, float(wind)


def run(duration=120.0, gui=True):
    mode = p.GUI if gui else p.DIRECT
    phys = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=phys)
    p.setGravity(0, 0, -9.81, physicsClientId=phys)
    p.setTimeStep(SIM_TIMESTEP, physicsClientId=phys)

    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=phys)
        p.resetDebugVisualizerCamera(
            cameraDistance=7, cameraYaw=30, cameraPitch=-25,
            cameraTargetPosition=[0, 0, 1],
            physicsClientId=phys,
        )

    p.loadURDF("plane.urdf", physicsClientId=phys)

    # Drone
    drone_id = p.loadURDF("cube_small.urdf",
                          basePosition=[0, 0, TARGET_ALTITUDE],
                          physicsClientId=phys)
    p.changeDynamics(drone_id, -1, mass=DRONE_MASS_KG,
                     linearDamping=LINEAR_DAMPING, angularDamping=0.9,
                     physicsClientId=phys)
    p.changeVisualShape(drone_id, -1, rgbaColor=[0.2, 0.5, 1.0, 1], physicsClientId=phys)

    # User — loaded as a sphere URDF-style via createMultiBody with colour
    user_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.15, physicsClientId=phys)
    user_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.15,
                                   rgbaColor=[1, 0.1, 0.1, 1], physicsClientId=phys)
    user_id  = p.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=user_col,
                                 baseVisualShapeIndex=user_vis,
                                 basePosition=[0, 0, 0.15],
                                 physicsClientId=phys)

    # Umbrella disc (flat cylinder above the drone)
    umb_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.7, height=0.04,
                                     physicsClientId=phys)
    umb_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.7, length=0.04,
                                  rgbaColor=[0.5, 0.5, 0.5, 0.4], physicsClientId=phys)
    umb_id  = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=umb_col,
                                baseVisualShapeIndex=umb_vis,
                                basePosition=[0, 0, TARGET_ALTITUDE + 0.35],
                                physicsClientId=phys)

    # Settle with hover thrust
    for _ in range(30):
        p.applyExternalForce(drone_id, -1, [0, 0, HOVER_FORCE_N],
                             [0, 0, 0], p.WORLD_FRAME, physicsClientId=phys)
        p.stepSimulation(physicsClientId=phys)
    p.resetBaseVelocity(drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=phys)

    # Subsystems
    tracker    = Tracker(DistanceEstimator())
    pid        = PIDFlightPolicy()
    umbrella   = UmbrellaClassifier()
    nav_policy = NavSafetyPolicy()

    battery_pct   = 100.0
    umbrella_cmd  = "STOW"
    nav_override  = "CONTINUE"
    confidence    = 0.0
    user_pos      = np.array([0.0, 0.0, 0.0])
    dt            = STEPS_PER_ACTION * SIM_TIMESTEP
    action_dt     = 1.0 / CONTROL_HZ

    # HUD text IDs — one slot per line, updated in-place
    hud_labels = [
        "Time    :", "Battery :", "Nav     :",
        "Umbrella:", "Tracking:", "HoverErr:", "Drone z :",
    ]
    hud_ids = []
    if gui:
        # Place HUD text at fixed world positions (column on left side of view)
        for i, label in enumerate(hud_labels):
            uid = p.addUserDebugText(
                label + " --", [-4.5, 0, 4.2 - i * 0.4],
                textColorRGB=[1, 1, 0.3], textSize=1.1,
                physicsClientId=phys,
            )
            hud_ids.append(uid)

    umb_deployed = False
    t_start = time.monotonic()
    tick    = 0

    print("SkyShade simulation running — Ctrl+C to stop.\n")
    print(f"{'Time':>6}  {'Bat':>6}  {'Nav':<10}  {'Umbrella':<8}  {'Conf':>5}  {'ErrXY':>6}  {'Z':>5}")
    print("-" * 65)

    try:
        while True:
            t_wall = time.monotonic() - t_start
            if t_wall >= duration:
                print(f"\nSimulation complete ({duration:.0f} s).")
                break

            # ── User walks figure-8 ───────────────────────────────────────────
            user_pos = figure8(t_wall * USER_WALK_SPEED)
            p.resetBasePositionAndOrientation(
                user_id, [user_pos[0], user_pos[1], 0.15],
                [0, 0, 0, 1], physicsClientId=phys,
            )

            # ── Battery ───────────────────────────────────────────────────────
            battery_pct = max(0.0, battery_pct - BATTERY_DRAIN_RATE * action_dt)

            # ── Sub-1: Perception — camera pointing down from drone ────────────
            d_pos, _ = p.getBasePositionAndOrientation(drone_id, physicsClientId=phys)
            drone_pos = np.array(d_pos)

            W, H = CAMERA_RES
            eye    = drone_pos.tolist()
            target = [drone_pos[0], drone_pos[1], drone_pos[2] - 1.0]
            view   = p.computeViewMatrix(eye, target, [0, 1, 0])
            proj   = p.computeProjectionMatrixFOV(CAMERA_FOV, W / H, 0.1, 20)
            _, _, px, _, _ = p.getCameraImage(
                W, H, view, proj,
                physicsClientId=phys,
                renderer=p.ER_TINY_RENDERER,
            )
            frame = np.array(px, dtype=np.uint8).reshape((H, W, 4))[:, :, :3]
            user_offset, confidence = tracker.process_frame(frame)

            # ── Sub-3: Env decision (1 Hz) ────────────────────────────────────
            if tick % CONTROL_HZ == 0:
                lux, rain, wind = weather_at(t_wall)
                cmd_int      = umbrella.predict(lux, rain, wind)
                umbrella_cmd = "DEPLOY" if cmd_int else "STOW"
                deployed     = cmd_int == 1
                if deployed != umb_deployed:
                    col = [0.1, 0.8, 0.2, 0.85] if deployed else [0.5, 0.5, 0.5, 0.4]
                    p.changeVisualShape(umb_id, -1, rgbaColor=col, physicsClientId=phys)
                    umb_deployed = deployed

            # ── Sub-4: Nav safety (5 Hz) ──────────────────────────────────────
            # The MDP is solved purely for safety (RTH always beats CONTINUE
            # because CONTINUE never earns REWARD_SAFE_COMPLETION).  For the
            # demo we suppress RTH while battery is healthy (>50%) so the
            # drone actually follows the user; the safety override only kicks
            # in when battery genuinely needs attention.
            if tick % max(1, CONTROL_HZ // 5) == 0:
                dist_home  = float(np.linalg.norm(drone_pos[:2]))
                action_idx = nav_policy.decide(battery_pct, dist_home)
                raw_override = ACTION_NAMES[action_idx]
                if battery_pct > 50.0 and raw_override == "RTH":
                    nav_override = "CONTINUE"   # battery healthy — keep following
                else:
                    nav_override = raw_override

            # ── Sub-2: PID flight ─────────────────────────────────────────────
            lin_vel, _ = p.getBaseVelocity(drone_id, physicsClientId=phys)
            drone_vel  = np.array(lin_vel)

            if nav_override == "LAND_NOW":
                pid_target = np.array([drone_pos[0], drone_pos[1], 0.3])
            elif nav_override == "RTH":
                pid_target = np.array([0.0, 0.0, TARGET_ALTITUDE])
            else:
                # Follow user position directly (Sub-1 camera offset used for display)
                pid_target = np.array([user_pos[0], user_pos[1], TARGET_ALTITUDE])

            force = pid.compute_force(drone_pos, drone_vel, pid_target, dt)
            force[2] += HOVER_FORCE_N

            for _ in range(STEPS_PER_ACTION):
                p.applyExternalForce(drone_id, -1, force.tolist(),
                                     [0, 0, 0], p.WORLD_FRAME, physicsClientId=phys)
                p.stepSimulation(physicsClientId=phys)

            # Keep umbrella attached to drone
            d_pos2, _ = p.getBasePositionAndOrientation(drone_id, physicsClientId=phys)
            p.resetBasePositionAndOrientation(
                umb_id,
                [d_pos2[0], d_pos2[1], d_pos2[2] + 0.35],
                [0, 0, 0, 1], physicsClientId=phys,
            )

            # ── HUD update (every 10 ticks) ───────────────────────────────────
            if gui and tick % 10 == 0 and hud_ids:
                err_xy = float(np.linalg.norm(np.array(d_pos2[:2]) - user_pos[:2]))
                values = [
                    f"{t_wall:6.1f} s",
                    f"{battery_pct:5.1f} %",
                    nav_override,
                    umbrella_cmd,
                    f"{confidence:.2f}  ({'OK' if confidence >= CONFIDENCE_THRESH else 'LOW'})",
                    f"{err_xy:.2f} m  ({'OK' if err_xy <= HOVER_RADIUS else '--'})",
                    f"{d_pos2[2]:.2f} m",
                ]
                for uid, label, val in zip(hud_ids, hud_labels, values):
                    p.addUserDebugText(
                        f"{label} {val}",
                        [-4.5, 0, 4.2 - hud_ids.index(uid) * 0.4],
                        textColorRGB=[1, 1, 0.3], textSize=1.1,
                        replaceItemUniqueId=uid,
                        physicsClientId=phys,
                    )

            # ── Console log every 10 s ────────────────────────────────────────
            if tick % (CONTROL_HZ * 10) == 0:
                err_xy = float(np.linalg.norm(np.array(d_pos2[:2]) - user_pos[:2]))
                print(f"{t_wall:6.1f}s  {battery_pct:5.1f}%  {nav_override:<10}  "
                      f"{umbrella_cmd:<8}  {confidence:.2f}  {err_xy:6.2f}m  "
                      f"{d_pos2[2]:.2f}m")

            # ── Frame-rate cap ────────────────────────────────────────────────
            if gui:
                elapsed = time.monotonic() - t_start - t_wall
                sleep_t = action_dt - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

            tick += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        p.disconnect(phys)
        print("Disconnected.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--no-gui",   action="store_true")
    args = ap.parse_args()
    run(duration=args.duration, gui=not args.no_gui)


if __name__ == "__main__":
    main()
