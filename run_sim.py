"""
SkyShade — full integrated simulation runner.

Starts a PyBullet GUI window and runs all four subsystems together:
  Sub-1  Perception       HSV tracker + distance estimator (camera feed)
  Sub-2  Flight Control   PID hover controller
  Sub-3  Env Decision     SVM umbrella classifier
  Sub-4  Nav Safety       MDP policy table

A simulated user walks a figure-8 path below the drone.  Weather sensors
randomly switch between cloudy periods, light rain, and full rain.
Battery drains and triggers the Sub-4 safety override.

Usage:
    python run_sim.py [--duration 120] [--no-gui]
"""

import argparse
import math
import subprocess
import sys
import threading
import time
import os
import cv2
import numpy as np

try:
    import tkinter as tk
except ImportError:
    tk = None

sys.path.insert(0, os.path.dirname(__file__))

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    sys.exit("pybullet not installed — run: pip install pybullet")

from sub1_perception.distance_estimator import DistanceEstimator
from sub1_perception.tracker import (
    Tracker, CAMERA_RES, CAMERA_FOV, CONFIDENCE_THRESH,
    HSV_LOWER, HSV_UPPER, HSV_LOWER2, HSV_UPPER2, MIN_CONTOUR_AREA,
)
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
WEATHER_MIN_SECONDS = 8.0
WEATHER_MAX_SECONDS = 20.0
DEBUG_WEATHER_LINES = False
SCENARIO_PARK = "park"
SCENARIO_FOREST = "forest"
AVOIDANCE_RADIUS = 0.75
AVOIDANCE_GAIN = 4.5
AVOIDANCE_MAX_FORCE = 5.5
FOLLOW_LEAD_SECONDS = 0.85
FOLLOW_LEAD_MAX_METERS = 1.05
GIMBAL_LOOKAHEAD_SECONDS = 0.95
GIMBAL_SEARCH_RADIUS = 0.55
USER_MARKER_HEIGHT = 1.62
POV_DISPLAY_SIZE = (960, 540)
FOREST_TRAIL_START_X = -9.0
FOREST_TRAIL_END_X = 9.0
FOREST_TRAIL_LENGTH = FOREST_TRAIL_END_X - FOREST_TRAIL_START_X


def figure8(t, scale=1.5):
    d = 1 + math.sin(t) ** 2
    return np.array([scale * math.cos(t) / d,
                     scale * math.sin(t) * math.cos(t) / d,
                     0.0])


def forest_walk(t):
    x = FOREST_TRAIL_START_X + (t * 0.65) % FOREST_TRAIL_LENGTH
    y = 0.55 * math.sin(t * 0.85) + 0.22 * math.sin(t * 1.7)
    return np.array([x, y, 0.0])


class RandomWeatherController:
    """Holds each weather mode briefly, then randomly switches to a new one."""

    MODES = {
        "Cloudy": {
            "lux": (28_000, 56_000),
            "rain": (0.0, 0.18),
            "wind": (1.4, 3.0),
            "weight": 0.40,
        },
        "Light rain": {
            "lux": (15_000, 38_000),
            "rain": (0.25, 0.55),
            "wind": (2.0, 4.0),
            "weight": 0.30,
        },
        "Full rain": {
            "lux": (4_000, 18_000),
            "rain": (0.85, 1.0),
            "wind": (3.2, 5.8),
            "weight": 0.30,
        },
    }

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.mode = None
        self.next_switch_t = 0.0
        self.next_sample_t = 0.0
        self.reading = None

    def update(self, t):
        if self.mode is None or t >= self.next_switch_t:
            self._choose_next_mode(t)

        if self.reading is None or t >= self.next_sample_t:
            self._sample_reading(t)

        return self.reading

    def _choose_next_mode(self, t):
        names = list(self.MODES)
        weights = np.array([self.MODES[name]["weight"] for name in names], dtype=float)
        if self.mode in names and len(names) > 1:
            weights[names.index(self.mode)] *= 0.25
        weights /= weights.sum()

        self.mode = str(self.rng.choice(names, p=weights))
        hold_for = self.rng.uniform(WEATHER_MIN_SECONDS, WEATHER_MAX_SECONDS)
        self.next_switch_t = t + hold_for
        self.next_sample_t = 0.0

    def _sample_reading(self, t):
        spec = self.MODES[self.mode]
        lux = float(self.rng.uniform(*spec["lux"]))
        rain = float(self.rng.uniform(*spec["rain"]))
        wind = float(self.rng.uniform(*spec["wind"]))
        self.reading = {
            "mode": self.mode,
            "lux": lux,
            "rain": rain,
            "wind": wind,
        }
        self.next_sample_t = t + 1.0


def _add_debug_ellipse(phys, center, radius_x, radius_z, color, line_width, lifetime):
    cx, cy, cz = center
    segments = 14
    prev = None
    for i in range(segments + 1):
        angle = 2.0 * math.pi * i / segments
        point = [
            cx + math.cos(angle) * radius_x,
            cy,
            cz + math.sin(angle) * radius_z,
        ]
        if prev is not None:
            p.addUserDebugLine(
                prev, point, color,
                lineWidth=line_width, lifeTime=lifetime,
                physicsClientId=phys,
            )
        prev = point


def _draw_cloud(phys, x, y, z, scale, color):
    lifetime = 0.55
    puffs = [
        (-0.55, 0.00, 0.42, 0.18),
        (-0.18, 0.14, 0.52, 0.24),
        (0.25, 0.06, 0.46, 0.20),
        (0.62, -0.02, 0.34, 0.16),
    ]
    for dx, dz, rx, rz in puffs:
        _add_debug_ellipse(
            phys,
            [x + dx * scale, y, z + dz * scale],
            rx * scale,
            rz * scale,
            color,
            3.0,
            lifetime,
        )

    base_left = [x - 0.95 * scale, y, z - 0.16 * scale]
    base_right = [x + 0.95 * scale, y, z - 0.16 * scale]
    p.addUserDebugLine(
        base_left, base_right, color,
        lineWidth=4.0, lifeTime=lifetime,
        physicsClientId=phys,
    )


def _draw_cloud_bank(phys, mode):
    if mode == "Cloudy":
        color = [0.68, 0.73, 0.80]
    else:
        color = [0.42, 0.48, 0.58]

    for cloud in [
        (-3.0, -3.9, 4.10, 1.10),
        (-0.7, -4.1, 4.35, 1.25),
        (1.7, -3.8, 4.05, 1.00),
        (3.4, -4.2, 4.30, 0.90),
    ]:
        _draw_cloud(phys, *cloud, color)


def draw_weather_visuals(phys, weather, rng):
    mode = weather["mode"]
    rain = weather["rain"]
    wind = weather["wind"]

    _draw_cloud_bank(phys, mode)

    if rain < 0.2:
        return

    drop_count = int(30 + rain * 90)
    for _ in range(drop_count):
        x = float(rng.uniform(-4.0, 4.0))
        y = float(rng.uniform(-4.0, 4.0))
        z = float(rng.uniform(2.4, 5.2))
        drift = 0.07 * wind
        p.addUserDebugLine(
            [x, y, z],
            [x + drift, y + drift * 0.35, z - 1.05],
            [0.18, 0.55, 1.0],
            lineWidth=2.2 + rain * 2.0,
            lifeTime=0.42,
            physicsClientId=phys,
        )


def _visual_body(phys, shape, rgba, base_position, **shape_kwargs):
    vis = p.createVisualShape(
        shape,
        rgbaColor=rgba,
        physicsClientId=phys,
        **shape_kwargs,
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=base_position,
        physicsClientId=phys,
    )


def _static_box(phys, half_extents, position, rgba):
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=half_extents, physicsClientId=phys,
    )
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba,
        physicsClientId=phys,
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=position,
        physicsClientId=phys,
    )


def _static_cylinder(phys, radius, height, position, rgba):
    col = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=phys,
    )
    vis = p.createVisualShape(
        p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgba,
        physicsClientId=phys,
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=position,
        physicsClientId=phys,
    )


def _rotate_xy(offset, yaw):
    x, y, z = offset
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([c * x - s * y, s * x + c * y, z], dtype=float)


def _tree(phys, x, y, trunk_radius=0.11, trunk_height=1.8, crown_radius=0.48):
    trunk = _static_cylinder(
        phys, trunk_radius, trunk_height, [x, y, trunk_height / 2.0],
        [0.33, 0.19, 0.09, 1],
    )
    crown = _visual_body(
        phys, p.GEOM_SPHERE, [0.06, 0.31, 0.13, 1],
        [x, y, trunk_height + crown_radius * 0.35], radius=crown_radius,
    )
    return {
        "id": trunk,
        "position": np.array([x, y], dtype=float),
        "radius": trunk_radius,
    }


def build_park_environment(phys):
    plane_id = p.loadURDF("plane.urdf", physicsClientId=phys)
    obstacles = []

    # Park-like ground details give scale and depth without affecting dynamics.
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.28, 0.43, 0.25, 1], physicsClientId=phys)
    _static_box(phys, [4.5, 0.08, 0.01], [0, 0, 0.012], [0.45, 0.42, 0.36, 1])
    _static_box(phys, [0.08, 4.5, 0.01], [0, 0, 0.014], [0.45, 0.42, 0.36, 1])

    for x, y, sx, sy in [
        (-3.3, 2.8, 0.35, 0.25),
        (3.1, -2.9, 0.45, 0.30),
        (-2.8, -3.1, 0.30, 0.32),
        (3.4, 2.7, 0.36, 0.28),
    ]:
        _static_box(phys, [sx, sy, 0.025], [x, y, 0.035], [0.23, 0.50, 0.21, 1])

    for x, y in [(-3.6, -1.4), (-2.9, 2.1), (2.8, 1.8), (3.5, -0.8)]:
        obstacles.append(_tree(phys, x, y, trunk_radius=0.08, trunk_height=0.9, crown_radius=0.38))

    for x, y, yaw in [(-1.7, 3.2, 0.0), (2.2, -3.0, math.pi / 2)]:
        bench = _static_box(phys, [0.55, 0.12, 0.08], [x, y, 0.42], [0.43, 0.24, 0.12, 1])
        p.resetBasePositionAndOrientation(
            bench, [x, y, 0.42], p.getQuaternionFromEuler([0, 0, yaw]),
            physicsClientId=phys,
        )

    return obstacles


def build_forest_environment(phys):
    plane_id = p.loadURDF("plane.urdf", physicsClientId=phys)
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.12, 0.25, 0.13, 1], physicsClientId=phys)

    obstacles = []

    # A winding dirt trail through denser trees. The user stays roughly on the
    # trail; the drone is high enough to see the marker but must avoid trunks.
    trail_xs = np.linspace(FOREST_TRAIL_START_X - 0.5, FOREST_TRAIL_END_X + 0.5, 35)
    for x in trail_xs:
        y = 0.55 * math.sin((x - FOREST_TRAIL_START_X) * 0.85)
        _static_box(
            phys, [0.36, 0.20, 0.012], [float(x), float(y), 0.018],
            [0.43, 0.34, 0.22, 1],
        )

    rng = np.random.default_rng(42)
    tree_specs = []
    for x in np.linspace(FOREST_TRAIL_START_X - 0.2, FOREST_TRAIL_END_X + 0.2, 30):
        trail_y = 0.55 * math.sin((x - FOREST_TRAIL_START_X) * 0.85)
        for side in (-1.0, 1.0):
            lateral_gap = rng.uniform(0.95, 1.75)
            jitter_x = rng.uniform(-0.18, 0.18)
            jitter_y = rng.uniform(-0.20, 0.20)
            radius = rng.uniform(0.09, 0.14)
            tree_specs.append((float(x + jitter_x), float(trail_y + side * lateral_gap + jitter_y), float(radius)))

    for x in np.linspace(FOREST_TRAIL_START_X, FOREST_TRAIL_END_X, 13):
        tree_specs.append((float(x), 2.45 + 0.28 * math.sin(x), 0.11))
        tree_specs.append((float(x + 0.35), -2.45 + 0.24 * math.cos(x * 0.7), 0.12))

    for x, y, radius in tree_specs:
        obstacles.append(
            _tree(
                phys, x, y, trunk_radius=radius,
                trunk_height=1.7 + 1.5 * radius,
                crown_radius=0.42 + 1.2 * radius,
            )
        )

    for x in np.linspace(FOREST_TRAIL_START_X + 1.0, FOREST_TRAIL_END_X - 1.0, 9):
        y = 0.55 * math.sin((x - FOREST_TRAIL_START_X) * 0.85)
        _static_box(phys, [0.18, 0.09, 0.05], [x, y, 0.055], [0.28, 0.25, 0.21, 1])

    return obstacles


def build_environment(phys, scenario):
    if scenario == SCENARIO_FOREST:
        return build_forest_environment(phys)
    return build_park_environment(phys)


def scenario_user_position(scenario, t_wall):
    if scenario == SCENARIO_FOREST:
        return forest_walk(t_wall)
    return figure8(t_wall * USER_WALK_SPEED)


def obstacle_avoidance_force(drone_pos, obstacles):
    if not obstacles:
        return np.zeros(3, dtype=float)

    force_xy = np.zeros(2, dtype=float)
    drone_xy = np.array(drone_pos[:2], dtype=float)
    nearby = []
    for obstacle in obstacles:
        away = drone_xy - obstacle["position"]
        distance = float(np.linalg.norm(away))
        clearance = distance - obstacle["radius"]
        nearby.append((clearance, distance, away, obstacle))

    for clearance, distance, away, obstacle in sorted(nearby, key=lambda item: item[0])[:6]:
        if distance < 1e-5 or clearance >= AVOIDANCE_RADIUS:
            continue

        direction = away / distance
        proximity = (AVOIDANCE_RADIUS - clearance) / AVOIDANCE_RADIUS
        strength = AVOIDANCE_GAIN * proximity * proximity
        force_xy += direction * strength

    magnitude = float(np.linalg.norm(force_xy))
    if magnitude > AVOIDANCE_MAX_FORCE:
        force_xy *= AVOIDANCE_MAX_FORCE / magnitude

    return np.array([force_xy[0], force_xy[1], 0.0], dtype=float)


def lead_follow_target(user_pos, user_velocity):
    lead = np.array(user_velocity[:2], dtype=float) * FOLLOW_LEAD_SECONDS
    lead_norm = float(np.linalg.norm(lead))
    if lead_norm > FOLLOW_LEAD_MAX_METERS:
        lead *= FOLLOW_LEAD_MAX_METERS / lead_norm
    return np.array([
        user_pos[0] + lead[0],
        user_pos[1] + lead[1],
        TARGET_ALTITUDE,
    ])


def predictive_gimbal_target(user_pos, user_velocity, confidence, t_wall):
    lead = np.array(user_velocity[:2], dtype=float) * GIMBAL_LOOKAHEAD_SECONDS
    lead_norm = float(np.linalg.norm(lead))
    if lead_norm > FOLLOW_LEAD_MAX_METERS:
        lead *= FOLLOW_LEAD_MAX_METERS / lead_norm

    target = np.array([
        user_pos[0] + lead[0],
        user_pos[1] + lead[1],
        USER_MARKER_HEIGHT,
    ], dtype=float)

    if confidence < CONFIDENCE_THRESH:
        # If the marker is weak or lost, sweep around the predicted position.
        sweep = t_wall * 2.2
        target[0] += GIMBAL_SEARCH_RADIUS * math.cos(sweep)
        target[1] += GIMBAL_SEARCH_RADIUS * math.sin(sweep)

    return target


def describe_gimbal_action(drone_pos, target):
    delta = target - np.array(drone_pos, dtype=float)
    directions = []
    if abs(delta[0]) > 0.25:
        directions.append("right" if delta[0] > 0 else "left")
    if abs(delta[1]) > 0.25:
        directions.append("forward" if delta[1] > 0 else "back")
    if abs(delta[2]) > 0.20:
        directions.append("up" if delta[2] > 0 else "down")
    return " + ".join(directions) if directions else "center"


def create_drone(phys, start_xy=None):
    if start_xy is None:
        start_xy = [0.0, 0.0]
    start_position = [float(start_xy[0]), float(start_xy[1]), TARGET_ALTITUDE]
    body_col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.18, 0.13, 0.06], physicsClientId=phys,
    )
    body_vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.18, 0.13, 0.06],
        rgbaColor=[0.08, 0.12, 0.16, 1], physicsClientId=phys,
    )
    drone_id = p.createMultiBody(
        baseMass=DRONE_MASS_KG,
        baseCollisionShapeIndex=body_col,
        baseVisualShapeIndex=body_vis,
        basePosition=start_position,
        physicsClientId=phys,
    )
    p.changeDynamics(
        drone_id, -1, mass=DRONE_MASS_KG,
        linearDamping=LINEAR_DAMPING, angularDamping=0.9,
        physicsClientId=phys,
    )

    parts = []
    for offset, half_extents in [
        ([0.38, 0.0, 0.0], [0.42, 0.025, 0.025]),
        ([0.0, 0.38, 0.0], [0.025, 0.42, 0.025]),
    ]:
        part_id = _visual_body(
            phys, p.GEOM_BOX, [0.12, 0.16, 0.20, 1],
            start_position, halfExtents=half_extents,
        )
        parts.append({"id": part_id, "offset": np.array(offset), "roll": 0.0})

    for x in (-0.44, 0.44):
        for y in (-0.44, 0.44):
            rotor = _visual_body(
                phys, p.GEOM_CYLINDER, [0.04, 0.04, 0.05, 0.65],
                [start_position[0] + x, start_position[1] + y, TARGET_ALTITUDE + 0.03],
                radius=0.18, length=0.018,
            )
            hub = _visual_body(
                phys, p.GEOM_SPHERE, [0.72, 0.76, 0.82, 1],
                [start_position[0] + x, start_position[1] + y, TARGET_ALTITUDE + 0.04],
                radius=0.045,
            )
            parts.append({"id": rotor, "offset": np.array([x, y, 0.035]), "roll": 0.0})
            parts.append({"id": hub, "offset": np.array([x, y, 0.045]), "roll": 0.0})

    for x in (-0.16, 0.16):
        skid = _visual_body(
            phys, p.GEOM_BOX, [0.75, 0.80, 0.85, 1],
            [start_position[0] + x, start_position[1], TARGET_ALTITUDE - 0.18],
            halfExtents=[0.018, 0.34, 0.018],
        )
        parts.append({"id": skid, "offset": np.array([x, 0.0, -0.18]), "roll": 0.0})

    return drone_id, parts


def update_drone_parts(phys, parts, drone_pos, rotor_angle):
    for part in parts:
        part_pos = np.array(drone_pos) + part["offset"]
        quat = [0, 0, 0, 1]
        if abs(part["offset"][0]) > 0.3 and abs(part["offset"][1]) > 0.3:
            quat = p.getQuaternionFromEuler([0, 0, rotor_angle])
        p.resetBasePositionAndOrientation(
            part["id"], part_pos.tolist(), quat, physicsClientId=phys,
        )


def create_person(phys):
    # The red marker on the cap is intentionally prominent: the perception
    # subsystem tracks it exactly like it tracked the old red sphere.
    specs = [
        ("torso", p.GEOM_BOX, [0.12, 0.07, 0.32], None, [0.12, 0.32, 0.82, 1], [0, 0, 1.02]),
        ("head", p.GEOM_SPHERE, None, 0.13, [0.86, 0.66, 0.50, 1], [0, 0, 1.43]),
        ("marker", p.GEOM_CYLINDER, None, 0.16, [1.0, 0.04, 0.02, 1], [0, 0, 1.62]),
        ("leg_l", p.GEOM_BOX, [0.045, 0.045, 0.31], None, [0.08, 0.09, 0.11, 1], [-0.06, 0, 0.43]),
        ("leg_r", p.GEOM_BOX, [0.045, 0.045, 0.31], None, [0.08, 0.09, 0.11, 1], [0.06, 0, 0.43]),
        ("arm_l", p.GEOM_BOX, [0.035, 0.045, 0.24], None, [0.12, 0.32, 0.82, 1], [-0.18, 0, 1.02]),
        ("arm_r", p.GEOM_BOX, [0.035, 0.045, 0.24], None, [0.12, 0.32, 0.82, 1], [0.18, 0, 1.02]),
    ]
    parts = {}
    for name, shape, half_extents, radius, color, offset in specs:
        kwargs = {}
        if half_extents is not None:
            kwargs["halfExtents"] = half_extents
        if radius is not None:
            kwargs["radius"] = radius
        if name == "marker":
            kwargs["length"] = 0.055
        parts[name] = {
            "id": _visual_body(phys, shape, color, offset, **kwargs),
            "offset": np.array(offset, dtype=float),
        }
    return parts


def update_person(phys, person, user_pos, yaw, walk_phase):
    for name, part in person.items():
        offset = part["offset"].copy()
        if name == "leg_l":
            offset[1] += 0.08 * math.sin(walk_phase)
        elif name == "leg_r":
            offset[1] -= 0.08 * math.sin(walk_phase)
        elif name == "arm_l":
            offset[1] -= 0.07 * math.sin(walk_phase)
        elif name == "arm_r":
            offset[1] += 0.07 * math.sin(walk_phase)

        pos = np.array([user_pos[0], user_pos[1], 0.0]) + _rotate_xy(offset, yaw)
        quat = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(
            part["id"], pos.tolist(), quat, physicsClientId=phys,
        )


class TelemetryWindow:
    def __init__(self):
        self.root = None
        self.closed = False
        self.values = {}
        self.badges = {}
        self.cards = {}
        self.card_notes = {}
        self.hero_canvas = None
        self.battery_bar = None
        self.battery_pct = 100.0
        self.hero_width = 820
        self.hero_height = 230
        self.test_buttons = []
        self.test_status_var = None
        self.test_running = False
        self.test_result = None
        self.test_lock = threading.Lock()
        self.vision_window = None
        self.vision_visible = True
        self.camera_label = None
        self.camera_image = None
        self.graph_canvas = None
        self.ai_note_var = None
        self.gimbal_status_var = None
        self.metric_history = {
            "confidence": [],
            "hover_error": [],
            "avoidance": [],
            "battery": [],
            "umbrella": [],
        }
        self.status_palette = {
            "OK": ("#064e3b", "#d1fae5"),
            "READY": ("#1e3a8a", "#dbeafe"),
            "WATCH": ("#854d0e", "#fef3c7"),
            "LOW": ("#7f1d1d", "#fee2e2"),
            "FAR": ("#7f1d1d", "#fee2e2"),
            "ACTIVE": ("#075985", "#e0f2fe"),
            "IDLE": ("#334155", "#e2e8f0"),
        }

        if tk is None:
            print("Telemetry GUI unavailable: tkinter is not installed.")
            return

        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            print(f"Telemetry GUI unavailable: {exc}")
            return

        self.root.title("SkyShade Dashboard")
        self.root.configure(bg="#0b1120")
        self.root.geometry("900x980+780+20")
        self.root.minsize(900, 920)
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        container = tk.Frame(self.root, bg="#0b1120", padx=18, pady=16)
        container.grid(row=0, column=0, sticky="nsew")

        self.hero_canvas = tk.Canvas(
            container, width=self.hero_width, height=self.hero_height,
            bg="#0b1726", highlightthickness=0,
        )
        self.hero_canvas.grid(row=0, column=0, sticky="ew")
        self._draw_hero(0.0)

        glance = tk.Frame(container, bg="#0b1120")
        glance.grid(row=1, column=0, sticky="ew", pady=(14, 10))
        for col, (title, key, note_key, accent) in enumerate([
            ("Mission", "mission", "mission_note", "#38bdf8"),
            ("Battery", "battery", "battery_note", "#22c55e"),
            ("Umbrella", "umbrella", "umbrella_note", "#a78bfa"),
        ]):
            self._card(glance, col, title, key, note_key, accent)

        body = tk.Frame(container, bg="#0b1120")
        body.grid(row=2, column=0, sticky="ew")

        row = 0
        row = self._section(body, row, "Flight At A Glance")
        for key, label_text in [
            ("time", "Run time"),
            ("nav", "Drone action"),
            ("hover_error", "Distance from user"),
            ("tracking", "Camera lock"),
        ]:
            row = self._metric(body, row, key, label_text)

        row = self._section(body, row, "Weather And Umbrella")
        for key, label_text in [
            ("weather", "Weather reading"),
            ("umbrella", "Umbrella state"),
            ("battery", "Battery level"),
            ("altitude", "Flying height"),
        ]:
            row = self._metric(body, row, key, label_text)

        row = self._section(body, row, "Behind The Scenes")
        for key, label_text in [
            ("location_summary", "Positions"),
            ("subsystem_summary", "AI modules"),
        ]:
            row = self._metric(body, row, key, label_text)

        row = self._section(body, row, "Validation Tests")
        self._test_panel(body, row)
        self._build_vision_window()

    def _draw_hero(self, t_wall):
        if self.hero_canvas is None:
            return

        c = self.hero_canvas
        c.delete("all")
        w = self.hero_width
        h = self.hero_height
        horizon_y = 154
        ground_y = 182

        # Level horizon and symmetric grid keep the visual easy to read.
        c.create_rectangle(0, 0, w, 52, fill="#07111f", outline="")
        c.create_rectangle(0, 52, w, 100, fill="#0b1b2f", outline="")
        c.create_rectangle(0, 100, w, horizon_y, fill="#113451", outline="")
        c.create_rectangle(0, horizon_y, w, h, fill="#12351f", outline="")
        c.create_rectangle(0, horizon_y, w, horizon_y + 2, fill="#5eead4", outline="")

        for y in (ground_y, 196, 210, 224):
            c.create_line(0, y, w, y, fill="#1f9f68", width=1)
        for x in range(-80, w + 120, 70):
            c.create_line(x, h, w / 2, horizon_y, fill="#1f9f68", width=1)

        c.create_text(
            26, 28, text="SkyShade", anchor="w",
            fill="#f8fafc", font=("Arial", 28, "bold"),
        )
        c.create_rectangle(28, 68, 116, 96, fill="#052e16", outline="#22c55e")
        c.create_text(72, 82, text="LIVE SIM", fill="#bbf7d0", font=("Arial", 10, "bold"))

        cx = 610
        cy = 100 + math.sin(t_wall * 2.0) * 7
        angle = t_wall * 2.8
        arm_x = 98
        arm_y = 38
        body_w = 72
        body_h = 34

        c.create_oval(cx - 130, 168, cx + 130, 202, fill="#06101d", outline="", stipple="gray50")
        c.create_oval(cx - 72, cy + 38, cx + 72, cy + 54, fill="#0f172a", outline="#1e293b")

        rotor_points = [
            (cx - arm_x, cy - arm_y),
            (cx + arm_x, cy - arm_y),
            (cx - arm_x, cy + arm_y),
            (cx + arm_x, cy + arm_y),
        ]
        for rx, ry in rotor_points:
            c.create_line(cx, cy, rx, ry, fill="#93c5fd", width=5)

        for i, (rx, ry) in enumerate(rotor_points):
            blade_angle = angle + i * 0.8
            c.create_oval(rx - 28, ry - 12, rx + 28, ry + 12, fill="#0f172a", outline="#60a5fa", width=2)
            c.create_line(
                rx - math.cos(blade_angle) * 36,
                ry - math.sin(blade_angle) * 11,
                rx + math.cos(blade_angle) * 36,
                ry + math.sin(blade_angle) * 11,
                fill="#e0f2fe", width=3,
            )
            c.create_line(
                rx - math.sin(blade_angle) * 36,
                ry + math.cos(blade_angle) * 11,
                rx + math.sin(blade_angle) * 36,
                ry - math.cos(blade_angle) * 11,
                fill="#bae6fd", width=2,
            )

        c.create_rectangle(
            cx - body_w / 2, cy - body_h / 2,
            cx + body_w / 2, cy + body_h / 2,
            fill="#2563eb", outline="#bfdbfe", width=2,
        )
        c.create_oval(cx - 19, cy - 11, cx + 19, cy + 11, fill="#0ea5e9", outline="#e0f2fe", width=2)
        canopy_top = cy - 72
        canopy_base = cy - 34
        mast_top = cy - 54
        mast_bottom = cy - body_h / 2
        c.create_line(cx, mast_top, cx, mast_bottom, fill="#e0f2fe", width=4)
        c.create_line(cx - 66, canopy_base, cx, mast_top, fill="#bae6fd", width=2)
        c.create_line(cx + 66, canopy_base, cx, mast_top, fill="#bae6fd", width=2)
        c.create_arc(
            cx - 86, canopy_top, cx + 86, cy + 12,
            start=0, extent=180,
            fill="#38bdf8", outline="#e0f2fe", width=2,
        )
        c.create_line(cx - 70, canopy_base, cx + 70, canopy_base, fill="#0f172a", width=2)
        c.create_oval(cx - 5, mast_top - 4, cx + 5, mast_top + 6, fill="#e0f2fe", outline="")

        width = max(0, min(w, int(w * self.battery_pct / 100)))
        if self.battery_pct > 50:
            color = "#22c55e"
        elif self.battery_pct > 30:
            color = "#f59e0b"
        else:
            color = "#ef4444"

        c.create_rectangle(0, h - 5, w, h, fill="#1e293b", outline="")
        self.battery_bar = c.create_rectangle(0, h - 5, width, h, fill=color, outline="")

    def _card(self, parent, col, title, key, note_key, accent):
        card = tk.Frame(parent, bg="#111c2e", padx=14, pady=12)
        card.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 8, 0))
        parent.grid_columnconfigure(col, weight=1, uniform="cards")

        tk.Label(
            card, text=title.upper(),
            fg=accent, bg="#111c2e",
            font=("Arial", 9, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")

        value = tk.StringVar(value="--")
        tk.Label(
            card, textvariable=value,
            fg="#f8fafc", bg="#111c2e",
            font=("Arial", 18, "bold"),
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", pady=(5, 1))
        self.cards[key] = value

        note = tk.StringVar(value="")
        tk.Label(
            card, textvariable=note,
            fg="#94a3b8", bg="#111c2e",
            font=("Arial", 9),
            anchor="w",
            wraplength=230,
            justify="left",
        ).grid(row=2, column=0, sticky="ew")
        self.card_notes[note_key] = note

    def _section(self, parent, row, title):
        tk.Label(
            parent, text=title,
            fg="#f8fafc", bg="#0b1120",
            font=("Arial", 15, "bold"),
            anchor="w",
            pady=7,
        ).grid(row=row, column=0, columnspan=3, sticky="ew")

        return row + 1

    def _metric(self, parent, row, key, label_text):
        row_bg = "#0f172a" if row % 2 == 0 else "#111c2e"
        frame = tk.Frame(parent, bg=row_bg, padx=10, pady=6)
        frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(3, 0))
        frame.grid_columnconfigure(1, weight=1)

        tk.Label(
            frame, text=label_text,
            fg="#94a3b8", bg=row_bg,
            font=("Arial", 10, "bold"),
            anchor="w",
            width=18,
        ).grid(row=0, column=0, sticky="w", padx=(0, 12))

        value_var = tk.StringVar(value="--")
        tk.Label(
            frame, textvariable=value_var,
            fg="#f8fafc", bg=row_bg,
            font=("Arial", 11),
            anchor="w",
            wraplength=470,
            justify="left",
        ).grid(row=0, column=1, sticky="ew")
        self.values[key] = value_var

        status = tk.Label(
            frame, text="",
            fg="#e2e8f0", bg="#334155",
            font=("Arial", 9, "bold"),
            anchor="center",
            padx=8, pady=3,
            width=8,
        )
        status.grid(row=0, column=2, sticky="e", padx=(12, 0))
        self.badges[key] = status

        return row + 1

    def _test_panel(self, parent, row):
        panel = tk.Frame(parent, bg="#0f172a", padx=10, pady=8)
        panel.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(3, 0))
        panel.grid_columnconfigure(0, weight=1)

        button_row = tk.Frame(panel, bg="#0f172a")
        button_row.grid(row=0, column=0, sticky="ew")

        tests = [
            ("Perception", [("Sub-1 Perception", "sub1_perception/test_perception.py")]),
            ("Flight", [("Sub-2 Flight", "sub2_flight/test_flight.py")]),
            ("Environment", [("Sub-3 Environment", "sub3_env/test_env_decision.py")]),
            ("Navigation", [("Sub-4 Navigation", "sub4_nav/test_nav_safety.py")]),
            ("All Tests", [
                ("Sub-1 Perception", "sub1_perception/test_perception.py"),
                ("Sub-2 Flight", "sub2_flight/test_flight.py"),
                ("Sub-3 Environment", "sub3_env/test_env_decision.py"),
                ("Sub-4 Navigation", "sub4_nav/test_nav_safety.py"),
            ]),
        ]

        for col, (label, test_group) in enumerate(tests):
            button = tk.Button(
                button_row,
                text=label,
                command=lambda title=label, group=test_group: self._start_tests(title, group),
                bg="#1e293b", fg="#e0f2fe",
                activebackground="#334155",
                activeforeground="#ffffff",
                relief="flat",
                font=("Arial", 10, "bold"),
                padx=8, pady=7,
            )
            button.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 6, 0))
            button_row.grid_columnconfigure(col, weight=1, uniform="test_buttons")
            self.test_buttons.append(button)

        stop = tk.Button(
            button_row,
            text="Stop",
            command=self.close,
            bg="#263242", fg="#fecaca",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief="flat",
            font=("Arial", 10, "bold"),
            padx=8, pady=7,
        )
        stop.grid(row=0, column=len(tests), sticky="ew", padx=(6, 0))
        button_row.grid_columnconfigure(len(tests), weight=1, uniform="test_buttons")

        self.test_status_var = tk.StringVar(value="Ready. Choose a subsystem test or run everything.")
        tk.Label(
            panel, textvariable=self.test_status_var,
            fg="#cbd5e1", bg="#111c2e",
            font=("Arial", 10),
            anchor="w",
            justify="left",
            wraplength=760,
            padx=10, pady=8,
        ).grid(row=1, column=0, sticky="ew", pady=(8, 0))

    def _build_vision_window(self):
        if self.root is None:
            return

        self.vision_window = tk.Toplevel(self.root)
        self.vision_window.title("SkyShade Drone POV And AI Evidence")
        self.vision_window.configure(bg="#07111f")
        self.vision_window.geometry("1180x980+20+60")
        self.vision_window.minsize(1000, 820)
        self.vision_window.protocol("WM_DELETE_WINDOW", self._hide_vision_window)

        tk.Label(
            self.vision_window,
            text="Drone POV: camera feed with red-marker detection",
            fg="#f8fafc", bg="#07111f",
            font=("Arial", 15, "bold"),
            anchor="w",
            padx=14, pady=10,
        ).grid(row=0, column=0, sticky="ew")

        self.camera_label = tk.Label(
            self.vision_window,
            bg="#020617",
            fg="#94a3b8",
            text="Waiting for camera frame...",
            font=("Arial", 12),
            width=POV_DISPLAY_SIZE[0],
            height=POV_DISPLAY_SIZE[1],
        )
        self.camera_label.grid(row=1, column=0, sticky="nsew", padx=14)

        self.gimbal_status_var = tk.StringVar(value="AI gimbal: waiting for prediction")
        tk.Label(
            self.vision_window,
            textvariable=self.gimbal_status_var,
            fg="#bae6fd", bg="#0f172a",
            font=("Arial", 11, "bold"),
            anchor="w",
            padx=12, pady=8,
        ).grid(row=2, column=0, sticky="ew", padx=14, pady=(8, 0))

        tk.Label(
            self.vision_window,
            text="Runtime AI evidence: this run uses pre-trained models; these traces show live tracking, control, avoidance, and decisions.",
            fg="#cbd5e1", bg="#07111f",
            font=("Arial", 10),
            anchor="w",
            justify="left",
            wraplength=1120,
            padx=14, pady=8,
        ).grid(row=3, column=0, sticky="ew")

        self.graph_canvas = tk.Canvas(
            self.vision_window,
            width=1120,
            height=360,
            bg="#0b1726",
            highlightthickness=0,
        )
        self.graph_canvas.grid(row=4, column=0, sticky="nsew", padx=14, pady=(0, 10))

        self.ai_note_var = tk.StringVar(value="Collecting live metrics...")
        tk.Label(
            self.vision_window,
            textvariable=self.ai_note_var,
            fg="#bae6fd", bg="#0f172a",
            font=("Arial", 10, "bold"),
            anchor="w",
            padx=12, pady=9,
            wraplength=1120,
            justify="left",
        ).grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 14))

        self.vision_window.grid_rowconfigure(1, weight=3)
        self.vision_window.grid_rowconfigure(4, weight=2)
        self.vision_window.grid_columnconfigure(0, weight=1)

    def _hide_vision_window(self):
        self.vision_visible = False
        if self.vision_window is not None:
            self.vision_window.withdraw()

    def _detect_marker_box(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
        mask2 = cv2.inRange(hsv, HSV_LOWER2, HSV_UPPER2)
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < MIN_CONTOUR_AREA:
            return None

        x, y, w, h = cv2.boundingRect(largest)
        return x, y, w, h, area

    def _frame_to_photo(self, frame):
        display = cv2.resize(frame, POV_DISPLAY_SIZE, interpolation=cv2.INTER_AREA)
        header = f"P6 {display.shape[1]} {display.shape[0]} 255\n".encode("ascii")
        return tk.PhotoImage(data=header + display.tobytes(), format="PPM")

    def update_vision(
        self, frame, confidence, hover_error, avoidance_force,
        battery_pct, umbrella_cmd, gimbal_action,
    ):
        if self.root is None or self.closed:
            return

        if self.vision_window is not None and self.vision_visible:
            try:
                self.vision_window.deiconify()
            except tk.TclError:
                return

        overlay = frame.copy()
        box = self._detect_marker_box(overlay)
        if box is not None:
            x, y, w, h, area = box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 80), 3)
            cv2.drawMarker(
                overlay, (x + w // 2, y + h // 2), (255, 255, 255),
                markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2,
            )
            cv2.putText(
                overlay, f"marker area {area:.0f}px conf {confidence:.2f}",
                (max(8, x), max(24, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2,
            )
        else:
            cv2.putText(
                overlay, "marker not detected",
                (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 80, 80), 2,
            )

        if self.camera_label is not None:
            self.camera_image = self._frame_to_photo(overlay)
            self.camera_label.configure(image=self.camera_image, text="")
        if self.gimbal_status_var is not None:
            mode = "search sweep" if confidence < CONFIDENCE_THRESH else "predictive track"
            self.gimbal_status_var.set(f"AI gimbal: {gimbal_action} ({mode})")

        self._append_metric("confidence", confidence)
        self._append_metric("hover_error", hover_error)
        self._append_metric("avoidance", float(np.linalg.norm(avoidance_force[:2])))
        self._append_metric("battery", battery_pct)
        self._append_metric("umbrella", 1.0 if umbrella_cmd == "DEPLOY" else 0.0)
        self._draw_live_graphs()

    def _append_metric(self, key, value, limit=180):
        series = self.metric_history[key]
        series.append(float(value))
        if len(series) > limit:
            del series[:len(series) - limit]

    def _draw_series(self, canvas, values, bounds, color, label, row):
        if not values:
            return

        x0, y0, x1, y1 = bounds
        min_v, max_v = {
            "confidence": (0.0, 1.0),
            "hover_error": (0.0, 3.0),
            "avoidance": (0.0, AVOIDANCE_MAX_FORCE),
            "battery": (0.0, 100.0),
            "umbrella": (0.0, 1.0),
        }[label]
        span = max(max_v - min_v, 1e-6)

        points = []
        count = len(values)
        for i, value in enumerate(values):
            x = x0 + (x1 - x0) * (i / max(1, count - 1))
            norm = max(0.0, min(1.0, (value - min_v) / span))
            y = y1 - (y1 - y0) * norm
            points.extend([x, y])

        if len(points) >= 4:
            canvas.create_line(*points, fill=color, width=2, smooth=True)
        latest = values[-1]
        label_text = {
            "confidence": f"confidence: {latest:.2f}",
            "hover_error": f"hover error: {latest:.2f} m",
            "avoidance": f"avoidance: {latest:.1f} N",
            "battery": f"battery: {latest:.0f}%",
            "umbrella": f"umbrella: {'deploy' if latest > 0.5 else 'stow'}",
        }[label]
        canvas.create_text(
            x0 + 8, y0 + 15,
            text=label_text, fill=color,
            font=("Arial", 9, "bold"), anchor="w",
        )

    def _draw_live_graphs(self):
        if self.graph_canvas is None:
            return

        c = self.graph_canvas
        c.delete("all")
        w = int(c.winfo_width() or 820)
        h = int(c.winfo_height() or 360)

        c.create_rectangle(0, 0, w, h, fill="#0b1726", outline="")

        graph_specs = [
            ("confidence", "#22c55e", "Marker tracking"),
            ("hover_error", "#f59e0b", "Coverage error"),
            ("avoidance", "#38bdf8", "Tree avoidance"),
            ("battery", "#a78bfa", "Battery"),
            ("umbrella", "#f472b6", "Umbrella decision"),
        ]
        cols = 2
        rows = 3
        gap = 12
        cell_w = (w - gap * (cols + 1)) / cols
        cell_h = (h - gap * (rows + 1)) / rows

        for index, (key, color, title) in enumerate(graph_specs):
            col = index % cols
            row = index // cols
            x0 = gap + col * (cell_w + gap)
            y0 = gap + row * (cell_h + gap)
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            c.create_rectangle(x0, y0, x1, y1, fill="#0f172a", outline="#334155")
            c.create_text(
                x0 + 8, y0 + 14,
                text=title, fill="#cbd5e1",
                font=("Arial", 9, "bold"), anchor="w",
            )
            plot = (x0 + 10, y0 + 28, x1 - 10, y1 - 12)
            for i in range(1, 3):
                gy = plot[1] + (plot[3] - plot[1]) * i / 3
                c.create_line(plot[0], gy, plot[2], gy, fill="#172554")
            self._draw_series(c, self.metric_history[key], plot, color, key, 0)

        if len(graph_specs) < cols * rows:
            index = len(graph_specs)
            col = index % cols
            row = index // cols
            x0 = gap + col * (cell_w + gap)
            y0 = gap + row * (cell_h + gap)
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            c.create_rectangle(x0, y0, x1, y1, fill="#0f172a", outline="#334155")
            c.create_text(
                x0 + 10, y0 + 16,
                text="What these show",
                fill="#cbd5e1", font=("Arial", 9, "bold"), anchor="w",
            )
            c.create_text(
                x0 + 10, y0 + 40,
                text="Lower coverage error means the drone is staying over the user.\nAvoidance should spike only near trunks.\nConfidence should stay high when the gimbal sees the marker.",
                fill="#94a3b8", font=("Arial", 9), anchor="nw", width=cell_w - 20,
            )

        conf = self.metric_history["confidence"][-1] if self.metric_history["confidence"] else 0.0
        err = self.metric_history["hover_error"][-1] if self.metric_history["hover_error"] else 0.0
        avoid = self.metric_history["avoidance"][-1] if self.metric_history["avoidance"] else 0.0
        if self.ai_note_var is not None:
            self.ai_note_var.set(
                "Live evidence, not live training: "
                f"tracker confidence {conf:.2f}, hover error {err:.2f}m, "
                f"tree avoidance force {avoid:.1f}N. "
                "Use the validation buttons on the main dashboard for held-out subsystem checks."
            )

    def _set_test_buttons_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in self.test_buttons:
            button.configure(state=state)

    def _start_tests(self, title, test_group):
        if self.test_running:
            return

        self.test_running = True
        self._set_test_buttons_enabled(False)
        if self.test_status_var is not None:
            self.test_status_var.set(f"Running {title}...")

        thread = threading.Thread(
            target=self._run_tests_worker,
            args=(title, test_group),
            daemon=True,
        )
        thread.start()

    def _run_tests_worker(self, title, test_group):
        repo_root = os.path.dirname(__file__)
        env = os.environ.copy()
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

        failures = []
        summaries = []
        for name, script in test_group:
            try:
                proc = subprocess.run(
                    [sys.executable, script],
                    cwd=repo_root,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=240,
                )
                output = (proc.stdout + "\n" + proc.stderr).strip()
                last_line = output.splitlines()[-1] if output else "No output"
                summaries.append(f"{name}: {'PASS' if proc.returncode == 0 else 'FAIL'}")
                if proc.returncode != 0:
                    failures.append(f"{name} failed - {last_line}")
            except subprocess.TimeoutExpired:
                summaries.append(f"{name}: TIMEOUT")
                failures.append(f"{name} timed out")
            except OSError as exc:
                summaries.append(f"{name}: ERROR")
                failures.append(f"{name} could not start - {exc}")

        if failures:
            message = f"{title} finished with issues. " + " | ".join(failures[:2])
            ok = False
        else:
            message = f"{title} passed. " + " | ".join(summaries)
            ok = True

        with self.test_lock:
            self.test_result = (ok, message)

    def _flush_test_result(self):
        with self.test_lock:
            result = self.test_result
            self.test_result = None

        if result is None:
            return

        ok, message = result
        if self.test_status_var is not None:
            prefix = "PASS" if ok else "CHECK"
            self.test_status_var.set(f"{prefix}: {message}")
        self.test_running = False
        self._set_test_buttons_enabled(True)

    def set_battery(self, battery_pct):
        if self.root is None:
            return

        self.battery_pct = battery_pct
        width = max(0, min(self.hero_width, int(self.hero_width * battery_pct / 100)))
        if battery_pct > 50:
            color = "#22c55e"
        elif battery_pct > 30:
            color = "#f59e0b"
        else:
            color = "#ef4444"

        if self.battery_bar is not None:
            self.hero_canvas.coords(self.battery_bar, 0, self.hero_height - 5, width, self.hero_height)
            self.hero_canvas.itemconfigure(self.battery_bar, fill=color)

    def set_status(self, key, status_text):
        label = self.badges.get(key)
        if label is None:
            return

        bg, fg = self.status_palette.get(status_text, ("#1f2937", "#d1d5db"))
        label.configure(text=status_text, bg=bg, fg=fg)

    def close(self):
        self.closed = True
        if self.root is not None:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            self.root = None

    def update(self, values, statuses=None, battery_pct=None):
        if self.root is None or self.closed:
            return

        for key, value in values.items():
            if key in self.values:
                self.values[key].set(value)
            if key in self.cards:
                self.cards[key].set(value)
            if key in self.card_notes:
                self.card_notes[key].set(value)

        for key, value in (statuses or {}).items():
            self.set_status(key, value)

        if battery_pct is not None:
            self.set_battery(battery_pct)

    def tick(self, t_wall):
        if self.root is None or self.closed:
            return

        self._flush_test_result()
        self._draw_hero(t_wall)

        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True
            self.root = None


class ConsoleTelemetry:
    closed = False

    def update(self, *args, **kwargs):
        pass

    def update_vision(self, *args, **kwargs):
        pass

    def tick(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _telemetry_payload(
    t_wall, battery_pct, nav_override, umbrella_cmd, confidence,
    err_xy, drone_pos, user_pos, user_offset, weather,
):
    lux = weather["lux"]
    rain = weather["rain"]
    wind = weather["wind"]
    weather_mode = weather["mode"]
    user_offset_text = "--"
    if user_offset is not None:
        user_offset_text = f"x {user_offset[0]:.2f}, y {user_offset[1]:.2f}, z {user_offset[2]:.2f}"

    nav_note = {
        "CONTINUE": "Following the user",
        "RTH": "Returning home",
        "LAND_NOW": "Landing now",
    }.get(nav_override, nav_override)
    umbrella_note = "Shade canopy is open" if umbrella_cmd == "DEPLOY" else "Shade canopy is stowed"
    battery_note = "Healthy flight time"
    if battery_pct <= 30:
        battery_note = "Low battery"
    elif battery_pct <= 50:
        battery_note = "Watch battery"
    location_summary = (
        f"Drone x {drone_pos[0]:.2f}, y {drone_pos[1]:.2f}; "
        f"User x {user_pos[0]:.2f}, y {user_pos[1]:.2f}; "
        f"Camera estimate {user_offset_text}"
    )
    subsystem_summary = (
        "Perception tracking the marker; flight controller holding hover; "
        "environment AI choosing shade; navigation AI checking safety."
    )

    return (
        {
            "mission": nav_note,
            "mission_note": "Autopilot state",
            "battery": f"{battery_pct:5.1f}%",
            "battery_note": battery_note,
            "umbrella": "Deployed" if umbrella_cmd == "DEPLOY" else "Stowed",
            "umbrella_note": umbrella_note,
            "time": f"{t_wall:5.1f} seconds",
            "nav": nav_note,
            "altitude": f"{drone_pos[2]:.2f} m above ground",
            "hover_error": f"{err_xy:.2f} m from target",
            "tracking": f"{confidence * 100:.0f}% confidence",
            "drone_position": f"x {drone_pos[0]:.2f}, y {drone_pos[1]:.2f}",
            "user_position": f"x {user_pos[0]:.2f}, y {user_pos[1]:.2f}",
            "user_offset": user_offset_text,
            "location_summary": location_summary,
            "weather": f"{weather_mode}: light {lux:.0f} lux, rain {rain:.2f}, wind {wind:.1f} m/s",
            "subsystem_summary": subsystem_summary,
            "perception": "Tracking the red user marker",
            "flight": "Holding hover and following the path",
            "env_decision": "Choosing deploy or stow",
            "nav_safety": "Checking battery and return-home safety",
        },
        {
            "battery": "OK" if battery_pct > 30 else "LOW",
            "nav": "ACTIVE" if nav_override != "CONTINUE" else "OK",
            "umbrella": "ACTIVE" if umbrella_cmd == "DEPLOY" else "IDLE",
            "tracking": "OK" if confidence >= CONFIDENCE_THRESH else "LOW",
            "hover_error": "OK" if err_xy <= HOVER_RADIUS else "FAR",
            "mission": "ACTIVE" if nav_override != "CONTINUE" else "OK",
            "location_summary": "READY",
            "subsystem_summary": "OK",
            "perception": "OK",
            "flight": "OK",
            "env_decision": "OK",
            "nav_safety": "OK",
        },
    )


def run(duration=120.0, gui=True, scenario=SCENARIO_PARK):
    mode = p.GUI if gui else p.DIRECT
    phys = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=phys)
    p.setGravity(0, 0, -9.81, physicsClientId=phys)
    p.setTimeStep(SIM_TIMESTEP, physicsClientId=phys)

    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=phys)
        if scenario == SCENARIO_FOREST:
            p.resetDebugVisualizerCamera(
                cameraDistance=14.0, cameraYaw=35, cameraPitch=-34,
                cameraTargetPosition=[0, 0, 1.3],
                physicsClientId=phys,
            )
        else:
            p.resetDebugVisualizerCamera(
                cameraDistance=7, cameraYaw=30, cameraPitch=-25,
                cameraTargetPosition=[0, 0, 1],
                physicsClientId=phys,
            )

    obstacles = build_environment(phys, scenario)
    initial_user_pos = scenario_user_position(scenario, 0.0)

    # Drone and user are assembled from primitive bodies so the scene has a
    # readable physical scale while the AI/control code keeps the same inputs.
    drone_id, drone_parts = create_drone(phys, initial_user_pos[:2])
    person_parts = create_person(phys)

    # Umbrella disc (flat cylinder above the drone)
    umb_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.7, height=0.04,
                                     physicsClientId=phys)
    umb_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.7, length=0.04,
                                  rgbaColor=[0.5, 0.5, 0.5, 0.4], physicsClientId=phys)
    umb_id  = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=umb_col,
                                baseVisualShapeIndex=umb_vis,
                                basePosition=[
                                    initial_user_pos[0],
                                    initial_user_pos[1],
                                    TARGET_ALTITUDE + 0.35,
                                ],
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
    if scenario == SCENARIO_FOREST:
        pid.KP_XY = 13.0
        pid.KI_XY = 0.15
        pid.KD_XY = 7.0
        pid.MAX_FORCE_XY = 15.0
    umbrella   = UmbrellaClassifier()
    nav_policy = NavSafetyPolicy()

    battery_pct   = 100.0
    umbrella_cmd  = "STOW"
    nav_override  = "CONTINUE"
    confidence    = 0.0
    user_pos      = initial_user_pos.copy()
    prev_user_pos = user_pos.copy()
    user_velocity = np.zeros(3, dtype=float)
    gimbal_target = np.array([user_pos[0], user_pos[1], USER_MARKER_HEIGHT], dtype=float)
    dt            = STEPS_PER_ACTION * SIM_TIMESTEP
    action_dt     = 1.0 / CONTROL_HZ

    telemetry = TelemetryWindow() if gui else ConsoleTelemetry()
    weather_controller = RandomWeatherController()
    weather_visual_rng = np.random.default_rng()
    weather_now = weather_controller.update(0.0)

    umb_deployed = False
    t_start = time.monotonic()
    tick    = 0

    print(f"SkyShade simulation running ({scenario} scenario) — Ctrl+C to stop.\n")
    print(f"{'Time':>6}  {'Bat':>6}  {'Nav':<10}  {'Umbrella':<8}  {'Conf':>5}  {'ErrXY':>6}  {'Z':>5}")
    print("-" * 65)

    try:
        while True:
            t_wall = time.monotonic() - t_start
            if t_wall >= duration:
                print(f"\nSimulation complete ({duration:.0f} s).")
                break
            if telemetry.closed:
                print("\nSimulation stopped from dashboard.")
                break
            weather_now = weather_controller.update(t_wall)

            # ── User path ────────────────────────────────────────────────────
            user_pos = scenario_user_position(scenario, t_wall)
            user_delta = user_pos - prev_user_pos
            user_velocity = user_delta / max(action_dt, 1e-6)
            user_yaw = math.atan2(user_delta[1], user_delta[0]) if np.linalg.norm(user_delta[:2]) > 1e-4 else 0.0
            update_person(phys, person_parts, user_pos, user_yaw, t_wall * 4.5)
            prev_user_pos = user_pos.copy()

            # ── Battery ───────────────────────────────────────────────────────
            battery_pct = max(0.0, battery_pct - BATTERY_DRAIN_RATE * action_dt)

            # ── Sub-1: Perception — camera pointing down from drone ────────────
            d_pos, _ = p.getBasePositionAndOrientation(drone_id, physicsClientId=phys)
            drone_pos = np.array(d_pos)

            W, H = CAMERA_RES
            eye = (drone_pos + np.array([0.0, 0.0, -0.08])).tolist()
            desired_marker = predictive_gimbal_target(
                user_pos, user_velocity, confidence, t_wall,
            )
            gimbal_target = 0.72 * gimbal_target + 0.28 * desired_marker
            view_direction = gimbal_target - np.array(eye, dtype=float)
            if np.linalg.norm(view_direction) < 0.05:
                gimbal_target = np.array([drone_pos[0], drone_pos[1], drone_pos[2] - 1.0], dtype=float)
            gimbal_action = describe_gimbal_action(drone_pos, gimbal_target)
            view = p.computeViewMatrix(eye, gimbal_target.tolist(), [0, 1, 0])
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
                lux = weather_now["lux"]
                rain = weather_now["rain"]
                wind = weather_now["wind"]
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
                # Follow slightly ahead of the user so coverage does not lag.
                pid_target = lead_follow_target(user_pos, user_velocity)

            force = pid.compute_force(drone_pos, drone_vel, pid_target, dt)
            avoidance = obstacle_avoidance_force(drone_pos, obstacles)
            force += avoidance
            force[2] += HOVER_FORCE_N
            avoidance_for_display = avoidance.copy()

            for _ in range(STEPS_PER_ACTION):
                p.applyExternalForce(drone_id, -1, force.tolist(),
                                     [0, 0, 0], p.WORLD_FRAME, physicsClientId=phys)
                p.stepSimulation(physicsClientId=phys)

            # Keep umbrella attached to drone
            d_pos2, _ = p.getBasePositionAndOrientation(drone_id, physicsClientId=phys)
            update_drone_parts(phys, drone_parts, d_pos2, t_wall * 35.0)
            p.resetBasePositionAndOrientation(
                umb_id,
                [d_pos2[0], d_pos2[1], d_pos2[2] + 0.35],
                [0, 0, 0, 1], physicsClientId=phys,
            )

            if DEBUG_WEATHER_LINES and gui and tick % 15 == 0:
                draw_weather_visuals(phys, weather_now, weather_visual_rng)

            # ── Telemetry update (every 10 ticks) ─────────────────────────────
            if tick % 10 == 0:
                err_xy = float(np.linalg.norm(np.array(d_pos2[:2]) - user_pos[:2]))
                values, statuses = _telemetry_payload(
                    t_wall, battery_pct, nav_override, umbrella_cmd, confidence,
                    err_xy, np.array(d_pos2), user_pos, user_offset, weather_now,
                )
                telemetry.update(values, statuses, battery_pct=battery_pct)

            if gui and tick % 5 == 0:
                err_xy = float(np.linalg.norm(np.array(d_pos2[:2]) - user_pos[:2]))
                telemetry.update_vision(
                    frame,
                    confidence,
                    err_xy,
                    avoidance_for_display,
                    battery_pct,
                    umbrella_cmd,
                    gimbal_action,
                )

            # ── Console log every 10 s ────────────────────────────────────────
            if tick % (CONTROL_HZ * 10) == 0:
                err_xy = float(np.linalg.norm(np.array(d_pos2[:2]) - user_pos[:2]))
                print(f"{t_wall:6.1f}s  {battery_pct:5.1f}%  {nav_override:<10}  "
                      f"{umbrella_cmd:<8}  {confidence:.2f}  {err_xy:6.2f}m  "
                      f"{d_pos2[2]:.2f}m")

            telemetry.tick(t_wall)

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
        telemetry.close()
        p.disconnect(phys)
        print("Disconnected.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--no-gui",   action="store_true")
    ap.add_argument(
        "--scenario",
        choices=[SCENARIO_PARK, SCENARIO_FOREST],
        default=SCENARIO_PARK,
        help="Simulation scene to run.",
    )
    args = ap.parse_args()
    run(duration=args.duration, gui=not args.no_gui, scenario=args.scenario)


if __name__ == "__main__":
    main()
