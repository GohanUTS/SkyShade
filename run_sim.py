"""
SkyShade — full integrated simulation runner.

Starts a PyBullet GUI window and runs all four subsystems together:
  Sub-1  Perception       HSV tracker + distance estimator (camera feed)
  Sub-2  Flight Control   Q-learning hover policy with PID fallback
  Sub-3  Env Decision     SVM umbrella classifier
  Sub-4  Nav Safety       MDP policy table

A simulated user walks a figure-8 path below the drone.  Weather sensors
randomly switch between cloudy periods, light rain, and full rain.
Battery drains and triggers the Sub-4 safety override.

Usage:
    python run_sim.py [--duration 120] [--no-gui] [--flight pid|q]
                      [--demo-low-battery]
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
from sub2_flight.runtime_control import RuntimeFlightController, obstacle_avoidance_force
from sub2_flight.env.hover_env import (
    TARGET_ALTITUDE, HOVER_FORCE_N, DRONE_MASS_KG,
    SIM_TIMESTEP, STEPS_PER_ACTION, LINEAR_DAMPING,
)
from sub3_env.classifier import UmbrellaClassifier
from sub4_nav.policy_table import NavSafetyPolicy
from sub4_nav.mdp import ACTION_NAMES

CONTROL_HZ         = 30
BATTERY_DRAIN_RATE = 0.5    # % per second
LOW_BATTERY_DEMO_START = 10.0
LOW_BATTERY_DEMO_DRAIN_RATE = 1.0
RUNTIME_PID_DAMPING = 0.5
WIND_FORCE_SCALE    = 0.35   # N per m/s of weather wind pushed on the drone
USER_WALK_SPEED    = 0.3    # rad/s for figure-8
HOVER_RADIUS       = 0.5    # m
WEATHER_MIN_SECONDS = 8.0
WEATHER_MAX_SECONDS = 20.0
DEBUG_WEATHER_LINES = False
SCENARIO_PARK = "park"
SCENARIO_FOREST = "forest"
SCENARIO_BUILDINGS = "buildings"
SCENARIO_CHOICES = (SCENARIO_PARK, SCENARIO_FOREST, SCENARIO_BUILDINGS)
SCENARIO_LABELS = {
    SCENARIO_PARK: "Park",
    SCENARIO_FOREST: "Forest Trail",
    SCENARIO_BUILDINGS: "Building District",
}
SCENARIO_DESCRIPTIONS = {
    SCENARIO_PARK: "Open park loop with light obstacles and figure-8 walking.",
    SCENARIO_FOREST: "Long wooded trail with tree avoidance and path reset.",
    SCENARIO_BUILDINGS: "City plaza with buildings, roads, vehicles, and pedestrians.",
}
AVOIDANCE_RADIUS = 0.75
AVOIDANCE_GAIN = 4.5
AVOIDANCE_MAX_FORCE = 5.5
FOLLOW_LEAD_MAX_METERS = 1.05
GIMBAL_LOOKAHEAD_SECONDS = 0.95
GIMBAL_SEARCH_RADIUS = 0.55
USER_MARKER_HEIGHT = 1.62
POV_DISPLAY_SIZE = (960, 540)
TRAINING_DISPLAY_SIZE = (360, 270)
LAUNCHER_TRAINING_DISPLAY_SIZE = (480, 360)
TRAINING_PIXEL_PASS = 18.0
TRAINING_GROUND_ITEMS = (
    ("camera", "Sub-1 Camera", "HSV tracker", "Live red-marker lock check.", "#22c55e"),
    ("flight", "Sub-2 Flight", "Q-learning", "Hover policy trainer slot.", "#60a5fa"),
    ("weather", "Sub-3 Weather", "SVM", "Umbrella classifier slot.", "#f472b6"),
    ("safety", "Sub-4 Safety", "MDP", "Battery and navigation policy slot.", "#f59e0b"),
    ("environment", "Environment", "Scenarios", "Park, forest, and buildings tests.", "#a78bfa"),
)
FOREST_TRAIL_START_X = -9.0
FOREST_TRAIL_END_X = 9.0
FOREST_TRAIL_LENGTH = FOREST_TRAIL_END_X - FOREST_TRAIL_START_X
FOREST_WALK_SPEED = 0.65
BUILDING_RING_MIN = 7.5
BUILDING_RING_MAX = 12.5
ROAD_HALF_WIDTH = 1.1
RING_ROAD_R = 6.2
HUMAN_STAND_Z = 0.62


def figure8(t, scale=1.5):
    d = 1 + math.sin(t) ** 2
    return np.array([scale * math.cos(t) / d,
                     scale * math.sin(t) * math.cos(t) / d,
                     0.0])


def forest_walk(t):
    progress = (t * FOREST_WALK_SPEED) % FOREST_TRAIL_LENGTH
    x = FOREST_TRAIL_START_X + progress
    y = 0.55 * math.sin(progress * 0.85) + 0.22 * math.sin(progress * 1.7)
    return np.array([x, y, 0.0])


def forest_lap_index(t):
    return int((t * FOREST_WALK_SPEED) // FOREST_TRAIL_LENGTH)


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


def _fixed_links(phys, base_mass, base_col, base_vis, base_pos, base_orn,
                 link_vis, link_pos, link_orn):
    """Create a rigid visual assembly from fixed PyBullet links."""
    n = len(link_vis)
    return p.createMultiBody(
        baseMass=base_mass,
        baseCollisionShapeIndex=base_col,
        baseVisualShapeIndex=base_vis,
        basePosition=list(base_pos),
        baseOrientation=base_orn,
        linkMasses=[0.0] * n,
        linkCollisionShapeIndices=[-1] * n,
        linkVisualShapeIndices=link_vis,
        linkPositions=link_pos,
        linkOrientations=link_orn,
        linkParentIndices=[0] * n,
        linkJointTypes=[p.JOINT_FIXED] * n,
        linkJointAxis=[[0, 0, 1]] * n,
        linkInertialFramePositions=[[0, 0, 0]] * n,
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * n,
        physicsClientId=phys,
    )


def _flat(phys, half_extents, color, x, y, z, yaw=0.0):
    """Thin visual-only slab for roads, sidewalks, lawns, and lane markings."""
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=list(color) + [1.0],
        physicsClientId=phys,
    )
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=[x, y, z],
        baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
        physicsClientId=phys,
    )


def make_city_car(phys, pos, yaw, color):
    body = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.34, 0.16, 0.10],
        rgbaColor=list(color) + [1.0],
        specularColor=[0.3, 0.3, 0.3],
        physicsClientId=phys,
    )
    cabin = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.17, 0.14, 0.085],
        rgbaColor=[0.6, 0.8, 0.92, 0.7],
        physicsClientId=phys,
    )

    def wheel():
        return p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.08,
            length=0.05,
            rgbaColor=[0.05, 0.05, 0.06, 1.0],
            physicsClientId=phys,
        )

    wheel_orn = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
    link_vis = [cabin, wheel(), wheel(), wheel(), wheel()]
    link_pos = [
        [-0.02, 0, 0.13],
        [0.22, 0.17, -0.03],
        [0.22, -0.17, -0.03],
        [-0.22, 0.17, -0.03],
        [-0.22, -0.17, -0.03],
    ]
    link_orn = [[0, 0, 0, 1], wheel_orn, wheel_orn, wheel_orn, wheel_orn]
    return _fixed_links(
        phys,
        0.0,
        -1,
        body,
        [pos[0], pos[1], 0.14],
        p.getQuaternionFromEuler([0, 0, yaw]),
        link_vis,
        link_pos,
        link_orn,
    )


def make_city_bus(phys, pos, yaw, color):
    body = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.66, 0.20, 0.18],
        rgbaColor=list(color) + [1.0],
        specularColor=[0.3, 0.3, 0.3],
        physicsClientId=phys,
    )
    windows = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.60, 0.205, 0.055],
        rgbaColor=[0.55, 0.72, 0.88, 0.85],
        physicsClientId=phys,
    )

    def wheel():
        return p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.10,
            length=0.06,
            rgbaColor=[0.05, 0.05, 0.06, 1.0],
            physicsClientId=phys,
        )

    wheel_orn = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
    link_vis = [windows, wheel(), wheel(), wheel(), wheel()]
    link_pos = [
        [0, 0, 0.07],
        [0.44, 0.21, -0.10],
        [0.44, -0.21, -0.10],
        [-0.44, 0.21, -0.10],
        [-0.44, -0.21, -0.10],
    ]
    link_orn = [[0, 0, 0, 1], wheel_orn, wheel_orn, wheel_orn, wheel_orn]
    return _fixed_links(
        phys,
        0.0,
        -1,
        body,
        [pos[0], pos[1], 0.22],
        p.getQuaternionFromEuler([0, 0, yaw]),
        link_vis,
        link_pos,
        link_orn,
    )


def make_city_tree(phys, x, y, scale=1.0):
    trunk = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.08 * scale,
        length=0.9 * scale,
        rgbaColor=[0.40, 0.26, 0.13, 1.0],
        physicsClientId=phys,
    )
    canopy_1 = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.42 * scale,
        rgbaColor=[0.16, 0.45, 0.18, 1.0],
        physicsClientId=phys,
    )
    canopy_2 = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.30 * scale,
        rgbaColor=[0.22, 0.55, 0.24, 1.0],
        physicsClientId=phys,
    )
    return _fixed_links(
        phys,
        0.0,
        -1,
        trunk,
        [x, y, 0.45 * scale],
        [0, 0, 0, 1],
        [canopy_1, canopy_2],
        [[0, 0, 0.62 * scale], [0.16 * scale, 0.10 * scale, 0.80 * scale]],
        [[0, 0, 0, 1], [0, 0, 0, 1]],
    )


def make_city_pedestrian(phys, pos, shirt, pants=(0.16, 0.20, 0.42),
                         skin=(0.95, 0.78, 0.66), scale=1.0):
    """Non-red pedestrian so the perception tracker stays locked to the user."""
    torso = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.13 * scale, 0.07 * scale, 0.20 * scale],
        rgbaColor=list(shirt) + [1.0],
        physicsClientId=phys,
    )
    head = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.11 * scale,
        rgbaColor=list(skin) + [1.0],
        physicsClientId=phys,
    )
    cap = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.135 * scale,
        length=0.06 * scale,
        rgbaColor=list(shirt) + [1.0],
        physicsClientId=phys,
    )
    arm_l = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.035 * scale,
        length=0.34 * scale,
        rgbaColor=list(shirt) + [1.0],
        physicsClientId=phys,
    )
    arm_r = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.035 * scale,
        length=0.34 * scale,
        rgbaColor=list(shirt) + [1.0],
        physicsClientId=phys,
    )
    leg_l = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.05 * scale,
        length=0.42 * scale,
        rgbaColor=list(pants) + [1.0],
        physicsClientId=phys,
    )
    leg_r = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.05 * scale,
        length=0.42 * scale,
        rgbaColor=list(pants) + [1.0],
        physicsClientId=phys,
    )
    return _fixed_links(
        phys,
        0.0,
        -1,
        torso,
        pos,
        [0, 0, 0, 1],
        [head, cap, arm_l, arm_r, leg_l, leg_r],
        [
            [0, 0, 0.31 * scale],
            [0, 0, 0.44 * scale],
            [-0.165 * scale, 0, 0.02 * scale],
            [0.165 * scale, 0, 0.02 * scale],
            [-0.06 * scale, 0, -0.41 * scale],
            [0.06 * scale, 0, -0.41 * scale],
        ],
        [[0, 0, 0, 1]] * 6,
    )


def spawn_buildings(phys, rng, n=12):
    """Spawn solid buildings around the clear central flight plaza."""
    palette = [
        [0.30, 0.34, 0.42],
        [0.24, 0.28, 0.36],
        [0.38, 0.40, 0.47],
        [0.22, 0.30, 0.41],
        [0.34, 0.30, 0.39],
        [0.28, 0.33, 0.38],
    ]
    obstacles = []
    for k in range(n):
        angle = 2 * math.pi * k / n + float(rng.uniform(-0.12, 0.12))
        radius = float(rng.uniform(BUILDING_RING_MIN, BUILDING_RING_MAX))
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        width = float(rng.uniform(0.45, 0.85))
        depth = float(rng.uniform(0.45, 0.85))
        height = float(rng.uniform(2.0, 6.0))
        color = palette[k % len(palette)] + [1.0]
        half = [width, depth, height / 2.0]

        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=phys)
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half,
            rgbaColor=color,
            specularColor=[0.25, 0.27, 0.33],
            physicsClientId=phys,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, height / 2.0],
            physicsClientId=phys,
        )

        for band_z in range(2, int(height) + 1):
            z = float(band_z) - 0.5
            if z > height - 0.25:
                break
            band = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[width + 0.015, depth + 0.015, 0.09],
                rgbaColor=[1.0, 0.91, 0.55, 1.0],
                physicsClientId=phys,
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=band,
                basePosition=[x, y, z],
                physicsClientId=phys,
            )

        obstacles.append({
            "position": np.array([x, y], dtype=float),
            "radius": max(width, depth) + 0.35,
        })

    return obstacles


def spawn_city_props(phys, rng):
    """Road network, parks, vehicles, and pedestrians around the city plaza."""
    radius = RING_ROAD_R
    road = (0.13, 0.13, 0.15)
    sidewalk = (0.62, 0.63, 0.66)
    grass = (0.18, 0.42, 0.20)
    plaza_radius = 2.3

    for quadrant in range(4):
        angle = math.pi / 4 + quadrant * math.pi / 2
        gx, gy = 8.4 * math.cos(angle), 8.4 * math.sin(angle)
        _flat(phys, [2.4, 2.4, 0.006], grass, gx, gy, 0.006, angle)
        for _ in range(4):
            tx = gx + float(rng.uniform(-1.7, 1.7))
            ty = gy + float(rng.uniform(-1.7, 1.7))
            make_city_tree(phys, tx, ty, scale=float(rng.uniform(0.85, 1.2)))

    segments = 30
    for k in range(segments):
        angle = 2 * math.pi * k / segments
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        segment_len = (2 * math.pi * radius / segments) * 0.62
        yaw = angle + math.pi / 2
        _flat(phys, [segment_len, ROAD_HALF_WIDTH, 0.012], road, x, y, 0.012, yaw)
        if k % 2 == 0:
            _flat(phys, [0.22, 0.035, 0.004], (0.92, 0.82, 0.25), x, y, 0.026, yaw)

    for direction in range(4):
        angle = direction * math.pi / 2
        mid = (plaza_radius + radius) / 2.0
        length = (radius - plaza_radius) / 2.0
        cx, cy = mid * math.cos(angle), mid * math.sin(angle)
        _flat(phys, [length, ROAD_HALF_WIDTH, 0.012], road, cx, cy, 0.012, angle)

        sidewalk_offset = ROAD_HALF_WIDTH + 0.22
        for side in (1, -1):
            sx = cx + sidewalk_offset * math.cos(angle + math.pi / 2) * side
            sy = cy + sidewalk_offset * math.sin(angle + math.pi / 2) * side
            _flat(phys, [length, 0.20, 0.02], sidewalk, sx, sy, 0.02, angle)

        for t in (-0.5, 0.0, 0.5):
            dx = (mid + t * length) * math.cos(angle)
            dy = (mid + t * length) * math.sin(angle)
            _flat(phys, [0.22, 0.035, 0.004], (0.92, 0.82, 0.25), dx, dy, 0.026, angle)

        for stripe in range(-3, 4):
            wx = plaza_radius * math.cos(angle) + 0.13 * stripe * math.cos(angle + math.pi / 2)
            wy = plaza_radius * math.sin(angle) + 0.13 * stripe * math.sin(angle + math.pi / 2)
            _flat(phys, [0.22, 0.05, 0.004], (0.9, 0.9, 0.92), wx, wy, 0.022, angle)

    car_colors = [
        (0.20, 0.40, 0.85),
        (0.85, 0.72, 0.20),
        (0.20, 0.62, 0.52),
        (0.72, 0.74, 0.78),
        (0.30, 0.30, 0.36),
        (0.55, 0.40, 0.70),
    ]
    for index, color in enumerate(car_colors):
        angle = 2 * math.pi * index / len(car_colors) + 0.25
        make_city_car(phys, [radius * math.cos(angle), radius * math.sin(angle)], angle + math.pi / 2, color)

    for direction, color in ((0, (0.85, 0.55, 0.15)), (2, (0.20, 0.50, 0.75))):
        angle = direction * math.pi / 2
        bus_radius = (plaza_radius + radius) / 2.0
        make_city_bus(phys, [bus_radius * math.cos(angle), bus_radius * math.sin(angle)], angle, color)

    ped_colors = [
        (0.20, 0.30, 0.72),
        (0.20, 0.60, 0.32),
        (0.25, 0.55, 0.62),
        (0.52, 0.52, 0.57),
        (0.62, 0.56, 0.22),
        (0.30, 0.45, 0.55),
    ]
    for index, color in enumerate(ped_colors):
        angle = math.pi / 5 + 2 * math.pi * index / len(ped_colors)
        ped_radius = 3.4 + float(rng.uniform(-0.3, 1.2))
        make_city_pedestrian(
            phys,
            [ped_radius * math.cos(angle), ped_radius * math.sin(angle), HUMAN_STAND_Z],
            shirt=color,
        )


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


def build_building_environment(phys):
    plane_id = p.loadURDF("plane.urdf", physicsClientId=phys)
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.34, 0.37, 0.39, 1], physicsClientId=phys)

    # Keep the centre clear for the follower, then wrap it with the city scene.
    _flat(phys, [2.25, 2.25, 0.008], (0.29, 0.32, 0.34), 0, 0, 0.018, 0.0)
    _flat(phys, [1.80, 0.06, 0.006], (0.86, 0.86, 0.80), 0, 0, 0.03, 0.0)
    _flat(phys, [0.06, 1.80, 0.006], (0.86, 0.86, 0.80), 0, 0, 0.031, 0.0)

    rng_buildings = np.random.default_rng(7)
    rng_props = np.random.default_rng(13)
    obstacles = spawn_buildings(phys, rng_buildings, n=14)
    spawn_city_props(phys, rng_props)
    return obstacles


def build_environment(phys, scenario):
    if scenario == SCENARIO_FOREST:
        return build_forest_environment(phys)
    if scenario == SCENARIO_BUILDINGS:
        return build_building_environment(phys)
    return build_park_environment(phys)


def scenario_user_position(scenario, t_wall):
    if scenario == SCENARIO_FOREST:
        return forest_walk(t_wall)
    return figure8(t_wall * USER_WALK_SPEED)


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
        linearDamping=RUNTIME_PID_DAMPING, angularDamping=0.9,
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


def launcher_photo_from_frame(frame, size):
    display = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    header = f"P6 {display.shape[1]} {display.shape[0]} 255\n".encode("ascii")
    return tk.PhotoImage(data=header + display.tobytes(), format="PPM")


def camera_training_frame(t_wall):
    width, height = CAMERA_RES
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    frame[:, :] = (10, 18, 31)
    cv2.rectangle(frame, (0, height // 2), (width, height), (19, 35, 49), -1)
    cv2.rectangle(frame, (0, 0), (width, 54), (15, 23, 38), -1)

    horizon = height // 2
    for x in range(-120, width + 120, 80):
        cv2.line(frame, (x, height), (width // 2, horizon), (32, 63, 82), 1)
    for y in range(horizon + 35, height, 46):
        cv2.line(frame, (0, y), (width, y), (28, 56, 75), 1)

    for x, y, w, h, color in [
        (24, 72, 92, 182, (35, 48, 63)),
        (500, 82, 78, 168, (38, 50, 68)),
        (436, 126, 44, 102, (49, 64, 82)),
    ]:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        for wy in range(y + 18, y + h - 10, 28):
            cv2.rectangle(frame, (x + 12, wy), (x + w - 12, wy + 7), (174, 196, 208), -1)

    marker_cx = int(width / 2 + math.sin(t_wall * 0.95) * 150)
    marker_cy = int(height / 2 + math.cos(t_wall * 1.18) * 58)
    radius = int(24 + 5 * math.sin(t_wall * 0.7))

    cv2.circle(frame, (150, 322), 18, (58, 105, 216), -1)
    cv2.circle(frame, (498, 328), 20, (66, 190, 116), -1)
    cv2.circle(frame, (marker_cx, marker_cy + radius + 24), 17, (198, 139, 86), -1)
    cv2.line(
        frame, (marker_cx, marker_cy + radius + 42),
        (marker_cx - 22, marker_cy + radius + 82),
        (51, 92, 173), 6,
    )
    cv2.line(
        frame, (marker_cx, marker_cy + radius + 42),
        (marker_cx + 23, marker_cy + radius + 82),
        (51, 92, 173), 6,
    )

    cv2.circle(frame, (marker_cx, marker_cy), radius, (230, 24, 34), -1)
    cv2.circle(
        frame,
        (marker_cx - radius // 3, marker_cy - radius // 3),
        max(4, radius // 5),
        (255, 100, 108),
        -1,
    )

    return frame, (marker_cx, marker_cy)


def camera_training_marker_box(frame):
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


def camera_training_step(tracker, t_wall):
    frame, expected = camera_training_frame(t_wall)
    pos, confidence = tracker.process_frame(frame)
    overlay = frame.copy()

    box = camera_training_marker_box(frame)
    pixel_error = float("inf")
    if box is not None:
        x, y, w, h, area = box
        detected = (x + w // 2, y + h // 2)
        pixel_error = float(np.hypot(detected[0] - expected[0], detected[1] - expected[1]))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 80), 3)
        cv2.drawMarker(
            overlay, detected, (255, 255, 255),
            markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2,
        )
        cv2.putText(
            overlay, f"detected {pixel_error:.1f}px",
            (max(8, x), max(26, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2,
        )
    else:
        cv2.putText(
            overlay, "searching",
            (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 90, 90), 2,
        )

    cv2.circle(overlay, expected, 7, (255, 255, 255), 2)
    cv2.putText(
        overlay, "expected",
        (expected[0] + 10, max(22, expected[1] - 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (226, 232, 240), 1,
    )

    passing = confidence >= CONFIDENCE_THRESH and pixel_error <= TRAINING_PIXEL_PASS
    return overlay, confidence, pixel_error, pos, passing


class CameraTrainingWindow:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Training Ground - Sub-1 Camera")
        self.window.configure(bg="#07111f")
        self.window.geometry("860x520+120+120")
        self.window.minsize(760, 460)
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.tracker = Tracker(DistanceEstimator())
        self.start_t = time.time()
        self.closed = False
        self.image = None

        container = tk.Frame(self.window, bg="#07111f", padx=16, pady=16)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_columnconfigure(0, weight=0)
        container.grid_columnconfigure(1, weight=1)
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)

        preview = tk.Frame(
            container,
            width=LAUNCHER_TRAINING_DISPLAY_SIZE[0],
            height=LAUNCHER_TRAINING_DISPLAY_SIZE[1],
            bg="#020617",
            highlightthickness=1,
            highlightbackground="#1f3a5f",
        )
        preview.grid(row=0, column=0, sticky="nw", padx=(0, 16))
        preview.grid_propagate(False)

        self.image_label = tk.Label(
            preview,
            bg="#020617",
            fg="#94a3b8",
            text="Starting camera trainer...",
            font=("Arial", 11, "bold"),
        )
        self.image_label.place(x=0, y=0, relwidth=1, relheight=1)

        info = tk.Frame(container, bg="#07111f")
        info.grid(row=0, column=1, sticky="nsew")
        info.grid_columnconfigure(0, weight=1)

        tk.Label(
            info,
            text="Sub-1 Camera",
            fg="#38bdf8", bg="#07111f",
            font=("Arial", 11, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")

        self.status_var = tk.StringVar(value="Camera trainer warming up")
        tk.Label(
            info,
            textvariable=self.status_var,
            fg="#f8fafc", bg="#0f172a",
            font=("Arial", 15, "bold"),
            anchor="w",
            padx=12, pady=12,
            wraplength=300,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(8, 0))

        self.detail_var = tk.StringVar(value="HSV tracker live check")
        tk.Label(
            info,
            textvariable=self.detail_var,
            fg="#cbd5e1", bg="#111c2e",
            font=("Arial", 10),
            anchor="nw",
            padx=12, pady=12,
            wraplength=300,
            justify="left",
        ).grid(row=2, column=0, sticky="ew", pady=(8, 0))

        tk.Button(
            info,
            text="Close",
            command=self.close,
            bg="#1e293b",
            fg="#cbd5e1",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief="flat",
            font=("Arial", 11, "bold"),
            padx=12,
            pady=10,
        ).grid(row=3, column=0, sticky="ew", pady=(18, 0))

        self._tick()

    def _tick(self):
        if self.closed:
            return

        t_wall = time.time() - self.start_t
        overlay, confidence, pixel_error, pos, passing = camera_training_step(self.tracker, t_wall)
        err_text = f"{pixel_error:.1f}px" if math.isfinite(pixel_error) else "no lock"
        verdict = "LOCKED" if passing else "SEARCHING"
        self.status_var.set(f"{verdict}: confidence {confidence:.2f}, error {err_text}")
        self.detail_var.set(
            "Same Sub-1 tracker as the main sim. White circle is the expected marker centre; "
            "green box is what the algorithm detected. "
            f"Estimated body offset: dx {pos[0]:+.2f} m, dy {pos[1]:+.2f} m, dz {pos[2]:.2f} m."
        )
        self.image = launcher_photo_from_frame(overlay, LAUNCHER_TRAINING_DISPLAY_SIZE)
        self.image_label.configure(image=self.image, text="")
        self.window.after(80, self._tick)

    def close(self):
        self.closed = True
        try:
            self.window.destroy()
        except tk.TclError:
            pass


class ScenarioLauncher:
    def __init__(self, default_duration=120.0):
        self.selection = None
        self.default_duration = default_duration
        self.scenario_cards = {}
        self.training_status_var = None
        self.training_window = None

        self.root = tk.Tk()
        self.root.title("SkyShade Launcher")
        self.root.configure(bg="#0b1120")
        self.root.geometry("1180x760")
        self.root.minsize(1040, 680)
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._cancel)

        self.scenario_var = tk.StringVar(value=SCENARIO_PARK)
        self.flight_var = tk.StringVar(value="q")
        self.duration_var = tk.StringVar(value=str(int(default_duration)))
        self.gui_var = tk.BooleanVar(value=True)
        self.error_var = tk.StringVar(value="")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        frame = tk.Frame(self.root, bg="#0b1120", padx=24, pady=22)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        hero = tk.Canvas(
            frame,
            height=150,
            bg="#08111f",
            highlightthickness=0,
        )
        hero.grid(row=0, column=0, sticky="ew")
        self._draw_launcher_hero(hero)

        content = tk.Frame(frame, bg="#0b1120")
        content.grid(row=1, column=0, sticky="nsew", pady=(18, 0))
        content.grid_columnconfigure(0, weight=4, uniform="launcher_columns")
        content.grid_columnconfigure(1, weight=5, uniform="launcher_columns")
        content.grid_rowconfigure(0, weight=1)

        training_panel = tk.Frame(content, bg="#0b1120")
        training_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        training_panel.grid_columnconfigure(0, weight=1)
        self._build_launcher_training_ground(training_panel, 0, base_bg="#0b1120", columns=1)

        right_panel = tk.Frame(content, bg="#0b1120")
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(1, weight=1)

        scenario_panel = tk.Frame(right_panel, bg="#0b1120")
        scenario_panel.grid(row=0, column=0, sticky="ew")
        scenario_panel.grid_columnconfigure(0, weight=1)

        settings_panel = tk.Frame(
            right_panel,
            bg="#101827",
            padx=16,
            pady=16,
            highlightthickness=1,
            highlightbackground="#26364f",
        )
        settings_panel.grid(row=1, column=0, sticky="nsew", pady=(14, 0))
        settings_panel.grid_columnconfigure(0, weight=1)

        tk.Label(
            scenario_panel,
            text="Scenario",
            fg="#f8fafc",
            bg="#0b1120",
            font=("Arial", 18, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")

        tk.Label(
            scenario_panel,
            text="Pick the world SkyShade should fly through.",
            fg="#aab8cf",
            bg="#0b1120",
            font=("Arial", 11),
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", pady=(2, 12))

        for row, scenario in enumerate(SCENARIO_CHOICES, start=2):
            self._scenario_card(scenario_panel, row, scenario)

        tk.Label(
            settings_panel,
            text="Mission Setup",
            fg="#f8fafc",
            bg="#101827",
            font=("Arial", 18, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")

        tk.Label(
            settings_panel,
            text="Runtime",
            fg="#38bdf8",
            bg="#101827",
            font=("Arial", 10, "bold"),
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", pady=(18, 5))

        duration_row = tk.Frame(settings_panel, bg="#101827")
        duration_row.grid(row=2, column=0, sticky="ew")
        duration_row.grid_columnconfigure(1, weight=1)
        tk.Label(
            duration_row,
            text="Duration",
            fg="#cbd5e1",
            bg="#101827",
            font=("Arial", 10, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=(0, 10))

        duration_entry = tk.Entry(
            duration_row,
            textvariable=self.duration_var,
            bg="#020617",
            fg="#f8fafc",
            insertbackground="#f8fafc",
            relief="flat",
            font=("Arial", 13, "bold"),
            width=8,
            justify="center",
        )
        duration_entry.grid(row=0, column=1, sticky="w")

        tk.Label(
            duration_row,
            text="seconds",
            fg="#94a3b8",
            bg="#101827",
            font=("Arial", 10),
            anchor="w",
        ).grid(row=0, column=2, sticky="w", padx=(8, 0))

        tk.Label(
            settings_panel,
            text="Flight Control",
            fg="#38bdf8",
            bg="#101827",
            font=("Arial", 10, "bold"),
            anchor="w",
        ).grid(row=3, column=0, sticky="ew", pady=(18, 5))

        flight_card = tk.Frame(
            settings_panel,
            bg="#0f172a",
            padx=12,
            pady=10,
            highlightthickness=1,
            highlightbackground="#2563eb",
        )
        flight_card.grid(row=4, column=0, sticky="ew")
        flight_card.grid_columnconfigure(0, weight=1)
        tk.Label(
            flight_card,
            text="Q-learning active",
            fg="#f8fafc",
            bg="#0f172a",
            font=("Arial", 12, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        tk.Label(
            flight_card,
            text="Sub-2 learned policy from qtable_v1.npy with velocity-aware state buckets.",
            fg="#bfdbfe",
            bg="#0f172a",
            font=("Arial", 10),
            anchor="w",
            justify="left",
            wraplength=520,
        ).grid(row=1, column=0, sticky="ew", pady=(3, 0))

        tk.Checkbutton(
            settings_panel,
            text="Show PyBullet GUI and dashboard",
            variable=self.gui_var,
            bg="#101827",
            fg="#cbd5e1",
            selectcolor="#020617",
            activebackground="#101827",
            activeforeground="#f8fafc",
            font=("Arial", 11),
            anchor="w",
        ).grid(row=5, column=0, sticky="w", pady=(18, 0))

        tk.Label(
            settings_panel,
            textvariable=self.error_var,
            fg="#fecaca",
            bg="#101827",
            font=("Arial", 10, "bold"),
            anchor="w",
            wraplength=300,
            justify="left",
        ).grid(row=6, column=0, sticky="ew", pady=(16, 0))

        actions = tk.Frame(settings_panel, bg="#101827")
        actions.grid(row=7, column=0, sticky="sew", pady=(24, 0))
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=1)
        settings_panel.grid_rowconfigure(7, weight=1)

        tk.Button(
            actions,
            text="Launch",
            command=self._launch,
            bg="#2563eb",
            fg="#eff6ff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            font=("Arial", 13, "bold"),
            padx=14,
            pady=12,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))

        tk.Button(
            actions,
            text="Cancel",
            command=self._cancel,
            bg="#1e293b",
            fg="#cbd5e1",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief="flat",
            font=("Arial", 13, "bold"),
            padx=14,
            pady=12,
        ).grid(row=0, column=1, sticky="ew")

        self.scenario_var.trace_add("write", lambda *_: self._refresh_scenario_cards())
        self._refresh_scenario_cards()

    def _build_launcher_training_ground(self, parent, row, base_bg="#0b1120", columns=2):
        panel = tk.Frame(parent, bg=base_bg)
        panel.grid(row=row, column=0, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(row, weight=1)

        header = tk.Frame(panel, bg=base_bg)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        tk.Label(
            header,
            text="Training Ground",
            fg="#f8fafc",
            bg=base_bg,
            font=("Arial", 20, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")

        tk.Button(
            header,
            text="Open Camera",
            command=lambda: self._open_training_ground("camera"),
            bg="#2563eb",
            fg="#eff6ff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief="flat",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=6,
        ).grid(row=0, column=1, sticky="e")

        tk.Label(
            panel,
            text="Subsystem trainers and validation surfaces.",
            fg="#aab8cf",
            bg=base_bg,
            font=("Arial", 11),
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", pady=(2, 8))

        grid = tk.Frame(panel, bg=base_bg)
        grid.grid(row=2, column=0, sticky="ew")
        for col in range(columns):
            grid.grid_columnconfigure(col, weight=1, uniform="training_cards")

        for index, item in enumerate(TRAINING_GROUND_ITEMS):
            self._training_ground_card(
                grid,
                index // columns,
                index % columns,
                item,
                compact=columns > 1,
            )

        self.training_status_var = tk.StringVar(
            value="Camera trainer is ready here. Press Open Camera to see the live Sub-1 lock check."
        )
        tk.Label(
            panel,
            textvariable=self.training_status_var,
            fg="#cbd5e1",
            bg="#111c2e",
            font=("Arial", 10),
            anchor="w",
            justify="left",
            wraplength=420 if columns == 1 else 690,
            padx=10,
            pady=8,
        ).grid(row=3, column=0, sticky="ew", pady=(8, 0))

    def _training_ground_card(self, parent, row, col, item, compact=False):
        key, title, tag, description, accent = item
        card = tk.Frame(
            parent,
            bg="#111827",
            padx=14 if not compact else 10,
            pady=11 if not compact else 8,
            highlightthickness=1,
            highlightbackground="#26364f",
        )
        card.grid(row=row, column=col, sticky="ew", padx=(0 if col == 0 else 8, 0), pady=(0, 8))
        card.grid_columnconfigure(0, weight=1)

        tk.Label(
            card,
            text=title,
            fg="#f8fafc",
            bg="#111827",
            font=("Arial", 12 if not compact else 11, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")

        tk.Label(
            card,
            text=tag,
            fg=accent,
            bg="#111827",
            font=("Arial", 10 if not compact else 9, "bold"),
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", pady=(1, 0))

        tk.Label(
            card,
            text=description,
            fg="#aab8cf",
            bg="#111827",
            font=("Arial", 10 if not compact else 9),
            anchor="w",
            justify="left",
            wraplength=360 if not compact else 255,
        ).grid(row=2, column=0, sticky="ew", pady=(3, 0))

        button_text = "Open" if key == "camera" else "Focus"
        tk.Button(
            card,
            text=button_text,
            command=lambda selected=key: self._open_training_ground(selected),
            bg="#0f172a",
            fg="#e0f2fe",
            activebackground="#1e293b",
            activeforeground="#ffffff",
            relief="flat",
            font=("Arial", 9, "bold"),
            padx=8,
            pady=5,
        ).grid(row=0, column=1, rowspan=3, sticky="e", padx=(8, 0))

    def _open_training_ground(self, key):
        messages = {
            "flight": (
                "Sub-2 Flight selected: Q-learning is the reinforcement-learning module. "
                "I set Flight to Q-learning; launch the sim to watch the learned policy run."
            ),
            "weather": (
                "Sub-3 Weather selected: this is supervised SVM classification for umbrella decisions."
            ),
            "safety": (
                "Sub-4 Safety selected: this uses an MDP policy table for battery and navigation overrides."
            ),
            "environment": (
                "Environment selected: Building District is the stress scene for roads, buildings, and pedestrians."
            ),
        }

        if key == "camera":
            if self.training_window is None or self.training_window.closed:
                self.training_window = CameraTrainingWindow(self.root)
            else:
                self.training_window.window.lift()
            if self.training_status_var is not None:
                self.training_status_var.set(
                    "Sub-1 Camera is open. It should show LOCKED when confidence stays high and marker error is under 18px."
                )
            return

        if key == "flight":
            self.flight_var.set("q")
        elif key == "environment":
            self._set_scenario(SCENARIO_BUILDINGS)

        if self.training_status_var is not None:
            self.training_status_var.set(messages[key])

    def _draw_launcher_hero(self, canvas):
        canvas.update_idletasks()
        width = int(canvas.winfo_width() or 980)
        height = 150
        canvas.create_rectangle(0, 0, width, height, fill="#08111f", outline="")
        canvas.create_rectangle(0, 80, width, height, fill="#0d2a3d", outline="")
        canvas.create_rectangle(0, 118, width, height, fill="#0f3a2a", outline="")

        for x in range(0, width, 42):
            top = 46 + (x * 7) % 38
            shade = "#14243a" if x % 84 else "#1b3150"
            canvas.create_rectangle(x, top, x + 30, 118, fill=shade, outline="")
            if x % 84 == 0:
                canvas.create_rectangle(x + 8, top + 12, x + 12, top + 18, fill="#facc15", outline="")
                canvas.create_rectangle(x + 18, top + 28, x + 22, top + 34, fill="#facc15", outline="")

        canvas.create_line(0, 118, width, 118, fill="#38bdf8", width=2)
        canvas.create_text(
            26, 34,
            text="SkyShade",
            anchor="w",
            fill="#f8fafc",
            font=("Arial", 30, "bold"),
        )
        canvas.create_text(
            28, 68,
            text="Autonomous drone umbrella simulation",
            anchor="w",
            fill="#b6c6dd",
            font=("Arial", 13),
        )

        cx = width - 160
        cy = 54
        canvas.create_line(cx - 62, cy, cx + 62, cy, fill="#93c5fd", width=5)
        canvas.create_line(cx, cy - 35, cx, cy + 35, fill="#93c5fd", width=5)
        for dx, dy in [(-62, 0), (62, 0), (0, -35), (0, 35)]:
            canvas.create_oval(cx + dx - 22, cy + dy - 10, cx + dx + 22, cy + dy + 10,
                               outline="#bfdbfe", width=2)
        canvas.create_rectangle(cx - 30, cy - 15, cx + 30, cy + 15,
                                fill="#2563eb", outline="#bfdbfe", width=2)
        canvas.create_arc(cx - 62, cy + 12, cx + 62, cy + 78,
                          start=0, extent=180, fill="#38bdf8", outline="#e0f2fe", width=2)

    def _scenario_card(self, parent, row, scenario):
        shell = tk.Frame(
            parent,
            bg="#111827",
            padx=10,
            pady=9,
            highlightthickness=2,
            highlightbackground="#23334d",
        )
        shell.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        shell.grid_columnconfigure(1, weight=1)
        self.scenario_cards[scenario] = shell

        preview = tk.Canvas(
            shell,
            width=112,
            height=62,
            bg="#07111f",
            highlightthickness=0,
        )
        preview.grid(row=0, column=0, rowspan=2, sticky="nw", padx=(0, 14))
        self._draw_scenario_preview(preview, scenario)

        tk.Label(
            shell,
            text=SCENARIO_LABELS[scenario],
            fg="#f8fafc",
            bg="#111827",
            font=("Arial", 13, "bold"),
            anchor="w",
        ).grid(row=0, column=1, sticky="ew")

        tk.Label(
            shell,
            text=SCENARIO_DESCRIPTIONS[scenario],
            fg="#aab8cf",
            bg="#111827",
            font=("Arial", 10),
            anchor="nw",
            justify="left",
            wraplength=360,
        ).grid(row=1, column=1, sticky="ew", pady=(4, 0))

        button = tk.Radiobutton(
            shell,
            text="Select",
            value=scenario,
            variable=self.scenario_var,
            indicatoron=False,
            command=self._refresh_scenario_cards,
            bg="#0f172a",
            fg="#f8fafc",
            selectcolor="#075985",
            activebackground="#1e293b",
            activeforeground="#ffffff",
            relief="flat",
            font=("Arial", 9, "bold"),
            anchor="center",
            padx=10,
            pady=5,
            width=10,
        )
        button.grid(row=0, column=2, rowspan=2, sticky="e", padx=(14, 0))

        for widget in (shell, preview):
            widget.bind("<Button-1>", lambda _event, value=scenario: self._set_scenario(value))

    def _draw_scenario_preview(self, canvas, scenario):
        canvas.create_rectangle(0, 0, 112, 62, fill="#07111f", outline="")
        canvas.create_rectangle(0, 37, 112, 62, fill="#12351f", outline="")
        if scenario == SCENARIO_FOREST:
            canvas.create_rectangle(0, 0, 112, 62, fill="#071a14", outline="")
            canvas.create_polygon(0, 62, 34, 34, 78, 34, 112, 62, fill="#5b4329", outline="")
            for x in (14, 32, 82, 98):
                canvas.create_rectangle(x, 27, x + 4, 58, fill="#6b3f1d", outline="")
                canvas.create_oval(x - 12, 9, x + 17, 36, fill="#1f6f3b", outline="")
            canvas.create_line(16, 53, 97, 41, fill="#94a3b8", width=2)
        elif scenario == SCENARIO_BUILDINGS:
            canvas.create_rectangle(0, 0, 112, 62, fill="#08111f", outline="")
            for x, top, color in [(7, 17, "#27364a"), (28, 8, "#334155"), (52, 21, "#1f2a44"), (80, 12, "#3b4251")]:
                canvas.create_rectangle(x, top, x + 18, 45, fill=color, outline="")
                for y in range(top + 7, 43, 10):
                    canvas.create_rectangle(x + 5, y, x + 7, y + 3, fill="#fde68a", outline="")
                    canvas.create_rectangle(x + 12, y, x + 14, y + 3, fill="#fde68a", outline="")
            canvas.create_rectangle(0, 45, 112, 62, fill="#1f2937", outline="")
            canvas.create_line(0, 53, 112, 53, fill="#facc15", width=2)
        else:
            canvas.create_rectangle(0, 0, 112, 62, fill="#0b2535", outline="")
            canvas.create_rectangle(0, 34, 112, 62, fill="#1f5b35", outline="")
            canvas.create_line(0, 51, 112, 33, fill="#9a7a4d", width=8)
            for x, y in [(16, 36), (84, 32), (96, 47)]:
                canvas.create_rectangle(x, y, x + 4, y + 15, fill="#6b3f1d", outline="")
                canvas.create_oval(x - 9, y - 15, x + 14, y + 6, fill="#2f7d42", outline="")
            canvas.create_oval(49, 14, 66, 30, outline="#dbeafe", width=2)

    def _set_scenario(self, scenario):
        self.scenario_var.set(scenario)
        self._refresh_scenario_cards()

    def _refresh_scenario_cards(self):
        selected = self.scenario_var.get()
        for scenario, card in self.scenario_cards.items():
            active = scenario == selected
            card.configure(
                bg="#10243a" if active else "#111827",
                highlightbackground="#38bdf8" if active else "#23334d",
            )
            for child in card.winfo_children():
                if isinstance(child, tk.Label):
                    child.configure(bg="#10243a" if active else "#111827")

    def _launch(self):
        try:
            duration = float(self.duration_var.get())
            if duration <= 0:
                raise ValueError
        except ValueError:
            self.error_var.set("Enter a positive duration in seconds.")
            return

        self.selection = {
            "scenario": self.scenario_var.get(),
            "flight": self.flight_var.get(),
            "duration": duration,
            "gui": bool(self.gui_var.get()),
        }
        self.root.destroy()

    def _cancel(self):
        self.selection = None
        self.root.destroy()

    def show(self):
        self.root.mainloop()
        return self.selection


def choose_launch_settings(default_duration=120.0):
    if tk is None:
        return {
            "scenario": SCENARIO_PARK,
            "flight": "q",
            "duration": default_duration,
            "gui": True,
        }

    try:
        return ScenarioLauncher(default_duration).show()
    except tk.TclError as exc:
        print(f"Launcher unavailable: {exc}")
        return {
            "scenario": SCENARIO_PARK,
            "flight": "q",
            "duration": default_duration,
            "gui": True,
        }


def launch_sim_process(settings):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--scenario",
        settings["scenario"],
        "--duration",
        str(settings["duration"]),
        "--flight",
        settings.get("flight", "pid"),
    ]
    if not settings["gui"]:
        cmd.append("--no-gui")

    print("Launching:", " ".join(cmd))
    subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))


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
        self.training_label = None
        self.training_image = None
        self.training_status_var = None
        self.training_detail_var = None
        self.training_tracker = Tracker(DistanceEstimator())
        self.training_last_update_t = -1.0
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
            "GAIN": ("#064e3b", "#d1fae5"),
            "COST": ("#7f1d1d", "#fee2e2"),
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
            ("controller", "Flight controller"),
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
            ("flight_reward", "Sub-2 reward"),
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
        self.vision_window.geometry("1240x1040+20+40")
        self.vision_window.minsize(1060, 900)
        self.vision_window.protocol("WM_DELETE_WINDOW", self._hide_vision_window)

        tk.Label(
            self.vision_window,
            text="Drone POV: camera feed with red-marker detection",
            fg="#f8fafc", bg="#07111f",
            font=("Arial", 15, "bold"),
            anchor="w",
            padx=14, pady=10,
        ).grid(row=0, column=0, sticky="ew")

        camera_preview = tk.Frame(
            self.vision_window,
            width=POV_DISPLAY_SIZE[0],
            height=POV_DISPLAY_SIZE[1],
            bg="#020617",
            highlightthickness=1,
            highlightbackground="#1f3a5f",
        )
        camera_preview.grid(row=1, column=0, sticky="n", padx=14)
        camera_preview.grid_propagate(False)

        self.camera_label = tk.Label(
            camera_preview,
            bg="#020617",
            fg="#94a3b8",
            text="Waiting for camera frame...",
            font=("Arial", 12),
        )
        self.camera_label.place(x=0, y=0, relwidth=1, relheight=1)

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
            height=230,
            bg="#0b1726",
            highlightthickness=0,
        )
        self.graph_canvas.grid(row=4, column=0, sticky="nsew", padx=14, pady=(0, 10))

        self._build_training_ground(self.vision_window, 5)

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
        ).grid(row=6, column=0, sticky="ew", padx=14, pady=(0, 14))

        self.vision_window.grid_rowconfigure(1, weight=3)
        self.vision_window.grid_rowconfigure(4, weight=1)
        self.vision_window.grid_rowconfigure(5, weight=1)
        self.vision_window.grid_columnconfigure(0, weight=1)

    def _build_training_ground(self, parent, row):
        ground = tk.Frame(parent, bg="#081322", padx=12, pady=12)
        ground.grid(row=row, column=0, sticky="ew", padx=14, pady=(0, 10))
        ground.grid_columnconfigure(0, weight=0)
        ground.grid_columnconfigure(1, weight=1)

        tk.Label(
            ground,
            text="Training Ground",
            fg="#f8fafc", bg="#081322",
            font=("Arial", 14, "bold"),
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="ew")

        preview = tk.Frame(
            ground,
            width=TRAINING_DISPLAY_SIZE[0],
            height=TRAINING_DISPLAY_SIZE[1],
            bg="#020617",
            highlightthickness=1,
            highlightbackground="#1f3a5f",
        )
        preview.grid(row=1, column=0, rowspan=2, sticky="nw", pady=(10, 0), padx=(0, 14))
        preview.grid_propagate(False)

        self.training_label = tk.Label(
            preview,
            bg="#020617",
            fg="#94a3b8",
            text="Preparing camera trainer...",
            font=("Arial", 10, "bold"),
        )
        self.training_label.place(x=0, y=0, relwidth=1, relheight=1)

        info = tk.Frame(ground, bg="#081322")
        info.grid(row=1, column=1, sticky="nsew", pady=(10, 0))
        info.grid_columnconfigure(0, weight=1)

        tk.Label(
            info,
            text="Sub-1 Perception",
            fg="#38bdf8", bg="#081322",
            font=("Arial", 10, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")

        self.training_status_var = tk.StringVar(value="Camera trainer warming up")
        tk.Label(
            info,
            textvariable=self.training_status_var,
            fg="#f8fafc", bg="#0f172a",
            font=("Arial", 12, "bold"),
            anchor="w",
            padx=10, pady=8,
            wraplength=690,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(6, 0))

        self.training_detail_var = tk.StringVar(
            value="HSV marker segmentation + contour tracking + EMA smoothing"
        )
        tk.Label(
            info,
            textvariable=self.training_detail_var,
            fg="#cbd5e1", bg="#111c2e",
            font=("Arial", 10),
            anchor="w",
            padx=10, pady=8,
            wraplength=690,
            justify="left",
        ).grid(row=2, column=0, sticky="ew", pady=(6, 0))

        stages = tk.Frame(info, bg="#081322")
        stages.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        stage_specs = [
            ("Sub-1 Camera", "live check", "#22c55e"),
            ("Sub-2 Flight", "Q-learning", "#60a5fa"),
            ("Sub-3 Weather", "SVM", "#f472b6"),
            ("Sub-4 Safety", "MDP", "#f59e0b"),
            ("Environment", "scenarios", "#a78bfa"),
        ]
        for col, (title, tag, color) in enumerate(stage_specs):
            tile = tk.Frame(stages, bg="#0f172a", padx=8, pady=7)
            tile.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 6, 0))
            stages.grid_columnconfigure(col, weight=1, uniform="training_stages")
            tk.Label(
                tile,
                text=title,
                fg="#f8fafc", bg="#0f172a",
                font=("Arial", 9, "bold"),
                anchor="center",
            ).grid(row=0, column=0, sticky="ew")
            tk.Label(
                tile,
                text=tag,
                fg=color, bg="#0f172a",
                font=("Arial", 8, "bold"),
                anchor="center",
            ).grid(row=1, column=0, sticky="ew", pady=(2, 0))

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

    def _frame_to_photo(self, frame, size=POV_DISPLAY_SIZE):
        display = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        header = f"P6 {display.shape[1]} {display.shape[0]} 255\n".encode("ascii")
        return tk.PhotoImage(data=header + display.tobytes(), format="PPM")

    def _make_training_frame(self, t_wall):
        width, height = CAMERA_RES
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        frame[:, :] = (10, 18, 31)
        cv2.rectangle(frame, (0, height // 2), (width, height), (19, 35, 49), -1)
        cv2.rectangle(frame, (0, 0), (width, 54), (15, 23, 38), -1)

        horizon = height // 2
        for x in range(-120, width + 120, 80):
            cv2.line(frame, (x, height), (width // 2, horizon), (32, 63, 82), 1)
        for y in range(horizon + 35, height, 46):
            cv2.line(frame, (0, y), (width, y), (28, 56, 75), 1)

        for x, y, w, h, color in [
            (24, 72, 92, 182, (35, 48, 63)),
            (500, 82, 78, 168, (38, 50, 68)),
            (436, 126, 44, 102, (49, 64, 82)),
        ]:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            for wy in range(y + 18, y + h - 10, 28):
                cv2.rectangle(frame, (x + 12, wy), (x + w - 12, wy + 7), (174, 196, 208), -1)

        marker_cx = int(width / 2 + math.sin(t_wall * 0.95) * 150)
        marker_cy = int(height / 2 + math.cos(t_wall * 1.18) * 58)
        radius = int(24 + 5 * math.sin(t_wall * 0.7))

        cv2.circle(frame, (150, 322), 18, (58, 105, 216), -1)
        cv2.circle(frame, (498, 328), 20, (66, 190, 116), -1)
        cv2.circle(frame, (marker_cx, marker_cy + radius + 24), 17, (198, 139, 86), -1)
        cv2.line(frame, (marker_cx, marker_cy + radius + 42), (marker_cx - 22, marker_cy + radius + 82), (51, 92, 173), 6)
        cv2.line(frame, (marker_cx, marker_cy + radius + 42), (marker_cx + 23, marker_cy + radius + 82), (51, 92, 173), 6)

        cv2.circle(frame, (marker_cx, marker_cy), radius, (230, 24, 34), -1)
        cv2.circle(frame, (marker_cx - radius // 3, marker_cy - radius // 3), max(4, radius // 5), (255, 100, 108), -1)

        return frame, (marker_cx, marker_cy)

    def _tick_training_ground(self, t_wall):
        if self.training_label is None:
            return
        if self.training_last_update_t >= 0 and t_wall - self.training_last_update_t < 0.08:
            return

        self.training_last_update_t = t_wall
        frame, expected = self._make_training_frame(t_wall)
        pos, confidence = self.training_tracker.process_frame(frame)
        overlay = frame.copy()

        box = self._detect_marker_box(frame)
        pixel_error = float("inf")
        if box is not None:
            x, y, w, h, area = box
            detected = (x + w // 2, y + h // 2)
            pixel_error = float(np.hypot(detected[0] - expected[0], detected[1] - expected[1]))
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 80), 3)
            cv2.drawMarker(
                overlay, detected, (255, 255, 255),
                markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2,
            )
            cv2.putText(
                overlay, f"detected {pixel_error:.1f}px",
                (max(8, x), max(26, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2,
            )
        else:
            cv2.putText(
                overlay, "searching",
                (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 90, 90), 2,
            )

        cv2.circle(overlay, expected, 7, (255, 255, 255), 2)
        cv2.putText(
            overlay, "expected",
            (expected[0] + 10, max(22, expected[1] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (226, 232, 240), 1,
        )

        passing = confidence >= CONFIDENCE_THRESH and pixel_error <= TRAINING_PIXEL_PASS
        verdict = "LOCKED" if passing else "SEARCHING"
        err_text = f"{pixel_error:.1f}px" if math.isfinite(pixel_error) else "no lock"

        if self.training_status_var is not None:
            self.training_status_var.set(
                f"{verdict}: confidence {confidence:.2f}, marker error {err_text}"
            )
        if self.training_detail_var is not None:
            self.training_detail_var.set(
                "Camera uses classic computer vision, not reinforcement learning: "
                "HSV thresholding finds the red marker, contours locate it, EMA smooths it, "
                f"and the distance estimator reports dx {pos[0]:+.2f} m, dy {pos[1]:+.2f} m, dz {pos[2]:.2f} m."
            )

        self.training_image = self._frame_to_photo(overlay, TRAINING_DISPLAY_SIZE)
        self.training_label.configure(image=self.training_image, text="")

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
        self._tick_training_ground(t_wall)

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
    flight_mode="PID", flight_action="PID_FORCE", flight_reward=None,
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
        f"Perception tracking the marker; {flight_mode} flight controller active; "
        "environment AI choosing shade; navigation AI checking safety."
    )
    flight_reward_text = (
        f"{flight_reward:+.2f} ({flight_action})"
        if flight_reward is not None
        else f"n/a ({flight_action})"
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
            "controller": flight_mode,
            "altitude": f"{drone_pos[2]:.2f} m above ground",
            "hover_error": f"{err_xy:.2f} m from target",
            "tracking": f"{confidence * 100:.0f}% confidence",
            "drone_position": f"x {drone_pos[0]:.2f}, y {drone_pos[1]:.2f}",
            "user_position": f"x {user_pos[0]:.2f}, y {user_pos[1]:.2f}",
            "user_offset": user_offset_text,
            "location_summary": location_summary,
            "flight_reward": flight_reward_text,
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
            "controller": "ACTIVE" if flight_mode != "PID" else "OK",
            "flight_reward": (
                "GAIN" if flight_reward >= 0 else "COST"
            ) if flight_reward is not None else "IDLE",
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


def run(
    duration=120.0,
    gui=True,
    scenario=SCENARIO_PARK,
    flight="pid",
    battery_start=100.0,
    battery_drain_rate=BATTERY_DRAIN_RATE,
):
    mode = p.GUI if gui else p.DIRECT
    phys = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=phys)
    p.setGravity(0, 0, -9.81, physicsClientId=phys)
    p.setTimeStep(SIM_TIMESTEP, physicsClientId=phys)

    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=phys)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1, physicsClientId=phys)
        if scenario == SCENARIO_FOREST:
            p.resetDebugVisualizerCamera(
                cameraDistance=14.0, cameraYaw=35, cameraPitch=-34,
                cameraTargetPosition=[0, 0, 1.3],
                physicsClientId=phys,
            )
        elif scenario == SCENARIO_BUILDINGS:
            p.resetDebugVisualizerCamera(
                cameraDistance=12.0, cameraYaw=42, cameraPitch=-31,
                cameraTargetPosition=[0, 0, 1.4],
                physicsClientId=phys,
            )
        else:
            p.resetDebugVisualizerCamera(
                cameraDistance=7, cameraYaw=30, cameraPitch=-25,
                cameraTargetPosition=[0, 0, 1],
                physicsClientId=phys,
            )

    flight_controller = RuntimeFlightController(flight)
    if flight_controller.fallback_reason:
        print(f"[flight] Q-learning unavailable; falling back to PID ({flight_controller.fallback_reason})")

    obstacles = build_environment(phys, scenario)
    initial_user_pos = scenario_user_position(scenario, 0.0)

    # Drone and user are assembled from primitive bodies so the scene has a
    # readable physical scale while the AI/control code keeps the same inputs.
    drone_id, drone_parts = create_drone(phys, initial_user_pos[:2])
    drone_damping = LINEAR_DAMPING if flight_controller.using_q else RUNTIME_PID_DAMPING
    p.changeDynamics(
        drone_id, -1, mass=DRONE_MASS_KG,
        linearDamping=drone_damping, angularDamping=0.9,
        physicsClientId=phys,
    )
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
    if scenario == SCENARIO_FOREST:
        flight_controller.tune_pid_for_forest()
    umbrella   = UmbrellaClassifier()
    nav_policy = NavSafetyPolicy()

    battery_pct   = float(np.clip(battery_start, 0.0, 100.0))
    umbrella_cmd  = "STOW"
    nav_override  = "CONTINUE"
    confidence    = 0.0
    user_pos      = initial_user_pos.copy()
    prev_user_pos = user_pos.copy()
    forest_lap    = forest_lap_index(0.0)
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
    prev_t_wall = 0.0

    print(f"SkyShade simulation running ({scenario} scenario) — Ctrl+C to stop.\n")
    print(f"Flight controller: {flight_controller.label}")
    print(
        f"Battery starts at {battery_pct:.1f}% and drains at "
        f"{battery_drain_rate:.2f}% per second.\n"
    )
    print(f"{'Time':>6}  {'Bat':>6}  {'Nav':<10}  {'Flight':<7}  {'Umbrella':<8}  {'Conf':>5}  {'ErrXY':>6}  {'Z':>5}")
    print("-" * 75)

    try:
        while True:
            t_wall = time.monotonic() - t_start
            loop_dt = max(0.0, t_wall - prev_t_wall)
            prev_t_wall = t_wall
            if t_wall >= duration:
                print(f"\nSimulation complete ({duration:.0f} s).")
                break
            if telemetry.closed:
                print("\nSimulation stopped from dashboard.")
                break
            weather_now = weather_controller.update(t_wall)

            # ── User path ────────────────────────────────────────────────────
            user_pos = scenario_user_position(scenario, t_wall)
            wrapped_forest_lap = False
            if scenario == SCENARIO_FOREST:
                next_lap = forest_lap_index(t_wall)
                wrapped_forest_lap = next_lap != forest_lap
                forest_lap = next_lap
                if wrapped_forest_lap:
                    prev_user_pos = user_pos.copy()
                    user_velocity = np.zeros(3, dtype=float)
                    gimbal_target = np.array(
                        [user_pos[0], user_pos[1], USER_MARKER_HEIGHT],
                        dtype=float,
                    )
                    reset_pos = [
                        user_pos[0],
                        user_pos[1],
                        TARGET_ALTITUDE,
                    ]
                    p.resetBasePositionAndOrientation(
                        drone_id, reset_pos, [0, 0, 0, 1],
                        physicsClientId=phys,
                    )
                    p.resetBaseVelocity(
                        drone_id, [0, 0, 0], [0, 0, 0],
                        physicsClientId=phys,
                    )
                    flight_controller.reset()

            user_delta = np.zeros(3, dtype=float) if wrapped_forest_lap else user_pos - prev_user_pos
            user_velocity = user_delta / max(action_dt, 1e-6)
            user_yaw = math.atan2(user_delta[1], user_delta[0]) if np.linalg.norm(user_delta[:2]) > 1e-4 else 0.0
            update_person(phys, person_parts, user_pos, user_yaw, t_wall * 4.5)
            prev_user_pos = user_pos.copy()

            # ── Battery ───────────────────────────────────────────────────────
            battery_pct = max(0.0, battery_pct - battery_drain_rate * loop_dt)

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

            # ── Sub-2: flight control (PID or learned Q-policy) ──────────────
            lin_vel, _ = p.getBaseVelocity(drone_id, physicsClientId=phys)
            drone_vel  = np.array(lin_vel)

            flight_cmd = flight_controller.compute_force(
                drone_pos=drone_pos,
                drone_vel=drone_vel,
                user_pos=user_pos,
                user_velocity=user_velocity,
                nav_override=nav_override,
                wind_speed=weather_now["wind"],
                dt=dt,
            )
            force = flight_cmd.force.copy()
            avoidance = obstacle_avoidance_force(
                drone_pos, obstacles,
                radius=AVOIDANCE_RADIUS,
                gain=AVOIDANCE_GAIN,
                max_force=AVOIDANCE_MAX_FORCE,
            )
            force += avoidance
            force[2] += HOVER_FORCE_N
            wind_mag = float(weather_now.get("wind", 0.0))
            gust_angle = 0.7 * t_wall + 2.0 * math.sin(0.23 * t_wall)
            gust = wind_mag * WIND_FORCE_SCALE
            force[0] += math.cos(gust_angle) * gust
            force[1] += math.sin(gust_angle) * gust
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
                    flight_mode=flight_cmd.controller,
                    flight_action=flight_cmd.action,
                    flight_reward=flight_cmd.reward,
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
                      f"{flight_cmd.controller:<7}  {umbrella_cmd:<8}  "
                      f"{confidence:.2f}  {err_xy:6.2f}m  {d_pos2[2]:.2f}m")

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
        "--flight",
        choices=["pid", "q"],
        default="q",
        help="Flight controller: 'q' for the learned Sub-2 policy, or 'pid' for the fallback baseline.",
    )
    ap.add_argument(
        "--battery-start",
        type=float,
        default=None,
        help="Initial battery percentage. Default: 100, or 10 with --demo-low-battery.",
    )
    ap.add_argument(
        "--battery-drain-rate",
        type=float,
        default=None,
        help="Battery drain percentage per second. Default: 0.5, or 1.0 with --demo-low-battery.",
    )
    ap.add_argument(
        "--demo-low-battery",
        action="store_true",
        help="Start near low battery so RTH/LAND_NOW appears quickly.",
    )
    ap.add_argument(
        "--launcher",
        action="store_true",
        help="Show the scenario launcher even if a scenario is provided.",
    )
    ap.add_argument(
        "--scenario",
        choices=SCENARIO_CHOICES,
        default=None,
        help="Simulation scene to run.",
    )
    args = ap.parse_args()

    if args.launcher or (args.scenario is None and not args.no_gui):
        settings = choose_launch_settings(args.duration)
        if settings is None:
            print("Launch cancelled.")
            return
        launch_sim_process(settings)
        return

    battery_start = args.battery_start
    battery_drain_rate = args.battery_drain_rate
    if args.demo_low_battery:
        if battery_start is None:
            battery_start = LOW_BATTERY_DEMO_START
        if battery_drain_rate is None:
            battery_drain_rate = LOW_BATTERY_DEMO_DRAIN_RATE

    if battery_start is None:
        battery_start = 100.0
    if battery_drain_rate is None:
        battery_drain_rate = BATTERY_DRAIN_RATE

    run(
        duration=args.duration,
        gui=not args.no_gui,
        scenario=args.scenario or SCENARIO_PARK,
        flight=args.flight,
        battery_start=battery_start,
        battery_drain_rate=battery_drain_rate,
    )


if __name__ == "__main__":
    main()
