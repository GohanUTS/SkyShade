"""
SkyShade — full integrated simulation runner.

Starts a PyBullet GUI window and runs all four subsystems together:
  Sub-1  Perception       HSV tracker + distance estimator (camera feed)
  Sub-2  Flight Control   PPO hover policy with Q/PID fallbacks
  Sub-3  Env Decision     SVM umbrella classifier
  Sub-4  Nav Safety       MDP policy table

A simulated user walks a figure-8 path below the drone.  Weather sensors
randomly switch between cloudy periods, light rain, and full rain.
Battery drains and triggers the Sub-4 safety override.

Usage:
    python run_sim.py [--duration 120] [--no-gui] [--flight ppo|q|pid]
                      [--demo-low-battery]
"""

import argparse
import json
import math
import queue
import subprocess
import sys
import threading
import time
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/skyshade_mpl")

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, messagebox as tk_messagebox
except ImportError:
    tk = None
    ttk = None
    tk_messagebox = None

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
from sub1_perception.training import (
    camera_training_frame, camera_training_marker_box, camera_training_step,
    TRAINING_PIXEL_PASS,
)
from sub2_flight.training_worker import FlightTrainingWorker as _FlightTrainingWorker
from sub4_nav.training_worker import MDPSolverWorker as _MDPSolverWorker

CONTROL_HZ         = 30
BATTERY_DRAIN_RATE = 0.5    # % per second
LOW_BATTERY_DEMO_START = 10.0
LOW_BATTERY_DEMO_DRAIN_RATE = 1.0
RUNTIME_PID_DAMPING = 0.5
WIND_FORCE_SCALE    = 0.35   # N per m/s of weather wind pushed on the drone
USER_WALK_SPEED    = 0.2    # rad/s for figure-8 (gentle, relaxed walking pace)
HOVER_RADIUS       = 0.5    # m
WEATHER_MIN_SECONDS = 8.0
WEATHER_MAX_SECONDS = 20.0
UMBRELLA_DEPLOY_RAIN_THRESHOLD = 0.20  # matches Sub-3 training labels
DEBUG_WEATHER_LINES = False
# In-scene rain/cloud/mist visuals.  Drawn with addUserDebugLine, so they cost
# GPU; redraw cadence + drop counts are tuned to stay smooth on integrated GPUs.
# Disable with env SKYSHADE_WEATHER_VIS=0 if a machine still struggles.
# On by default: clouds are now static (drawn once) and rain is light, so the
# old integrated-GPU lag is gone.  Still disable with SKYSHADE_WEATHER_VIS=0.
WEATHER_VISUALS = os.environ.get("SKYSHADE_WEATHER_VIS", "1") != "0"
WEATHER_VIS_EVERY = 16   # redraw rain every N ticks (clouds are static, drawn once)
SCENARIO_PARK = "park"
SCENARIO_FOREST = "forest"
SCENARIO_BUILDINGS = "buildings"
SCENARIO_TRAIL = "trail"
SCENARIO_CHOICES = (SCENARIO_PARK, SCENARIO_FOREST, SCENARIO_BUILDINGS, SCENARIO_TRAIL)
SCENARIO_LABELS = {
    SCENARIO_PARK:      "Park",
    SCENARIO_FOREST:    "Forest Trail",
    SCENARIO_BUILDINGS: "Building District",
    SCENARIO_TRAIL:     "Urban Trail",
}
SCENARIO_DESCRIPTIONS = {
    SCENARIO_PARK:      "Open park loop with light obstacles and figure-8 walking.",
    SCENARIO_FOREST:    "Long wooded trail with tree avoidance and path reset.",
    SCENARIO_BUILDINGS: "City plaza with buildings, roads, vehicles, and pedestrians.",
    SCENARIO_TRAIL:     "Urban trail: bridge/underpass to fly over, crowd pedestrians as distractors.",
}

# ── Urban Trail scenario constants ────────────────────────────────────────────
TRAIL_WALK_SPEED       = 0.40    # m/s — steady walk along the trail
TRAIL_LENGTH           = 26.0    # metres end to end before looping
TRAIL_START_X          = -13.0
TRAIL_END_X            = 13.0
TRAIL_BRIDGE_CX        = 0.0     # bridge centre x
TRAIL_BRIDGE_HALF      = 2.8     # half-span of bridge along trail (x-axis)
TRAIL_BRIDGE_DECK_Z    = 3.0     # underside of bridge deck (drone 2.5 m → would hit at 3 m)
TRAIL_BRIDGE_WALL_Y    = 2.3     # y position of side walls (interior width 4.6 m)
TRAIL_FLY_OVER_ALT     = 5.5     # altitude drone climbs to when crossing bridge zone
TRAIL_FLY_OVER_X_HALF  = 4.5    # x half-range that triggers altitude boost
AVOIDANCE_RADIUS = 0.45
AVOIDANCE_GAIN = 3.0
AVOIDANCE_MAX_FORCE = 3.0
# Building-proximity alerts (Building District).  The drone keeps its avoidance
# AI; these only surface how close it gets so you can watch it handle buildings.
# clearance = distance(drone, building) − building.radius  (radius includes a
# ~0.35 m safety buffer), so clearance ≤ 0 means the drone is inside the
# footprint = a contact/collision.
NEAR_MISS_CLEARANCE = 1.2     # m — warn "NEAR BUILDING" below this gap
COLLISION_CLEARANCE = -0.15   # m — at/under this the drone is touching the wall
                              # (with avoidance on, this should rarely happen)
FOLLOW_LEAD_MAX_METERS = 1.05
GIMBAL_LOOKAHEAD_SECONDS = 0.95
GIMBAL_SEARCH_RADIUS = 0.55
USER_MARKER_HEIGHT = 1.62
POV_DISPLAY_SIZE = (960, 540)
TRAINING_DISPLAY_SIZE = (360, 270)
LAUNCHER_TRAINING_DISPLAY_SIZE = (480, 360)
# TRAINING_PIXEL_PASS imported from sub1_perception.training
TRAINING_GROUND_ITEMS = (
    ("camera",   "Sub-1 Perception", "HSV tracker", "3D calibration room — live red-marker lock check.", "#22c55e"),
    ("flight",   "Sub-2 Flight",     "PPO",          "3D hover arena — PPO wind-gust curriculum training.", "#60a5fa"),
    ("weather",  "Sub-3 Weather",    "SVM",          "3D weather scene — umbrella deploy/stow SVM decision.", "#f472b6"),
    ("safety",   "Sub-4 Nav Safety", "MDP + SAC",    "3D obstacle room — lidar navigation (SAC) + battery safety.", "#f59e0b"),
)


def _artifact_ready(path: str, min_bytes: int = 1) -> bool:
    """True only when a model artefact exists and is not an empty/stale file."""
    try:
        return os.path.exists(path) and os.path.getsize(path) >= min_bytes
    except OSError:
        return False


# ── Persistent history helpers ────────────────────────────────────────────────
_REPORTS_DIR          = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
_RUN_HISTORY_PATH     = os.path.join(_REPORTS_DIR, "run_metrics_history.json")
_TRAINING_HIST_PATH   = os.path.join(_REPORTS_DIR, "training_history.json")


def _load_json_file(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _save_json_file(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _append_run_history(entry, max_kept=20):
    history = _load_json_file(_RUN_HISTORY_PATH, [])
    if not isinstance(history, list):
        history = []
    history.append(entry)
    _save_json_file(_RUN_HISTORY_PATH, history[-max_kept:])
FOREST_TRAIL_START_X = -9.0
FOREST_TRAIL_END_X = 9.0
FOREST_TRAIL_LENGTH = FOREST_TRAIL_END_X - FOREST_TRAIL_START_X
FOREST_WALK_SPEED = 0.32   # slower walk — realistic pace through dense canopy
# City (Building District) — enlarged so the drone has room to roam toward
# buildings.  Buildings sit in a ring; the ring road and footpaths scale with it.
BUILDING_RING_MIN = 11.0
BUILDING_RING_MAX = 17.0
BUILDING_COUNT    = 18
ROAD_HALF_WIDTH = 1.1
RING_ROAD_R = 8.5
HUMAN_STAND_Z = 0.62

# City pedestrian tour — the user leaves the central plaza, walks the footpath
# right up to one building, holds, returns, then heads to the next building, so
# the follower drone actually approaches buildings (see city_walk).
CITY_INNER_COUNT   = 8      # buildings on the close, walk-up-able inner ring
CITY_VISIT_SECONDS = 24.0   # seconds per out-and-back building visit (relaxed pace)
CITY_APPROACH_GAP  = 0.9    # how close (m) to a building wall the user stops
# Fallback orbit (used only if the city layout wasn't recorded for some reason)
CITY_WALK_R_MIN = 1.2
CITY_WALK_R_MAX = 8.6
CITY_WALK_ORBIT = 0.11
CITY_WALK_BREATHE = 0.13
# Approach points (np.array([x, y])) the user walks out to, one per inner
# building, sorted by angle so the tour circles the city.  Populated by
# spawn_buildings() when the Building District is built.
_CITY_APPROACH: list = []


def _set_city_approach(points):
    global _CITY_APPROACH
    _CITY_APPROACH = list(points)


def figure8(t, scale=1.5):
    d = 1 + math.sin(t) ** 2
    return np.array([scale * math.cos(t) / d,
                     scale * math.sin(t) * math.cos(t) / d,
                     0.0])


def park_walk(t):
    """Wider figure-8 so the user actually roams the enlarged park."""
    return figure8(t * USER_WALK_SPEED, scale=3.0)


def _city_orbit_fallback(t):
    """Breathing orbit used only if the city layout wasn't recorded."""
    ang = CITY_WALK_ORBIT * t
    breathe = 0.5 - 0.5 * math.cos(CITY_WALK_BREATHE * t)
    r = CITY_WALK_R_MIN + (CITY_WALK_R_MAX - CITY_WALK_R_MIN) * breathe
    return np.array([r * math.cos(ang), r * math.sin(ang), 0.0])


def city_walk(t):
    """City tour: walk out of the plaza along the footpath right up to a
    building, hold, return to the plaza, then head to the next building.

    Instead of staying boxed in the central plaza (old figure-8), the user now
    visits the inner-ring buildings one after another, so the follower drone is
    led close to each building — where its avoidance AI steers around and the
    near-miss/collision alerts fire (see the main loop).
    """
    pts = _CITY_APPROACH
    if not pts:
        return _city_orbit_fallback(t)

    n = len(pts)
    seg    = t / CITY_VISIT_SECONDS
    k      = int(seg) % n
    frac   = seg - math.floor(seg)          # 0 → 1 within this visit
    target = pts[k]

    # Out-and-back with a brief hold next to the building.
    if frac < 0.42:
        s = frac / 0.42                     # plaza → building
    elif frac < 0.58:
        s = 1.0                             # hold at the building
    else:
        s = 1.0 - (frac - 0.58) / 0.42      # building → plaza
    s = 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, s)))   # ease in/out

    # Small per-visit plaza offset so the user doesn't always pivot on dead centre.
    home = np.array([0.5 * math.cos(2.3 * k), 0.5 * math.sin(2.3 * k)])
    pos  = home + (target - home) * s
    return np.array([pos[0], pos[1], 0.0])


def forest_walk(t):
    progress = (t * FOREST_WALK_SPEED) % FOREST_TRAIL_LENGTH
    x = FOREST_TRAIL_START_X + progress
    y = 0.55 * math.sin(progress * 0.85) + 0.22 * math.sin(progress * 1.7)
    return np.array([x, y, 0.0])


def forest_lap_index(t):
    return int((t * FOREST_WALK_SPEED) // FOREST_TRAIL_LENGTH)


def trail_walk(t):
    """Walk along the urban trail — straight with gentle y weave, loops at end."""
    progress = (t * TRAIL_WALK_SPEED) % TRAIL_LENGTH
    x = TRAIL_START_X + progress
    # Gentle sinusoidal weave so the path isn't perfectly straight
    y = 0.30 * math.sin(progress * 0.38) + 0.12 * math.sin(progress * 0.9)
    return np.array([x, y, 0.0])


def trail_lap_index(t):
    return int((t * TRAIL_WALK_SPEED) // TRAIL_LENGTH)


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
    segments = 10
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


def _draw_cloud(phys, x, y, z, scale, color, lifetime=0.0):
    # lifetime=0.0 → permanent (drawn once, no per-frame churn, no flicker/"swirl")
    puffs = [
        (-0.45, 0.04, 0.50, 0.20),
        (0.10, 0.12, 0.56, 0.22),
        (0.58, 0.00, 0.40, 0.18),
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


def draw_static_clouds(phys, mode="Cloudy"):
    """Draw the cloud bank ONCE as permanent geometry (called at sim start).

    Static clouds don't churn debug lines every frame, so they neither lag the
    sim nor flicker/"swirl" — they just sit in the sky like real clouds.
    """
    _draw_cloud_bank(phys, mode, lifetime=0.0)


def _draw_cloud_bank(phys, mode, lifetime=0.0):
    """Draw a light cloud bank around the scene (static by default).

    Fewer, larger clouds spread further out so they read as a calm overcast sky
    rather than a churning ring near the action.
    """
    if mode == "Clear":
        color = [0.80, 0.83, 0.88]   # a few wispy clouds even when clear
    elif mode == "Cloudy":
        color = [0.72, 0.77, 0.84]
    else:   # Rainy / Storm
        color = [0.46, 0.50, 0.60]
    clouds = [
        (-12.0, -11.0, 6.2, 1.7),
        ( -4.0, -12.0, 6.6, 1.9),
        (  5.0, -11.5, 6.0, 1.6),
        ( 12.0, -10.5, 6.4, 1.7),
        ( -9.0,  11.5, 6.3, 1.8),
        (  3.0,  12.0, 6.1, 1.6),
        ( 11.0,  11.0, 6.5, 1.7),
    ]
    for cloud in clouds:
        _draw_cloud(phys, *cloud, color, lifetime=lifetime)


def draw_weather_visuals(phys, weather, rng, t_wall: float = 0.0, focus_xy=None):
    """Draw rain, wind streamers, mist and clouds in the PyBullet scene.

    All visuals use addUserDebugLine with a short lifeTime so they animate
    naturally — each call refreshes them for the current frame.
    """
    rain = float(weather.get("rain", 0.0))
    wind = float(weather.get("wind", 0.0))

    # Clouds are drawn once as static geometry (see draw_static_clouds) — not
    # here — so they don't churn every frame.

    # Slowly rotating wind direction (cycles over ~42 s)
    wind_angle = t_wall * 0.15
    wd_x = math.cos(wind_angle)
    wd_y = math.sin(wind_angle) * 0.45

    # ── Rain drops ───────────────────────────────────────────────────────────
    # Small, thin streaks kept light so the sim stays smooth on integrated GPUs.
    if rain >= 0.03:
        drop_count = int(40 + rain * 110)   # 40–150 drops
        rain_r     = 13.0                   # half-width of rain area (m)
        drop_len   = 0.30 + rain * 0.40     # 0.30–0.70 m — much smaller drops
        lw         = 0.8 + rain * 1.0       # 0.8–1.8 — thin lines
        brightness = min(1.0, 0.45 + rain * 0.90)
        colour     = [0.35 * brightness, 0.72 * brightness, 1.0 * brightness]

        for _ in range(drop_count):
            x  = float(rng.uniform(-rain_r, rain_r))
            y  = float(rng.uniform(-rain_r, rain_r))
            z  = float(rng.uniform(2.3, 8.5))
            dx = wd_x * wind * 0.14
            dy = wd_y * wind * 0.14
            p.addUserDebugLine(
                [x, y, z],
                [x + dx, y + dy, z - drop_len],
                colour,
                lineWidth=lw,
                lifeTime=0.65,
                physicsClientId=phys,
            )

        # Extra local curtain so rain stays visible near the active drone/user.
        if focus_xy is not None:
            fx, fy = float(focus_xy[0]), float(focus_xy[1])
            local_count = int(15 + rain * 45)
            for _ in range(local_count):
                x = fx + float(rng.uniform(-3.0, 3.0))
                y = fy + float(rng.uniform(-3.0, 3.0))
                z = float(rng.uniform(2.0, 5.8))
                p.addUserDebugLine(
                    [x, y, z],
                    [x + wd_x * wind * 0.16, y + wd_y * wind * 0.16, z - drop_len],
                    [0.55 * brightness, 0.84 * brightness, 1.0],
                    lineWidth=lw + 0.4,
                    lifeTime=0.65,
                    physicsClientId=phys,
                )

    # ── Wind streamers ───────────────────────────────────────────────────────
    # Horizontal streaks at varying heights that show wind speed + direction
    if wind >= 1.5:
        n_streaks = int(wind * 2)           # 3–14 streaks at 1.5–7 m/s (light)
        streak_len = wind * 0.40            # longer = faster wind
        alpha      = min(0.85, wind / 8.0)
        col        = [0.75 * alpha, 0.82 * alpha, 0.92 * alpha]

        for _ in range(n_streaks):
            x  = float(rng.uniform(-10.0, 10.0))
            y  = float(rng.uniform(-10.0, 10.0))
            z  = float(rng.uniform(0.4, 4.2))
            p.addUserDebugLine(
                [x, y, z],
                [x + wd_x * streak_len, y + wd_y * streak_len, z],
                col,
                lineWidth=1.2 + wind * 0.18,
                lifeTime=0.35,
                physicsClientId=phys,
            )

    # ── Ground mist (heavy rain / storm only) ───────────────────────────────
    if rain >= 0.55:
        mist_n = int((rain - 0.55) * 60)   # 0–27 mist wisps
        for _ in range(mist_n):
            x  = float(rng.uniform(-8.0, 8.0))
            y  = float(rng.uniform(-8.0, 8.0))
            z  = float(rng.uniform(0.03, 0.35))
            ex = x + float(rng.uniform(-1.2, 1.2))
            ey = y + float(rng.uniform(-1.2, 1.2))
            p.addUserDebugLine(
                [x,  y,  z],
                [ex, ey, z + float(rng.uniform(0.1, 0.45))],
                [0.60, 0.64, 0.72],
                lineWidth=3.5,
                lifeTime=0.48,
                physicsClientId=phys,
            )


def _draw_weather_cv(frame, weather, rng, t_wall):
    """Stamp rain and wind streaks onto an OpenCV (NumPy) frame.

    All drawing is pure NumPy / cv2 — no PyBullet calls, no GPU overhead.
    Returns the frame modified in-place.
    """
    rain = float(weather.get("rain", 0.0))
    wind = float(weather.get("wind", 0.0))
    if rain < 0.05 and wind < 1.5:
        return frame

    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Wind direction rotates slowly, matching the 3D scene
    wind_angle = t_wall * 0.15
    lean_x = math.cos(wind_angle) * wind * 0.14
    lean_y = math.sin(wind_angle) * wind * 0.06

    # Rain drops
    if rain >= 0.05:
        n_drops  = int(12 + rain * 48)              # 12 – 60 drops
        drop_len = int(h * (0.04 + rain * 0.09))    # 2 – 11 % of frame height
        lw       = max(1, int(1 + rain * 1.5))
        bright   = min(255, int(130 + rain * 125))
        color    = (bright, int(bright * 0.75), int(bright * 0.45))  # blue-white in RGB

        for _ in range(n_drops):
            x0 = int(rng.integers(0, w))
            y0 = int(rng.integers(0, h))
            x1 = int(x0 + lean_x * drop_len)
            y1 = int(y0 + drop_len)
            cv2.line(overlay, (x0, y0),
                     (max(0, min(w - 1, x1)), max(0, min(h - 1, y1))),
                     color, lw)

    # Wind streaks (horizontal motion blur wisps)
    if wind >= 2.0:
        n_streaks  = int(wind * 1.5)
        streak_len = int(w * wind * 0.012)
        for _ in range(n_streaks):
            x0 = int(rng.integers(0, max(1, w - streak_len)))
            y0 = int(rng.integers(0, h))
            cv2.line(overlay, (x0, y0), (x0 + streak_len, y0), (200, 210, 225), 1)

    # Blend: rain/wind marks at partial transparency over original
    alpha = min(0.60, 0.20 + rain * 0.40)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    return frame


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


def spawn_buildings(phys, rng, n=BUILDING_COUNT):
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
    approach = []   # (angle, point) for the close inner-ring buildings
    n_inner = min(CITY_INNER_COUNT, n)
    for k in range(n):
        if k < n_inner:
            # Evenly-spaced inner ring the drone can be walked up to.
            angle = 2 * math.pi * k / n_inner + float(rng.uniform(-0.08, 0.08))
            radius = BUILDING_RING_MIN + float(rng.uniform(0.0, 1.5))
        else:
            # Outer scatter — city depth / backdrop, not visited.
            m = n - n_inner
            angle = (2 * math.pi * (k - n_inner) / m + math.pi / m
                     + float(rng.uniform(-0.12, 0.12)))
            radius = float(rng.uniform(BUILDING_RING_MIN + 2.5, BUILDING_RING_MAX))
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        width = float(rng.uniform(0.55, 1.05))
        depth = float(rng.uniform(0.55, 1.05))
        height = float(rng.uniform(2.5, 8.0))
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

        if k < n_inner:
            # Stop point on the footpath ~CITY_APPROACH_GAP m from the wall.
            d_appr = max(2.0, radius - max(width, depth) - CITY_APPROACH_GAP)
            approach.append((angle,
                             np.array([d_appr * math.cos(angle),
                                       d_appr * math.sin(angle)], dtype=float)))

    approach.sort(key=lambda item: item[0])
    _set_city_approach([pt for _angle, pt in approach])
    return obstacles


def spawn_city_props(phys, rng):
    """Road network, parks, vehicles, and pedestrians around the city plaza."""
    radius = RING_ROAD_R
    road = (0.13, 0.13, 0.15)
    sidewalk = (0.62, 0.63, 0.66)
    grass = (0.18, 0.42, 0.20)
    plaza_radius = 2.3

    grass_r = (RING_ROAD_R + BUILDING_RING_MIN) / 2.0   # between ring road & buildings
    for quadrant in range(4):
        angle = math.pi / 4 + quadrant * math.pi / 2
        gx, gy = grass_r * math.cos(angle), grass_r * math.sin(angle)
        _flat(phys, [2.7, 2.7, 0.006], grass, gx, gy, 0.006, angle)
        for _ in range(5):
            tx = gx + float(rng.uniform(-2.0, 2.0))
            ty = gy + float(rng.uniform(-2.0, 2.0))
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
        ped_radius = 5.2 + float(rng.uniform(-0.4, 1.6))
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
    # Enlarged so the user (park_walk) has a big lawn to roam across.
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.28, 0.43, 0.25, 1], physicsClientId=phys)
    _static_box(phys, [8.0, 0.10, 0.01], [0, 0, 0.012], [0.45, 0.42, 0.36, 1])
    _static_box(phys, [0.10, 8.0, 0.01], [0, 0, 0.014], [0.45, 0.42, 0.36, 1])

    for x, y, sx, sy in [
        (-5.6, 4.9, 0.55, 0.40),
        (5.4, -5.0, 0.70, 0.46),
        (-4.9, -5.4, 0.48, 0.50),
        (5.8, 4.7, 0.56, 0.44),
        (0.0, 6.4, 0.80, 0.36),
        (-6.6, 0.0, 0.36, 0.80),
    ]:
        _static_box(phys, [sx, sy, 0.025], [x, y, 0.035], [0.23, 0.50, 0.21, 1])

    for x, y in [(-6.2, -2.4), (-5.0, 3.6), (4.8, 3.1), (6.1, -1.4),
                 (-2.6, 6.0), (3.0, -6.2), (6.8, 1.8), (-6.9, 1.2)]:
        obstacles.append(_tree(phys, x, y, trunk_radius=0.09, trunk_height=1.1, crown_radius=0.46))

    for x, y, yaw in [(-2.9, 5.5, 0.0), (3.8, -5.2, math.pi / 2),
                      (5.2, 2.0, 0.0), (-5.4, -1.0, math.pi / 2)]:
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
            lateral_gap = rng.uniform(1.4, 2.2)   # wider gap — more sky visible above path
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
    _flat(phys, [2.6, 2.6, 0.008], (0.29, 0.32, 0.34), 0, 0, 0.018, 0.0)
    _flat(phys, [2.10, 0.06, 0.006], (0.86, 0.86, 0.80), 0, 0, 0.03, 0.0)
    _flat(phys, [0.06, 2.10, 0.006], (0.86, 0.86, 0.80), 0, 0, 0.031, 0.0)

    rng_buildings = np.random.default_rng(7)
    rng_props = np.random.default_rng(13)
    obstacles = spawn_buildings(phys, rng_buildings, n=BUILDING_COUNT)
    spawn_city_props(phys, rng_props)
    return obstacles


def build_trail_environment(phys):
    """Urban trail with a bridge/underpass, crowd pedestrians, trees, and street furniture.

    Layout (x-axis = direction of travel, -13 → +13 m):
      • Paved trail surface
      • Bridge underpass centred at x=0  (drone must climb to TRAIL_FLY_OVER_ALT to clear)
      • 8 crowd pedestrians (non-red) — tracker distractors & drone obstacles
      • Tree rows along both sides (gap where bridge is)
      • Lamp posts, bollards, benches
    """
    plane_id = p.loadURDF("plane.urdf", physicsClientId=phys)
    p.changeVisualShape(plane_id, -1, rgbaColor=[0.46, 0.48, 0.44, 1], physicsClientId=phys)

    obstacles = []
    rng = np.random.default_rng(77)

    # ── Trail surface (paved slabs along path) ────────────────────────────────
    for seg_x in np.linspace(TRAIL_START_X, TRAIL_END_X, 52):
        trail_y = 0.30 * math.sin((seg_x - TRAIL_START_X) * 0.38)
        _static_box(phys, [0.52, 1.1, 0.012],
                    [float(seg_x), float(trail_y), 0.016],
                    [0.54, 0.52, 0.48, 1])

    # ── Bridge / underpass ────────────────────────────────────────────────────
    bx   = TRAIL_BRIDGE_CX
    bh   = TRAIL_BRIDGE_HALF
    dz   = TRAIL_BRIDGE_DECK_Z
    wy   = TRAIL_BRIDGE_WALL_Y
    wall_c = [0.38, 0.35, 0.31, 1]
    deck_c = [0.32, 0.30, 0.28, 1]
    rail_c = [0.55, 0.53, 0.50, 1]

    # Left and right side walls (solid stone/concrete)
    for side in (-1, 1):
        _static_box(phys, [bh, 0.38, dz / 2],
                    [bx, side * wy, dz / 2], wall_c)
        # Decorative vertical grooves
        for gx in (-bh * 0.6, 0.0, bh * 0.6):
            _static_box(phys, [0.06, 0.42, dz / 2 - 0.1],
                        [bx + gx, side * wy, dz / 2], [0.28, 0.26, 0.24, 1])

    # Bridge deck (what the drone would hit at normal altitude)
    col = p.createCollisionShape(p.GEOM_BOX,
                                 halfExtents=[bh, wy + 0.38, 0.30],
                                 physicsClientId=phys)
    vis = p.createVisualShape(p.GEOM_BOX,
                              halfExtents=[bh, wy + 0.38, 0.30],
                              rgbaColor=deck_c, physicsClientId=phys)
    p.createMultiBody(0, col, vis, [bx, 0.0, dz + 0.30], physicsClientId=phys)

    # Bridge parapet / handrail
    for side in (-1, 1):
        _static_box(phys, [bh, 0.08, 0.55],
                    [bx, side * (wy + 0.38 + 0.08), dz + 0.60 + 0.27], rail_c)
    # Arch face plates (visual front/back edges)
    for ex in (-bh, bh):
        _static_box(phys, [0.22, wy, 0.24],
                    [bx + ex, 0.0, dz + 0.24], deck_c)

    # Bridge collision as lateral obstacle (sides keep drone from hitting walls)
    for side in (-1, 1):
        obstacles.append({
            "position": np.array([bx, side * wy], dtype=float),
            "radius": 0.55,
        })

    # ── Trees lining both sides ───────────────────────────────────────────────
    for x in np.linspace(TRAIL_START_X + 0.5, TRAIL_END_X - 0.5, 24):
        if abs(x - bx) < bh + 1.5:
            continue   # clear zone around bridge
        trail_y = 0.30 * math.sin((x - TRAIL_START_X) * 0.38)
        for side in (-1, 1):
            ty = trail_y + side * (2.0 + float(rng.uniform(0.2, 0.9)))
            r = float(rng.uniform(0.07, 0.12))
            obs = _tree(phys, float(x), float(ty),
                        trunk_radius=r,
                        trunk_height=float(rng.uniform(1.5, 2.4)),
                        crown_radius=float(rng.uniform(0.35, 0.65)))
            obstacles.append(obs)

    # ── Lamp posts ────────────────────────────────────────────────────────────
    for lx in np.linspace(TRAIL_START_X + 2.0, TRAIL_END_X - 2.0, 7):
        if abs(lx - bx) < bh + 0.5:
            continue
        trail_y = 0.30 * math.sin((lx - TRAIL_START_X) * 0.38)
        for side in (-1, 1):
            px, py = float(lx), float(trail_y) + side * 1.6
            _static_cylinder(phys, 0.045, 3.8, [px, py, 1.9], [0.20, 0.20, 0.22, 1])
            # Lamp head (small box on top)
            _static_box(phys, [0.12, 0.06, 0.06], [px, py, 3.85], [1.0, 0.95, 0.65, 1])
        obstacles.append({"position": np.array([float(lx), float(trail_y)]), "radius": 0.18})

    # ── Street bollards (near bridge entry) ──────────────────────────────────
    for boll_x in [bx - bh - 0.4, bx + bh + 0.4]:
        for boll_y in [-0.8, 0.0, 0.8]:
            trail_y = 0.30 * math.sin((boll_x - TRAIL_START_X) * 0.38)
            _static_cylinder(phys, 0.08, 0.9,
                             [boll_x, trail_y + boll_y, 0.45], [0.22, 0.22, 0.72, 1])

    # ── Benches along the trail ───────────────────────────────────────────────
    for bx_b, by_b in [(-8.0, -1.5), (-4.0, 1.6), (4.5, -1.4), (9.0, 1.5)]:
        trail_y = 0.30 * math.sin((bx_b - TRAIL_START_X) * 0.38)
        bench = _static_box(phys, [0.55, 0.12, 0.08],
                            [bx_b, trail_y + by_b, 0.42], [0.43, 0.24, 0.12, 1])
        p.resetBasePositionAndOrientation(
            bench, [bx_b, trail_y + by_b, 0.42], [0, 0, 0, 1], physicsClientId=phys)

    return obstacles


def build_trail_crowd(phys):
    """Return list of 8 non-red pedestrians spread along the urban trail.

    They act as both collision obstacles (drone avoids them) and Sub-1 tracker
    distractors (non-red shirts — tracker stays locked to the user's red cap).
    Returns obstacle dicts for the avoidance system.
    """
    crowd_obstacles = []
    crowd_specs = [
        # (x, y_offset_from_trail, shirt_colour)
        (-9.5,  1.2, (0.18, 0.32, 0.72)),   # blue jacket
        (-6.0, -1.1, (0.22, 0.55, 0.30)),   # green top
        (-2.0,  2.0, (0.52, 0.50, 0.55)),   # grey
        ( 1.5, -1.8, (0.62, 0.50, 0.20)),   # tan/beige
        ( 4.0,  1.3, (0.22, 0.45, 0.58)),   # teal
        ( 6.5, -1.0, (0.38, 0.28, 0.55)),   # purple
        ( 8.5,  0.6, (0.25, 0.58, 0.42)),   # seafoam
        (11.0, -1.4, (0.60, 0.40, 0.25)),   # brown
    ]
    pants_opts = [(0.10, 0.12, 0.18), (0.22, 0.16, 0.10), (0.08, 0.08, 0.10)]

    for i, (cx, cy_off, shirt) in enumerate(crowd_specs):
        trail_y = 0.30 * math.sin((cx - TRAIL_START_X) * 0.38)
        world_pos = [cx, trail_y + cy_off, HUMAN_STAND_Z]
        pants = pants_opts[i % len(pants_opts)]
        make_city_pedestrian(phys, world_pos, shirt=shirt, pants=pants)
        crowd_obstacles.append({
            "position": np.array([cx, trail_y + cy_off], dtype=float),
            "radius": 0.38,
        })

    return crowd_obstacles


def build_environment(phys, scenario):
    if scenario == SCENARIO_FOREST:
        return build_forest_environment(phys)
    if scenario == SCENARIO_BUILDINGS:
        return build_building_environment(phys)
    if scenario == SCENARIO_TRAIL:
        obs = build_trail_environment(phys)
        obs += build_trail_crowd(phys)
        return obs
    return build_park_environment(phys)


def scenario_user_position(scenario, t_wall):
    if scenario == SCENARIO_FOREST:
        return forest_walk(t_wall)
    if scenario == SCENARIO_TRAIL:
        return trail_walk(t_wall)
    if scenario == SCENARIO_BUILDINGS:
        return city_walk(t_wall)
    if scenario == SCENARIO_PARK:
        return park_walk(t_wall)
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


# ── Drone umbrella (the shade canopy) ─────────────────────────────────────────
# A proper umbrella that visibly OPENS when the Sub-3 SVM says DEPLOY and folds
# CLOSED when it says STOW.  Visual-only (no collision) so it never snags on a
# building.  Two part groups share one pole:
#   • "open"   — wide canopy disc + domed cap + 8 ribs + finial (rain)
#   • "closed" — a slim folded wrap along the pole (dry)
#   • "always" — the pole itself
_UMB_CANOPY    = [0.20, 0.62, 0.78, 1.0]   # teal canopy
_UMB_RIB       = [0.13, 0.45, 0.58, 1.0]   # darker ribs / folded wrap
_UMB_POLE      = [0.20, 0.20, 0.22, 1.0]
_UMB_FINIAL    = [0.90, 0.90, 0.93, 1.0]


def make_umbrella(phys):
    """Build the openable umbrella rig.  Returns a dict with a flat part list."""
    parts = []   # each: {"id", "offset", "yaw", "group", "rgba"}

    def _add(body_id, offset, group, rgba, yaw=0.0):
        parts.append({"id": body_id, "offset": np.array(offset, dtype=float),
                      "yaw": yaw, "group": group, "rgba": list(rgba)})

    # The whole umbrella only appears when DEPLOYed (rain).  When stowed it is
    # fully hidden so there is no stray "block"/stick left on the drone.

    # Short pole the canopy sits on
    _add(_visual_body(phys, p.GEOM_CYLINDER, _UMB_POLE, [0, 0, 0],
                      radius=0.022, length=0.34),
         [0.0, 0.0, 0.30], "open", _UMB_POLE)

    # Open canopy: wide disc + smaller domed cap + finial
    _add(_visual_body(phys, p.GEOM_CYLINDER, _UMB_CANOPY, [0, 0, 0],
                      radius=0.64, length=0.05),
         [0.0, 0.0, 0.44], "open", _UMB_CANOPY)
    _add(_visual_body(phys, p.GEOM_CYLINDER, _UMB_CANOPY, [0, 0, 0],
                      radius=0.34, length=0.10),
         [0.0, 0.0, 0.50], "open", _UMB_CANOPY)
    _add(_visual_body(phys, p.GEOM_SPHERE, _UMB_FINIAL, [0, 0, 0], radius=0.035),
         [0.0, 0.0, 0.58], "open", _UMB_FINIAL)

    # Open ribs: 8 thin boxes radiating to the rim
    for k in range(8):
        ang = 2 * math.pi * k / 8
        _add(_visual_body(phys, p.GEOM_BOX, _UMB_RIB, [0, 0, 0],
                          halfExtents=[0.34, 0.013, 0.013]),
             [0.30 * math.cos(ang), 0.30 * math.sin(ang), 0.435], "open",
             _UMB_RIB, yaw=ang)

    return {"parts": parts}


def update_umbrella(phys, umb, drone_pos):
    """Keep every umbrella part anchored above the drone."""
    base = np.array(drone_pos, dtype=float)
    for prt in umb["parts"]:
        pos  = (base + prt["offset"]).tolist()
        quat = (p.getQuaternionFromEuler([0, 0, prt["yaw"]])
                if prt["yaw"] else [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(prt["id"], pos, quat,
                                          physicsClientId=phys)


def set_umbrella_open(phys, umb, deployed: bool):
    """Show the open umbrella only when deployed; hide it completely otherwise."""
    for prt in umb["parts"]:
        visible = (prt["group"] == "always") or (prt["group"] == "open" and deployed)
        rgba = list(prt["rgba"])
        rgba[3] = rgba[3] if visible else 0.0
        p.changeVisualShape(prt["id"], -1, rgbaColor=rgba, physicsClientId=phys)


def create_person(phys):
    # The red marker on the cap is intentionally prominent: the perception
    # subsystem tracks it exactly like it tracked the old red sphere.
    specs = [
        ("torso", p.GEOM_BOX, [0.12, 0.07, 0.32], None, [0.12, 0.32, 0.82, 1], [0, 0, 1.02]),
        ("head", p.GEOM_SPHERE, None, 0.13, [0.86, 0.66, 0.50, 1], [0, 0, 1.43]),
        ("marker", p.GEOM_CYLINDER, None, 0.30, [1.0, 0.04, 0.02, 1], [0, 0, 1.68]),
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


# camera_training_frame / camera_training_marker_box / camera_training_step
# are imported from sub1_perception.training


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


# _FlightTrainingWorker imported from sub2_flight.training_worker as alias


class FlightTrainingWindow:
    """Live PPO training window for Sub-2 Flight.

    Shows a scrolling reward curve (colour-coded by curriculum stage),
    real-time stats, and Start / Stop controls.  Training runs in a
    background thread so the UI stays responsive throughout.
    """

    _STEPS_DEFAULT = 1_500_000
    _PLOT_W = 510
    _PLOT_H = 230
    _MAX_PLOT_POINTS = 400

    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Training Ground — Sub-2 Flight (PPO)")
        self.window.configure(bg="#07111f")
        self.window.geometry("960x560+140+140")
        self.window.minsize(840, 480)
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.closed = False
        self._thread = None
        self._stop_event = threading.Event()
        self._queue = queue.Queue()

        self._rewards = []      # (timestep, reward, stage)
        self._stage = 1
        self._current_step = 0
        self._total_steps = self._STEPS_DEFAULT
        self._training_active = False

        self._canvas = None
        self._status_var = None
        self._stat_vars = {}
        self._steps_var = None
        self._start_btn = None
        self._stop_btn = None

        self._build_ui()
        self._tick()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        container = tk.Frame(self.window, bg="#07111f", padx=16, pady=14)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_columnconfigure(0, weight=0)
        container.grid_columnconfigure(1, weight=1)
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)

        # Left — reward curve
        plot_frame = tk.Frame(
            container, bg="#020617",
            highlightthickness=1, highlightbackground="#1f3a5f",
            width=self._PLOT_W + 4,
            height=self._PLOT_H + 64,
        )
        plot_frame.grid(row=0, column=0, sticky="nw", padx=(0, 16))
        plot_frame.grid_propagate(False)

        tk.Label(
            plot_frame, text="Mean Episode Reward",
            fg="#60a5fa", bg="#020617",
            font=("Arial", 10, "bold"), anchor="w",
        ).pack(side="top", fill="x", padx=8, pady=(6, 0))

        self._canvas = tk.Canvas(
            plot_frame, bg="#020617", highlightthickness=0,
            width=self._PLOT_W, height=self._PLOT_H,
        )
        self._canvas.pack(side="top", fill="both", expand=True, padx=4, pady=4)

        tk.Label(
            plot_frame,
            text="Blue = Stage 1  •  Pink = Stage 2  •  Purple = Stage 3",
            fg="#475569", bg="#020617",
            font=("Arial", 9),
        ).pack(side="bottom", pady=(0, 6))

        # Right — stats + controls
        info = tk.Frame(container, bg="#07111f")
        info.grid(row=0, column=1, sticky="nsew")
        info.grid_columnconfigure(0, weight=1)

        tk.Label(
            info, text="Sub-2 Flight — PPO",
            fg="#60a5fa", bg="#07111f",
            font=("Arial", 12, "bold"), anchor="w",
        ).grid(row=0, column=0, sticky="ew")

        self._status_var = tk.StringVar(value="Ready to train")
        tk.Label(
            info, textvariable=self._status_var,
            fg="#f8fafc", bg="#0f172a",
            font=("Arial", 14, "bold"),
            anchor="w", padx=12, pady=10,
            wraplength=300, justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(8, 0))

        stats_frame = tk.Frame(info, bg="#111c2e", padx=10, pady=8)
        stats_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        stats_frame.grid_columnconfigure(1, weight=1)

        for i, (key, default) in enumerate([
            ("Timestep",     "0 / 1 500 000"),
            ("Stage",        "1 — stationary, calm"),
            ("Mean reward",  "—"),
            ("Model",        "not yet saved"),
        ]):
            tk.Label(
                stats_frame, text=key + ":",
                fg="#64748b", bg="#111c2e",
                font=("Arial", 9, "bold"), anchor="w",
            ).grid(row=i, column=0, sticky="w", pady=2)
            v = tk.StringVar(value=default)
            self._stat_vars[key] = v
            tk.Label(
                stats_frame, textvariable=v,
                fg="#cbd5e1", bg="#111c2e",
                font=("Arial", 9), anchor="w",
            ).grid(row=i, column=1, sticky="w", padx=(8, 0), pady=2)

        ep_row = tk.Frame(info, bg="#07111f")
        ep_row.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        ep_row.grid_columnconfigure(1, weight=1)

        tk.Label(
            ep_row, text="Timesteps",
            fg="#94a3b8", bg="#07111f",
            font=("Arial", 10, "bold"), anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=(0, 10))

        self._steps_var = tk.StringVar(value=str(self._STEPS_DEFAULT))
        self._ep_entry = tk.Entry(
            ep_row, textvariable=self._steps_var,
            bg="#020617", fg="#f8fafc",
            insertbackground="#f8fafc",
            relief="flat", font=("Arial", 12, "bold"),
            width=9, justify="center",
        )
        self._ep_entry.grid(row=0, column=1, sticky="w")

        btn_frame = tk.Frame(info, bg="#07111f")
        btn_frame.grid(row=4, column=0, sticky="ew", pady=(14, 0))
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)

        self._start_btn = tk.Button(
            btn_frame, text="Start Training",
            command=self._start_training,
            bg="#2563eb", fg="#eff6ff",
            activebackground="#1d4ed8", activeforeground="#ffffff",
            relief="flat", font=("Arial", 11, "bold"),
            padx=10, pady=9,
        )
        self._start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self._stop_btn = tk.Button(
            btn_frame, text="Stop",
            command=self._stop_training,
            bg="#334155", fg="#94a3b8",
            activebackground="#475569", activeforeground="#ffffff",
            relief="flat", font=("Arial", 11, "bold"),
            padx=10, pady=9, state="disabled",
        )
        self._stop_btn.grid(row=0, column=1, sticky="ew")

        tk.Button(
            info, text="Close",
            command=self.close,
            bg="#1e293b", fg="#cbd5e1",
            activebackground="#334155", activeforeground="#ffffff",
            relief="flat", font=("Arial", 11, "bold"),
            padx=12, pady=9,
        ).grid(row=5, column=0, sticky="ew", pady=(12, 0))

    # ── Training control ──────────────────────────────────────────────────────

    def _start_training(self):
        if self._training_active:
            return
        try:
            total_steps = max(2048, int(self._steps_var.get()))
        except ValueError:
            total_steps = self._STEPS_DEFAULT
        self._total_steps = total_steps
        self._rewards.clear()
        self._current_step = 0

        self._stop_event.clear()
        self._thread = _FlightTrainingWorker(total_steps, self._queue, self._stop_event)
        self._thread.start()

        self._training_active = True
        self._ep_entry.configure(state="disabled")
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal", bg="#dc2626", fg="#ffffff",
                                 activebackground="#b91c1c")
        self._status_var.set("Training…")

    def _stop_training(self):
        if not self._training_active:
            return
        self._stop_event.set()
        self._status_var.set("Stopping…")
        self._stop_btn.configure(state="disabled")

    # ── UI update loop ────────────────────────────────────────────────────────

    def _tick(self):
        if self.closed:
            return

        changed = False
        while True:
            try:
                msg = self._queue.get_nowait()
            except queue.Empty:
                break

            kind = msg[0]
            if kind == "progress":
                _, timestep, stage, mean_reward = msg
                self._rewards.append((timestep, mean_reward, stage))
                self._stage = stage
                self._current_step = timestep
                changed = True

            elif kind in ("done", "stopped"):
                _, _output_path, steps_done = msg
                self._training_active = False
                self._ep_entry.configure(state="normal")
                self._start_btn.configure(state="normal")
                self._stop_btn.configure(state="disabled", bg="#334155",
                                         fg="#94a3b8", activebackground="#475569")
                verb = "Done" if kind == "done" else "Stopped"
                self._status_var.set(f"{verb} — {steps_done:,} steps trained")
                changed = True

            elif kind == "error":
                _, err = msg
                self._training_active = False
                self._ep_entry.configure(state="normal")
                self._start_btn.configure(state="normal")
                self._stop_btn.configure(state="disabled", bg="#334155",
                                         fg="#94a3b8", activebackground="#475569")
                self._status_var.set(f"Error: {err[:80]}")
                changed = True

        if changed or self._training_active:
            self._update_stats()
            self._draw_plot()

        self.window.after(160, self._tick)

    def _update_stats(self):
        self._stat_vars["Timestep"].set(
            f"{self._current_step:,} / {self._total_steps:,}"
        )
        stage_labels = {
            1: "1 — stationary, calm",
            2: "2 — stationary, wind",
            3: "3 — walking, wind",
        }
        self._stat_vars["Stage"].set(stage_labels.get(self._stage, str(self._stage)))

        if self._rewards:
            recent = [r for _, r, _ in self._rewards[-20:]]
            self._stat_vars["Mean reward"].set(f"{np.mean(recent):+.1f}")

        model_path = _FlightTrainingWorker.OUTPUT_PATH + ".zip"
        if os.path.exists(model_path):
            self._stat_vars["Model"].set("ppo_flight_v1.zip  (saved)")
        else:
            self._stat_vars["Model"].set("not yet saved")

    def _draw_plot(self):
        canvas = self._canvas
        w = int(canvas.winfo_width()) or self._PLOT_W
        h = int(canvas.winfo_height()) or self._PLOT_H
        canvas.delete("all")

        if not self._rewards:
            canvas.create_text(
                w // 2, h // 2,
                text="No training data yet — press Start Training",
                fill="#334155", font=("Arial", 11),
            )
            return

        # Downsample for rendering
        step = max(1, len(self._rewards) // self._MAX_PLOT_POINTS)
        pts = self._rewards[::step]
        vals = [r for _, r, _ in pts]
        stages = [s for _, _, s in pts]

        min_r = min(vals)
        max_r = max(vals)
        if max_r - min_r < 1:
            max_r = min_r + 1

        pad_x, pad_y = 38, 18
        plot_w = w - 2 * pad_x
        plot_h = h - 2 * pad_y

        # Grid
        for frac in (0.25, 0.5, 0.75):
            y = pad_y + int((1 - frac) * plot_h)
            canvas.create_line(pad_x, y, w - pad_x, y,
                               fill="#1e3a5f", width=1, dash=(3, 4))
            val = min_r + frac * (max_r - min_r)
            canvas.create_text(pad_x - 4, y, text=f"{val:+.0f}",
                               fill="#475569", font=("Arial", 8), anchor="e")

        # Axis labels
        canvas.create_text(pad_x - 4, pad_y, text=f"{max_r:+.0f}",
                           fill="#475569", font=("Arial", 8), anchor="e")
        canvas.create_text(pad_x - 4, h - pad_y, text=f"{min_r:+.0f}",
                           fill="#475569", font=("Arial", 8), anchor="e")
        canvas.create_text(w - pad_x, h - pad_y + 2,
                           text=f"step {self._current_step:,}",
                           fill="#475569", font=("Arial", 8), anchor="se")

        stage_fg = {1: "#3b82f6", 2: "#ec4899", 3: "#8b5cf6"}

        n = len(pts)
        if n < 2:
            return

        for i in range(1, n):
            x0 = pad_x + int((i - 1) / (n - 1) * plot_w)
            x1 = pad_x + int(i / (n - 1) * plot_w)
            y0 = pad_y + int((1 - (vals[i - 1] - min_r) / (max_r - min_r)) * plot_h)
            y1 = pad_y + int((1 - (vals[i] - min_r) / (max_r - min_r)) * plot_h)
            color = stage_fg.get(stages[i], "#60a5fa")
            canvas.create_line(x0, y0, x1, y1, fill=color, width=1)

        # Current position marker
        last_x = pad_x + plot_w
        last_y = pad_y + int((1 - (vals[-1] - min_r) / (max_r - min_r)) * plot_h)
        canvas.create_oval(last_x - 4, last_y - 4, last_x + 4, last_y + 4,
                           fill=stage_fg.get(stages[-1], "#60a5fa"), outline="")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self):
        self._stop_event.set()
        self.closed = True
        try:
            self.window.destroy()
        except tk.TclError:
            pass


# _MDPSolverWorker imported from sub4_nav.training_worker as alias


# ── Matplotlib dark-theme helper (shared by TrainingGroundsHub) ───────────────
_MPL_BG      = "#020617"
_MPL_FIG_BG  = "#07111f"
_MPL_EDGE    = "#1f3a5f"
_MPL_TICK    = "#94a3b8"   # slate-400 — bright enough to read on dark bg
_MPL_LABEL   = "#cbd5e1"   # slate-300 — axis label colour
_MPL_TITLE   = "#7dd3fc"   # sky-300  — chart title colour
_MPL_GRID    = "#1e3a5f"


def _mpl_dark_axes(ax, title="", xlabel="", ylabel=""):
    """Apply the SkyShade dark colour theme to a matplotlib Axes."""
    ax.set_facecolor(_MPL_BG)
    for sp in ax.spines.values():
        sp.set_color(_MPL_EDGE)
    ax.tick_params(colors=_MPL_TICK, labelsize=12)
    ax.xaxis.label.set_color(_MPL_LABEL)
    ax.yaxis.label.set_color(_MPL_LABEL)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, color=_MPL_TITLE, fontsize=13, pad=7)
    ax.grid(True, color=_MPL_GRID, linestyle="--", alpha=0.4)


class TrainingGroundsHub:
    """Unified training-grounds window with live matplotlib graphs.

    Tabs
    ────
    Sub-1 Perception : HSV tracker calibration — rolling confidence + pixel-error chart
    Sub-2 Flight PPO : PPO reward curve coloured by curriculum stage
    Sub-4 Nav Safety : MDP convergence curve + learned policy heatmap

    Training-only window — no sim launch here.
    Each tab has its own Start / Stop controls.
    The footer shows which models are trained.
    Launch the main sim from the SkyShade Launcher window.
    """

    _PPO_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", "ppo_flight_v1.zip"
    )
    _SVM_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", "svm_v1.pkl"
    )
    _MDP_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", "policy_table_v1.npy"
    )
    _NAV_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", "ppo_nav_v1.zip"
    )

    def __init__(self, parent, launch_callback=None):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.window = tk.Toplevel(parent)
        self.window.title("SkyShade — Training Grounds")
        self.window.configure(bg="#07111f")
        self.window.geometry("1160x760+60+60")
        self.window.minsize(900, 620)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.closed      = False
        self._launch_cb  = launch_callback
        self._nb         = None   # ttk.Notebook — set in _build_ui

        # ── Sub-1 (PyBullet perception room + HSV tracker, main thread) ─────────
        self._sub1_perc_env  = None   # PerceptionTrainingEnv, lazy-init
        self._sub1_tracker   = Tracker(DistanceEstimator())
        self._sub1_running   = False
        self._sub1_history   = []     # [(frame_idx, confidence, pixel_error)]
        self._sub1_frame_idx = 0
        self._sub1_start_t   = 0.0
        self._sub1_ready     = True   # tracker always available
        self._sub1_img_ref   = None   # matplotlib imshow AxesImage (reused)

        # ── Sub-3 (SVM umbrella classifier) ──────────────────────────────────
        self._sub3_queue    = queue.Queue()
        self._sub3_stop     = threading.Event()
        self._sub3_thread   = None
        self._sub3_training = False
        self._sub3_ready    = _artifact_ready(self._SVM_PATH, 256)
        self._sub3_accuracy = None
        self._sub3_cm       = None
        # If model already exists, compute CM from training data so it shows immediately
        if self._sub3_ready:
            try:
                self._sub3_accuracy, self._sub3_cm = self._eval_sub3_on_disk()
            except Exception:
                pass
        self._sub3_status_var = None
        self._sub3_summary_var = None
        self._sub3_fig = self._sub3_canvas = None
        self._sub3_ax_3d = self._sub3_ax_chart = self._sub3_ax_matrix = None
        self._sub3_train_btn = None
        # Shared classifier for live demo (loaded lazily)
        self._sub3_clf      = None
        self._sub3_umbrella = "STOW"    # current SVM prediction

        # ── Sub-2 (PPO flight training) ───────────────────────────────────────
        self._sub2_queue    = queue.Queue()
        self._sub2_stop     = threading.Event()
        self._sub2_thread   = None
        self._sub2_history  = []     # [(timestep, mean_reward, stage)]
        self._sub2_step           = 0
        self._sub2_run_start_step = None   # cumulative step count at run start (for % calc)
        self._sub2_total          = 1_500_000
        self._sub2_training       = False
        self._sub2_ready          = _artifact_ready(self._PPO_PATH, 1024)
        self._sub2_start_time     = None
        # Load persisted runs from disk so history survives across restarts
        _th = _load_json_file(_TRAINING_HIST_PATH, {})
        self._sub2_runs = [list(map(tuple, r)) for r in _th.get("sub2_runs", [])]

        # ── Sub-4 MDP (battery safety, value iteration) ───────────────────────
        self._sub4_queue    = queue.Queue()
        self._sub4_stop     = threading.Event()
        self._sub4_thread   = None
        self._sub4_deltas   = []
        self._sub4_policy   = None
        self._sub4_iters    = 0
        self._sub4_training = False
        self._sub4_ready    = _artifact_ready(self._MDP_PATH)
        if self._sub4_ready:
            try:
                self._sub4_policy = np.load(self._MDP_PATH)
            except Exception:
                pass

        # ── Sub-4 Nav (obstacle avoidance, PPO) ──────────────────────────────
        self._nav_queue    = queue.Queue()
        self._nav_stop     = threading.Event()
        self._nav_thread   = None
        self._nav_history  = [tuple(p) for p in _th.get("nav_history_last", [])]
        self._nav_step     = 0
        self._nav_total    = 500_000
        self._nav_training = False
        self._nav_ready    = _artifact_ready(self._NAV_PATH, 1024)
        self._nav_viz      = None   # latest viz state dict from worker

        # ── Matplotlib figures & axes (populated in _build_ui) ──────────────
        self._sub1_fig = self._sub1_canvas = None
        self._sub1_ax_cam = self._sub1_ax_chart = self._sub1_ax_err = None
        self._sub2_fig = self._sub2_canvas = None
        self._sub2_ax_3d = self._sub2_ax_reward = None   # 3D arena | reward
        self._sub4_fig = self._sub4_canvas = None
        self._sub4_ax_3d = self._sub4_ax_reward = None   # 3D nav room | reward
        # Auto-rotation azimuth angles (degrees) — incremented each tick
        self._sub2_azim = -55.0
        self._sub4_azim = -60.0

        # ── Auto-train sequence ───────────────────────────────────────────────
        # Stages: 0=sub2, 1=sub3, 2=sub4_mdp, 3=sub4_nav, 4=done, -1=idle
        self._auto_stage      = -1
        self._auto_retrain    = False
        self._auto_started    = False   # True once current stage has been kicked off
        self._auto_stage_started_at = None
        self._auto_failed_msg = ""
        self._auto_on_complete = None   # callback when all done
        self._auto_scenario    = None   # train Sub-4 nav on this scenario's layout
        self._auto_banner_var  = None   # StringVar for top banner
        self._auto_total_steps = [50_000, 0, 0, 12_000]  # quick auto-train steps

        # ── Evaluation state ──────────────────────────────────────────────────
        self._sub2_eval_queue  = queue.Queue()
        self._sub2_eval_stop   = threading.Event()
        self._sub2_eval_thread = None
        self._sub2_eval_active = False
        self._sub2_eval_path   = []
        self._sub2_eval_result = ""
        self._sub2_efficiency  = ""   # "Model trained ✓  Hover efficiency ~73%" etc.

        self._nav_eval_queue   = queue.Queue()
        self._nav_eval_stop    = threading.Event()
        self._nav_eval_thread  = None
        self._nav_eval_active  = False
        self._nav_eval_path    = []
        self._nav_eval_result  = ""
        self._nav_efficiency   = ""   # "Model trained ✓  Navigation success ~61%"

        # ── Tkinter vars & widgets ────────────────────────────────────────────
        self._sub1_status_var = None
        self._sub2_status_var = None
        self._sub4_status_var = None
        self._nav_status_var  = None
        self._sub2_steps_var  = None
        self._nav_steps_var   = None
        self._gate_var        = None
        self._sub1_btn        = None
        self._sub2_start_btn   = self._sub2_stop_btn  = self._sub2_entry = None
        self._sub2_retrain_btn = None
        self._sub2_eval_btn    = None
        self._sub4_solve_btn   = self._sub4_stop_btn = None
        self._nav_start_btn    = self._nav_stop_btn  = self._nav_entry = None
        self._nav_retrain_btn  = None
        self._nav_eval_btn     = None

        self._build_ui(Figure, FigureCanvasTkAgg)
        self._tick()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self, Figure, FigureCanvasTkAgg):
        # Header
        hdr = tk.Frame(self.window, bg="#07111f", padx=20, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="SkyShade — Training Grounds",
                 fg="#60a5fa", bg="#07111f", font=("Arial", 14, "bold")).pack(side="left")
        tk.Label(hdr, text="Train each subsystem before launching the simulation.",
                 fg="#475569", bg="#07111f", font=("Arial", 10)).pack(side="left", padx=(14, 0))

        # Auto-train progress banner (hidden until auto-train starts)
        self._auto_banner_var = tk.StringVar(value="")
        self._auto_banner_lbl = tk.Label(
            self.window, textvariable=self._auto_banner_var,
            fg="#4ade80", bg="#0f172a",
            font=("Arial", 11, "bold"), anchor="center", pady=7,
        )
        # Packed dynamically when auto-train starts

        # Notebook
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",        background="#07111f", borderwidth=0)
        style.configure("TNotebook.Tab",    background="#0f172a", foreground="#94a3b8",
                        font=("Arial", 10, "bold"), padding=[16, 7])
        style.map("TNotebook.Tab",
                  background=[("selected", "#1e3a5f")],
                  foreground=[("selected", "#7dd3fc")])
        style.configure("TFrame", background="#07111f")

        self._nb = ttk.Notebook(self.window)
        self._nb.pack(fill="both", expand=True, padx=14, pady=(0, 6))

        tab1 = tk.Frame(self._nb, bg="#07111f")
        tab2 = tk.Frame(self._nb, bg="#07111f")
        tab3 = tk.Frame(self._nb, bg="#07111f")
        tab4 = tk.Frame(self._nb, bg="#07111f")
        self._nb.add(tab1, text="  Sub-1  Perception  ")
        self._nb.add(tab2, text="  Sub-2  Flight PPO  ")
        self._nb.add(tab3, text="  Sub-3  Weather SVM  ")
        self._nb.add(tab4, text="  Sub-4  Nav Safety  ")

        self._build_sub1_tab(tab1, Figure, FigureCanvasTkAgg)
        self._build_sub2_tab(tab2, Figure, FigureCanvasTkAgg)
        self._build_sub3_tab(tab3, Figure, FigureCanvasTkAgg)
        self._build_sub4_tab(tab4, Figure, FigureCanvasTkAgg)

        # Footer — training status only, no sim launch
        footer = tk.Frame(self.window, bg="#0f172a", padx=20, pady=10)
        footer.pack(fill="x", side="bottom")
        footer.grid_columnconfigure(0, weight=1)

        self._gate_var = tk.StringVar(value=self._gate_text())
        tk.Label(footer, textvariable=self._gate_var,
                 fg="#94a3b8", bg="#0f172a", font=("Arial", 10)).grid(row=0, column=0, sticky="w")
        tk.Label(footer, text="Launch the main sim from the SkyShade Launcher window.",
                 fg="#334155", bg="#0f172a", font=("Arial", 9, "italic")).grid(row=0, column=1, sticky="e")

        if self._sub4_policy is not None:
            self._update_sub4_plot()

    def _build_sub1_tab(self, frame, Figure, FigureCanvasTkAgg):
        # Left: camera imshow.  Right: dual-axis chart (confidence + pixel error).
        # constrained_layout handles spacing automatically with twinx axes.
        fig = Figure(figsize=(9, 3.8), dpi=90, facecolor=_MPL_FIG_BG,
                     layout="constrained")
        gs      = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.05)
        ax_cam   = fig.add_subplot(gs[0])
        ax_chart = fig.add_subplot(gs[1])
        # Pixel-error right axis — created ONCE here, reused every tick
        ax_err   = ax_chart.twinx()

        ax_cam.set_facecolor("#000000")
        ax_cam.set_xticks([]); ax_cam.set_yticks([])
        ax_cam.set_title("Live Camera View", color=_MPL_TITLE, fontsize=11, pad=5)

        self._sub1_fig        = fig
        self._sub1_ax_cam     = ax_cam
        self._sub1_ax_chart   = ax_chart
        self._sub1_ax_err     = ax_err   # pixel-error right axis (twinx, permanent)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(10, 2))
        self._sub1_canvas = canvas

        ctrl = tk.Frame(frame, bg="#07111f", padx=14, pady=8)
        ctrl.pack(fill="x")
        self._sub1_status_var = tk.StringVar(value="Idle — click Start Live View to open the perception room")
        tk.Label(ctrl, textvariable=self._sub1_status_var,
                 fg="#94a3b8", bg="#07111f", font=("Arial", 10, "bold"), anchor="w"
                 ).pack(side="left", fill="x", expand=True)
        self._sub1_btn = tk.Button(ctrl, text="Start Live View",
                                   command=self._toggle_sub1,
                                   bg="#166534", fg="#dcfce7", relief="flat",
                                   font=("Arial", 10, "bold"), padx=12, pady=6)
        self._sub1_btn.pack(side="right")

    def _build_sub2_tab(self, frame, Figure, FigureCanvasTkAgg):
        # Left: 3D hover arena.  Right: PPO reward curve.
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
        fig = Figure(figsize=(9, 4.4), dpi=90, facecolor=_MPL_FIG_BG)
        gs  = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.08,
                               left=0.02, right=0.98, top=0.93, bottom=0.09)
        ax_3d     = fig.add_subplot(gs[0], projection="3d")
        ax_reward = fig.add_subplot(gs[1])
        _mpl_dark_axes(ax_reward, "PPO Reward — Training History", "Timestep", "Mean reward")
        self._style_3d_ax(ax_3d, "Hover Arena")
        self._sub2_fig     = fig
        self._sub2_ax_3d   = ax_3d
        self._sub2_ax_reward = ax_reward

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(10, 2))
        self._sub2_canvas = canvas

        ctrl = tk.Frame(frame, bg="#07111f", padx=14, pady=6)
        ctrl.pack(fill="x")
        ctrl.grid_columnconfigure(1, weight=1)

        self._sub2_status_var = tk.StringVar(value=self._model_label(2))
        tk.Label(ctrl, textvariable=self._sub2_status_var,
                 fg="#94a3b8", bg="#07111f", font=("Arial", 10, "bold"), anchor="w"
                 ).grid(row=0, column=0, columnspan=4, sticky="w")

        tk.Label(ctrl, text="Timesteps:", fg="#64748b", bg="#07111f", font=("Arial", 9)
                 ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self._sub2_steps_var = tk.StringVar(value="500000")
        self._sub2_entry = tk.Entry(ctrl, textvariable=self._sub2_steps_var,
                                    bg="#020617", fg="#f8fafc", insertbackground="#f8fafc",
                                    relief="flat", font=("Arial", 11, "bold"), width=10)
        self._sub2_entry.grid(row=1, column=1, sticky="w", padx=(8, 24), pady=(6, 0))

        self._sub2_start_btn = tk.Button(ctrl, text="Start Training",
                                         command=self._start_sub2,
                                         bg="#2563eb", fg="#eff6ff", relief="flat",
                                         font=("Arial", 10, "bold"), padx=12, pady=6)
        self._sub2_start_btn.grid(row=1, column=2, sticky="e", pady=(6, 0))

        self._sub2_stop_btn = tk.Button(ctrl, text="Stop",
                                        command=self._stop_sub2,
                                        bg="#334155", fg="#94a3b8", relief="flat",
                                        font=("Arial", 10, "bold"), padx=12, pady=6,
                                        state="disabled")
        self._sub2_stop_btn.grid(row=1, column=3, sticky="e", padx=(8, 0), pady=(6, 0))

        self._sub2_retrain_btn = tk.Button(ctrl, text="🔄  Retrain from Scratch",
                                           command=self._retrain_sub2,
                                           bg="#92400e", fg="#fef3c7", relief="flat",
                                           font=("Arial", 10, "bold"), padx=12, pady=6)
        self._sub2_retrain_btn.grid(row=2, column=0, columnspan=2, sticky="ew",
                                     pady=(8, 0))

        self._sub2_eval_btn = tk.Button(ctrl, text="Evaluate Model",
                                        command=self._run_sub2_eval,
                                        bg="#0e7490", fg="#cffafe", relief="flat",
                                        font=("Arial", 10, "bold"), padx=12, pady=6)
        self._sub2_eval_btn.grid(row=2, column=2, columnspan=2, sticky="ew",
                                  padx=(8, 0), pady=(8, 0))

    def _build_sub3_tab(self, frame, Figure, FigureCanvasTkAgg):
        # Left: 3D animated weather scene (drone + umbrella + rain + wind).
        # Right: Weather gauges + SVM decision + confusion matrix.
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = Figure(figsize=(10.5, 5.2), dpi=100, facecolor=_MPL_FIG_BG)
        gs  = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0],
                               height_ratios=[0.44, 0.56],
                               wspace=0.08, hspace=0.18,
                               left=0.02, right=0.98, top=0.93, bottom=0.08)
        ax_3d     = fig.add_subplot(gs[:, 0], projection="3d")
        ax_chart  = fig.add_subplot(gs[0, 1])
        ax_matrix = fig.add_subplot(gs[1, 1])
        self._style_3d_ax(ax_3d, "Weather Scene")
        _mpl_dark_axes(ax_chart, "Live SVM Decision", "", "")
        _mpl_dark_axes(ax_matrix, "Validation Results", "", "")
        self._sub3_fig      = fig
        self._sub3_ax_3d    = ax_3d
        self._sub3_ax_chart = ax_chart
        self._sub3_ax_matrix = ax_matrix

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(10, 2))
        self._sub3_canvas = canvas

        ctrl = tk.Frame(frame, bg="#07111f", padx=14, pady=6)
        ctrl.pack(fill="x")
        ctrl.grid_columnconfigure(0, weight=1)

        self._sub3_status_var = tk.StringVar(
            value=("● READY — svm_v1.pkl found" if self._sub3_ready
                   else "○ NOT TRAINED — click Train SVM (takes < 2 seconds)"))
        tk.Label(ctrl, textvariable=self._sub3_status_var,
                 fg="#94a3b8", bg="#07111f", font=("Arial", 10, "bold"), anchor="w"
                 ).grid(row=0, column=0, sticky="w")

        self._sub3_summary_var = tk.StringVar(
            value=("Output: STOW keeps the umbrella closed; DEPLOY opens it. "
                   "Green matrix cells are correct decisions, red cells are mistakes."))
        tk.Label(ctrl, textvariable=self._sub3_summary_var,
                 fg="#cbd5e1", bg="#07111f", font=("Arial", 9), anchor="w",
                 justify="left", wraplength=820
                 ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self._sub3_train_btn = tk.Button(
            ctrl, text="🔄  Train / Retrain SVM",
            command=self._start_sub3,
            bg="#831843", fg="#fce7f3", relief="flat",
            font=("Arial", 10, "bold"), padx=12, pady=6)
        self._sub3_train_btn.grid(row=0, column=1, sticky="e")

    def _build_sub4_tab(self, frame, Figure, FigureCanvasTkAgg):
        # Left: 3D obstacle navigation room.  Right: PPO nav reward curve.
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = Figure(figsize=(9, 4.4), dpi=90, facecolor=_MPL_FIG_BG)
        gs  = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.06,
                               left=0.02, right=0.98, top=0.93, bottom=0.07)
        ax_3d     = fig.add_subplot(gs[0], projection="3d")
        ax_reward = fig.add_subplot(gs[1])
        _mpl_dark_axes(ax_reward, "Nav PPO Reward Curve", "Timestep", "Mean reward")
        self._style_3d_ax(ax_3d, "Obstacle Navigation Room")
        self._sub4_fig     = fig
        self._sub4_ax_3d   = ax_3d
        self._sub4_ax_reward = ax_reward

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(10, 2))
        self._sub4_canvas = canvas

        # Nav training controls
        nav_ctrl = tk.Frame(frame, bg="#07111f", padx=14, pady=4)
        nav_ctrl.pack(fill="x")
        nav_ctrl.grid_columnconfigure(1, weight=1)

        self._nav_status_var = tk.StringVar(
            value="● READY — nav model found (SAC)" if self._nav_ready
            else "○ NOT TRAINED — click Train Navigation (SAC, ~3× faster than PPO)")
        tk.Label(nav_ctrl, textvariable=self._nav_status_var,
                 fg="#94a3b8", bg="#07111f", font=("Arial", 10, "bold"), anchor="w"
                 ).grid(row=0, column=0, columnspan=4, sticky="w")

        tk.Label(nav_ctrl, text="Steps:", fg="#64748b", bg="#07111f", font=("Arial", 9)
                 ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._nav_steps_var = tk.StringVar(value="12000")
        self._nav_entry = tk.Entry(nav_ctrl, textvariable=self._nav_steps_var,
                                   bg="#020617", fg="#f8fafc", insertbackground="#f8fafc",
                                   relief="flat", font=("Arial", 11, "bold"), width=9)
        self._nav_entry.grid(row=1, column=1, sticky="w", padx=(8, 20), pady=(4, 0))

        self._nav_start_btn = tk.Button(nav_ctrl, text="Train Navigation",
                                        command=self._start_nav,
                                        bg="#0f766e", fg="#ccfbf1", relief="flat",
                                        font=("Arial", 10, "bold"), padx=10, pady=5)
        self._nav_start_btn.grid(row=1, column=2, sticky="e", pady=(4, 0))

        self._nav_stop_btn = tk.Button(nav_ctrl, text="Stop",
                                       command=self._stop_nav,
                                       bg="#334155", fg="#94a3b8", relief="flat",
                                       font=("Arial", 10, "bold"), padx=10, pady=5,
                                       state="disabled")
        self._nav_stop_btn.grid(row=1, column=3, sticky="e", padx=(6, 0), pady=(4, 0))

        self._nav_retrain_btn = tk.Button(nav_ctrl, text="🔄  Retrain from Scratch",
                                          command=self._retrain_nav,
                                          bg="#92400e", fg="#fef3c7", relief="flat",
                                          font=("Arial", 10, "bold"), padx=10, pady=5)
        self._nav_retrain_btn.grid(row=2, column=0, columnspan=2, sticky="ew",
                                    pady=(6, 0))

        self._nav_eval_btn = tk.Button(nav_ctrl, text="Evaluate Model",
                                       command=self._run_nav_eval,
                                       bg="#0e7490", fg="#cffafe", relief="flat",
                                       font=("Arial", 10, "bold"), padx=10, pady=5)
        self._nav_eval_btn.grid(row=2, column=2, columnspan=2, sticky="ew",
                                 padx=(6, 0), pady=(6, 0))

        # MDP solver — compact secondary row
        mdp_ctrl = tk.Frame(frame, bg="#0f172a", padx=14, pady=4)
        mdp_ctrl.pack(fill="x")
        mdp_ctrl.grid_columnconfigure(0, weight=1)

        self._sub4_status_var = tk.StringVar(value=self._model_label(4))
        tk.Label(mdp_ctrl, textvariable=self._sub4_status_var,
                 fg="#64748b", bg="#0f172a", font=("Arial", 9), anchor="w"
                 ).grid(row=0, column=0, sticky="w")

        self._sub4_solve_btn = tk.Button(mdp_ctrl, text="Solve Battery-Safety MDP",
                                         command=self._start_sub4,
                                         bg="#4c1d95", fg="#ddd6fe", relief="flat",
                                         font=("Arial", 9, "bold"), padx=8, pady=4)
        self._sub4_solve_btn.grid(row=0, column=1, sticky="e")

        self._sub4_stop_btn = tk.Button(mdp_ctrl, text="Stop",
                                        command=self._stop_sub4,
                                        bg="#1e293b", fg="#94a3b8", relief="flat",
                                        font=("Arial", 9, "bold"), padx=8, pady=4,
                                        state="disabled")
        self._sub4_stop_btn.grid(row=0, column=2, sticky="e", padx=(4, 0))

    # ── Status helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _style_3d_ax(ax, title: str = ""):
        """Apply SkyShade dark theme to a 3D Axes3D instance."""
        ax.set_facecolor(_MPL_BG)
        ax.set_title(title, color=_MPL_TITLE, fontsize=12, pad=5)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = True
            pane.set_facecolor("#07111f")
            pane.set_edgecolor("#1e3a5f")
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.zaxis.label.set_color("#94a3b8")
        ax.grid(True, color="#1e3a5f", lw=0.4)

    def _model_label(self, sub: int) -> str:
        if sub == 2:
            return ("● READY — ppo_flight_v1.zip found" if self._sub2_ready
                    else "○ NOT TRAINED — click Start Training to generate the PPO model")
        if sub == 4:
            return ("● READY — policy_table_v1.npy found" if self._sub4_ready
                    else "○ NOT SOLVED — click Solve MDP to run value iteration")
        return ""

    def _all_ready(self) -> bool:
        return self._sub2_ready and self._sub3_ready and self._sub4_ready and self._nav_ready

    def _gate_text(self) -> str:
        s1 = "●"
        s2 = "●" if self._sub2_ready  else "○"
        s3 = "●" if self._sub3_ready  else "○"
        s4 = "●" if self._sub4_ready  else "○"
        sn = "●" if self._nav_ready   else "○"
        return (f"Sub-1 {s1}   Sub-2 PPO {s2}   Sub-3 SVM {s3}"
                f"   Sub-4 MDP {s4}   Sub-4 Nav {sn}")

    def select_tab(self, idx: int):
        """Select a tab by 0-based index (0=Sub-1, 1=Sub-2, 2=Sub-3, 3=Sub-4)."""
        if self._nb is not None:
            self._nb.select(idx)

    # ── Auto-Train sequence ───────────────────────────────────────────────────

    def start_auto_train(self, retrain: bool = False, stages=None,
                         on_complete=None, scenario=None):
        """Fully-automatic sequential training.

        Stages (pass a set to run only specific ones):
          0 — Sub-2 Flight PPO   50k steps or skip if already trained
          1 — Sub-3 Weather SVM  always < 5 sec
          2 — Sub-4 Battery MDP  always < 2 sec
          3 — Sub-4 Nav SAC      skip if trained, otherwise 12k quick steps

        `scenario` makes Sub-4 Nav SAC train on that launch scenario's obstacle
        layout (city / park / forest / trail), so the nav policy is tuned to the
        world it will fly in.
        """
        if self._auto_stage >= 0:
            return  # already running

        self._auto_stages      = set(stages) if stages is not None else {0, 1, 2, 3}
        self._auto_retrain     = retrain
        self._auto_scenario    = scenario
        self._auto_stage       = 0
        self._auto_started     = False
        self._auto_stage_started_at = None
        self._auto_failed_msg  = ""
        self._auto_on_complete = on_complete

        if retrain:
            if 0 in self._auto_stages:
                for path in [self._PPO_PATH, self._PPO_PATH.replace(".zip", "")]:
                    try:
                        if os.path.exists(path): os.remove(path)
                    except Exception: pass
                self._sub2_runs.clear()
                self._sub2_efficiency = ""
            if 3 in self._auto_stages:
                nav = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "models", "ppo_nav_v1")
                for path in [nav, nav + ".zip"]:
                    try:
                        if os.path.exists(path): os.remove(path)
                    except Exception: pass

        self._auto_banner_lbl.pack(fill="x", before=self._nb)
        selected = sorted(self._auto_stages)
        first_label = {0: "Sub-2 Flight PPO", 1: "Sub-3 Weather SVM",
                       2: "Sub-4 Battery MDP", 3: "Sub-4 Nav SAC"}
        first = first_label.get(selected[0], "subsystems") if selected else "nothing"
        self._auto_banner_var.set(f"🚀  Auto-Train  Starting with {first}…")
        self.window.lift()
        self.window.focus_force()

    def _auto_tick(self):
        """Drive the auto-train state machine — called every hub tick."""
        stage = self._auto_stage
        if stage < 0 or stage >= 4:
            return
        if self._auto_failed_msg:
            self._show_auto_complete()
            return

        LABELS = [
            "Step 1 / 4 — Sub-2 Flight PPO  (quick)",
            "Step 2 / 4 — Sub-3 Weather SVM  (~5 sec)",
            "Step 3 / 4 — Sub-4 Battery MDP  (~2 sec)",
            "Step 4 / 4 — Sub-4 Nav SAC      (skip if trained)",
        ]

        if stage == 0:
            if not self._auto_started:
                # Skip if not selected
                if 0 not in self._auto_stages:
                    self._auto_stage = 1
                    self._auto_started = False
                    return
                self.select_tab(1)
                if self._sub2_ready and not self._auto_retrain:
                    self._sub2_efficiency = "Sub-2 PPO already trained ✓"
                    self._sub2_status_var.set("Already trained — skipping PPO fine-tune.")
                    self._auto_stage = 1
                    self._auto_started = False
                    return
                self._sub2_steps_var.set(str(self._auto_total_steps[0]))
                self._start_sub2()
                self._auto_started = True
                self._auto_stage_started_at = time.time()
                self._auto_banner_var.set(f"🚀  Auto-Train  {LABELS[0]}")
            elif not self._sub2_training:
                self._auto_stage = 1
                self._auto_started = False
                self._auto_stage_started_at = None

        elif stage == 1:
            if not self._auto_started:
                if 1 not in self._auto_stages:
                    self._auto_stage = 2
                    self._auto_started = False
                    return
                self.select_tab(2)
                self._start_sub3()
                self._auto_started = True
                self._auto_stage_started_at = time.time()
                self._auto_banner_var.set(f"🚀  Auto-Train  {LABELS[1]}")
            elif not self._sub3_training:
                self._auto_stage = 2
                self._auto_started = False
                self._auto_stage_started_at = None

        elif stage == 2:
            if not self._auto_started:
                if 2 not in self._auto_stages:
                    self._auto_stage = 3
                    self._auto_started = False
                    return
                self.select_tab(3)
                self._start_sub4()
                self._auto_started = True
                self._auto_stage_started_at = time.time()
                self._auto_banner_var.set(f"🚀  Auto-Train  {LABELS[2]}")
            elif not self._sub4_training:
                self._auto_stage = 3
                self._auto_started = False
                self._auto_stage_started_at = None

        elif stage == 3:
            if not self._auto_started:
                if 3 not in self._auto_stages:
                    self._auto_stage = 4
                    self._show_auto_complete()
                    return
                if self._nav_ready and not self._auto_retrain:
                    self._nav_efficiency = "Sub-4 Nav SAC already trained ✓"
                    self._nav_status_var.set("Already trained — skipping long SAC fine-tune.")
                    self._auto_stage = 4
                    self._show_auto_complete()
                    return
                self._nav_steps_var.set(str(self._auto_total_steps[3]))
                self._start_nav(scenario=self._auto_scenario)
                self._auto_started = True
                self._auto_stage_started_at = time.time()
                self._auto_banner_var.set(f"🚀  Auto-Train  {LABELS[3]}")
            elif self._nav_training:
                elapsed = time.time() - (self._auto_stage_started_at or time.time())
                if elapsed > 90:
                    self._nav_stop.set()
                    self._nav_training = False
                    self._nav_ready = _artifact_ready(self._NAV_PATH, 1024)
                    self._nav_entry.configure(state="normal")
                    self._nav_start_btn.configure(state="normal")
                    self._nav_retrain_btn.configure(state="normal")
                    self._nav_stop_btn.configure(state="disabled", bg="#334155",
                                                  fg="#94a3b8", activebackground="#475569")
                    self._nav_efficiency = (
                        "⏱ Nav SAC fine-tune timed out after 90 sec — using existing "
                        "or partial model.")
                    self._nav_status_var.set(self._nav_efficiency)
                    self._auto_stage = 4
                    self._show_auto_complete()
            elif not self._nav_training:
                self._auto_stage = 4
                self._show_auto_complete()

    def _show_auto_complete(self):
        """Replace banner with a green completion summary."""
        self._auto_stage = -1

        if self._auto_failed_msg:
            self._auto_banner_var.set(
                f"⚠ Auto-Train stopped before completion: {self._auto_failed_msg}  "
                "Fix that subsystem, then run Auto-Train again.")
            self._auto_banner_lbl.configure(fg="#fed7aa", bg="#7c2d12")
            return

        s2 = self._sub2_efficiency or "Sub-2 PPO ✓"
        s3 = (f"Sub-3 SVM {self._sub3_accuracy:.0%} ✓"
              if self._sub3_accuracy else "Sub-3 SVM ✓")
        s4n = self._nav_efficiency or "Sub-4 Nav ✓"

        self._auto_banner_var.set(
            f"✅  Training complete!   {s2}  ·  {s3}  ·  Sub-4 MDP ✓  ·  {s4n}"
            "   →  Select a scenario in the Launcher and click Launch.")
        self._auto_banner_lbl.configure(fg="#4ade80", bg="#052e16")

        if self._auto_on_complete:
            try: self._auto_on_complete()
            except Exception: pass

    # ── Sub-1 controls & plot ─────────────────────────────────────────────────

    def _toggle_sub1(self):
        self._sub1_running = not self._sub1_running
        if self._sub1_running:
            self._sub1_start_t = time.time()
            self._sub1_frame_idx = 0
            self._sub1_history.clear()
            self._sub1_img_ref = None
            self._sub1_btn.configure(text="Stop Live View")
            self._sub1_status_var.set("Starting perception room…")
        else:
            self._sub1_btn.configure(text="Start Live View")
            self._sub1_status_var.set("Stopped.")
            if self._sub1_perc_env is not None:
                self._sub1_perc_env.close()
                self._sub1_perc_env = None

    def _sub1_step(self):
        if not self._sub1_running:
            return
        try:
            from sub1_perception.training_env import PerceptionTrainingEnv
            from sub1_perception.training import camera_training_marker_box
            if self._sub1_perc_env is None:
                self._sub1_perc_env = PerceptionTrainingEnv()

            t = time.time() - self._sub1_start_t
            frame_rgb, sphere_3d = self._sub1_perc_env.step(t)

            # Project the sphere's known 3D position to 2D pixel — this is the
            # "expected" target centre the tracker should land on.
            expected_px = self._sub1_perc_env.project_to_pixel(sphere_3d)

            # Run HSV detection on the rendered frame
            _pos, confidence = self._sub1_tracker.process_frame(frame_rgb)
            box = camera_training_marker_box(frame_rgb)
            overlay = frame_rgb.copy()
            pixel_error = float("inf")

            if box is not None:
                x, y, w, h, _area = box
                detected_px = (x + w // 2, y + h // 2)
                if expected_px is not None:
                    pixel_error = float(np.hypot(
                        detected_px[0] - expected_px[0],
                        detected_px[1] - expected_px[1],
                    ))
                # Bounding box — green if passing, orange if not
                passing_det = pixel_error <= TRAINING_PIXEL_PASS
                box_col = (0, 220, 80) if passing_det else (255, 140, 0)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), box_col, 3)
                # Detection crosshair (white)
                cv2.drawMarker(overlay, detected_px, (255, 255, 255),
                               cv2.MARKER_CROSS, 20, 2)
                # Pixel error label
                err_txt = f"{pixel_error:.0f}px" if math.isfinite(pixel_error) else "?"
                cv2.putText(overlay, f"err {err_txt}",
                            (max(4, x), max(18, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, box_col, 2)

            # Draw the TARGET zone: yellow circle at sphere's projected position,
            # blue ring = acceptance radius (tracker must stay within this)
            if expected_px is not None:
                epx, epy = expected_px
                if 0 <= epx < frame_rgb.shape[1] and 0 <= epy < frame_rgb.shape[0]:
                    # Acceptance-zone ring (pixel radius = TRAINING_PIXEL_PASS)
                    zone_r = max(1, int(TRAINING_PIXEL_PASS))
                    zone_col = (80, 180, 255)   # blue
                    cv2.circle(overlay, (epx, epy), zone_r, zone_col, 2)
                    # Target centre dot (yellow)
                    cv2.circle(overlay, (epx, epy), 6, (255, 210, 0), -1)
                    cv2.putText(overlay, "target",
                                (epx + 8, max(14, epy - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 210, 0), 1)

            # Status
            passing = (confidence >= CONFIDENCE_THRESH
                       and math.isfinite(pixel_error)
                       and pixel_error <= TRAINING_PIXEL_PASS)
            err_str = f"{pixel_error:.1f}px" if math.isfinite(pixel_error) else "no detection"
            verdict  = "LOCKED ✓" if passing else ("FOUND – off target" if box else "SEARCHING")
            self._sub1_status_var.set(
                f"{verdict}   confidence {confidence:.2f}   pixel error {err_str}")

            # Store history (frame, confidence, pixel_error)
            self._sub1_history.append(
                (self._sub1_frame_idx,
                 float(confidence),
                 float(pixel_error) if math.isfinite(pixel_error) else 50.0))
            self._sub1_frame_idx += 1
            if len(self._sub1_history) > 200:
                self._sub1_history = self._sub1_history[-200:]

            # ── Camera view (imshow) ──────────────────────────────────────────
            ax_cam = self._sub1_ax_cam
            if self._sub1_img_ref is None:
                self._sub1_img_ref = ax_cam.imshow(overlay, aspect="auto")
                ax_cam.set_xticks([]); ax_cam.set_yticks([])
            else:
                self._sub1_img_ref.set_data(overlay)

            # ── Confidence + pixel-error chart (right) ────────────────────────
            ax2 = self._sub1_ax_chart
            ax3 = self._sub1_ax_err   # permanent twinx — clear and redraw each tick
            ax2.cla()
            ax3.cla()

            # Re-apply styles after cla()
            ax2.set_facecolor(_MPL_BG)
            for sp in ax2.spines.values():
                sp.set_color(_MPL_EDGE)
            ax2.tick_params(colors="#94a3b8", labelsize=11)
            ax2.set_xlabel("Frame", color="#94a3b8", fontsize=12)
            ax2.set_ylabel("Confidence", color="#60a5fa", fontsize=12)
            ax2.set_title("Tracking Accuracy", color=_MPL_TITLE, fontsize=12, pad=6)
            ax2.grid(True, color=_MPL_GRID, linestyle="--", alpha=0.4)

            for sp in ax3.spines.values():
                sp.set_color(_MPL_EDGE)
            ax3.tick_params(colors="#f97316", labelsize=11)
            ax3.set_ylabel("Pixel error (px)", color="#f97316", fontsize=12)
            ax3.set_facecolor(_MPL_BG)

            if len(self._sub1_history) > 1:
                xs = [h[0] for h in self._sub1_history]
                cs = [h[1] for h in self._sub1_history]
                es = [h[2] for h in self._sub1_history]

                ax2.plot(xs, cs, color="#60a5fa", lw=2.0, label="Confidence")
                ax2.axhline(CONFIDENCE_THRESH, color="#60a5fa",
                            ls="--", lw=1.2, alpha=0.6,
                            label=f"threshold {CONFIDENCE_THRESH:.2f}")
                ax2.set_ylim(0, 1.12)

                ax3.plot(xs, es, color="#f97316", lw=1.8, label="Pixel error")
                ax3.axhline(TRAINING_PIXEL_PASS, color="#f97316",
                            ls="--", lw=1.2, alpha=0.6,
                            label=f"pass ≤ {int(TRAINING_PIXEL_PASS)}px")
                ax3.set_ylim(-2, 56)

                # Shade passing region green
                ax2.fill_between(xs,
                                 [c if c >= CONFIDENCE_THRESH else 0 for c in cs],
                                 CONFIDENCE_THRESH,
                                 where=[c >= CONFIDENCE_THRESH for c in cs],
                                 color="#22c55e", alpha=0.12)

                # Combined legend, larger font
                h1, l1 = ax2.get_legend_handles_labels()
                h2, l2 = ax3.get_legend_handles_labels()
                ax2.legend(h1 + h2, l1 + l2,
                           facecolor="#0f172a", edgecolor="#334155",
                           labelcolor="#e2e8f0", fontsize=10,
                           loc="lower right")
            else:
                ax2.text(0.5, 0.5, "Waiting for frames…",
                         ha="center", va="center", color="#475569",
                         transform=ax2.transAxes, fontsize=11)

            self._sub1_canvas.draw_idle()
        except Exception:
            pass

    def _update_sub1_plot(self):
        pass  # all updates handled inline in _sub1_step()

    # ── Sub-2 controls & plot ─────────────────────────────────────────────────

    # ── Sub-3 SVM controls & plot ─────────────────────────────────────────────

    def _start_sub3(self):
        if self._sub3_training:
            return
        from sub3_env.training_worker import SVMTrainingWorker
        self._sub3_stop.clear()
        self._sub3_thread = SVMTrainingWorker(self._sub3_queue, self._sub3_stop)
        self._sub3_thread.start()
        self._sub3_training = True
        self._sub3_train_btn.configure(state="disabled", text="Training…")
        self._sub3_status_var.set("Training SVM on weather sensor data…")

    @staticmethod
    def _eval_sub3_on_disk():
        """Load the saved SVM and compute accuracy + CM on training CSV.

        Called once at hub startup so the confusion matrix is always visible
        even if training didn't happen in this session.
        """
        import csv as _csv
        import pickle
        from sklearn.metrics import confusion_matrix

        _DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "env_sensor_log.csv")
        _MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "models", "svm_v1.pkl")
        if not os.path.exists(_DATA) or not os.path.exists(_MODEL):
            return None, None

        with open(_MODEL, "rb") as f:
            clf = pickle.load(f)

        lux_l, rain_l, wind_l, y_l = [], [], [], []
        with open(_DATA, newline="") as f:
            for row in _csv.DictReader(f):
                lux_l.append(float(row["lux"]))
                rain_l.append(float(row["rain_raw"]))
                wind_l.append(float(row["wind_speed"]))
                y_l.append(int(row["label"]))

        lux = np.array(lux_l); rain = np.array(rain_l)
        wind = np.array(wind_l); y = np.array(y_l)
        lux_d  = np.diff(lux,  prepend=lux[0])
        rain_d = np.diff(rain, prepend=rain[0])
        wind_d = np.diff(wind, prepend=wind[0])
        prev1  = np.roll(y, 1); prev1[0]  = 0
        prev2  = np.roll(y, 2); prev2[:2] = 0
        prev3  = np.roll(y, 3); prev3[:3] = 0
        X = np.column_stack([lux, rain, wind, lux_d, rain_d, wind_d, prev1, prev2, prev3])

        preds = clf.predict(X)
        cm    = confusion_matrix(y, preds)
        acc   = float((preds == y).mean())
        return acc, cm

    def _load_sub3_clf(self):
        """Lazy-load the SVM classifier for the live demo."""
        if self._sub3_clf is None and _artifact_ready(self._SVM_PATH, 256):
            try:
                import pickle
                from sub3_env.feature_engineering import FeatureBuilder
                with open(self._SVM_PATH, "rb") as f:
                    self._sub3_clf = pickle.load(f)
                self._sub3_fb = FeatureBuilder()
            except Exception:
                self._sub3_clf = None

    def _update_sub3_plot(self):
        import math as _math
        t = time.time()

        # ── Animated weather cycle: clear → cloudy → rainy → storm → clear ──
        period = 30.0    # seconds per full cycle
        phase  = (t % period) / period   # 0-1

        if phase < 0.25:       # clear
            lux   = 85000 - phase * 4 * 40000
            rain  = 0.01 + phase * 4 * 0.05
            wind  = 1.0  + phase * 4 * 1.0
        elif phase < 0.50:     # cloudy
            p2    = (phase - 0.25) * 4
            lux   = 45000 - p2 * 25000
            rain  = 0.06  + p2 * 0.30
            wind  = 2.0   + p2 * 2.0
        elif phase < 0.75:     # rainy
            p3    = (phase - 0.50) * 4
            lux   = 20000 - p3 * 12000
            rain  = 0.36  + p3 * 0.50
            wind  = 4.0   + p3 * 3.0
        else:                  # clearing
            p4    = (phase - 0.75) * 4
            lux   = 8000  + p4 * 77000
            rain  = 0.86  - p4 * 0.85
            wind  = 7.0   - p4 * 6.0

        lux  = float(np.clip(lux,  1000, 100000))
        rain = float(np.clip(rain, 0.0,  1.0))
        wind = float(np.clip(wind, 0.0,  10.0))

        # Run SVM prediction if model loaded
        self._load_sub3_clf()
        if self._sub3_clf is not None:
            try:
                feat = self._sub3_fb.build(lux, rain, wind)
                pred = int(self._sub3_clf.predict(feat.reshape(1, -1))[0])
                self._sub3_umbrella = "DEPLOY" if pred else "STOW"
                self._sub3_fb.push_action(pred)
            except Exception:
                pass

        umbrella_open = self._sub3_umbrella == "DEPLOY"
        if self._sub3_status_var is not None and not self._sub3_training:
            acc = (f" | CV accuracy {self._sub3_accuracy:.1%}"
                   if self._sub3_accuracy is not None else "")
            self._sub3_status_var.set(
                f"Live SVM output: {self._sub3_umbrella} | "
                f"lux {lux/1000:.0f}k, rain {rain:.2f}, wind {wind:.1f} m/s{acc}")
        if self._sub3_summary_var is not None:
            self._sub3_summary_var.set(
                f"The SVM reads lux, rain, and wind. DEPLOY opens the umbrella "
                f"when rain is about {UMBRELLA_DEPLOY_RAIN_THRESHOLD:.2f}+; "
                "STOW keeps it closed. Green cells below are correct validation "
                "decisions; red cells are mistakes.")

        # ── 3D Weather Scene ─────────────────────────────────────────────────
        ax = self._sub3_ax_3d
        ax.cla()
        self._style_3d_ax(ax, "Live Weather Scene")

        ROOM   = 4.0
        ALT    = 2.5
        rain_n = int(rain * 60)
        th     = np.linspace(0, 2*_math.pi, 40)

        # Ground floor (visible grid)
        for x in np.linspace(-ROOM, ROOM, 7):
            ax.plot([x, x], [-ROOM, ROOM], [0, 0], color="#334155", lw=0.8, alpha=0.7)
        for y in np.linspace(-ROOM, ROOM, 7):
            ax.plot([-ROOM, ROOM], [y, y], [0, 0], color="#334155", lw=0.8, alpha=0.7)
        ax.plot([-ROOM, ROOM, ROOM, -ROOM, -ROOM],
                [-ROOM, -ROOM, ROOM, ROOM, -ROOM],
                [0]*5, color="#475569", lw=1.5)

        # Cloud — filled disc at ceiling with colour based on rain intensity
        cloud_r   = 1.8 + rain * 2.2
        cloud_col = (max(0.25, 0.55 - rain*0.40),
                     max(0.25, 0.55 - rain*0.40),
                     max(0.25, 0.60 - rain*0.35))
        # Filled ellipse approximated with many thin slices
        for frac in np.linspace(0.3, 1.0, 6):
            ax.plot(cloud_r*frac*np.cos(th)*1.6, cloud_r*frac*np.sin(th),
                    np.full(40, ROOM + 0.1),
                    color=cloud_col, lw=1.5, alpha=0.25)
        ax.plot(cloud_r*np.cos(th)*1.6, cloud_r*np.sin(th),
                np.full(40, ROOM + 0.1),
                color=cloud_col, lw=3.0, alpha=0.9)

        # Rain particles — thicker, brighter, more visible
        if rain_n > 0:
            rng_r  = np.random.default_rng(int(t * 8) % 9999)
            rx     = rng_r.uniform(-ROOM*0.85, ROOM*0.85, rain_n)
            ry     = rng_r.uniform(-ROOM*0.85, ROOM*0.85, rain_n)
            rz_top = rng_r.uniform(ALT + 0.5, ROOM, rain_n)
            rz_bot = np.clip(rz_top - 0.9, 0, ROOM)
            alpha  = min(0.95, 0.4 + rain * 0.6)
            for i in range(rain_n):
                ax.plot([rx[i], rx[i]], [ry[i], ry[i]], [rz_top[i], rz_bot[i]],
                        color="#bfdbfe", lw=1.4, alpha=alpha)

        # Wind arrows at drone altitude (pointing right, varying strength)
        n_wind = int(wind / 2) + 1
        for i in range(n_wind):
            wy = -ROOM*0.6 + i * (ROOM * 1.2 / max(1, n_wind - 1))
            ax.quiver(-ROOM*0.7, wy, ALT, wind * 0.25, 0, 0,
                      length=0.8, color="#f97316", alpha=0.7,
                      arrow_length_ratio=0.4, linewidth=1.5)

        # Drone (X-frame with rotors)
        arm = 0.38
        for ang in [45, 135, 225, 315]:
            rad = _math.radians(ang)
            ex, ey = arm*_math.cos(rad), arm*_math.sin(rad)
            ax.plot([0, ex], [0, ey], [ALT, ALT], color="#7dd3fc", lw=2.5, zorder=10)
            th_r = np.linspace(0, 2*_math.pi, 12)
            ax.plot(ex + 0.14*np.cos(th_r), ey + 0.14*np.sin(th_r),
                    [ALT]*12, color="#60a5fa", lw=1.2, alpha=0.8)
        ax.scatter([0], [0], [ALT], color="#7dd3fc", s=60, depthshade=False, zorder=11)

        # Umbrella canopy above drone
        umb_h = ALT + 0.65
        if umbrella_open:
            # Open umbrella — disc with spokes
            umb_r = 1.2
            ax.plot(umb_r*np.cos(th), umb_r*np.sin(th), np.full(40, umb_h),
                    color="#22c55e", lw=3.0, zorder=12)
            for ang in range(0, 360, 45):
                rad = _math.radians(ang)
                ax.plot([0, umb_r*_math.cos(rad)], [0, umb_r*_math.sin(rad)],
                        [umb_h, umb_h], color="#22c55e", lw=1.5, alpha=0.8)
            ax.plot([0, 0], [0, 0], [ALT, umb_h], color="#22c55e", lw=2, alpha=0.9)
        else:
            # Folded umbrella — thin vertical line
            ax.plot([0, 0], [0, 0], [ALT, umb_h + 0.3], color="#475569", lw=2.5, alpha=0.7)

        # Decision label in 3D scene
        dec_col = "#4ade80" if umbrella_open else "#94a3b8"
        dec_txt = "☂ DEPLOY" if umbrella_open else "✕ STOW"
        ax.text2D(0.5, 0.03, dec_txt, transform=ax.transAxes,
                  ha="center", color=dec_col, fontsize=14, fontweight="bold")

        ax.set_xlim(-ROOM, ROOM); ax.set_ylim(-ROOM, ROOM); ax.set_zlim(0, ROOM + 0.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([0, ALT, ROOM])
        ax.zaxis.set_ticklabels(["ground", f"{ALT}m", "sky"], fontsize=8, color="#94a3b8")
        self._sub2_azim  # reuse azim counter — use a separate one
        if not hasattr(self, "_sub3_azim"):
            self._sub3_azim = -50.0
        self._sub3_azim = (self._sub3_azim + 0.3) % 360
        ax.view_init(elev=18, azim=self._sub3_azim)

        # ── Upper-right panel: weather gauges + live SVM decision ────────────
        ax_c = self._sub3_ax_chart
        ax_c.cla()
        ax_c.set_facecolor(_MPL_BG)
        for sp in ax_c.spines.values():
            sp.set_color(_MPL_EDGE)
        ax_c.set_xticks([]); ax_c.set_yticks([])

        phases      = ["Clear ☀", "Cloudy ⛅", "Rainy 🌧", "Storm ⛈"]
        phase_idx   = int(phase * 4) % 4
        phase_label = phases[phase_idx]

        ax_c.set_title(f"Live SVM Decision  ·  {phase_label}",
                       color=_MPL_TITLE, fontsize=13, pad=6, fontweight="bold")

        # ── Weather gauges (bigger bars, bigger labels) ───────────────────────
        bar_data = [
            ("Lux",  min(1.0, lux/100000), "#fbbf24", f"{lux/1000:.0f}k lux"),
            ("Rain", rain,                 "#60a5fa", f"{rain:.2f}"),
            ("Wind", min(1.0, wind/10),    "#67e8f9", f"{wind:.1f} m/s"),
        ]
        for i, (lbl, val, col, display) in enumerate(bar_data):
            y = 0.82 - i * 0.17
            ax_c.barh(y, val, height=0.13, color=col, alpha=0.85,
                      transform=ax_c.transAxes, left=0.16)
            ax_c.text(0.01, y, lbl, transform=ax_c.transAxes,
                      color=col, fontsize=12, fontweight="bold", va="center")
            ax_c.text(0.99, y, display, transform=ax_c.transAxes,
                      color=col, fontsize=11, va="center", ha="right")

        # ── SVM decision banner (very large) ─────────────────────────────────
        bg_col  = "#14532d" if umbrella_open else "#1e293b"
        ax_c.add_patch(__import__("matplotlib.patches", fromlist=["FancyBboxPatch"])
                       .FancyBboxPatch((0.05, 0.08), 0.90, 0.20,
                                       boxstyle="round,pad=0.02",
                                       transform=ax_c.transAxes,
                                       facecolor=bg_col, edgecolor=dec_col, linewidth=2))
        ax_c.text(0.5, 0.19, dec_txt, transform=ax_c.transAxes,
                  ha="center", va="center",
                  color=dec_col, fontsize=24, fontweight="bold")

        # ── Explanation line ──────────────────────────────────────────────────
        explain = ("Model output: DEPLOY because rain crossed the learned boundary"
                   if umbrella_open
                   else "Model output: STOW because conditions are below the deploy boundary")
        ax_c.text(0.5, 0.015, explain, transform=ax_c.transAxes,
                  ha="center", color="#94a3b8", fontsize=9, style="italic")

        # ── Lower-right panel: validation confusion matrix ───────────────────
        ax_m = self._sub3_ax_matrix
        ax_m.cla()
        ax_m.set_facecolor(_MPL_BG)
        for sp in ax_m.spines.values():
            sp.set_color(_MPL_EDGE)
        ax_m.set_xticks([]); ax_m.set_yticks([])

        if self._sub3_cm is not None:
            cm = self._sub3_cm
            from matplotlib.patches import FancyBboxPatch
            ax_m.set_title("Validation Confusion Matrix",
                           color=_MPL_TITLE, fontsize=13, pad=8, fontweight="bold")
            ax_m.text(0.36, 0.88, "Predicted STOW", transform=ax_m.transAxes,
                      ha="center", color="#93c5fd", fontsize=10, fontweight="bold")
            ax_m.text(0.76, 0.88, "Predicted DEPLOY", transform=ax_m.transAxes,
                      ha="center", color="#93c5fd", fontsize=10, fontweight="bold")

            cells = [
                (0, 0, 0.18, 0.55, "Actual STOW", "Correct stow", "#16a34a"),
                (0, 1, 0.58, 0.55, "Actual STOW", "False deploy", "#dc2626"),
                (1, 0, 0.18, 0.25, "Actual DEPLOY", "Missed rain", "#dc2626"),
                (1, 1, 0.58, 0.25, "Actual DEPLOY", "Correct deploy", "#16a34a"),
            ]
            for ri, ci, bx, by, row_lbl, label, color in cells:
                if ci == 0:
                    ax_m.text(0.02, by + 0.10, row_lbl, transform=ax_m.transAxes,
                              ha="left", va="center", color="#cbd5e1",
                              fontsize=9, fontweight="bold")
                ax_m.add_patch(FancyBboxPatch(
                    (bx, by), 0.34, 0.20,
                    boxstyle="round,pad=0.01",
                    transform=ax_m.transAxes,
                    facecolor=color, alpha=0.82,
                    edgecolor="#334155", linewidth=1.5))
                ax_m.text(bx + 0.17, by + 0.125, str(cm[ri, ci]),
                          transform=ax_m.transAxes, ha="center", va="center",
                          color="white", fontsize=20, fontweight="bold")
                ax_m.text(bx + 0.17, by + 0.050, label,
                          transform=ax_m.transAxes, ha="center", va="center",
                          color="white", fontsize=9)

            # CV accuracy prominently
            acc_col = "#4ade80" if (self._sub3_accuracy or 0) >= 0.9 else "#f97316"
            acc_txt = (f"Accuracy: {self._sub3_accuracy:.1%}  ✓ passes 90% target"
                       if (self._sub3_accuracy or 0) >= 0.9
                       else f"Accuracy: {self._sub3_accuracy:.1%}  ✗ below 90% target")
            ax_m.text(0.5, 0.05, acc_txt, transform=ax_m.transAxes,
                      ha="center", color=acc_col, fontsize=12, fontweight="bold")
        else:
            if self._sub3_ready:
                # Model exists but CM not yet computed — try loading now
                try:
                    self._sub3_accuracy, self._sub3_cm = self._eval_sub3_on_disk()
                except Exception:
                    pass
            if self._sub3_cm is None:
                ax_m.text(0.5, 0.50,
                          "Click  '🔄 Train / Retrain SVM'\nto train the model\nand see accuracy results here.",
                          transform=ax_m.transAxes, ha="center", va="center",
                          color="#64748b", fontsize=13)

        self._sub3_canvas.draw_idle()

    def _retrain_sub2(self):
        """Delete the existing PPO model and train from scratch."""
        if self._sub2_training:
            return
        # Remove saved model so warm-start won't trigger
        for p in [self._PPO_PATH, self._PPO_PATH.replace(".zip", "")]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        # Clear multi-run history — fresh slate on the chart
        self._sub2_runs.clear()
        self._sub2_efficiency = ""
        self._sub2_ready = False
        self._sub2_status_var.set("Deleted previous model — starting from scratch…")
        self._start_sub2()

    def _start_sub2(self):
        if self._sub2_training:
            return
        try:
            total = max(2048, int(self._sub2_steps_var.get()))
        except ValueError:
            total = 1_500_000
        self._sub2_total = total
        self._sub2_history.clear()
        self._sub2_step           = 0
        self._sub2_run_start_step = None   # set from first progress message this run
        self._sub2_start_time     = time.time()
        self._sub2_stop.clear()
        self._sub2_thread = _FlightTrainingWorker(total, self._sub2_queue, self._sub2_stop)
        self._sub2_thread.start()
        self._sub2_training   = True
        self._sub2_efficiency = ""
        self._sub2_history    = []   # fresh history for this run
        self._sub2_entry.configure(state="disabled")
        self._sub2_start_btn.configure(state="disabled"); self._sub2_retrain_btn.configure(state="disabled")
        self._sub2_stop_btn.configure(state="normal", bg="#dc2626", fg="#ffffff",
                                      activebackground="#b91c1c")
        warm = _artifact_ready(self._PPO_PATH, 1024)
        run_n = len(self._sub2_runs) + 1
        warm_note = f"Run {run_n} — fine-tuning…" if warm else "Run 1 — training from scratch…"
        mins = max(1, total // 120_000)   # ~2000 steps/sec with 4 parallel envs
        self._sub2_status_var.set(
            f"{warm_note}  4 parallel envs  ~{mins} min  "
            "(each run is a new colour on the chart)")

    def _stop_sub2(self):
        if not self._sub2_training:
            return
        self._sub2_stop.set()
        self._sub2_stop_btn.configure(state="disabled")
        self._sub2_status_var.set("Stopping…")

    def _update_sub2_plot(self):
        from matplotlib.patches import Patch
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        stage = (self._sub2_history[-1][2] if self._sub2_history else 1)
        t_now = time.time()

        ROOM_W     = 7.0    # square room half-width (m)
        TARGET_ALT = 2.5    # hover altitude (m)
        CEIL_H     = 4.0    # ceiling height (m)
        HOVER_R    = 0.5    # target hover radius (m)

        # Stage descriptions for the overlay text
        STAGE_DESC = {
            1: ("Stage 1 / 3  —  Calm air",
                "Drone learns to fly up and\nhold position above the target.\nNo wind, stationary user."),
            2: ("Stage 2 / 3  —  Gusty wind",
                "Random gusts hit the drone\nfrom every direction. It must\nresist and stay on target."),
            3: ("Stage 3 / 3  —  Walking user",
                "The user walks; drone must\ntrack them in gusty wind\nwhile maintaining altitude."),
        }

        # ── 3D Training Room ─────────────────────────────────────────────────
        ax = self._sub2_ax_3d
        ax.cla()
        self._style_3d_ax(ax)

        # Floor — dark grid
        for x in np.linspace(-ROOM_W, ROOM_W, 8):
            ax.plot([x, x], [-ROOM_W, ROOM_W], [0, 0],
                    color="#1e293b", lw=0.6, alpha=0.6)
        for y in np.linspace(-ROOM_W, ROOM_W, 8):
            ax.plot([-ROOM_W, ROOM_W], [y, y], [0, 0],
                    color="#1e293b", lw=0.6, alpha=0.6)
        # Floor outline
        ax.plot([-ROOM_W, ROOM_W, ROOM_W, -ROOM_W, -ROOM_W],
                [-ROOM_W, -ROOM_W, ROOM_W, ROOM_W, -ROOM_W],
                [0]*5, color="#334155", lw=1.5)

        # 4 semi-transparent walls (as polygon collections)
        wall_col = "#0f2235"
        wall_alpha = 0.25
        for (xs, ys) in [
            ([-ROOM_W, ROOM_W, ROOM_W, -ROOM_W], [-ROOM_W,-ROOM_W,-ROOM_W,-ROOM_W]),
            ([-ROOM_W, ROOM_W, ROOM_W, -ROOM_W], [ ROOM_W, ROOM_W, ROOM_W, ROOM_W]),
            ([-ROOM_W,-ROOM_W,-ROOM_W,-ROOM_W], [-ROOM_W, ROOM_W, ROOM_W,-ROOM_W]),
            ([ ROOM_W, ROOM_W, ROOM_W, ROOM_W], [-ROOM_W, ROOM_W, ROOM_W,-ROOM_W]),
        ]:
            verts = list(zip(xs, ys, [0, 0, CEIL_H, CEIL_H]))
            poly  = Poly3DCollection([verts], alpha=wall_alpha,
                                     facecolor=wall_col, edgecolor="#1e3a5f")
            ax.add_collection3d(poly)

        # Ceiling outline (faint)
        ax.plot([-ROOM_W, ROOM_W, ROOM_W, -ROOM_W, -ROOM_W],
                [-ROOM_W,-ROOM_W, ROOM_W, ROOM_W,-ROOM_W],
                [CEIL_H]*5, color="#1e3a5f", lw=0.8, alpha=0.3)

        # Landing-pad circle on the floor below hover target
        theta = np.linspace(0, 2*math.pi, 40)
        ax.plot(HOVER_R*1.5*np.cos(theta), HOVER_R*1.5*np.sin(theta),
                np.zeros(40), color="#22c55e", lw=1.5, alpha=0.5, ls="--")

        # Green hover-zone ring at target altitude
        ax.plot(HOVER_R*np.cos(theta), HOVER_R*np.sin(theta),
                np.full(40, TARGET_ALT), color="#22c55e", lw=2.5, alpha=0.9)
        # Faint vertical guide from landing pad to hover ring
        for t_ in theta[::8]:
            ax.plot([HOVER_R*math.cos(t_)]*2, [HOVER_R*math.sin(t_)]*2,
                    [0, TARGET_ALT], color="#22c55e", lw=0.4, alpha=0.25)
        # Target star
        ax.scatter([0], [0], [TARGET_ALT],
                   color="#22c55e", s=90, marker="*", depthshade=False, zorder=9)

        # Animated drone (drifts more in higher stages)
        noise = {1: 0.06, 2: 0.30, 3: 0.55}.get(stage, 0.06)
        drx = math.sin(t_now * 1.1) * noise
        dry = math.cos(t_now * 0.8) * noise
        drz = TARGET_ALT + math.sin(t_now * 2.0) * noise * 0.08

        # Drone body — cross frame + rotor circles
        arm = 0.38
        for ang in [45, 135, 225, 315]:
            rad = math.radians(ang)
            ex, ey = drx + arm*math.cos(rad), dry + arm*math.sin(rad)
            ax.plot([drx, ex], [dry, ey], [drz, drz], color="#7dd3fc", lw=2.5, zorder=10)
            # Rotor circle
            rr = np.linspace(0, 2*math.pi, 14)
            ax.plot(ex + 0.14*np.cos(rr), ey + 0.14*np.sin(rr),
                    [drz]*14, color="#60a5fa", lw=1.2, alpha=0.8, zorder=10)
        ax.scatter([drx], [dry], [drz],
                   color="#7dd3fc", s=60, depthshade=False, zorder=11)
        # Blue line from drone down to landing pad
        ax.plot([drx, 0], [dry, 0], [drz, TARGET_ALT],
                color="#3b82f6", lw=1.0, alpha=0.35, ls="--")

        # Wind arrows (stage 2+)
        n_arrows = {1: 0, 2: 6, 3: 10}.get(stage, 0)
        for i in range(n_arrows):
            ang  = i*(2*math.pi/n_arrows) + t_now*0.22
            r    = ROOM_W * 0.65
            ox, oy = r*math.cos(ang), r*math.sin(ang)
            ax.quiver(ox, oy, TARGET_ALT,
                      -math.cos(ang)*0.8, -math.sin(ang)*0.8, 0,
                      length=0.9, color="#f97316", alpha=0.75,
                      arrow_length_ratio=0.35, linewidth=1.5)

        # Walking user on the floor (stage 3)
        if stage == 3:
            ux = 2.2*math.sin(t_now*0.5)
            uy = 2.2*math.cos(t_now*0.5)
            # Person body (line)
            ax.plot([ux, ux], [uy, uy], [0, 1.8], color="#e879f9", lw=3, alpha=0.9)
            ax.scatter([ux], [uy], [2.0], color="#e879f9", s=80, depthshade=False)
            # Dashed tracking line from drone down to user
            ax.plot([drx, ux], [dry, uy], [drz, 2.0],
                    color="#8b5cf6", lw=0.9, ls="--", alpha=0.5)

        # Eval path overlay
        if self._sub2_eval_path:
            ep = self._sub2_eval_path
            ax.plot([p[0] for p in ep], [p[1] for p in ep], [p[2] for p in ep],
                    color="#4ade80", lw=2.2, alpha=0.9, zorder=12)
            ax.scatter([ep[-1][0]], [ep[-1][1]], [ep[-1][2]],
                       color="#4ade80", s=70, marker="D", depthshade=False, zorder=13)

        # Stage / progress overlay text (top-left of 3D pane)
        title_txt, desc_txt = STAGE_DESC.get(stage, ("", ""))
        ax.text2D(0.02, 0.97, title_txt, transform=ax.transAxes,
                  color="#7dd3fc", fontsize=10, fontweight="bold", va="top")
        ax.text2D(0.02, 0.84, desc_txt, transform=ax.transAxes,
                  color="#cbd5e1", fontsize=8, va="top")

        # Efficiency banner (bottom centre of 3D pane)
        banner = self._sub2_efficiency or self._sub2_eval_result
        if banner:
            col = ("#4ade80" if ("✓" in banner or "PASS" in banner)
                   else "#f97316" if "⏹" in banner else "#f87171")
            ax.text2D(0.5, 0.02, banner,
                      transform=ax.transAxes, ha="center",
                      color=col, fontsize=9, fontweight="bold")

        ax.set_xlim(-ROOM_W, ROOM_W); ax.set_ylim(-ROOM_W, ROOM_W)
        ax.set_zlim(0, CEIL_H)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_zticks([0, TARGET_ALT, CEIL_H])
        ax.zaxis.set_ticklabels(["floor", f"{TARGET_ALT}m\nhover", f"{CEIL_H}m"],
                                fontsize=8, color="#94a3b8")
        self._sub2_azim = (self._sub2_azim + 0.35) % 360
        ax.view_init(elev=20, azim=self._sub2_azim)

        # ── Reward chart — completed runs + current run ───────────────────────
        ax_r = self._sub2_ax_reward
        ax_r.cla()
        _mpl_dark_axes(ax_r, "PPO Reward — Training History", "Timestep", "Mean reward")

        RUN_COLS = ["#60a5fa", "#4ade80", "#fb923c", "#c084fc",
                    "#f472b6", "#facc15", "#22d3ee", "#f87171"]
        has_any = self._sub2_runs or self._sub2_history

        if not has_any:
            ax_r.text(0.5, 0.5,
                      "Press  'Start Training'  to begin.\n\n"
                      "Each run appears as a new colour.\n"
                      "Keep retraining — watch the lines\n"
                      "climb higher each session.",
                      ha="center", va="center", color="#64748b",
                      transform=ax_r.transAxes, fontsize=11)
        else:
            legend_handles = []

            # Completed runs — faded, one colour each
            for i, run in enumerate(self._sub2_runs[-7:]):
                col = RUN_COLS[i % len(RUN_COLS)]
                pts = run[::max(1, len(run) // 300)]
                rxs = [p[0] for p in pts]
                rys = [p[1] for p in pts]
                ax_r.plot(rxs, rys, color=col, lw=1.2, alpha=0.30)
                if len(rys) >= 8:
                    win = min(12, len(rys))
                    ma  = np.convolve(rys, np.ones(win)/win, mode="valid")
                    ax_r.plot(rxs[win-1:], ma, color=col, lw=2.4, alpha=0.80)
                legend_handles.append(Patch(color=col, alpha=0.9, label=f"Run {i+1}"))

            # Current active run — bright
            if self._sub2_history:
                run_idx  = len(self._sub2_runs)
                cur_col  = RUN_COLS[run_idx % len(RUN_COLS)]
                stage_col = {1: cur_col, 2: cur_col, 3: cur_col}

                pts = self._sub2_history[::max(1, len(self._sub2_history) // 400)]
                xs  = [p[0] for p in pts]
                ys  = [p[1] for p in pts]
                ss  = [p[2] for p in pts]

                ax_r.plot(xs, ys, color=cur_col, lw=1.6, alpha=0.5)
                if len(ys) >= 8:
                    win = min(15, len(ys))
                    ma  = np.convolve(ys, np.ones(win)/win, mode="valid")
                    ax_r.plot(xs[win-1:], ma, color=cur_col, lw=3.0, alpha=1.0)

                ax_r.set_xlim(0, max(self._sub2_total, xs[-1]))

                # ETA title
                if self._sub2_start_time and self._sub2_step > 100:
                    elapsed   = time.time() - self._sub2_start_time
                    rate      = self._sub2_step / elapsed
                    remaining = max(0, self._sub2_total - self._sub2_step)
                    eta_sec   = remaining / max(rate, 1)
                    eta_str   = (f"{int(eta_sec//3600)}h {int((eta_sec%3600)//60)}m"
                                 if eta_sec > 3600
                                 else f"{int(eta_sec//60)}m {int(eta_sec%60)}s")
                    start = self._sub2_run_start_step or self._sub2_step
                    steps_this_run = max(0, self._sub2_step - start)
                    pct = min(100, 100 * steps_this_run / max(1, self._sub2_total))
                    ax_r.set_title(
                        f"Run {run_idx+1}  ·  {pct:.0f}%  ·  ~{eta_str} left  ·  {rate:.0f} steps/s",
                        color="#94a3b8", fontsize=11, pad=5)

                # Trend + explanation text
                if len(ys) >= 20:
                    trend  = ys[-1] - ys[max(0, len(ys) - 20)]
                    latest = ys[-1]
                    if latest < -800:
                        expl, t_col = "Adjusting weights — dip is normal, reward rises after", "#f97316"
                    elif trend > 50:
                        expl, t_col = "↑ Improving fast", "#22c55e"
                    elif trend > 0:
                        expl, t_col = "↑ Improving", "#22c55e"
                    else:
                        expl, t_col = "→ Flat — run more steps or retrain", "#f97316"
                    ax_r.text(0.98, 0.04, expl, transform=ax_r.transAxes,
                              ha="right", color=t_col, fontsize=11, fontweight="bold")

                # Dip explanation annotation
                if len(ys) > 5 and ys[0] > -400 and min(ys) < ys[0] - 300:
                    ax_r.annotate(
                        "Initial dip: fine-tuning temporarily\n"
                        "disrupts existing weights before\n"
                        "settling on a better policy",
                        xy=(xs[ys.index(min(ys))], min(ys)),
                        xytext=(0.55, 0.12), textcoords="axes fraction",
                        color="#94a3b8", fontsize=9,
                        arrowprops=dict(arrowstyle="->", color="#475569", lw=1),
                    )

                legend_handles.append(
                    Patch(color=cur_col, label=f"Run {run_idx+1} (active)"))

            if legend_handles:
                ax_r.legend(handles=legend_handles,
                            facecolor="#0f172a", edgecolor=_MPL_EDGE,
                            labelcolor="#e2e8f0", fontsize=10,
                            loc="upper left", framealpha=0.85)

        self._sub2_canvas.draw_idle()

    # ── Sub-4 controls & plot ─────────────────────────────────────────────────

    def _start_sub4(self):
        if self._sub4_training:
            return
        self._sub4_deltas.clear()
        self._sub4_iters = 0
        self._sub4_stop.clear()
        self._sub4_thread = _MDPSolverWorker(self._sub4_queue, self._sub4_stop)
        self._sub4_thread.start()
        self._sub4_training = True
        self._sub4_solve_btn.configure(state="disabled")
        self._sub4_stop_btn.configure(state="normal", bg="#dc2626", fg="#ffffff",
                                       activebackground="#b91c1c")
        self._sub4_status_var.set("Running value iteration…")

    def _stop_sub4(self):
        if not self._sub4_training:
            return
        self._sub4_stop.set()
        self._sub4_stop_btn.configure(state="disabled")

    def _update_sub4_plot(self):
        try:
            from sub4_nav.obstacle_env import (ROOM_W, ROOM_D, OBSTACLES, GOAL_XY,
                                               N_LIDAR, LIDAR_R, NAV_ALT, WALL_H)
        except ImportError:
            ROOM_W, ROOM_D = 10.0, 8.0
            OBSTACLES = [(-2.5,2.5,0.35),(-2.5,-2.5,0.35),(0,1.5,0.4),
                         (0,-1.5,0.4),(0,0,0.3),(2.5,2.5,0.35),(2.5,-2.5,0.35)]
            GOAL_XY = (4.5, 0.0); N_LIDAR = 8; LIDAR_R = 5.0
            NAV_ALT = 1.5; WALL_H = 3.5

        # ── 3D Obstacle Room ─────────────────────────────────────────────────
        ax = self._sub4_ax_3d
        ax.cla()
        self._style_3d_ax(ax, "Obstacle Navigation Room")

        hw, hd = ROOM_W / 2, ROOM_D / 2
        theta = np.linspace(0, 2 * math.pi, 32)

        # Room floor outline
        ax.plot([-hw, hw, hw, -hw, -hw], [-hd, -hd, hd, hd, -hd],
                [0]*5, color="#1e3a5f", lw=1.5)

        # Vertical corner edges and top frame
        for cx, cy in [(-hw,-hd),(-hw,hd),(hw,-hd),(hw,hd)]:
            ax.plot([cx,cx],[cy,cy],[0,WALL_H], color="#1e3a5f", lw=1.0, alpha=0.5)
        ax.plot([-hw,hw,hw,-hw,-hw],[-hd,-hd,hd,hd,-hd],
                [WALL_H]*5, color="#1e3a5f", lw=0.7, alpha=0.35)

        # Red obstacle cylinders
        for ox, oy, r in OBSTACLES:
            ax.plot(ox + r*np.cos(theta), oy + r*np.sin(theta),
                    np.zeros(32), color="#dc2626", lw=1.5, alpha=0.9)
            ax.plot(ox + r*np.cos(theta), oy + r*np.sin(theta),
                    np.full(32, WALL_H), color="#dc2626", lw=0.8, alpha=0.4)
            for t_ in theta[::4]:
                ax.plot([ox+r*math.cos(t_)]*2, [oy+r*math.sin(t_)]*2,
                        [0, WALL_H], color="#dc2626", lw=0.7, alpha=0.35)

        # Goal zone ring at nav altitude
        ax.plot(GOAL_XY[0] + 0.9*np.cos(theta), GOAL_XY[1] + 0.9*np.sin(theta),
                np.full(32, NAV_ALT), color="#22c55e", lw=2.5)
        ax.scatter([GOAL_XY[0]], [GOAL_XY[1]], [NAV_ALT],
                   color="#22c55e", s=80, marker="*", depthshade=False, zorder=9)

        # Start arrow
        ax.quiver(-hw+0.3, 0, NAV_ALT, 0.9, 0, 0,
                  length=1.0, color="#60a5fa", arrow_length_ratio=0.4, linewidth=2)

        # Live drone + lidar + trail
        if self._nav_viz:
            viz   = self._nav_viz
            dxy   = viz.get("drone_xy", [0.0, 0.0])
            dx, dy = float(dxy[0]), float(dxy[1])
            path  = viz.get("path", [])
            if len(path) > 1:
                ax.plot([p[0] for p in path], [p[1] for p in path],
                        [NAV_ALT]*len(path), color="#3b82f6", lw=1.2, alpha=0.6)
            for i, frac in enumerate(viz.get("lidar", [])):
                ang = i * (2 * math.pi / N_LIDAR)
                ex = dx + math.cos(ang) * frac * LIDAR_R
                ey = dy + math.sin(ang) * frac * LIDAR_R
                ax.plot([dx, ex], [dy, ey], [NAV_ALT, NAV_ALT],
                        color="#f97316", lw=1.0, alpha=0.75)
            ax.scatter([dx], [dy], [NAV_ALT],
                       color="#7dd3fc", s=160, depthshade=False, zorder=10)
        else:
            ax.scatter([-hw+0.5], [0], [NAV_ALT], color="#7dd3fc", s=120,
                       marker="o", depthshade=False)
            ax.text2D(0.5, 0.05, "Click 'Train Navigation' (SAC) — drone + lidar rays appear live",
                      transform=ax.transAxes, ha="center", color="#94a3b8", fontsize=9)

        # Evaluation path overlay (bright green — best eval episode)
        if self._nav_eval_path:
            ep = self._nav_eval_path
            ax.plot([p[0] for p in ep], [p[1] for p in ep],
                    [NAV_ALT] * len(ep), color="#4ade80", lw=2.2, alpha=0.95, zorder=11)
            ax.scatter([ep[-1][0]], [ep[-1][1]], [NAV_ALT],
                       color="#4ade80", s=80, marker="D", depthshade=False, zorder=12)
        nav_banner = self._nav_efficiency or self._nav_eval_result
        if nav_banner:
            col = ("#4ade80" if ("✓" in nav_banner or "PASS" in nav_banner)
                   else "#f97316" if "⏹" in nav_banner else "#f87171")
            ax.text2D(0.5, 0.01, nav_banner,
                      transform=ax.transAxes, ha="center",
                      color=col, fontsize=9, fontweight="bold")

        ax.set_xlim(-hw, hw); ax.set_ylim(-hd, hd); ax.set_zlim(0, WALL_H)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_zticks([0, NAV_ALT, WALL_H])
        ax.zaxis.set_ticklabels(["0", f"{NAV_ALT}m", f"{WALL_H}m"],
                                fontsize=8, color="#94a3b8")
        self._sub4_azim = (self._sub4_azim + 0.4) % 360
        ax.view_init(elev=20, azim=self._sub4_azim)

        # ── Nav reward curve (right) ──────────────────────────────────────────
        ax_r = self._sub4_ax_reward
        ax_r.cla()
        _mpl_dark_axes(ax_r, "Nav SAC Reward Curve", "Timestep", "Mean reward")
        if self._nav_history:
            xs = [p[0] for p in self._nav_history]
            ys = [p[1] for p in self._nav_history]

            # "Goal reached" threshold line — reward > 0 means goal bonus kicking in
            ax_r.axhline(0, color="#22c55e", ls="--", lw=1.2, alpha=0.6,
                         label="Goal reached zone (reward > 0)")
            ax_r.axhline(-150, color="#dc2626", ls="--", lw=0.8, alpha=0.4,
                         label="Stuck / crashing zone")

            ax_r.plot(xs, ys, color="#14b8a6", lw=2.0)
            ax_r.set_xlim(0, max(self._nav_total, xs[-1]))

            # Trend label
            if len(ys) >= 5:
                latest = ys[-1]
                if latest > 50:
                    lbl, col = "✓ Goal being reached consistently!", "#4ade80"
                elif latest > 0:
                    lbl, col = "↑ Approaching goal zone — keep going", "#86efac"
                elif ys[-1] > ys[max(0, len(ys)-10)]:
                    lbl, col = "↑ Learning — reward rising", "#fde68a"
                else:
                    lbl, col = "Still exploring — reward will rise", "#94a3b8"
                ax_r.text(0.98, 0.05, lbl, transform=ax_r.transAxes,
                          ha="right", color=col, fontsize=11, fontweight="bold")

            ax_r.legend(facecolor="#0f172a", edgecolor=_MPL_EDGE,
                        labelcolor="#94a3b8", fontsize=9, loc="upper left")
        else:
            ax_r.text(0.5, 0.5,
                      "SAC reward curve appears here during training.\n\n"
                      "Reward > 0  →  drone is reaching the goal\n"
                      "Reward < 0  →  still learning to navigate",
                      ha="center", va="center", color="#475569",
                      transform=ax_r.transAxes, fontsize=10)

        self._sub4_canvas.draw_idle()

    # ── Evaluation controls ───────────────────────────────────────────────────

    def _run_sub2_eval(self):
        if self._sub2_eval_active or self._sub2_training:
            return
        from sub2_flight.eval_worker import FlightEvalWorker
        if not os.path.exists(_FlightTrainingWorker.OUTPUT_PATH + ".zip"):
            self._sub2_status_var.set("No trained model found — train PPO first.")
            return
        self._sub2_eval_stop.clear()
        self._sub2_eval_thread = FlightEvalWorker(self._sub2_eval_queue,
                                                   self._sub2_eval_stop)
        self._sub2_eval_thread.start()
        self._sub2_eval_active = True
        self._sub2_eval_path   = []
        self._sub2_eval_result = ""
        self._sub2_eval_btn.configure(state="disabled", text="Evaluating…")
        self._sub2_status_var.set("Running 5 evaluation episodes…")

    def _run_nav_eval(self):
        if self._nav_eval_active or self._nav_training:
            return
        from sub4_nav.eval_worker import NavEvalWorker
        if not _artifact_ready(self._NAV_PATH, 1024):
            self._nav_status_var.set("No trained model found — train navigation first.")
            return
        self._nav_eval_stop.clear()
        self._nav_eval_thread = NavEvalWorker(self._nav_eval_queue,
                                               self._nav_eval_stop)
        self._nav_eval_thread.start()
        self._nav_eval_active = True
        self._nav_eval_path   = []
        self._nav_eval_result = ""
        self._nav_eval_btn.configure(state="disabled", text="Evaluating…")
        self._nav_status_var.set("Running 5 evaluation episodes…")

    # ── Sub-4 nav training controls ───────────────────────────────────────────

    def _retrain_nav(self):
        """Delete the existing SAC nav model and train from scratch."""
        if self._nav_training:
            return
        nav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models", "ppo_nav_v1")
        for p in [nav_path, nav_path + ".zip"]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        self._nav_ready = False
        self._nav_efficiency = ""
        self._nav_status_var.set("Deleted previous model — starting from scratch…")
        self._start_nav()

    def _start_nav(self, scenario=None):
        if self._nav_training:
            return
        from sub4_nav.nav_training_worker import NavTrainingWorker
        try:
            total = max(2048, int(self._nav_steps_var.get()))
        except ValueError:
            total = 12_000
        self._nav_total = total
        self._nav_history.clear()
        self._nav_viz  = None
        self._nav_step = 0
        self._nav_stop.clear()
        self._nav_thread = NavTrainingWorker(total, self._nav_queue, self._nav_stop,
                                             scenario=scenario)
        self._nav_thread.start()
        self._nav_training   = True
        self._nav_efficiency = ""
        self._nav_entry.configure(state="disabled")
        self._nav_start_btn.configure(state="disabled"); self._nav_retrain_btn.configure(state="disabled")
        self._nav_stop_btn.configure(state="normal", bg="#dc2626", fg="#ffffff",
                                     activebackground="#b91c1c")
        warm_note = "SAC fine-tuning from existing model…" if _artifact_ready(self._NAV_PATH, 1024) else "SAC training from scratch…"
        eta = "~15-30 sec" if total <= 15_000 else f"~{max(1, total // 50000)} min"
        self._nav_status_var.set(
            f"{warm_note}  {eta} on CPU  "
            "· SAC off-policy (replay buffer) · auto-entropy exploration")

    def _stop_nav(self):
        if not self._nav_training:
            return
        self._nav_stop.set()
        self._nav_stop_btn.configure(state="disabled")
        self._nav_status_var.set("Stopping…")

    # ── Main tick — drain queues, step Sub-1, refresh plots ──────────────────

    def _tick(self):
        if self.closed:
            return

        changed2 = changed4 = False

        # Which tab is the user currently looking at? (0=Sub-1, 1=Sub-2, 2=Sub-3, 3=Sub-4)
        try:
            active_tab = self._nb.index(self._nb.select())
        except Exception:
            active_tab = -1

        # Sub-1 camera step only runs when that tab is visible (it's expensive)
        if active_tab == 0 and self._sub1_running:
            self._sub1_step()

        # Drain Sub-2 queue
        while True:
            try:
                msg = self._sub2_queue.get_nowait()
            except queue.Empty:
                break
            k = msg[0]
            if k == "progress":
                _, ts, stage, mr = msg
                self._sub2_history.append((ts, mr, stage))
                self._sub2_step = ts
                if self._sub2_run_start_step is None:
                    self._sub2_run_start_step = ts   # record starting offset
                changed2 = True
            elif k in ("done", "stopped"):
                # New format: (kind, path, steps, efficiency_pct, was_warm)
                parts = msg[1:]
                _path, steps = parts[0], parts[1]
                eff  = parts[2] if len(parts) > 2 else 0
                warm = parts[3] if len(parts) > 3 else False
                self._sub2_training = False
                self._sub2_ready = _artifact_ready(self._PPO_PATH, 1024)
                # Archive completed run for multi-line history
                if self._sub2_history:
                    self._sub2_runs.append(list(self._sub2_history))
                    self._sub2_history = []
                    # Persist so history survives hub restarts (keep last 10 runs)
                    _th_data = _load_json_file(_TRAINING_HIST_PATH, {})
                    _th_data["sub2_runs"] = [list(r) for r in self._sub2_runs[-10:]]
                    _save_json_file(_TRAINING_HIST_PATH, _th_data)
                self._sub2_entry.configure(state="normal")
                run_n = len(self._sub2_runs)
                btn_lbl = f"Fine-tune  (Run {run_n + 1})" if self._sub2_ready else "Start Training"
                self._sub2_start_btn.configure(state="normal", text=btn_lbl)
                self._sub2_retrain_btn.configure(state="normal")
                self._sub2_stop_btn.configure(state="disabled", bg="#334155",
                                               fg="#94a3b8", activebackground="#475569")
                warm_str = "fine-tuned from previous model" if warm else "trained from scratch"
                eff_str  = (f"predicted hover efficiency  ~{eff}%"
                            if k == "done" else "partial save")
                self._sub2_efficiency = (
                    f"✓ Run {run_n} done — {eff_str}  ({warm_str})" if k == "done"
                    else f"⏹ Run {run_n} stopped — {eff_str}")
                self._sub2_status_var.set(self._sub2_efficiency)
                changed2 = True
            elif k == "error":
                self._sub2_training = False
                if self._auto_stage == 0:
                    self._auto_failed_msg = f"Sub-2 Flight PPO error: {msg[1][:100]}"
                self._sub2_entry.configure(state="normal")
                self._sub2_start_btn.configure(state="normal")
                self._sub2_retrain_btn.configure(state="normal")
                self._sub2_stop_btn.configure(state="disabled", bg="#334155",
                                               fg="#94a3b8", activebackground="#475569")
                self._sub2_status_var.set(f"Error: {msg[1][:120]}")
                changed2 = True

        # Drain Sub-4 queue
        while True:
            try:
                msg = self._sub4_queue.get_nowait()
            except queue.Empty:
                break
            k = msg[0]
            if k == "iter":
                _, itr, delta = msg
                self._sub4_deltas.append(delta)
                self._sub4_iters = itr
                changed4 = True
            elif k in ("done", "stopped"):
                _, policy, _vals = msg
                self._sub4_policy  = policy
                self._sub4_training = False
                self._sub4_ready   = _artifact_ready(self._MDP_PATH)
                self._sub4_solve_btn.configure(state="normal")
                self._sub4_stop_btn.configure(state="disabled", bg="#334155",
                                               fg="#94a3b8", activebackground="#475569")
                verb = "Solved" if k == "done" else "Stopped"
                self._sub4_status_var.set(
                    f"{verb} — {self._sub4_iters} iterations. {self._model_label(4)}")
                changed4 = True
            elif k == "error":
                self._sub4_training = False
                if self._auto_stage == 2:
                    self._auto_failed_msg = f"Sub-4 Battery MDP error: {msg[1][:100]}"
                self._sub4_solve_btn.configure(state="normal")
                self._sub4_stop_btn.configure(state="disabled", bg="#334155",
                                               fg="#94a3b8", activebackground="#475569")
                self._sub4_status_var.set(f"Error: {msg[1][:120]}")
                changed4 = True

        # Drain nav queue
        changed_nav = False
        while True:
            try:
                msg = self._nav_queue.get_nowait()
            except queue.Empty:
                break
            k = msg[0]
            if k == "progress":
                _, ts, mr, viz = msg
                if ts == -1:
                    # Worker detected old PPO model — deleted it, restarting fresh
                    self._nav_status_var.set(
                        "⚠ Old PPO model was incompatible with SAC — deleted automatically. "
                        "Training fresh SAC model from scratch…")
                else:
                    self._nav_history.append((ts, mr))
                    self._nav_step = ts
                    self._nav_viz  = viz
                    changed_nav = True
            elif k in ("done", "stopped"):
                parts = msg[1:]
                _path, steps = parts[0], parts[1]
                eff  = parts[2] if len(parts) > 2 else 0
                warm = parts[3] if len(parts) > 3 else False
                self._nav_training = False
                self._nav_ready    = _artifact_ready(self._NAV_PATH, 1024)
                # Persist nav history
                if self._nav_history:
                    _th_data = _load_json_file(_TRAINING_HIST_PATH, {})
                    _th_data["nav_history_last"] = list(self._nav_history[-500:])
                    _save_json_file(_TRAINING_HIST_PATH, _th_data)
                self._nav_entry.configure(state="normal")
                self._nav_start_btn.configure(state="normal")
                self._nav_retrain_btn.configure(state="normal")
                self._nav_stop_btn.configure(state="disabled", bg="#334155",
                                              fg="#94a3b8", activebackground="#475569")
                warm_str = "SAC fine-tuned from previous model" if warm else "SAC trained from scratch"
                eff_str  = (f"predicted navigation success  ~{eff}%"
                            if k == "done" else "partial save")
                self._nav_efficiency = (
                    f"✓ SAC model trained — {eff_str}  ({warm_str})" if k == "done"
                    else f"⏹ Stopped at {steps:,} steps — {eff_str}")
                if not self._nav_ready:
                    self._nav_efficiency = (
                        "⚠ Nav SAC did not produce a valid model file. "
                        "Run Train Navigation again.")
                    if self._auto_stage == 3:
                        self._auto_failed_msg = "Sub-4 Nav SAC did not save a valid model."
                self._nav_status_var.set(self._nav_efficiency)
                changed_nav = True
            elif k == "error":
                self._nav_training = False
                if self._auto_stage == 3:
                    self._auto_failed_msg = f"Sub-4 Nav SAC error: {msg[1][:100]}"
                self._nav_entry.configure(state="normal")
                self._nav_start_btn.configure(state="normal")
                self._nav_retrain_btn.configure(state="normal")
                self._nav_stop_btn.configure(state="disabled", bg="#334155",
                                              fg="#94a3b8", activebackground="#475569")
                self._nav_status_var.set(f"Error: {msg[1][:100]}")
                changed_nav = True

        # Drain Sub-3 SVM queue
        while True:
            try:
                msg = self._sub3_queue.get_nowait()
            except queue.Empty:
                break
            k = msg[0]
            if k == "progress":
                self._sub3_status_var.set(msg[1])
            elif k == "done":
                _, path, acc, cm = msg
                self._sub3_training  = False
                self._sub3_ready     = _artifact_ready(self._SVM_PATH, 256)
                self._sub3_accuracy  = acc
                self._sub3_cm        = cm
                self._sub3_clf       = None   # reload on next tick
                self._sub3_train_btn.configure(state="normal", text="Train SVM")
                self._sub3_status_var.set(
                    f"✓ SVM trained — CV accuracy {acc:.1%}  "
                    f"({'≥90% target met' if acc >= 0.9 else 'below 90% target'})")
            elif k == "error":
                self._sub3_training = False
                if self._auto_stage == 1:
                    self._auto_failed_msg = f"Sub-3 Weather SVM error: {msg[1][:100]}"
                self._sub3_train_btn.configure(state="normal", text="Train SVM")
                self._sub3_status_var.set(f"Error: {msg[1][:100]}")

        # Only redraw the visible tab — 3D matplotlib renders are expensive;
        # drawing all three every tick caused severe lag.
        if active_tab == 1:
            self._update_sub2_plot()
        elif active_tab == 2:
            self._update_sub3_plot()
        elif active_tab == 3:
            self._update_sub4_plot()
        elif active_tab == 0 and self._sub1_running:
            pass  # Sub-1 plot updated inline in _sub1_step()

        # Drain Sub-2 eval queue
        while True:
            try:
                msg = self._sub2_eval_queue.get_nowait()
            except queue.Empty:
                break
            k = msg[0]
            if k == "eval_ep":
                _, ep, rw, hr, passed, path = msg
                if path:
                    self._sub2_eval_path = path
                pct = int(hr * 100)
                mark = "✓" if passed else "✗"
                self._sub2_status_var.set(
                    f"Eval ep {ep}/5  {mark}  reward {rw:+.0f}  hover {pct}%  {self._sub2_eval_result}")
            elif k == "eval_done":
                _, rewards, passes, best_path = msg
                self._sub2_eval_active = False
                self._sub2_eval_path   = best_path
                n_pass = sum(passes)
                mean_r = float(np.mean(rewards)) if rewards else 0.0
                verdict = "PASS ✓" if n_pass >= 3 else "FAIL ✗"
                self._sub2_eval_result = (
                    f"{verdict}  {n_pass}/5 episodes hovered  mean reward {mean_r:+.0f}")
                self._sub2_status_var.set("Evaluation complete — " + self._sub2_eval_result)
                label = f"Evaluate Model  ({n_pass}/5 ✓)" if n_pass >= 3 else f"Evaluate Model  ({n_pass}/5 ✗)"
                self._sub2_eval_btn.configure(state="normal", text=label)
            elif k == "eval_error":
                self._sub2_eval_active = False
                self._sub2_eval_btn.configure(state="normal", text="Evaluate Model")
                self._sub2_status_var.set(f"Eval error: {msg[1][:80]}")

        # Drain Nav eval queue
        while True:
            try:
                msg = self._nav_eval_queue.get_nowait()
            except queue.Empty:
                break
            k = msg[0]
            if k == "eval_ep":
                _, ep, rw, reached, steps, path = msg
                if path:
                    self._nav_eval_path = path
                mark = "✓ GOAL" if reached else "✗ miss"
                self._nav_status_var.set(
                    f"Eval ep {ep}/5  {mark}  reward {rw:+.0f}  {steps} steps")
            elif k == "eval_done":
                _, rewards, successes, best_path = msg
                self._nav_eval_active = False
                self._nav_eval_path   = best_path
                n_pass  = sum(successes)
                mean_r  = float(np.mean(rewards)) if rewards else 0.0
                verdict = "PASS ✓" if n_pass >= 3 else "FAIL ✗"
                self._nav_eval_result = (
                    f"{verdict}  {n_pass}/5 episodes reached goal  mean reward {mean_r:+.0f}")
                self._nav_status_var.set("Evaluation complete — " + self._nav_eval_result)
                label = f"Evaluate Model  ({n_pass}/5 ✓)" if n_pass >= 3 else f"Evaluate Model  ({n_pass}/5 ✗)"
                self._nav_eval_btn.configure(state="normal", text=label)
            elif k == "eval_error":
                self._nav_eval_active = False
                self._nav_eval_btn.configure(state="normal", text="Evaluate Model")
                self._nav_status_var.set(f"Eval error: {msg[1][:80]}")

        self._gate_var.set(self._gate_text())
        # Advance auto-train sequence if active
        if self._auto_stage >= 0:
            self._auto_tick()
        self.window.after(350, self._tick)

    # _do_launch removed — hub is training-only; sim launch uses main Launcher

    def close(self):
        self._sub2_stop.set()
        self._sub3_stop.set()
        self._sub4_stop.set()
        self._nav_stop.set()
        self._sub2_eval_stop.set()
        self._nav_eval_stop.set()
        if self._sub1_perc_env is not None:
            try:
                self._sub1_perc_env.close()
            except Exception:
                pass
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
        self.flight_training_window = None
        self.training_hub = None
        self.training_model_status_vars = {}

        self.root = tk.Tk()
        self.root.title("SkyShade Launcher")
        self.root.configure(bg="#0b1120")
        self.root.geometry("1180x760")
        self.root.minsize(1040, 680)
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._cancel)

        self.scenario_var = tk.StringVar(value=SCENARIO_PARK)
        self.flight_var = tk.StringVar(value="ppo")
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
            text="PPO flight model active",
            fg="#f8fafc",
            bg="#0f172a",
            font=("Arial", 12, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        tk.Label(
            flight_card,
            text="Uses ppo_flight_v1.zip. If PPO is unavailable, runtime falls back to the legacy Q-table, then PID.",
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
            value="Click any card to train individually, or use Auto-Train to train all systems at once."
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

        # ── Auto-Train buttons ─────────────────────────────────────────────────
        auto_frame = tk.Frame(panel, bg="#0b1120")
        auto_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        auto_frame.grid_columnconfigure(0, weight=1)
        auto_frame.grid_columnconfigure(1, weight=1)

        tk.Button(
            auto_frame,
            text="🚀  Auto-Train All  (quick)",
            command=lambda: self._launch_auto_train(retrain=False),
            bg="#166534", fg="#dcfce7",
            activebackground="#14532d", activeforeground="#ffffff",
            relief="flat", font=("Arial", 11, "bold"),
            padx=14, pady=10,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))

        tk.Button(
            auto_frame,
            text="🔄  Auto-Retrain from Scratch",
            command=lambda: self._launch_auto_train(retrain=True),
            bg="#78350f", fg="#fef3c7",
            activebackground="#92400e", activeforeground="#ffffff",
            relief="flat", font=("Arial", 11, "bold"),
            padx=14, pady=10,
        ).grid(row=0, column=1, sticky="ew")

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

        status_var = tk.StringVar(value=self._model_status_for_key(key))
        self.training_model_status_vars[key] = status_var
        tk.Label(
            card,
            textvariable=status_var,
            fg="#94a3b8",
            bg="#111827",
            font=("Arial", 9),
            anchor="w",
            justify="left",
            wraplength=360 if not compact else 255,
        ).grid(row=3, column=0, sticky="ew", pady=(5, 0))

        _BTN = {
            "camera":      ("Calibrate",  "#166534", "#dcfce7"),
            "flight":      ("Train PPO",  "#1e3a8a", "#bfdbfe"),
            "weather":     ("Train SVM",  "#831843", "#fce7f3"),
            "safety":      ("Solve MDP",  "#4c1d95", "#ddd6fe"),
            "environment": ("Select",     "#1e293b", "#94a3b8"),
        }
        btn_label, btn_bg, btn_fg = _BTN.get(key, ("Open", "#0f172a", "#e0f2fe"))
        tk.Button(
            card,
            text=btn_label,
            command=lambda selected=key: self._open_training_ground(selected),
            bg=btn_bg,
            fg=btn_fg,
            activebackground="#334155",
            activeforeground="#ffffff",
            relief="flat",
            font=("Arial", 9, "bold"),
            padx=10,
            pady=6,
        ).grid(row=0, column=1, rowspan=4, sticky="e", padx=(8, 0))

    def _model_status_for_key(self, key):
        repo_root = os.path.dirname(os.path.abspath(__file__))
        specs = {
            "camera": None,
            "flight": ("models/ppo_flight_v1.zip", "runtime active"),
            "weather": ("models/svm_v1.pkl", "runtime active"),
        }
        if key == "camera":
            return "No saved model: deterministic CV calibration."
        if key == "environment":
            return "Stress scene available: Building District."
        if key == "safety":
            mdp_path = os.path.join(repo_root, "models/policy_table_v1.npy")
            nav_path = os.path.join(repo_root, "models/ppo_nav_v1.zip")
            mdp_ok = _artifact_ready(mdp_path)
            nav_ok = _artifact_ready(nav_path, 1024)
            if mdp_ok and nav_ok:
                nav_meta = self._read_model_meta(nav_path)
                if nav_meta and "efficiency_pct" in nav_meta:
                    return f"Trained: MDP + SAC nav (eff {int(nav_meta['efficiency_pct'])}%)."
                return "Trained: MDP battery table + SAC navigation model."
            missing = []
            if not mdp_ok:
                missing.append("policy_table_v1.npy")
            if not nav_ok:
                missing.append("ppo_nav_v1.zip")
            return "Missing " + ", ".join(missing) + "."

        spec = specs.get(key)
        if not spec:
            return ""
        rel_path, role = spec
        path = os.path.join(repo_root, rel_path)
        min_bytes = 1024 if path.endswith(".zip") else 256
        if not _artifact_ready(path, min_bytes):
            return f"Missing {os.path.basename(path)}."

        meta = self._read_model_meta(path)
        if meta:
            if "efficiency_pct" in meta:
                metric = f"eff {int(meta['efficiency_pct'])}%"
            elif "cv_accuracy" in meta:
                metric = f"CV {float(meta['cv_accuracy']) * 100:.1f}%"
            else:
                metric = "trained"
            return f"Trained: {meta.get('algorithm', os.path.basename(path))} ({metric}, {role})."

        age_seconds = max(0.0, time.time() - os.path.getmtime(path))
        if age_seconds < 3600:
            age = f"{int(age_seconds // 60)} min ago"
        elif age_seconds < 86400:
            age = f"{int(age_seconds // 3600)} hr ago"
        else:
            age = f"{int(age_seconds // 86400)} days ago"
        return f"Trained: {os.path.basename(path)} ({age}, {role})."

    @staticmethod
    def _read_model_meta(model_path):
        meta_candidates = [model_path + ".meta.json"]
        if model_path.endswith(".zip"):
            meta_candidates.append(model_path[:-4] + ".meta.json")
        meta_path = next((path for path in meta_candidates if os.path.exists(path)), None)
        if meta_path is None:
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError, TypeError):
            return None

    def _refresh_model_statuses(self):
        for key, var in list(self.training_model_status_vars.items()):
            try:
                var.set(self._model_status_for_key(key))
            except tk.TclError:
                pass

    def _launch_auto_train(self, retrain: bool = False):
        """Open the hub and immediately start the auto-train sequence."""
        if self.training_hub is None or self.training_hub.closed:
            self.training_hub = TrainingGroundsHub(self.root)
        else:
            self.training_hub.window.lift()
        scenario = None
        try:
            scenario = self.scenario_var.get()
        except (AttributeError, tk.TclError):
            pass
        self.training_hub.start_auto_train(retrain=retrain,
                                            on_complete=self._on_auto_train_complete,
                                            scenario=scenario)

    def _on_auto_train_complete(self):
        """Called by the hub when all auto-training is done."""
        try:
            if self.training_status_var:
                self.training_status_var.set(
                    "✓ All systems trained! Select a scenario and click Launch.")
            self._refresh_model_statuses()
        except tk.TclError:
            pass

    def _open_training_hub(self, tab_idx: int = 0):
        """Open (or focus) the unified TrainingGroundsHub and select a tab."""
        if self.training_hub is None or self.training_hub.closed:
            self.training_hub = TrainingGroundsHub(self.root)
        else:
            self.training_hub.window.lift()
        self.training_hub.select_tab(tab_idx)



    def _open_training_ground(self, key):
        tab_map = {"camera": 0, "flight": 1, "weather": 2, "safety": 3}
        self._open_training_hub(tab_map.get(key, 0))

        if key == "environment":
            self._set_scenario(SCENARIO_BUILDINGS)

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

    def _missing_models(self) -> list:
        """Return list of subsystem names whose model files are absent."""
        missing = []
        ppo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models", "ppo_flight_v1.zip")
        mdp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models", "policy_table_v1.npy")
        svm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models", "svm_v1.pkl")
        nav_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models", "ppo_nav_v1.zip")
        if not _artifact_ready(ppo_path, 1024):
            missing.append("Sub-2 Flight PPO (ppo_flight_v1.zip)")
        if not _artifact_ready(svm_path, 256):
            missing.append("Sub-3 Weather SVM (svm_v1.pkl)")
        if not _artifact_ready(mdp_path):
            missing.append("Sub-4 Nav Safety (policy_table_v1.npy)")
        if not _artifact_ready(nav_path, 1024):
            missing.append("Sub-4 Navigation SAC (ppo_nav_v1.zip)")
        return missing

    def _launch(self):
        try:
            duration = float(self.duration_var.get())
            if duration <= 0:
                raise ValueError
        except ValueError:
            self.error_var.set("Enter a positive duration in seconds.")
            return

        missing = self._missing_models()
        if missing:
            msg = ("The following subsystems are not trained:\n\n  • "
                   + "\n  • ".join(missing)
                   + "\n\nOpen Training Grounds to train them, or launch anyway?")
            proceed = tk_messagebox.askyesno(
                "Models not trained", msg,
                icon="warning", default="no", parent=self.root,
            )
            if not proceed:
                self._open_training_hub(1 if "Sub-2" in missing[0] else 2)
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
            "flight": "ppo",
            "duration": default_duration,
            "gui": True,
        }

    try:
        return ScenarioLauncher(default_duration).show()
    except tk.TclError as exc:
        print(f"Launcher unavailable: {exc}")
        return {
            "scenario": SCENARIO_PARK,
            "flight": "ppo",
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
        settings.get("flight", "ppo"),
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
        self._weather_rng = np.random.default_rng()
        # Load persistent run history for ghost-line overlays
        self._run_history = _load_json_file(_RUN_HISTORY_PATH, [])
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
        self.vision_window.geometry("1420x1300+10+10")
        self.vision_window.minsize(1320, 1100)
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
            wraplength=1380,
            padx=14, pady=8,
        ).grid(row=3, column=0, sticky="ew")

        self.graph_canvas = tk.Canvas(
            self.vision_window,
            width=1380,
            height=520,
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
            wraplength=1380,
            justify="left",
        ).grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 14))

        self.vision_window.grid_rowconfigure(1, weight=2)
        self.vision_window.grid_rowconfigure(4, weight=3)
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
        weather=None, t_wall=0.0,
    ):
        if self.root is None or self.closed:
            return

        if self.vision_window is not None and self.vision_visible:
            try:
                self.vision_window.deiconify()
            except tk.TclError:
                return

        # Rain/wind camera overlay — gated by same flag as PyBullet visuals
        if WEATHER_VISUALS and weather and (
                weather.get("rain", 0) >= 0.05 or weather.get("wind", 0) >= 1.5):
            frame = _draw_weather_cv(frame.copy(), weather, self._weather_rng, t_wall)

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

    def _draw_series(self, canvas, values, bounds, color, label, row,
                     ghost=False, ghost_alpha_tag=""):
        if not values:
            return
        x0, y0, x1, y1 = bounds
        min_v, max_v = {
            "confidence": (0.0, 1.0),
            "hover_error": (0.0, 3.0),
            "avoidance": (0.0, AVOIDANCE_MAX_FORCE),
            "battery": (0.0, 100.0),
            "umbrella": (0.0, 1.0),
        }.get(label, (0.0, 1.0))
        span = max(max_v - min_v, 1e-6)

        # Thin out very long series so the canvas stays fast
        step = max(1, len(values) // 300)
        pts_sub = values[::step]
        points = []
        count = len(pts_sub)
        for i, v in enumerate(pts_sub):
            x = x0 + (x1 - x0) * (i / max(1, count - 1))
            norm = max(0.0, min(1.0, (v - min_v) / span))
            points.extend([x, y1 - (y1 - y0) * norm])

        if len(points) < 4:
            return

        if ghost:
            # Past-run ghost line: dimmed colour, dashed appearance via segments
            canvas.create_line(*points, fill=color, width=1,
                               smooth=True, dash=(4, 6))
        else:
            canvas.create_line(*points, fill=color, width=2, smooth=True)
            latest = values[-1]
            label_text = {
                "confidence": f"confidence: {latest:.2f}",
                "hover_error": f"hover error: {latest:.2f} m",
                "avoidance":   f"avoidance: {latest:.1f} N",
                "battery":     f"battery: {latest:.0f}%",
                "umbrella":    f"umbrella: {'deploy' if latest > 0.5 else 'stow'}",
            }.get(label, "")
            canvas.create_text(x0 + 8, y0 + 15, text=label_text,
                                fill=color, font=("Arial", 9, "bold"), anchor="w")

    def _draw_live_graphs(self):
        if self.graph_canvas is None:
            return

        c = self.graph_canvas
        c.delete("all")
        w = int(c.winfo_width() or 1120)
        h = int(c.winfo_height() or 400)
        c.create_rectangle(0, 0, w, h, fill="#0b1726", outline="")

        # ── Layout: 4 sub graphs + 1 trend row ──────────────────────────────
        # Ghost-line palette for up to 5 past runs (most recent = brightest)
        GHOST_COLS = {
            "confidence": ["#064e3b", "#065f46", "#047857", "#059669", "#10b981"],
            "hover_error":["#451a03", "#78350f", "#92400e", "#b45309", "#d97706"],
            "avoidance":  ["#082f49", "#0c4a6e", "#075985", "#0369a1", "#0284c7"],
            "battery":    ["#2e1065", "#3b0764", "#4a044e", "#6b21a8", "#7c3aed"],
            "umbrella":   ["#500724", "#881337", "#9f1239", "#be123c", "#e11d48"],
        }

        graph_specs = [
            ("confidence", "#22c55e",  "Sub-1  Marker tracking confidence"),
            ("hover_error","#f59e0b",  "Sub-2  Hover error (m)"),
            ("avoidance",  "#38bdf8",  "Sub-2  Tree avoidance force (N)"),
            ("battery",    "#a78bfa",  "Sub-4  Battery (%)"),
            ("umbrella",   "#f472b6",  "Sub-3  Umbrella decision"),
        ]
        cols, n_graph_rows = 3, 2
        trend_h   = 72      # height for the trend row at the bottom
        graph_h   = h - trend_h - 8
        gap       = 12
        cell_w    = (w - gap * (cols + 1)) / cols
        cell_h    = (graph_h - gap * (n_graph_rows + 1)) / n_graph_rows

        past_runs = self._run_history[-5:] if self._run_history else []

        for index, (key, color, title) in enumerate(graph_specs):
            col  = index % cols
            row  = index // cols
            x0   = gap + col * (cell_w + gap)
            y0   = gap + row  * (cell_h + gap)
            x1, y1 = x0 + cell_w, y0 + cell_h
            c.create_rectangle(x0, y0, x1, y1, fill="#0f172a", outline="#1e3a5f")
            c.create_text(x0 + 10, y0 + 13, text=title,
                          fill="#64748b", font=("Arial", 8, "bold"), anchor="w")
            plot = (x0 + 10, y0 + 26, x1 - 10, y1 - 8)
            # Grid lines
            for gi in range(1, 3):
                gy = plot[1] + (plot[3] - plot[1]) * gi / 3
                c.create_line(plot[0], gy, plot[2], gy, fill="#172554", dash=(2, 6))

            # Ghost lines — past runs, oldest = most faded
            ghost_palette = GHOST_COLS.get(key, ["#334155"] * 5)
            for ri, run in enumerate(past_runs):
                series = run.get("series", {}).get(key, [])
                if series:
                    gc = ghost_palette[ri]
                    self._draw_series(c, series, plot, gc, key, 0, ghost=True)

            # Current run — bright
            self._draw_series(c, self.metric_history[key], plot, color, key, 0)

        # 6th slot: info panel (5 graphs + 1 info fills 3×2 perfectly)
        index = len(graph_specs)
        col, row = index % cols, index // cols
        x0  = gap + col * (cell_w + gap)
        y0  = gap + row * (cell_h + gap)
        x1, y1 = x0 + cell_w, y0 + cell_h
        c.create_rectangle(x0, y0, x1, y1, fill="#0f172a", outline="#1e3a5f")
        n = len(past_runs)
        ghost_note = (f"Dashed lines = {n} past run{'s' if n != 1 else ''}"
                      if n else "No past runs yet — they'll appear here.")
        c.create_text(x0 + 12, y0 + 18, text="How to read these graphs",
                      fill="#cbd5e1", font=("Arial", 10, "bold"), anchor="w")
        c.create_text(x0 + 12, y0 + 42,
                      text=(ghost_note + "\n\nBright line = this run.\n"
                            "↓ hover error = better.\n↑ confidence = better."),
                      fill="#64748b", font=("Arial", 9), anchor="nw",
                      width=cell_w - 24)

        # ── Trend row: overall% per past run ────────────────────────────────
        ty = graph_h + 4
        c.create_rectangle(gap, ty, w - gap, ty + trend_h - 4,
                           fill="#0d1b2a", outline="#1e3a5f")
        c.create_text(gap + 10, ty + 10, text="Performance trend  (overall % across all runs)",
                      fill="#38bdf8", font=("Arial", 8, "bold"), anchor="w")

        all_runs = _load_json_file(_RUN_HISTORY_PATH, [])
        if isinstance(all_runs, list) and len(all_runs) >= 2:
            scores  = [r.get("overall_pct", 0) for r in all_runs]
            grades  = [r.get("grade", "?")     for r in all_runs]
            grade_c = {"A": "#4ade80", "B": "#60a5fa", "C": "#fbbf24", "D": "#f87171"}
            tx0, tx1 = gap + 10, w - gap - 10
            bar_y0, bar_y1 = ty + 24, ty + trend_h - 10
            bw = (tx1 - tx0) / max(len(scores), 1)
            for i, (sc, gr) in enumerate(zip(scores, grades)):
                bx = tx0 + i * bw
                bh = (bar_y1 - bar_y0) * sc / 100
                col = grade_c.get(gr, "#64748b")
                is_last = (i == len(scores) - 1)
                c.create_rectangle(bx + 2, bar_y1 - bh, bx + bw - 2, bar_y1,
                                   fill=col, outline="" if not is_last else "#ffffff")
                if bw > 18:
                    c.create_text(bx + bw / 2, bar_y1 - bh - 6,
                                  text=f"{sc:.0f}%", fill=col,
                                  font=("Arial", 7, "bold"), anchor="s")
        elif all_runs:
            c.create_text(w // 2, ty + trend_h // 2,
                          text="Complete one more run to see improvement trend.",
                          fill="#475569", font=("Arial", 8), anchor="center")
        else:
            c.create_text(w // 2, ty + trend_h // 2,
                          text="No run history yet.",
                          fill="#475569", font=("Arial", 8), anchor="center")

        conf  = self.metric_history["confidence"][-1] if self.metric_history["confidence"] else 0.0
        err   = self.metric_history["hover_error"][-1] if self.metric_history["hover_error"] else 0.0
        avoid = self.metric_history["avoidance"][-1]   if self.metric_history["avoidance"]   else 0.0
        if self.ai_note_var is not None:
            self.ai_note_var.set(
                "Live evidence, not live training: "
                f"tracker confidence {conf:.2f}, hover error {err:.2f}m, "
                f"avoidance {avoid:.1f}N.  "
                f"Past-run ghost lines: {len(past_runs)} run(s) overlaid."
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


def _make_sim_tb_writer(scenario):
    """Create a TensorBoard SummaryWriter for live simulation metrics.
    Returns None gracefully if torch / tensorboard are unavailable."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        _ts      = time.strftime("%Y%m%d_%H%M%S")
        _runs    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "runs", "sim")
        _tb_dir  = os.path.join(_runs, f"{scenario}_{_ts}")
        os.makedirs(_tb_dir, exist_ok=True)
        writer   = SummaryWriter(_tb_dir)
        _logdir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
        print(f"\n  \033[36m[TensorBoard]\033[0m  tensorboard --logdir {_logdir}"
              f"\n  Logging sim/{scenario} → {_tb_dir}\n")
        return writer
    except Exception:
        return None


def run(
    duration=120.0,
    gui=True,
    scenario=SCENARIO_PARK,
    flight="ppo",
    battery_start=100.0,
    battery_drain_rate=BATTERY_DRAIN_RATE,
):
    tb_writer = _make_sim_tb_writer(scenario)   # TensorBoard live logging

    mode = p.GUI if gui else p.DIRECT
    phys = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=phys)
    p.setGravity(0, 0, -9.81, physicsClientId=phys)
    p.setTimeStep(SIM_TIMESTEP, physicsClientId=phys)

    if gui:
        # Hide PyBullet's built-in side panels; SkyShade has its own dashboard.
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=phys)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1, physicsClientId=phys)
        if scenario == SCENARIO_FOREST:
            p.resetDebugVisualizerCamera(
                cameraDistance=14.0, cameraYaw=35, cameraPitch=-34,
                cameraTargetPosition=[0, 0, 1.3],
                physicsClientId=phys,
            )
        elif scenario == SCENARIO_BUILDINGS:
            p.resetDebugVisualizerCamera(
                cameraDistance=19.0, cameraYaw=42, cameraPitch=-33,
                cameraTargetPosition=[0, 0, 1.4],
                physicsClientId=phys,
            )
        elif scenario == SCENARIO_TRAIL:
            p.resetDebugVisualizerCamera(
                cameraDistance=22.0, cameraYaw=0, cameraPitch=-22,
                cameraTargetPosition=[0, 0, 2.0],
                physicsClientId=phys,
            )
        else:
            p.resetDebugVisualizerCamera(
                cameraDistance=11, cameraYaw=30, cameraPitch=-27,
                cameraTargetPosition=[0, 0, 1],
                physicsClientId=phys,
            )

    flight_controller = RuntimeFlightController(flight)
    if flight_controller.fallback_reason:
        print(f"[flight] Learned flight fallback: {flight_controller.fallback_reason}")

    obstacles = build_environment(phys, scenario)
    initial_user_pos = scenario_user_position(scenario, 0.0)

    # Draw the cloud bank ONCE as static geometry — no per-frame churn (no lag,
    # no flicker/"swirl"); they just sit in the sky like real clouds.
    if gui and WEATHER_VISUALS:
        draw_static_clouds(phys, "Cloudy")

    # Drone and user are assembled from primitive bodies so the scene has a
    # readable physical scale while the AI/control code keeps the same inputs.
    drone_id, drone_parts = create_drone(phys, initial_user_pos[:2])
    drone_damping = LINEAR_DAMPING if flight_controller.using_learned else RUNTIME_PID_DAMPING
    p.changeDynamics(
        drone_id, -1, mass=DRONE_MASS_KG,
        linearDamping=drone_damping, angularDamping=0.9,
        physicsClientId=phys,
    )
    person_parts = create_person(phys)

    # Openable shade umbrella above the drone (opens on DEPLOY, folds on STOW)
    umbrella_rig = make_umbrella(phys)
    update_umbrella(phys, umbrella_rig,
                    [initial_user_pos[0], initial_user_pos[1],
                     TARGET_ALTITUDE + 0.30])
    set_umbrella_open(phys, umbrella_rig, False)   # starts folded (dry)

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
    if scenario == SCENARIO_TRAIL:
        # Trail also benefits from no look-ahead (straight path with crowds)
        flight_controller._lead_seconds = 0.25
        flight_controller._max_lead_m   = 0.4
    umbrella   = UmbrellaClassifier()
    nav_policy = NavSafetyPolicy()

    battery_pct      = float(np.clip(battery_start, 0.0, 100.0))
    umbrella_cmd     = "STOW"
    nav_override     = "CONTINUE"
    confidence       = 0.0
    user_pos         = initial_user_pos.copy()
    prev_user_pos    = user_pos.copy()
    forest_lap       = forest_lap_index(0.0)
    trail_lap        = trail_lap_index(0.0)
    user_velocity    = np.zeros(3, dtype=float)
    _bridge_active   = False    # True when drone is in the bridge fly-over zone
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

    # ── Stats accumulators for end-of-sim efficiency report ──────────────────
    _stats_err_xy   = []   # hover error each tick (m)
    _stats_conf     = []   # tracker confidence
    _stats_umb      = []   # umbrella state per tick (1=deploy, 0=stow)
    _stats_umb_need = []   # should umbrella be deployed? mirrors Sub-3 label threshold
    _stats_in_hover = []   # 1 if drone within 0.5m of user, else 0

    # ── Building-proximity tracking (city) ───────────────────────────────────
    _collision_count    = 0            # times the drone entered a building footprint
    _near_miss_count    = 0            # times it got within NEAR_MISS_CLEARANCE
    _in_collision       = False        # debounce: one event per entry
    _in_near            = False
    _min_building_clear = float("inf")  # closest the drone ever got (m)

    # Downsampled series for persistent history (1 sample per second)
    _hist_series: dict = {"confidence": [], "hover_error": [], "battery": [], "umbrella": []}

    # ── ANSI helpers ─────────────────────────────────────────────────────────
    _R = "\033[0m"          # reset
    _B = "\033[1m"          # bold
    _GR = "\033[32m"        # green
    _YL = "\033[33m"        # yellow
    _RD = "\033[31m"        # red
    _CY = "\033[36m"        # cyan
    _MG = "\033[35m"        # magenta
    _BL = "\033[34m"        # blue
    _WH = "\033[97m"        # bright white

    def _bat_col(pct):
        return _GR if pct > 50 else _YL if pct > 30 else _RD

    def _conf_col(c):
        return _GR if c >= 0.8 else _YL if c >= 0.5 else _RD

    def _err_col(e):
        return _GR if e <= 0.5 else _YL if e <= 1.0 else _RD

    def _nav_col(n):
        return _GR if n == "CONTINUE" else _YL if n == "RTH" else _RD

    # ── State tracking for event-driven prints ────────────────────────────────
    _prev_weather_mode  = None
    _prev_nav_override  = "CONTINUE"
    _prev_umbrella_cmd  = None
    _prev_conf_ok       = True     # True when confidence >= CONFIDENCE_THRESH
    _bat_warned         = set()    # thresholds already printed {50, 30, 10}

    print(f"\n{_B}{_CY}SkyShade{_R}  {_WH}{SCENARIO_LABELS.get(scenario, scenario)}{_R}"
          f"  ·  {flight_controller.label}  ·  {duration:.0f} s\n")
    print(f"  Battery: {_bat_col(battery_pct)}{battery_pct:.1f}%{_R}"
          f"  drain {battery_drain_rate:.2f}%/s\n")
    print(f"  {'Time':>7}  {'Battery':>8}  {'Nav':^10}  {'Flight':^9}"
          f"  {'Umbrella':^8}  {'Conf':>5}  {'Err':>6}  {'Alt':>5}")
    print("  " + "─" * 73)

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

            # Event: weather mode changed
            _wmode = weather_now["mode"]
            if _wmode != _prev_weather_mode and _prev_weather_mode is not None:
                _rain  = weather_now["rain"]
                _wind  = weather_now["wind"]
                _wcol  = _BL if "rain" in _wmode.lower() else _CY
                print(f"\n  {_wcol}[Weather]{_R} {_prev_weather_mode} → {_B}{_wmode}{_R}"
                      f"  rain {_rain:.2f}  wind {_wind:.1f} m/s")
            _prev_weather_mode = _wmode

            # ── User path ────────────────────────────────────────────────────
            user_pos = scenario_user_position(scenario, t_wall)
            wrapped_lap = False
            if scenario == SCENARIO_FOREST:
                next_lap = forest_lap_index(t_wall)
                wrapped_lap = next_lap != forest_lap
                forest_lap = next_lap
            elif scenario == SCENARIO_TRAIL:
                next_lap = trail_lap_index(t_wall)
                wrapped_lap = next_lap != trail_lap
                trail_lap = next_lap

            if wrapped_lap:
                prev_user_pos = user_pos.copy()
                user_velocity = np.zeros(3, dtype=float)
                gimbal_target = np.array(
                    [user_pos[0], user_pos[1], USER_MARKER_HEIGHT], dtype=float)
                p.resetBasePositionAndOrientation(
                    drone_id,
                    [user_pos[0], user_pos[1], TARGET_ALTITUDE],
                    [0, 0, 0, 1], physicsClientId=phys)
                p.resetBaseVelocity(
                    drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=phys)
                flight_controller.reset()

            user_delta = np.zeros(3, dtype=float) if wrapped_lap else user_pos - prev_user_pos
            user_velocity = user_delta / max(action_dt, 1e-6)
            user_yaw = math.atan2(user_delta[1], user_delta[0]) if np.linalg.norm(user_delta[:2]) > 1e-4 else 0.0
            update_person(phys, person_parts, user_pos, user_yaw, t_wall * 4.5)
            prev_user_pos = user_pos.copy()

            # ── Battery ───────────────────────────────────────────────────────
            battery_pct = max(0.0, battery_pct - battery_drain_rate * loop_dt)
            for _thresh, _msg, _col in ((50, "Watch battery", _YL),
                                        (30, "LOW battery — RTH likely soon", _RD),
                                        (10, "CRITICAL battery — LAND NOW", _RD + _B)):
                if battery_pct <= _thresh and _thresh not in _bat_warned:
                    _bat_warned.add(_thresh)
                    print(f"\n  {_col}[Sub-4 Battery]{_R} {_msg}: {_bat_col(battery_pct)}{battery_pct:.1f}%{_R}")

            # ── Sub-1: Perception — camera pointing down from drone ────────────
            d_pos, _ = p.getBasePositionAndOrientation(drone_id, physicsClientId=phys)
            drone_pos = np.array(d_pos)

            W, H = CAMERA_RES
            # Sit the camera just below the drone's landing skids (z ≈ -0.2) so
            # the drone's own gear doesn't poke into the downward POV as "blocks".
            eye = (drone_pos + np.array([0.0, 0.0, -0.26])).tolist()
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

            # ── Sub-1 Visual servo: nudge gimbal_target to centre the marker ──
            # When the marker is detected we know exactly where it sits in the
            # image.  Convert that pixel error to a world-space correction and
            # apply it so the next frame captures the marker closer to centre.
            if tracker.pixel_centroid is not None and confidence >= CONFIDENCE_THRESH:
                _cx_px, _cy_px = tracker.pixel_centroid
                _err_x = _cx_px - W * 0.5   # pixels right of centre
                _err_y = _cy_px - H * 0.5   # pixels below centre

                # Angular fraction of the field of view
                _fov_h_rad = math.radians(CAMERA_FOV)
                _fov_v_rad = _fov_h_rad * H / W
                _frac_x = _err_x / W
                _frac_y = _err_y / H

                # World-space displacement at the current view distance
                _eye_arr = np.array(eye, dtype=float)
                _view_dist = float(np.linalg.norm(gimbal_target - _eye_arr))
                _view_dist = max(0.3, _view_dist)
                _delta_x = 2.0 * _frac_x * _view_dist * math.tan(_fov_h_rad / 2)
                _delta_y = 2.0 * _frac_y * _view_dist * math.tan(_fov_v_rad / 2)

                # Camera right and down axes in world frame
                _gvec = gimbal_target - _eye_arr
                _gn = float(np.linalg.norm(_gvec))
                if _gn > 1e-3:
                    _gvec /= _gn
                    _up = np.array([0., 1., 0.])
                    _cam_right = np.cross(_gvec, _up)
                    _cr_n = float(np.linalg.norm(_cam_right))
                    if _cr_n > 1e-3:
                        _cam_right /= _cr_n
                        _cam_down = np.cross(_cam_right, _gvec)
                        _servo_correction = _delta_x * _cam_right + _delta_y * _cam_down
                        gimbal_target = gimbal_target + _servo_correction * 0.35

            # Event: tracker confidence crosses threshold
            _conf_ok_now = confidence >= CONFIDENCE_THRESH
            if _conf_ok_now != _prev_conf_ok:
                if _conf_ok_now:
                    print(f"\n  {_GR}[Sub-1 Tracker]{_R} Lock {_B}RECOVERED{_R}"
                          f"  conf {_conf_col(confidence)}{confidence:.2f}{_R}")
                else:
                    print(f"\n  {_YL}[Sub-1 Tracker]{_R} Lock {_B}LOST{_R}"
                          f"  conf {_conf_col(confidence)}{confidence:.2f}{_R} — gimbal searching")
                _prev_conf_ok = _conf_ok_now

            # ── Sub-3: Env decision (1 Hz) ────────────────────────────────────
            if tick % CONTROL_HZ == 0:
                lux = weather_now["lux"]
                rain = weather_now["rain"]
                wind = weather_now["wind"]
                cmd_int      = umbrella.predict(lux, rain, wind)
                umbrella_cmd = "DEPLOY" if cmd_int else "STOW"
                deployed     = cmd_int == 1
                if deployed != umb_deployed:
                    set_umbrella_open(phys, umbrella_rig, deployed)
                    umb_deployed = deployed
                # Event: umbrella state changed
                if umbrella_cmd != _prev_umbrella_cmd and _prev_umbrella_cmd is not None:
                    _ucol = _MG if deployed else _BL
                    print(f"\n  {_ucol}[Sub-3 Umbrella]{_R} {_prev_umbrella_cmd} → {_B}{umbrella_cmd}{_R}"
                          f"  rain {rain:.2f}  lux {lux:.0f}  wind {wind:.1f}")
                _prev_umbrella_cmd = umbrella_cmd

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
                # Event: nav override changed
                if nav_override != _prev_nav_override:
                    print(f"\n  {_nav_col(nav_override)}[Sub-4 Nav]{_R}"
                          f" {_prev_nav_override} → {_B}{nav_override}{_R}"
                          f"  battery {_bat_col(battery_pct)}{battery_pct:.1f}%{_R}"
                          f"  dist-home {dist_home:.1f} m")
                    _prev_nav_override = nav_override

            # ── Sub-2: flight control (PPO, legacy Q-table, or PID fallback) ──
            lin_vel, _ = p.getBaseVelocity(drone_id, physicsClientId=phys)
            drone_vel  = np.array(lin_vel)

            # Bridge fly-over: raise target altitude when drone or user is inside
            # the bridge zone so it climbs over the deck instead of colliding.
            _alt_override = None
            if scenario == SCENARIO_TRAIL:
                in_bridge = (abs(drone_pos[0] - TRAIL_BRIDGE_CX) < TRAIL_FLY_OVER_X_HALF or
                             abs(user_pos[0]  - TRAIL_BRIDGE_CX) < TRAIL_FLY_OVER_X_HALF)
                if in_bridge != _bridge_active:
                    _bridge_active = in_bridge
                    tag = "ACTIVE" if in_bridge else "CLEAR"
                    print(f"\n  {_CY}[Bridge]{_R} fly-over {_B}{tag}{_R}"
                          f"  drone-x {drone_pos[0]:.1f}  alt → "
                          f"{'%.1f' % TRAIL_FLY_OVER_ALT if in_bridge else '%.1f' % TARGET_ALTITUDE} m")
                if _bridge_active:
                    _alt_override = TRAIL_FLY_OVER_ALT

            flight_cmd = flight_controller.compute_force(
                drone_pos=drone_pos,
                drone_vel=drone_vel,
                user_pos=user_pos,
                user_velocity=user_velocity,
                nav_override=nav_override,
                wind_speed=weather_now["wind"],
                dt=dt,
                altitude_override=_alt_override,
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
            update_umbrella(phys, umbrella_rig,
                            [d_pos2[0], d_pos2[1], d_pos2[2] + 0.30])

            # ── Building-proximity alerts (city only) ─────────────────────────
            # The avoidance AI still runs; this just reports how close the drone
            # gets so you can watch what it does near a building.
            if scenario == SCENARIO_BUILDINGS and obstacles:
                _d_xy = np.array(d_pos2[:2], dtype=float)
                _min_clear = min(
                    float(np.linalg.norm(_d_xy - o["position"]) - o["radius"])
                    for o in obstacles)
                _min_building_clear = min(_min_building_clear, _min_clear)
                if _min_clear <= COLLISION_CLEARANCE:
                    if not _in_collision:
                        _collision_count += 1
                        _in_collision = True
                        _in_near = True
                        print(f"\n  {_RD}{_B}[Collision]{_R} drone CONTACTED a "
                              f"building (#{_collision_count})  clearance "
                              f"{_min_clear:+.2f} m — avoidance couldn't fully clear it")
                    if gui:
                        p.addUserDebugText(
                            "COLLISION", [d_pos2[0], d_pos2[1], d_pos2[2] + 0.85],
                            textColorRGB=[1.0, 0.25, 0.20], textSize=1.7,
                            lifeTime=0.6, physicsClientId=phys)
                elif _min_clear < NEAR_MISS_CLEARANCE:
                    _in_collision = False
                    if not _in_near:
                        _near_miss_count += 1
                        _in_near = True
                        print(f"\n  {_YL}[Near miss]{_R} drone within "
                              f"{_min_clear:.2f} m of a building "
                              f"(#{_near_miss_count}) — avoidance steering around")
                    if gui:
                        p.addUserDebugText(
                            "NEAR BUILDING", [d_pos2[0], d_pos2[1], d_pos2[2] + 0.85],
                            textColorRGB=[1.0, 0.80, 0.20], textSize=1.4,
                            lifeTime=0.5, physicsClientId=phys)
                else:
                    _in_collision = False
                    _in_near = False

            if gui and WEATHER_VISUALS and tick % WEATHER_VIS_EVERY == 0:
                focus_xy = (
                    (np.array(d_pos2[:2], dtype=float) + user_pos[:2]) * 0.5
                )
                draw_weather_visuals(
                    phys, weather_now, weather_visual_rng, t_wall,
                    focus_xy=focus_xy,
                )

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

            if gui and tick % 10 == 0:   # 3 Hz — was 6 Hz, halved for performance
                err_xy = float(np.linalg.norm(np.array(d_pos2[:2]) - user_pos[:2]))
                telemetry.update_vision(
                    frame,
                    confidence,
                    err_xy,
                    avoidance_for_display,
                    battery_pct,
                    umbrella_cmd,
                    gimbal_action,
                    weather=weather_now,
                    t_wall=t_wall,
                )

            # ── Accumulate stats ──────────────────────────────────────────────
            _err = float(np.linalg.norm(np.array(d_pos2[:2]) - user_pos[:2]))
            _stats_err_xy.append(_err)
            _stats_conf.append(float(confidence))
            _stats_umb.append(1.0 if umbrella_cmd == "DEPLOY" else 0.0)
            _stats_umb_need.append(
                1.0 if weather_now.get("rain", 0) >= UMBRELLA_DEPLOY_RAIN_THRESHOLD else 0.0)
            _stats_in_hover.append(1.0 if _err <= 0.5 else 0.0)

            # Downsample at 1 Hz for history; TensorBoard at 0.33 Hz (every 3 s)
            if tick % CONTROL_HZ == 0:
                _hist_series["confidence"].append(round(float(confidence), 3))
                _hist_series["hover_error"].append(round(_err, 3))
                _hist_series["battery"].append(round(float(battery_pct), 1))
                _hist_series["umbrella"].append(1.0 if umbrella_cmd == "DEPLOY" else 0.0)

                if tb_writer is not None and tick % (CONTROL_HZ * 3) == 0:
                    _step = int(t_wall)
                    # Sub-1 Perception
                    tb_writer.add_scalar("sub1/tracker_confidence",    float(confidence),      _step)
                    # Sub-2 Flight Control
                    tb_writer.add_scalar("sub2/hover_error_m",         _err,                   _step)
                    tb_writer.add_scalar("sub2/avoidance_force_n",
                                         float(np.linalg.norm(avoidance_for_display[:2])),     _step)
                    if flight_cmd.reward is not None:
                        tb_writer.add_scalar("sub2/flight_reward",
                                             float(flight_cmd.reward),                         _step)
                    # Sub-3 Environmental Decision
                    tb_writer.add_scalar("sub3/umbrella_deployed",
                                         1.0 if umbrella_cmd == "DEPLOY" else 0.0,            _step)
                    tb_writer.add_scalar("sub3/weather_rain",
                                         float(weather_now.get("rain", 0)),                    _step)
                    tb_writer.add_scalar("sub3/weather_wind",
                                         float(weather_now.get("wind", 0)),                    _step)
                    # Sub-4 Navigation Safety
                    tb_writer.add_scalar("sub4/battery_pct",           float(battery_pct),     _step)
                    tb_writer.add_scalar("sub4/nav_override",
                                         {"CONTINUE": 0, "RTH": 1,
                                          "LAND_NOW": 2}.get(nav_override, 0),                 _step)

            # ── Console log every 5 s ─────────────────────────────────────────
            if tick % (CONTROL_HZ * 5) == 0:
                _e = _err
                _pct_done = min(100, t_wall / duration * 100)
                _bar_len  = 12
                _filled   = int(_bar_len * _pct_done / 100)
                _prog_bar = f"[{'█' * _filled}{'░' * (_bar_len - _filled)}] {_pct_done:4.0f}%"
                print(f"  {_CY}{t_wall:7.1f}s{_R}  "
                      f"{_bat_col(battery_pct)}{battery_pct:5.1f}%{_R}  "
                      f"{_nav_col(nav_override)}{nav_override:<10}{_R}  "
                      f"{flight_cmd.controller:<9}  "
                      f"{'DEPLOY' if umbrella_cmd == 'DEPLOY' else 'stow':<8}  "
                      f"{_conf_col(confidence)}{confidence:.2f}{_R}  "
                      f"{_err_col(_e)}{_e:5.2f}m{_R}  "
                      f"{d_pos2[2]:.2f}m  {_prog_bar}")

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
        if tb_writer is not None:
            tb_writer.flush()

    # ── End-of-sim efficiency report ─────────────────────────────────────────
    if _stats_err_xy:
        N = len(_stats_err_xy)
        mean_err    = float(np.mean(_stats_err_xy))
        hover_pct   = float(np.mean(_stats_in_hover)) * 100
        mean_conf   = float(np.mean(_stats_conf))
        umb_acc_pct = (float(np.mean(
            [a == b for a, b in zip(_stats_umb, _stats_umb_need)])) * 100
            if _stats_umb_need else 100.0)

        # Colour helpers (ANSI)
        def _ok(s):  return f"\033[32m{s}\033[0m"
        def _warn(s): return f"\033[33m{s}\033[0m"
        def _bad(s):  return f"\033[31m{s}\033[0m"
        def _grade(val, good, ok):
            return (_ok if val >= good else _warn if val >= ok else _bad)

        print("\n" + "═" * 60)
        print("  SkyShade — Post-Run Efficiency Report")
        print("═" * 60)
        print(f"  Sub-2  Hover accuracy   : {hover_pct:5.1f}%  "
              + _grade(hover_pct, 70, 40)(
                  f"({'✓ good' if hover_pct >= 70 else '~ ok' if hover_pct >= 40 else '✗ train more'})"))
        print(f"         Mean hover error  : {mean_err:5.2f} m  "
              + _grade(100 - mean_err*100, 50, 0)(
                  f"({'✓ within 0.5m' if mean_err <= 0.5 else '~ close' if mean_err <= 1.0 else '✗ far off target'})"))
        print(f"  Sub-1  Tracker lock     : {mean_conf*100:5.1f}%  "
              + _grade(mean_conf*100, 80, 60)(
                  f"({'✓ reliable' if mean_conf >= 0.8 else '~ ok' if mean_conf >= 0.6 else '✗ poor lock'})"))
        print(f"  Sub-3  Umbrella correct : {umb_acc_pct:5.1f}%  "
              + _grade(umb_acc_pct, 85, 65)(
                  f"({'✓ accurate' if umb_acc_pct >= 85 else '~ ok' if umb_acc_pct >= 65 else '✗ check SVM'})"))
        print(f"  Sub-4  Battery at end   : {battery_pct:5.1f}%  "
              + _grade(battery_pct, 30, 10)(
                  f"({'✓ safe' if battery_pct >= 30 else '~ low' if battery_pct >= 10 else '✗ critical'})"))
        if scenario == SCENARIO_BUILDINGS:
            _clear_txt = ("n/a" if _min_building_clear == float("inf")
                          else f"{_min_building_clear:+.2f} m")
            _bld_grade = (_bad if _collision_count else
                          _warn if _near_miss_count else _ok)
            print(f"  Bldg   Near-miss/collide : "
                  f"{_near_miss_count} near / {_collision_count} hit  "
                  + _bld_grade(
                      f"(closest {_clear_txt}; "
                      f"{'✗ hit a building' if _collision_count else '✓ avoided all' if not _near_miss_count else '~ flew close, avoided'})"))
        print("═" * 60)

        overall = (hover_pct + mean_conf*100 + umb_acc_pct) / 3
        grade   = "A" if overall >= 80 else "B" if overall >= 65 else "C" if overall >= 50 else "D"
        print(f"  Overall system score: {overall:.0f}%  Grade: {grade}")
        if hover_pct < 50:
            print("  → Train Sub-2 PPO more (hover accuracy is the bottleneck)")
        if mean_conf < 0.7:
            print("  → Check Sub-1 calibration (tracker confidence is low)")
        if umb_acc_pct < 75:
            print("  → Retrain Sub-3 SVM (umbrella decisions are inaccurate)")
        print("═" * 60 + "\n")

        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        latest_report = {
            "scenario": scenario,
            "flight_requested": flight,
            "flight_controller": flight_controller.label,
            "duration_s": float(duration),
            "samples": int(N),
            "mean_hover_error_m": mean_err,
            "hover_accuracy_pct": hover_pct,
            "tracker_lock_pct": mean_conf * 100,
            "umbrella_correct_pct": umb_acc_pct,
            "battery_end_pct": float(battery_pct),
            "overall_pct": float(overall),
            "grade": grade,
            "building_near_misses": int(_near_miss_count),
            "building_collisions": int(_collision_count),
            "closest_building_m": (None if _min_building_clear == float("inf")
                                   else float(_min_building_clear)),
            "created_at": time.time(),
        }
        with open(os.path.join(reports_dir, "run_summary_latest.json"), "w", encoding="utf-8") as f:
            json.dump(latest_report, f, indent=2)

        # Append to persistent multi-run history (used for graph overlays)
        _append_run_history({
            "timestamp":            time.time(),
            "scenario":             scenario,
            "overall_pct":          float(overall),
            "grade":                grade,
            "hover_pct":            hover_pct,
            "tracker_lock_pct":     mean_conf * 100,
            "umbrella_correct_pct": umb_acc_pct,
            "battery_end_pct":      float(battery_pct),
            "series":               _hist_series,
        })

        # Write final efficiency stats to TensorBoard then close the writer
        if tb_writer is not None:
            _grade_num = {"A": 4, "B": 3, "C": 2, "D": 1}.get(grade, 0)
            # Final scores grouped by subsystem
            tb_writer.add_scalar("sub1/run_tracker_lock_pct",      mean_conf * 100,   0)
            tb_writer.add_scalar("sub2/run_hover_accuracy_pct",    hover_pct,         0)
            tb_writer.add_scalar("sub2/run_mean_hover_error_m",    mean_err,          0)
            tb_writer.add_scalar("sub3/run_umbrella_correct_pct",  umb_acc_pct,       0)
            tb_writer.add_scalar("sub4/run_battery_end_pct",       float(battery_pct),0)
            tb_writer.add_scalar("system/overall_score_pct",       float(overall),    0)
            tb_writer.add_scalar("system/grade_numeric",           _grade_num,        0)
            tb_writer.flush()
            tb_writer.close()

        return {
            "hover_pct":   hover_pct,
            "mean_err":    mean_err,
            "mean_conf":   mean_conf * 100,
            "umb_acc_pct": umb_acc_pct,
            "battery_pct": float(battery_pct),
            "overall":     float(overall),
            "grade":       grade,
        }

    # No stats — still close writer if open
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
    return None


class ScenarioCompleteDialog:
    """Post-run dialog with per-subsystem retrain checkboxes and scenario picker.

    show() returns (action, scenario) where action is 'retrain', 'run', or None.
    Clicking 'Retrain Selected' opens the full Training Grounds hub for only the
    ticked subsystems, then auto-launches the chosen scenario.
    """

    def __init__(self, scenario, duration, flight, battery_start, battery_drain_rate,
                 stats=None):
        self._scenario = scenario
        self._duration = duration
        self._flight = flight
        self._battery_start = battery_start
        self._battery_drain_rate = battery_drain_rate
        self._stats = stats or {}

        self._result = None               # "retrain" | "run" | None
        self._selected_scenario = scenario # may be changed by picker
        self._retrain_vars = {}           # stage_int → tk.BooleanVar
        self._scenario_btns = {}          # scenario_key → tk.Button

        self.root = None

    # ── public ────────────────────────────────────────────────────────────────

    def show(self):
        """Block until the user acts. Returns (action, scenario)."""
        if tk is None:
            return None, self._scenario
        try:
            self.root = tk.Tk()
        except tk.TclError:
            return None, self._scenario

        self.root.title("SkyShade — Simulation Complete")
        self.root.configure(bg="#07111f")
        self.root.resizable(False, False)
        self.root.minsize(520, 380)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()

        # Measure content then centre
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        rw = max(520, self.root.winfo_reqwidth())
        rh = max(380, self.root.winfo_reqheight())
        self.root.geometry(f"{rw}x{rh}+{(sw - rw) // 2}+{(sh - rh) // 2}")

        self.root.mainloop()
        return self._result, self._selected_scenario

    # ── UI construction ───────────────────────────────────────────────────────

    _W = 600   # fixed inner content width

    def _build_ui(self):
        pad = tk.Frame(self.root, bg="#07111f", padx=22, pady=16)
        pad.pack(fill="both", expand=True)

        scenario_label = SCENARIO_LABELS.get(self._scenario, self._scenario)
        tk.Label(pad, text="Simulation Complete",
                 fg="#38bdf8", bg="#07111f",
                 font=("Arial", 15, "bold"), anchor="w",
                 ).pack(fill="x")
        tk.Label(pad,
                 text=(f"Scenario: {scenario_label}  ·  {self._duration:.0f} s  "
                       f"·  Flight: {self._flight.upper()}"),
                 fg="#64748b", bg="#07111f",
                 font=("Arial", 9), anchor="w",
                 ).pack(fill="x", pady=(1, 10))

        self._build_report(pad)
        self._build_scenario_picker(pad)
        self._build_buttons(pad)

    def _build_report(self, parent):
        card = tk.Frame(parent, bg="#111c2e")
        card.pack(fill="x", pady=(0, 8))

        # Fixed column widths — label expands, value and verdict are fixed
        # col0=checkbox(28), col1=label(expands), col2=value(72), col3=verdict(150)
        card.columnconfigure(0, weight=0, minsize=28)
        card.columnconfigure(1, weight=1)
        card.columnconfigure(2, weight=0, minsize=72)
        card.columnconfigure(3, weight=0, minsize=150)

        tk.Label(card,
                 text="Post-Run Efficiency Report  —  tick to retrain",
                 fg="#38bdf8", bg="#111c2e",
                 font=("Arial", 10, "bold"), anchor="w",
                 padx=12, pady=6,
                 ).grid(row=0, column=0, columnspan=4, sticky="ew")
        tk.Frame(card, bg="#1e3a5f", height=1).grid(
            row=1, column=0, columnspan=4, sticky="ew")

        s = self._stats

        def _bg_fg(val, good, ok):
            if val >= good: return "#052e16", "#4ade80"
            if val >= ok:   return "#451a03", "#fbbf24"
            return "#3b0a0a", "#f87171"

        def _vtext(val, good, ok, gm, om, bm):
            if val >= good: return f"✓ {gm}"
            if val >= ok:   return f"~ {om}"
            return f"✗ {bm}"

        def _row(r, stage, lbl, val_s, verdict_s, rbg, rfg, pre=False, sub=False):
            if stage is not None:
                var = tk.BooleanVar(value=pre)
                self._retrain_vars[stage] = var
                tk.Checkbutton(card, variable=var, bg="#111c2e",
                               activebackground="#111c2e",
                               selectcolor="#1e3a5f",
                               ).grid(row=r, column=0, padx=(8, 0))
            else:
                tk.Frame(card, bg="#111c2e", width=28).grid(row=r, column=0)

            tk.Label(card, text=("    " if sub else "") + lbl,
                     fg="#64748b" if sub else "#94a3b8",
                     bg="#111c2e",
                     font=("Arial", 9 if sub else 9, "bold"),
                     anchor="w", padx=4, pady=4,
                     ).grid(row=r, column=1, sticky="ew")
            tk.Label(card, text=val_s,
                     fg="#cbd5e1", bg="#111c2e",
                     font=("Arial", 9, "bold"),
                     anchor="e", padx=6, pady=4,
                     ).grid(row=r, column=2, sticky="ew")
            tk.Label(card, text=verdict_s,
                     fg=rfg, bg=rbg,
                     font=("Arial", 9, "bold"),
                     anchor="w", padx=8, pady=4,
                     ).grid(row=r, column=3, sticky="ew", padx=(3, 0))

        if not s:
            tk.Label(card, text="No efficiency data available.",
                     fg="#64748b", bg="#111c2e",
                     font=("Arial", 9), anchor="w", padx=12, pady=6,
                     ).grid(row=2, column=0, columnspan=4, sticky="ew")
        else:
            hover   = s.get("hover_pct",   0.0)
            err     = s.get("mean_err",    0.0)
            conf    = s.get("mean_conf",   0.0)
            umb     = s.get("umb_acc_pct", 0.0)
            bat     = s.get("battery_pct", 0.0)
            overall = s.get("overall",     0.0)
            grade   = s.get("grade",       "?")

            bg, fg = _bg_fg(conf, 80, 60)
            _row(2, None, "Sub-1  Tracker lock", f"{conf:.1f}%",
                 _vtext(conf, 80, 60, "reliable", "ok", "poor lock"), bg, fg)

            bg, fg = _bg_fg(hover, 70, 40)
            _row(3, 0, "Sub-2  Hover accuracy", f"{hover:.1f}%",
                 _vtext(hover, 70, 40, "good", "ok", "train more"),
                 bg, fg, pre=(hover < 50))

            e_bg = "#052e16" if err<=0.5 else "#451a03" if err<=1.0 else "#3b0a0a"
            e_fg = "#4ade80" if err<=0.5 else "#fbbf24" if err<=1.0 else "#f87171"
            e_v  = ("✓ within 0.5 m" if err<=0.5 else
                    "~ close" if err<=1.0 else "✗ far off target")
            _row(4, None, "Mean hover error", f"{err:.2f} m", e_v, e_bg, e_fg, sub=True)

            bg, fg = _bg_fg(umb, 85, 65)
            _row(5, 1, "Sub-3  Umbrella correct", f"{umb:.1f}%",
                 _vtext(umb, 85, 65, "accurate", "ok", "check SVM"),
                 bg, fg, pre=(umb < 75))

            bg, fg = _bg_fg(bat, 30, 10)
            _row(6, 2, "Sub-4  Battery MDP + Nav SAC", f"{bat:.1f}% left",
                 _vtext(bat, 30, 10, "safe", "low", "critical"),
                 bg, fg, pre=(bat < 10))
            self._retrain_vars[3] = self._retrain_vars[2]

            tk.Frame(card, bg="#1e3a5f", height=1).grid(
                row=7, column=0, columnspan=4, sticky="ew")

            gfg = ("#4ade80" if grade=="A" else "#60a5fa" if grade=="B"
                   else "#fbbf24" if grade=="C" else "#f87171")
            tk.Label(card,
                     text=f"Overall score:  {overall:.0f}%     Grade: {grade}",
                     fg=gfg, bg="#0b1726",
                     font=("Arial", 11, "bold"), anchor="w",
                     padx=12, pady=6,
                     ).grid(row=8, column=0, columnspan=4, sticky="ew")

            hints = []
            if hover < 50: hints.append("→ Sub-2 PPO needs more hover training")
            if conf  < 70: hints.append("→ Sub-1 tracker confidence is low")
            if umb   < 75: hints.append("→ Sub-3 SVM umbrella accuracy is low")
            if hints:
                tk.Label(card, text="  ".join(hints),
                         fg="#f59e0b", bg="#0b1726",
                         font=("Arial", 8), anchor="w",
                         padx=12, pady=4, justify="left",
                         wraplength=self._W - 30,
                         ).grid(row=9, column=0, columnspan=4, sticky="ew")

            # Performance trend chart
            all_runs = _load_json_file(_RUN_HISTORY_PATH, [])
            if isinstance(all_runs, list) and len(all_runs) >= 1:
                tk.Label(card,
                         text=f"Performance trend  ({len(all_runs)} run{'s' if len(all_runs)!=1 else ''})",
                         fg="#38bdf8", bg="#0d1b2a",
                         font=("Arial", 8, "bold"), anchor="w",
                         padx=12, pady=4,
                         ).grid(row=10, column=0, columnspan=4, sticky="ew")

                tc = tk.Canvas(card, bg="#0d1b2a", height=72,
                               highlightthickness=0)
                tc.grid(row=11, column=0, columnspan=4,
                        sticky="ew", padx=12, pady=(0, 6))

                def _draw(event=None, _tc=tc, _runs=all_runs):
                    try:
                        _tc.winfo_exists()
                    except Exception:
                        return
                    if not _tc.winfo_exists():
                        return
                    _tc.delete("all")
                    tw = int(_tc.winfo_width()) or (self._W - 24)
                    th = 72
                    scores = [r.get("overall_pct", 0) for r in _runs]
                    grades = [r.get("grade", "?")     for r in _runs]
                    n = len(scores)
                    gcol = {"A":"#4ade80","B":"#60a5fa","C":"#fbbf24","D":"#f87171"}
                    raw_bw = tw / max(n, 1)
                    bw = min(raw_bw, 48)          # cap bar width so few bars look tidy
                    x_off = (tw - bw * n) / 2     # centre bars when capped
                    plot_h = th - 22
                    for i, (sc, gr) in enumerate(zip(scores, grades)):
                        bx = x_off + i * bw
                        bh = max(2, plot_h * sc / 100)
                        col = gcol.get(gr, "#64748b")
                        is_last = (i == n - 1)
                        _tc.create_rectangle(
                            bx + 2, th - 14 - bh, bx + bw - 2, th - 14,
                            fill=col, outline="#e2e8f0" if is_last else "")
                        _tc.create_text(bx + bw/2, th - 14 - bh - 3,
                                        text=f"{sc:.0f}", fill=col,
                                        font=("Arial", 7, "bold"), anchor="s")
                        _tc.create_text(bx + bw/2, th - 4,
                                        text=gr, fill="#475569",
                                        font=("Arial", 7), anchor="s")
                    if n >= 2:
                        pts = []
                        for i, sc in enumerate(scores):
                            pts.extend([x_off + i*bw + bw/2,
                                        (th-14) - plot_h*sc/100])
                        _tc.create_line(*pts, fill="#94a3b8", width=1,
                                        smooth=True, dash=(3,4))

                tc.bind("<Configure>", _draw)
                tc.after(60, _draw)

    def _build_scenario_picker(self, parent):
        outer = tk.Frame(parent, bg="#111c2e")
        outer.pack(fill="x", pady=(0, 8))

        tk.Label(outer, text="Run next scenario",
                 fg="#64748b", bg="#111c2e",
                 font=("Arial", 8, "bold"), anchor="w",
                 padx=12, pady=5,
                 ).pack(fill="x")

        btn_row = tk.Frame(outer, bg="#111c2e")
        btn_row.pack(fill="x", padx=10, pady=(0, 8))
        n_scenarios = len(SCENARIO_LABELS)
        for i in range(n_scenarios):
            btn_row.columnconfigure(i, weight=1)

        for col, (key, label) in enumerate(SCENARIO_LABELS.items()):
            is_cur = (key == self._scenario)
            btn = tk.Button(
                btn_row,
                text=("● " if is_cur else "") + label,
                command=lambda k=key: self._pick_scenario(k),
                bg="#1e3a8a" if is_cur else "#1e293b",
                fg="#bfdbfe" if is_cur else "#64748b",
                activebackground="#2563eb", activeforeground="#ffffff",
                relief="flat", font=("Arial", 9, "bold"),
                padx=6, pady=7,
            )
            btn.grid(row=0, column=col, sticky="ew",
                     padx=(0, 5) if col < n_scenarios - 1 else 0)
            self._scenario_btns[key] = btn

    def _build_buttons(self, parent):
        row = tk.Frame(parent, bg="#07111f")
        row.pack(fill="x", pady=(4, 0))
        for i in range(3):
            row.columnconfigure(i, weight=1)

        tk.Button(row, text="Retrain Selected & Run",
                  command=self._on_retrain_run,
                  bg="#2563eb", fg="#eff6ff",
                  activebackground="#1d4ed8", activeforeground="#ffffff",
                  relief="flat", font=("Arial", 10, "bold"),
                  padx=8, pady=9,
                  ).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        tk.Button(row, text="Run Again",
                  command=self._on_run_again,
                  bg="#064e3b", fg="#d1fae5",
                  activebackground="#065f46", activeforeground="#ffffff",
                  relief="flat", font=("Arial", 10, "bold"),
                  padx=8, pady=9,
                  ).grid(row=0, column=1, sticky="ew", padx=(0, 5))
        tk.Button(row, text="Close",
                  command=self._on_close,
                  bg="#1e293b", fg="#cbd5e1",
                  activebackground="#334155", activeforeground="#ffffff",
                  relief="flat", font=("Arial", 10, "bold"),
                  padx=8, pady=9,
                  ).grid(row=0, column=2, sticky="ew")

    def _pick_scenario(self, key):
        self._selected_scenario = key
        for k, btn in self._scenario_btns.items():
            active = (k == key)
            btn.configure(
                bg="#1e3a8a" if active else "#1e293b",
                fg="#bfdbfe" if active else "#94a3b8",
                text=("● " if active else "  ") + SCENARIO_LABELS[k],
            )

    # ── button handlers ───────────────────────────────────────────────────────

    def _on_retrain_run(self):
        stages = {s for s, var in self._retrain_vars.items() if var.get()}
        if not stages:
            # Nothing ticked → just run again with the selected scenario
            self._result = "run"
            self.root.destroy()
            return
        self._result = "retrain"
        self.root.withdraw()
        hub = TrainingGroundsHub(self.root)
        hub.start_auto_train(retrain=True, stages=stages,
                             on_complete=lambda: self._on_hub_done(hub),
                             scenario=self._selected_scenario)

    def _on_hub_done(self, hub):
        hub.close()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _on_run_again(self):
        self._result = "run"
        self.root.destroy()

    def _on_close(self):
        self._result = None
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--no-gui",   action="store_true")
    ap.add_argument(
        "--flight",
        choices=["ppo", "q", "pid"],
        default="ppo",
        help="Flight controller: 'ppo' for the active learned policy, 'q' for legacy Q-table, or 'pid' for fallback baseline.",
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

    gui = not args.no_gui
    scenario = args.scenario or SCENARIO_PARK

    while True:
        stats = run(
            duration=args.duration,
            gui=gui,
            scenario=scenario,
            flight=args.flight,
            battery_start=battery_start,
            battery_drain_rate=battery_drain_rate,
        )

        if not gui:
            break

        dialog = ScenarioCompleteDialog(
            scenario=scenario,
            duration=args.duration,
            flight=args.flight,
            battery_start=battery_start,
            battery_drain_rate=battery_drain_rate,
            stats=stats,
        )
        action, next_scenario = dialog.show()
        if action is None:
            break
        scenario = next_scenario  # honour any scenario switch from the picker


if __name__ == "__main__":
    main()
