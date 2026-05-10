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
WEATHER_MIN_SECONDS = 8.0
WEATHER_MAX_SECONDS = 20.0


def figure8(t, scale=1.5):
    d = 1 + math.sin(t) ** 2
    return np.array([scale * math.cos(t) / d,
                     scale * math.sin(t) * math.cos(t) / d,
                     0.0])


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

    telemetry = TelemetryWindow() if gui else ConsoleTelemetry()
    weather_controller = RandomWeatherController()
    weather_visual_rng = np.random.default_rng()
    weather_now = weather_controller.update(0.0)

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
            if telemetry.closed:
                print("\nSimulation stopped from dashboard.")
                break
            weather_now = weather_controller.update(t_wall)

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

            if gui and tick % 5 == 0:
                draw_weather_visuals(phys, weather_now, weather_visual_rng)

            # ── Telemetry update (every 10 ticks) ─────────────────────────────
            if tick % 10 == 0:
                err_xy = float(np.linalg.norm(np.array(d_pos2[:2]) - user_pos[:2]))
                values, statuses = _telemetry_payload(
                    t_wall, battery_pct, nav_override, umbrella_cmd, confidence,
                    err_xy, np.array(d_pos2), user_pos, user_offset, weather_now,
                )
                telemetry.update(values, statuses, battery_pct=battery_pct)

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
    args = ap.parse_args()
    run(duration=args.duration, gui=not args.no_gui)


if __name__ == "__main__":
    main()
