"""
Sub-1 Perception — PyBullet calibration room.

A 6×6 m room rendered with PyBullet's tiny renderer.  A red sphere follows a
Lissajous path and the camera pans slightly to track it (simulating a gimbal).
The HSV tracker runs on the rendered frames so calibration is done against a
3-D scene rather than a flat procedural image.

Usage (standalone smoke-test):
    python sub1_perception/training_env.py
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import pybullet as p
    import pybullet_data
    _PB = True
except ImportError:
    _PB = False

IMG_W = 640
IMG_H = 480

# Sphere path parameters (Lissajous)
_AX, _AY, _AZ = 2.2, 1.6, 0.5   # amplitudes (m)
_FX, _FY, _FZ = 0.7, 1.1, 0.4   # frequencies (rad/s)
_Z_BASE        = 1.6              # centre height (m)

# Camera base position (mounted high on south wall, looking inward)
_CAM_BASE = [0.0, -2.7, 2.4]


class PerceptionTrainingEnv:
    """PyBullet perception-calibration room with a moving red sphere."""

    def __init__(self):
        self._phys_id   = None
        self._sphere_id = None
        self._last_view = None   # stored for 3D→2D projection
        self._last_proj = None

    def _init(self):
        self._phys_id = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self._phys_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self._phys_id)

        # Room walls — grey-blue panels
        half, wall_h = 3.0, 4.0
        for pos, he in [
            ([0,  -half, wall_h/2], [half, 0.06, wall_h]),
            ([0,   half, wall_h/2], [half, 0.06, wall_h]),
            ([-half, 0,  wall_h/2], [0.06, half, wall_h]),
            ([ half, 0,  wall_h/2], [0.06, half, wall_h]),
        ]:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=he,
                                         physicsClientId=self._phys_id)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=he,
                                      rgbaColor=[0.38, 0.45, 0.55, 1.0],
                                      physicsClientId=self._phys_id)
            p.createMultiBody(0, col, vis, pos, physicsClientId=self._phys_id)

        # Floor grid markers
        for gx in range(-2, 3):
            for gy in range(-2, 3):
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.005],
                                          rgbaColor=[0.3, 0.3, 0.35, 0.8],
                                          physicsClientId=self._phys_id)
                p.createMultiBody(0, -1, vis, [gx, gy, 0.005],
                                  physicsClientId=self._phys_id)

        # Red target sphere
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.18,
                                     physicsClientId=self._phys_id)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.18,
                                  rgbaColor=[0.93, 0.06, 0.06, 1.0],
                                  physicsClientId=self._phys_id)
        self._sphere_id = p.createMultiBody(0, col, vis, [0, 0, _Z_BASE],
                                             physicsClientId=self._phys_id)

    def step(self, t_wall: float):
        """Advance the scene by one frame.

        Returns
        -------
        frame : np.ndarray  shape (IMG_H, IMG_W, 3) uint8 RGB
        sphere_pos : (x, y, z) world-space position of the sphere
        """
        if not _PB:
            return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8), (0.0, 0.0, _Z_BASE)

        if self._phys_id is None:
            self._init()

        # Move sphere
        sx = _AX * math.sin(_FX * t_wall)
        sy = _AY * math.sin(_FY * t_wall + 0.5)
        sz = _Z_BASE + _AZ * math.sin(_FZ * t_wall)
        p.resetBasePositionAndOrientation(
            self._sphere_id, [sx, sy, sz], [0, 0, 0, 1],
            physicsClientId=self._phys_id,
        )

        # Gentle camera pan toward sphere (simulates drone gimbal tracking)
        pan_x = sx * 0.25
        pan_z = sz * 0.35
        cam_target = [pan_x, 0.0, pan_z]

        view = p.computeViewMatrix(
            cameraEyePosition=_CAM_BASE,
            cameraTargetPosition=cam_target,
            cameraUpVector=[0, 0, 1],
            physicsClientId=self._phys_id,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=68, aspect=IMG_W / IMG_H,
            nearVal=0.1, farVal=20.0,
            physicsClientId=self._phys_id,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            IMG_W, IMG_H,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._phys_id,
        )
        self._last_view = view
        self._last_proj = proj
        frame = np.array(rgba, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, :3]
        return frame, (sx, sy, sz)

    def project_to_pixel(self, xyz) -> tuple | None:
        """Project a 3D world-space point to (px, py) image coordinates.

        Uses the view/projection matrices from the last call to step().
        Returns None if the point is behind the camera or matrices aren't set.
        """
        if self._last_view is None:
            return None
        # PyBullet returns matrices as 16-element tuples in column-major order.
        # numpy reshape gives row-major, so .T converts to the standard 4×4 form.
        V = np.array(self._last_view).reshape(4, 4).T
        P = np.array(self._last_proj).reshape(4, 4).T
        pt  = np.array([xyz[0], xyz[1], xyz[2], 1.0])
        clip = P @ V @ pt
        if abs(clip[3]) < 1e-7 or clip[3] < 0:
            return None
        ndc = clip[:3] / clip[3]
        px = int((ndc[0] + 1.0) * 0.5 * IMG_W)
        py = int((1.0 - (ndc[1] + 1.0) * 0.5) * IMG_H)
        return px, py

    def close(self):
        if _PB and self._phys_id is not None:
            p.disconnect(self._phys_id)
            self._phys_id = None


if __name__ == "__main__":
    import time
    env = PerceptionTrainingEnv()
    t0  = time.time()
    for _ in range(30):
        frame, pos = env.step(time.time() - t0)
        print(f"frame {frame.shape}  sphere {pos[0]:+.2f} {pos[1]:+.2f} {pos[2]:+.2f}")
        time.sleep(1/10)
    env.close()
