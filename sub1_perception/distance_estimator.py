"""
Sub-1 Perception — Pinhole focal-length distance estimator.

Given the apparent pixel width of a marker with known physical width and the
camera's field of view, computes the slant distance to the marker and then
decomposes that into dx, dy, dz body-frame offsets.

Pinhole model:
    focal_length_px = (image_width_px / 2) / tan(FOV_rad / 2)
    distance_m      = (MARKER_WIDTH_M * focal_length_px) / apparent_width_px

The camera is assumed to be mounted pointing straight down from the drone's
centre, so:
    dz = distance_m                     (positive = below drone)
    dx = (cx - image_cx) * dz / f      (positive = East)
    dy = (image_cy - cy) * dz / f      (positive = North; y-axis flipped)
"""

import math
import numpy as np


class DistanceEstimator:
    """Stateless pinhole distance estimator."""

    # ── Public API ────────────────────────────────────────────────────────────

    @staticmethod
    def focal_length_px(image_width_px: int, fov_deg: float) -> float:
        """Return the horizontal focal length in pixels."""
        return (image_width_px / 2.0) / math.tan(math.radians(fov_deg / 2.0))

    def estimate(
        self,
        centroid_px: tuple,
        apparent_width_px: int,
        camera_res: tuple,
        fov_deg: float,
        marker_width_m: float,
    ) -> np.ndarray:
        """
        Convert a pixel detection to a (dx, dy, dz) body-frame vector in metres.

        Parameters
        ----------
        centroid_px      : (cx, cy) pixel location of the marker centre
        apparent_width_px: width of the marker bounding box in pixels
        camera_res       : (W, H) camera resolution in pixels
        fov_deg          : horizontal field of view in degrees
        marker_width_m   : known physical width of the marker in metres

        Returns
        -------
        np.ndarray shape (3,): (dx, dy, dz) in metres
        """
        if apparent_width_px <= 0:
            return np.zeros(3)

        W, H = camera_res
        cx, cy = centroid_px

        f = self.focal_length_px(W, fov_deg)  # focal length in pixels

        # Slant distance along the camera optical axis (pointing down)
        dz = (marker_width_m * f) / apparent_width_px

        # Image-plane offsets from the principal point (image centre)
        dx = (cx - W / 2.0) * dz / f   # East positive
        dy = (H / 2.0 - cy) * dz / f   # North positive (image y is downward)

        return np.array([dx, dy, dz], dtype=float)
