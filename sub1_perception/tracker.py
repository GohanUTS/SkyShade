"""
Sub-1 Perception — HSV marker tracker with EMA smoothing.

Receives an RGB frame (H x W x 3 uint8), isolates the user's coloured marker
via HSV thresholding, computes pixel-space centroid, then hands the centroid
to DistanceEstimator to get a 3-D body-frame offset (dx, dy, dz).

Occlusion handling: if no marker is found, the last known position is held for
up to MAX_OCCLUSION_FRAMES frames before confidence drops to zero.
"""

import cv2
import numpy as np

# ── Tunable parameters ────────────────────────────────────────────────────────
EMA_ALPHA = 0.3          # EMA weight on new observation; higher = less smoothing
CONFIDENCE_THRESH = 0.7  # Minimum confidence to publish a position
MAX_OCCLUSION_FRAMES = 20  # Frames to hold last known position before giving up
MARKER_WIDTH_M = 0.1     # Known physical width of the user's marker (metres)

# Simulated camera intrinsics (must match PyBullet virtual camera setup)
CAMERA_FOV = 60          # Degrees, horizontal field of view
CAMERA_RES = (640, 480)  # (width, height) in pixels
CAMERA_FPS = 30

# HSV colour bounds for the user's marker (bright red by default)
# Adjust these if you change the marker colour in the simulation.
HSV_LOWER = np.array([0, 120, 70], dtype=np.uint8)
HSV_UPPER = np.array([10, 255, 255], dtype=np.uint8)
# Red wraps around 180°; include the upper-hue band as well.
HSV_LOWER2 = np.array([170, 120, 70], dtype=np.uint8)
HSV_UPPER2 = np.array([180, 255, 255], dtype=np.uint8)

# Minimum contour area (px²) to consider a detection valid
MIN_CONTOUR_AREA = 50
# ─────────────────────────────────────────────────────────────────────────────


class Tracker:
    """Stateful HSV tracker.  One instance lives for the life of the node."""

    def __init__(self, distance_estimator):
        self._estimator = distance_estimator

        # EMA state — initialised on first valid detection
        self._ema_pos = np.zeros(3, dtype=float)  # (dx, dy, dz) in metres
        self._ema_initialised = False

        self._occlusion_count = 0  # Frames since last valid detection
        self._last_confidence = 0.0

        # Exposed for the visual servo in the sim loop:
        # pixel coordinates (cx, cy) of the last detected centroid, or None.
        self.pixel_centroid: tuple = None

    # ── Public API ────────────────────────────────────────────────────────────

    def process_frame(self, rgb_frame: np.ndarray):
        """
        Process one RGB frame and return (position, confidence).

        position : np.ndarray shape (3,) — (dx, dy, dz) in the drone body frame
                   dx > 0 → user is to the right (East)
                   dy > 0 → user is ahead (North)
                   dz > 0 → user is below the drone (expected: always positive)
        confidence: float in [0, 1]
        """
        centroid, apparent_width_px, area_px2 = self._detect_marker(rgb_frame)
        self.pixel_centroid = centroid  # expose for visual servo (None when lost)

        if centroid is not None:
            # Valid detection — compute 3-D offset
            raw_pos = self._estimator.estimate(
                centroid, apparent_width_px, CAMERA_RES, CAMERA_FOV, MARKER_WIDTH_M
            )
            self._update_ema(raw_pos)
            self._occlusion_count = 0
            confidence = self._compute_confidence(area_px2)
        else:
            # No detection — increment occlusion counter
            self._occlusion_count += 1
            # Confidence decays linearly to 0 over MAX_OCCLUSION_FRAMES
            confidence = max(
                0.0,
                self._last_confidence * (1 - self._occlusion_count / MAX_OCCLUSION_FRAMES),
            )

        self._last_confidence = confidence
        return self._ema_pos.copy(), confidence

    # ── Private helpers ───────────────────────────────────────────────────────

    def _detect_marker(self, rgb_frame: np.ndarray):
        """
        Run HSV segmentation and find the largest qualifying contour.

        Returns (centroid_px, width_px, area_px2) or (None, None, None).
        centroid_px is (cx, cy) in image coordinates (top-left origin).
        """
        hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)

        mask1 = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
        mask2 = cv2.inRange(hsv, HSV_LOWER2, HSV_UPPER2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morphological cleanup to remove salt-and-pepper noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < MIN_CONTOUR_AREA:
            return None, None, None

        x, y, w, h = cv2.boundingRect(largest)
        cx = x + w // 2
        cy = y + h // 2
        return (cx, cy), w, area  # Use bounding-box width as apparent marker width

    def _update_ema(self, raw_pos: np.ndarray):
        if not self._ema_initialised:
            self._ema_pos = raw_pos.copy()
            self._ema_initialised = True
        else:
            self._ema_pos = EMA_ALPHA * raw_pos + (1 - EMA_ALPHA) * self._ema_pos

    def _compute_confidence(self, area_px2: float) -> float:
        """
        Heuristic confidence based on detected blob area.

        A blob clearly above the minimum detection area gets high confidence.
        Confidence saturates at 1.0 once the blob is ≥ FULL_CONF_AREA pixels².
        This is distance-independent: a clearly segmented blob is trustworthy
        regardless of how far away the user is.
        """
        # Area at which confidence saturates to 1.0 (~10× minimum detection area)
        FULL_CONF_AREA = MIN_CONTOUR_AREA * 10.0
        ratio = area_px2 / FULL_CONF_AREA
        return float(np.clip(ratio, 0.0, 1.0))
