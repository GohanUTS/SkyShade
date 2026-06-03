"""
Sub-1 Perception — HSV marker tracker with EMA smoothing.

Receives an RGB frame (H x W x 3 uint8), isolates the user's coloured marker
via HSV thresholding, computes pixel-space centroid, then hands the centroid
to DistanceEstimator to get a 3-D body-frame offset (dx, dy, dz).

Occlusion handling: if no marker is found, the last known position is held for
up to MAX_OCCLUSION_FRAMES frames before confidence drops to zero.

Shadow handling: a secondary HSV band (lower Value floor) catches the marker
in underpass / forest shade where the red appears darker than normal.
"""

import cv2
import numpy as np

# ── Tunable parameters ────────────────────────────────────────────────────────
EMA_ALPHA = 0.3          # EMA weight on new observation; higher = less smoothing
CONFIDENCE_THRESH = 0.7  # Minimum confidence to publish a position
MAX_OCCLUSION_FRAMES = 35  # Frames to hold last known position before giving up
                            # Increased from 20 → 35 to survive bridge/underpass shadow
MARKER_WIDTH_M = 0.1     # Known physical width of the user's marker (metres)

# Simulated camera intrinsics (must match PyBullet virtual camera setup)
CAMERA_FOV = 60          # Degrees, horizontal field of view
CAMERA_RES = (640, 480)  # (width, height) in pixels
CAMERA_FPS = 30

# HSV colour bounds for the user's marker (bright red by default)
# Primary band — well-lit conditions (Value ≥ 55 catches moderate shadow too)
HSV_LOWER  = np.array([0,   120,  55], dtype=np.uint8)
HSV_UPPER  = np.array([10,  255, 255], dtype=np.uint8)
HSV_LOWER2 = np.array([170, 120,  55], dtype=np.uint8)
HSV_UPPER2 = np.array([180, 255, 255], dtype=np.uint8)

# Shadow band — bridge/underpass/canopy shadow; looser saturation + lower value.
# Only fires when primary detection misses, to avoid false positives in sunlight.
HSV_SHADOW_LOWER  = np.array([0,   80, 30], dtype=np.uint8)
HSV_SHADOW_UPPER  = np.array([12, 255, 80], dtype=np.uint8)
HSV_SHADOW_LOWER2 = np.array([168, 80, 30], dtype=np.uint8)
HSV_SHADOW_UPPER2 = np.array([180, 255, 80], dtype=np.uint8)

# Minimum contour area (px²) to consider a detection valid
MIN_CONTOUR_AREA = 50
# Shadow detections are noisier — require a slightly larger blob
MIN_SHADOW_AREA  = 80
# Area (px²) at which confidence saturates to 1.0.
# Reduced from 500 (10× min) to 250 (5× min) so distant/partially-occluded targets
# that produce small-but-valid blobs still report high confidence.
FULL_CONF_AREA   = 250
# EMA alpha for confidence smoothing — damps single-frame occlusion dips
CONF_EMA_ALPHA   = 0.35
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
        self._conf_ema = 0.0       # EMA-smoothed confidence (published value)

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
        centroid, apparent_width_px, area_px2, shadowed = self._detect_marker(rgb_frame)
        self.pixel_centroid = centroid  # expose for visual servo (None when lost)

        if centroid is not None:
            # Valid detection — compute 3-D offset
            raw_pos = self._estimator.estimate(
                centroid, apparent_width_px, CAMERA_RES, CAMERA_FOV, MARKER_WIDTH_M
            )
            self._update_ema(raw_pos)
            self._occlusion_count = 0
            confidence = self._compute_confidence(area_px2, shadowed)
        else:
            # No detection — increment occlusion counter
            self._occlusion_count += 1
            # Confidence decays linearly to 0 over MAX_OCCLUSION_FRAMES
            confidence = max(
                0.0,
                self._last_confidence * (1 - self._occlusion_count / MAX_OCCLUSION_FRAMES),
            )

        self._last_confidence = confidence
        # EMA smoothing on published confidence to dampen single-frame dips
        self._conf_ema = (CONF_EMA_ALPHA * confidence
                          + (1 - CONF_EMA_ALPHA) * self._conf_ema)
        return self._ema_pos.copy(), self._conf_ema

    # ── Private helpers ───────────────────────────────────────────────────────

    def _detect_marker(self, rgb_frame: np.ndarray):
        """
        Run HSV segmentation and find the largest qualifying contour.

        Returns (centroid_px, width_px, area_px2, shadowed) or (None, None, None, False).
        centroid_px is (cx, cy) in image coordinates (top-left origin).
        shadowed is True when the detection came from the low-light band.
        """
        hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)

        # Primary detection (normal / well-lit conditions)
        mask1 = cv2.inRange(hsv, HSV_LOWER,  HSV_UPPER)
        mask2 = cv2.inRange(hsv, HSV_LOWER2, HSV_UPPER2)
        mask  = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Note: no dilation on the primary band — dilation shifts the centroid
        # when branches or posts partially occlude the marker (forest/rooftop),
        # degrading hover position accuracy even though blob area increases.

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = self._best_contour(contours, MIN_CONTOUR_AREA)
        if result is not None:
            return (*result, False)   # not shadowed

        # Shadow fallback — only if primary missed, to avoid sunlit false positives
        smask1 = cv2.inRange(hsv, HSV_SHADOW_LOWER,  HSV_SHADOW_UPPER)
        smask2 = cv2.inRange(hsv, HSV_SHADOW_LOWER2, HSV_SHADOW_UPPER2)
        smask  = cv2.bitwise_or(smask1, smask2)
        smask  = cv2.morphologyEx(smask, cv2.MORPH_OPEN, kernel)
        # Dilate shadow band to merge fragmented low-lit detections
        smask  = cv2.dilate(smask, kernel, iterations=1)

        scontours, _ = cv2.findContours(smask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = self._best_contour(scontours, MIN_SHADOW_AREA)
        if result is not None:
            return (*result, True)    # shadowed

        return None, None, None, False

    @staticmethod
    def _best_contour(contours, min_area):
        """Return (centroid, width, area) for the largest contour above min_area."""
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < min_area:
            return None
        x, y, w, h = cv2.boundingRect(largest)
        cx = x + w // 2
        cy = y + h // 2
        return (cx, cy), w, area

    def _update_ema(self, raw_pos: np.ndarray):
        if not self._ema_initialised:
            self._ema_pos = raw_pos.copy()
            self._ema_initialised = True
        else:
            self._ema_pos = EMA_ALPHA * raw_pos + (1 - EMA_ALPHA) * self._ema_pos

    def _compute_confidence(self, area_px2: float, shadowed: bool = False) -> float:
        """
        Heuristic confidence based on detected blob area.

        Uses FULL_CONF_AREA (250 px²) so distant / partially-occluded targets
        that produce smaller-than-ideal blobs still score above the 0.7 lock
        threshold, reducing spurious track-lost events in crowd environments.
        Shadow detections are capped at 0.85 to reflect lower band reliability.
        """
        ratio = area_px2 / FULL_CONF_AREA
        conf  = float(np.clip(ratio, 0.0, 1.0))
        if shadowed:
            conf = min(conf, 0.85)
        return conf
