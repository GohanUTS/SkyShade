"""
Sub-1 Perception — camera calibration helpers.

Generates synthetic training frames with a moving red marker and runs the
HSV tracker against them to measure confidence and pixel error.  Used by
the Training Grounds hub and the standalone CameraTrainingWindow.

Keeping this here rather than in run_sim.py lets the tracker be tested
and debugged independently of the full simulation.
"""

import math

import cv2
import numpy as np

from sub1_perception.tracker import (
    Tracker,
    CAMERA_RES,
    CONFIDENCE_THRESH,
    HSV_LOWER,
    HSV_UPPER,
    HSV_LOWER2,
    HSV_UPPER2,
    MIN_CONTOUR_AREA,
)

TRAINING_PIXEL_PASS = 18.0   # px — detection within this of the marker centre = pass


def camera_training_frame(t_wall):
    """Generate a synthetic scene with a moving red marker.

    Returns (frame: np.ndarray BGR uint8, expected_centre: (cx, cy)).
    The marker follows a Lissajous path so the tracker must actively track
    rather than stay locked on a static target.
    """
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

    marker_cx = int(width / 2  + math.sin(t_wall * 0.95) * 150)
    marker_cy = int(height / 2 + math.cos(t_wall * 1.18) * 58)
    radius    = int(24 + 5 * math.sin(t_wall * 0.7))

    cv2.circle(frame, (150, 322), 18, (58, 105, 216), -1)
    cv2.circle(frame, (498, 328), 20, (66, 190, 116), -1)
    cv2.circle(frame, (marker_cx, marker_cy + radius + 24), 17, (198, 139, 86), -1)
    cv2.line(frame,
             (marker_cx, marker_cy + radius + 42),
             (marker_cx - 22, marker_cy + radius + 82),
             (51, 92, 173), 6)
    cv2.line(frame,
             (marker_cx, marker_cy + radius + 42),
             (marker_cx + 23, marker_cy + radius + 82),
             (51, 92, 173), 6)

    cv2.circle(frame, (marker_cx, marker_cy), radius, (230, 24, 34), -1)
    cv2.circle(frame,
               (marker_cx - radius // 3, marker_cy - radius // 3),
               max(4, radius // 5), (255, 100, 108), -1)

    return frame, (marker_cx, marker_cy)


def camera_training_marker_box(frame):
    """Detect the red marker bounding box via HSV thresholding.

    Returns (x, y, w, h, area) or None if nothing found above MIN_CONTOUR_AREA.
    """
    hsv   = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, HSV_LOWER,  HSV_UPPER)
    mask2 = cv2.inRange(hsv, HSV_LOWER2, HSV_UPPER2)
    mask  = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)
    if area < MIN_CONTOUR_AREA:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h, area


def camera_training_step(tracker: Tracker, t_wall: float):
    """Run one calibration frame through the tracker.

    Returns (overlay, confidence, pixel_error, pos, passing) where:
        overlay     : annotated BGR frame for display
        confidence  : tracker confidence ∈ [0, 1]
        pixel_error : distance from detected centre to ground-truth centre (px)
        pos         : estimated 3-D body offset from tracker
        passing     : True when confidence ≥ CONFIDENCE_THRESH and pixel_error ≤ TRAINING_PIXEL_PASS
    """
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
        cv2.drawMarker(overlay, detected, (255, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
        cv2.putText(overlay, f"detected {pixel_error:.1f}px",
                    (max(8, x), max(26, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 2)
    else:
        cv2.putText(overlay, "searching",
                    (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 90, 90), 2)

    cv2.circle(overlay, expected, 7, (255, 255, 255), 2)
    cv2.putText(overlay, "expected",
                (expected[0] + 10, max(22, expected[1] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (226, 232, 240), 1)

    passing = confidence >= CONFIDENCE_THRESH and pixel_error <= TRAINING_PIXEL_PASS
    return overlay, confidence, pixel_error, pos, passing
