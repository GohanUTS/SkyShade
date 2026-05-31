# Sub-1: Perception

## Responsibility

Identify the user in the simulated downward-facing camera feed and publish their position relative to the drone, along with a tracking confidence score.

---

## How it works

1. PyBullet renders an RGB frame from a virtual camera mounted below the drone
2. OpenCV HSV segmentation isolates the user's coloured marker
3. The largest detected contour's centroid gives the pixel-space offset
4. A pinhole focal-length model converts pixel offset to metres using the known marker width
5. An Exponential Moving Average smooths the (Δx, Δy, Δz) output to suppress per-frame jitter
6. If confidence falls below threshold the last known position is held for up to 20 frames (occlusion holdout)
7. Position and confidence are published to `/skyshade/user_position` and `/skyshade/tracking_confidence`

---

## Key parameters

| Parameter | Value | Description |
|---|---|---|
| `EMA_ALPHA` | 0.3 | Smoothing factor — higher = less smoothing |
| `CONFIDENCE_THRESH` | 0.7 | Minimum confidence to publish a position |
| `MAX_OCCLUSION_FRAMES` | 20 | Frames to hold last known position when marker is lost |
| `MARKER_WIDTH_M` | 0.1 | Known physical width of the user's marker (metres) |
| `CAMERA_FOV` | 60° | Virtual camera field of view |
| `CAMERA_RES` | 640 × 480 | Virtual camera resolution |
| `CAMERA_FPS` | 30 | Frames per second |
| `TRAINING_PIXEL_PASS` | 18 px | Detection must land within this many pixels of truth to pass |

---

## Training Grounds — calibration room

Sub-1 uses a deterministic HSV tracker — no neural network model is trained. The Training Grounds hub provides a live calibration check to verify the tracker works correctly before the main sim is launched.

### What the calibration room shows

Open the launcher → **Calibrate** → Sub-1 Perception tab.

A PyBullet room (`sub1_perception/training_env.py`) renders a live camera view in the left panel:

| Visual element | Meaning |
|---|---|
| **Green bounding box** | Where the HSV tracker detected the red sphere |
| **White crosshair** | Centre of the detected region |
| **Yellow dot** | Where the sphere actually is (projected from its 3D world position) |
| **Blue acceptance ring** | 18 px radius — detection centre must fall inside this to pass |
| **"err Xpx" label** | Pixel error — distance between detected and true centres |
| Box turns **orange** | Detection found but outside the acceptance zone |

**Status bar** shows: `LOCKED ✓  confidence 1.00  pixel error 0.0px` when tracking correctly, `SEARCHING` when the sphere is not detected.

### Right chart

- **Blue line** — tracker confidence [0–1] over time, with dashed threshold
- **Orange line** — pixel error (px) over time, with dashed pass threshold (18 px)
- A green shaded region fills above the confidence threshold while locked

### The calibration environment

`sub1_perception/training_env.py` builds a 6×6 m room with:
- Grey-blue walls, floor with grid markers
- A **red sphere** (0.18 m radius) following a 3D Lissajous path
- A **camera** mounted on the south wall that gently pans to simulate a drone gimbal

The sphere's projected pixel position is computed from the PyBullet camera matrices so the acceptance-zone ring always shows the true target.

---

## Validation target

| Metric | Target |
|---|---|
| Tracking continuity | > 70% of frames above `CONFIDENCE_THRESH` |
| Position MAE | < 0.15 m in simulation |

---

## Known limitations

- HSV thresholds are tuned for the simulated marker colour; different lighting or colours require retuning
- Confidence drops during sharp user turns — mitigated by the 20-frame occlusion holdout
- No depth sensor; altitude (Δz) is estimated purely from apparent marker size via the pinhole model
- Calibration room uses a synthetic scene; real-camera performance is unverified

---

## Relevant files

| File | Purpose |
|---|---|
| `sub1_perception/tracker.py` | HSV segmentation, EMA, occlusion holdout |
| `sub1_perception/distance_estimator.py` | Pinhole focal-length distance model |
| `sub1_perception/training.py` | Camera calibration helpers: `camera_training_frame()`, `camera_training_step()`, `TRAINING_PIXEL_PASS` |
| `sub1_perception/training_env.py` | PyBullet calibration room — red sphere + camera render + 3D→2D projection |
| `sub1_perception/test_perception.py` | Accuracy test on held-out frames |
| `ros2_ws/src/skyshade/skyshade/perception_node.py` | ROS 2 wrapper |
