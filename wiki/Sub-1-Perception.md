# Sub-1: Perception

## Responsibility

Identify the user in the simulated downward-facing camera feed and publish their position relative to the drone, along with a tracking confidence score.

## How it works

1. PyBullet renders an RGB frame from a virtual camera mounted below the drone
2. OpenCV HSV segmentation isolates the user's coloured marker
3. The largest detected contour's centroid gives the pixel-space offset
4. A pinhole focal-length model converts pixel offset to metres using the known marker width
5. An Exponential Moving Average smooths the (Δx, Δy, Δz) output to suppress per-frame jitter
6. If confidence falls below threshold the last known position is held for up to 20 frames (occlusion holdout)
7. Position and confidence are published to `/skyshade/user_position` and `/skyshade/tracking_confidence`

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

## Validation target

| Metric | Target |
|---|---|
| Tracking continuity | > 70% of frames above confidence threshold |
| Position MAE | < 0.15 m in simulation |

## Known limitations

- HSV thresholds are tuned for the simulated marker colour; different lighting conditions or marker colours require retuning
- Confidence drops during sharp user turns — mitigated by the 20-frame occlusion holdout
- No depth sensor; altitude (Δz) is estimated purely from apparent marker size via the pinhole model

## Relevant files

- `sub1_perception/tracker.py` — HSV segmentation, EMA, occlusion holdout
- `sub1_perception/distance_estimator.py` — pinhole focal-length distance model
- `sub1_perception/test_perception.py` — accuracy test on held-out frames
- `ros2_ws/src/skyshade/skyshade/perception_node.py` — ROS 2 wrapper
