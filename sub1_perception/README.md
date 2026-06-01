# Sub-1 Perception

Sub-1 Perception is the vision subsystem for SkyShade. Its job is to detect the simulated user marker in the drone camera feed and convert that image detection into a 3-D position offset that the flight controller can use.

The final implementation uses a simple RGB camera pipeline with HSV colour tracking instead of a deep-learning detector. This is a good fit for the simulation because the user is represented by a clear red marker, so the system can be fast, explainable, and reliable without needing a large image dataset.

## What It Does

1. Receives an RGB frame from the simulated downward-facing camera.
2. Converts the frame into HSV colour space.
3. Thresholds the red marker colour range.
4. Cleans the mask with morphology to reduce noise.
5. Finds the largest valid contour.
6. Calculates the marker centre and apparent pixel width.
7. Estimates the user's 3-D offset using a pinhole camera model.
8. Smooths the output using an exponential moving average.
9. Handles short occlusions by holding the last known position briefly.

The tracker returns:

```text
position:   (dx, dy, dz) in metres
confidence: value from 0.0 to 1.0
```

## Main Files

| File | Purpose |
|---|---|
| `tracker.py` | Detects the red marker, estimates confidence, smooths position, and handles occlusion. |
| `distance_estimator.py` | Converts marker pixel position and width into real-world body-frame offsets. |
| `test_perception.py` | Runs a synthetic validation test with known ground-truth marker positions. |

## Key Design Choices

### HSV Marker Tracking

The simulated user marker is bright red, so HSV thresholding is used to isolate it from the background. Red wraps around the HSV hue boundary, so the tracker checks both low-red and high-red hue ranges.

### Pinhole Distance Estimation

The system knows the marker's physical width, the camera resolution, and the camera field of view. From this, it estimates focal length and distance:

```text
focal_length_px = (image_width_px / 2) / tan(FOV / 2)
distance_m = (marker_width_m * focal_length_px) / apparent_width_px
```

This gives the user's offset from the drone:

```text
dx: user left/right offset
dy: user forward/back offset
dz: user distance below the drone
```

### Smoothing And Occlusion Handling

The tracker uses exponential moving average smoothing with `EMA_ALPHA = 0.3`. This reduces jitter between frames. If the marker disappears briefly, the system holds the last known position for up to `MAX_OCCLUSION_FRAMES = 20` frames while confidence decays.

## Important Parameters

| Parameter | Value | Meaning |
|---|---:|---|
| `CAMERA_RES` | `(640, 480)` | Simulated camera resolution. |
| `CAMERA_FOV` | `60` degrees | Horizontal camera field of view. |
| `CAMERA_FPS` | `30` | Camera frame rate. |
| `MARKER_WIDTH_M` | `0.1 m` | Known physical marker width. |
| `CONFIDENCE_THRESH` | `0.7` | Minimum confidence used for reliable tracking. |
| `MIN_CONTOUR_AREA` | `50 px^2` | Minimum blob area accepted as a valid marker. |
| `MAX_OCCLUSION_FRAMES` | `20` | Number of frames to hold last position after marker loss. |

## Validation

The subsystem is tested using synthetic frames, so it can be validated without launching PyBullet. The test draws a red marker at known positions and checks whether the tracker can recover the expected offset.

Run:

```bash
python sub1_perception/test_perception.py
```

Current validation result:

```text
Tracking continuity : 100.00%  (target > 70%)
Position MAE        : 0.0810 m  (target < 0.15 m)

Sub-1 PASSED all checks.
```

## Integrated Simulation Feed

In the full simulation, `run_sim.py` now displays the same PyBullet RGB frame that is passed into `Tracker.process_frame(frame)`. The SkyShade dashboard includes a **Sub-1 Perception Camera** panel showing the live virtual camera feed, confidence percentage, and estimated body-frame offset.

The PyBullet scene also draws a labelled **Sub-1 RGB camera** footprint under the drone so it is clear where the virtual downward camera is looking.

Run:

```bash
venv/bin/python run_sim.py
```

## Why This Approach Works

This perception subsystem is intentionally simple. In a controlled simulation, a trained object detector would add training time, extra dependencies, and more failure modes without giving a major benefit. HSV tracking gives a clear, real-time signal that is easy to explain and works well with the rest of the SkyShade pipeline.

The main limitation is that this method depends on the red marker being visible and visually distinct. In a real outdoor system, this would likely need to be replaced or extended with a more robust detector.
