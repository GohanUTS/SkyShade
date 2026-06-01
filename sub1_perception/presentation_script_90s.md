# Sub-1 Perception Presentation Script

Approximate timing: 1 minute 30 seconds.

Hi, I am presenting Subsystem One, which is the perception part of SkyShade.

This subsystem lets the drone find the user in simulation. The drone has a downward-facing RGB camera, and the user is shown as a red marker. My code detects that marker and turns its image position into a 3-D offset, so the flight controller can follow the user.

I used HSV colour tracking rather than deep learning. In this environment, the marker is controlled and easy to distinguish, so a neural detector would add complexity without much benefit. Each frame is converted from RGB to HSV, then the red colour range is thresholded. Because red wraps around the HSV hue scale, the tracker combines two red masks.

Next, the mask is cleaned, the largest valid contour is selected, and the marker centre and width are measured. Since the real marker width is known, the distance estimator uses a pinhole camera model, camera field of view, and pixel width to estimate distance in metres. The output is dx, dy, and dz: sideways offset, forward offset, and distance below the drone.

I also added smoothing to reduce jitter, plus short occlusion handling so brief marker loss does not immediately break tracking.

For testing, I generated 200 synthetic frames with known ground truth. It achieved 100 percent continuity and 0.081 metres mean error, beating the 0.15 metre target.

Overall, Subsystem One gives SkyShade a reliable and explainable way to track the user.
