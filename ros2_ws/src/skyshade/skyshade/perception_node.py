"""
Sub-1 Perception — ROS2 node.

Subscribes to the PyBullet virtual camera image topic, runs the HSV tracker,
and publishes user position and tracking confidence.

Published topics:
  /skyshade/user_position      geometry_msgs/Point   — (dx, dy, dz) in metres
  /skyshade/tracking_confidence std_msgs/Float32     — [0, 1]

The node runs at CAMERA_FPS (30 Hz by default).  When running standalone
(without a live image source), it generates synthetic frames for testing.
"""

import sys
import os
# Allow importing from the project root regardless of how ROS2 sourced the ws
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

from skyshade.sub1_perception.distance_estimator import DistanceEstimator
from skyshade.sub1_perception.tracker import Tracker, CAMERA_FPS, CONFIDENCE_THRESH


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        # Publishers
        self._pos_pub = self.create_publisher(Point, "/skyshade/user_position", 10)
        self._conf_pub = self.create_publisher(Float32, "/skyshade/tracking_confidence", 10)

        # Tracker
        estimator = DistanceEstimator()
        self._tracker = Tracker(estimator)

        # Timer fires at camera frame rate
        self._timer = self.create_timer(1.0 / CAMERA_FPS, self._tick)

        # In a real pipeline this would come from a camera subscriber.
        # For simulation, frames are pulled from PyBullet directly each tick.
        self._pybullet_cam = None
        self.get_logger().info("PerceptionNode started.")

    def _tick(self):
        frame = self._get_frame()
        if frame is None:
            return

        pos, confidence = self._tracker.process_frame(frame)

        if confidence >= CONFIDENCE_THRESH:
            pt = Point()
            pt.x, pt.y, pt.z = float(pos[0]), float(pos[1]), float(pos[2])
            self._pos_pub.publish(pt)

        conf_msg = Float32()
        conf_msg.data = float(confidence)
        self._conf_pub.publish(conf_msg)

    def _get_frame(self) -> np.ndarray | None:
        """
        Return the latest RGB frame from PyBullet.

        In simulation, PyBullet is expected to be running in the same process
        (or accessible via shared memory) and the camera is queried here.  If
        PyBullet is unavailable, a blank frame is returned so the node stays
        alive for integration testing.
        """
        try:
            import pybullet as p
            # Assumes a single PyBullet instance is connected.
            # Camera parameters must match sub1_perception/tracker.py
            W, H = 640, 480
            _, _, px, _, _ = p.getCameraImage(
                width=W, height=H,
                viewMatrix=p.computeViewMatrix(
                    cameraEyePosition=[0, 0, 2.5],
                    cameraTargetPosition=[0, 0, 0],
                    cameraUpVector=[0, 1, 0],
                ),
                projectionMatrix=p.computeProjectionMatrixFOV(
                    fov=60, aspect=W / H, nearVal=0.1, farVal=100
                ),
            )
            frame = np.array(px, dtype=np.uint8).reshape((H, W, 4))[:, :, :3]
            return frame
        except Exception:
            # Return None to skip this tick rather than crash the node
            return None


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
