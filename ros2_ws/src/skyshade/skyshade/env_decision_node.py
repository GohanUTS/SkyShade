"""
Sub-3 Environmental Decision — ROS2 node.

Reads simulated weather sensor data from the PyBullet environment, runs the
SVM classifier with hysteresis, and publishes umbrella commands.

Published topics:
  /skyshade/umbrella_cmd    std_msgs/String    — "DEPLOY" or "STOW"

The node runs at 1 Hz — weather decisions do not need high frequency.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from skyshade.sub3_env.classifier import UmbrellaClassifier

DECISION_HZ = 1.0   # 1 decision per second is sufficient for weather


class EnvDecisionNode(Node):
    def __init__(self):
        super().__init__("env_decision_node")

        self._classifier = UmbrellaClassifier()

        self._pub = self.create_publisher(String, "/skyshade/umbrella_cmd", 10)
        self._timer = self.create_timer(1.0 / DECISION_HZ, self._tick)

        self.get_logger().info("EnvDecisionNode started.")

    def _tick(self):
        lux, rain, wind = self._read_sensors()

        action = self._classifier.predict(lux, rain, wind)
        msg = String()
        msg.data = "DEPLOY" if action == 1 else "STOW"
        self._pub.publish(msg)
        self.get_logger().debug(f"Umbrella: {msg.data}  (lux={lux:.0f}, rain={rain:.2f}, wind={wind:.1f})")

    def _read_sensors(self) -> tuple:
        """
        Pull lux / rain / wind readings from the PyBullet simulation.

        Sensor values are stored as PyBullet user debug parameters or read
        from a shared memory segment published by the main sim loop.  If
        PyBullet is not available, synthetic values are returned so the node
        can be started independently for unit testing.
        """
        try:
            import pybullet as p
            # These parameter IDs are registered by the main sim initialiser.
            # The sim stores them in /tmp/skyshade_sensor_ids.json on startup.
            import json
            ids_path = "/tmp/skyshade_sensor_ids.json"
            if os.path.exists(ids_path):
                with open(ids_path) as f:
                    ids = json.load(f)
                lux  = p.readUserDebugParameter(ids["lux"])
                rain = p.readUserDebugParameter(ids["rain"])
                wind = p.readUserDebugParameter(ids["wind"])
            else:
                raise RuntimeError("Sensor IDs not found.")
        except Exception:
            # Fallback: clear-sky defaults (will produce STOW)
            lux  = 80_000.0
            rain = 0.01
            wind = 1.0

        return float(lux), float(rain), float(wind)


def main(args=None):
    rclpy.init(args=args)
    node = EnvDecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
