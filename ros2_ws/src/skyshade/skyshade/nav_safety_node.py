"""
Sub-4 Navigation and Safety — ROS2 node.

Subscribes to:
  /skyshade/tracking_confidence  std_msgs/Float32  — tracking confidence [0,1]
  /skyshade/battery_level        std_msgs/Float32  — battery percentage [0,100]

Publishes:
  /skyshade/nav_override         std_msgs/String   — CONTINUE | RTH | LAND_NOW

Runs at 5 Hz.  On each tick:
  1. Reads battery level and estimates distance to home (from a global position
     estimate maintained by the flight node or the simulator).
  2. Looks up the MDP policy table to get the safety action.
  3. Publishes the override command.

If tracking confidence drops below a threshold for too long, RTH is triggered
regardless of the MDP policy (fail-safe).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String

from sub4_nav.policy_table import NavSafetyPolicy
from sub4_nav.mdp import ACTION_NAMES, ACTION_CONTINUE

SAFETY_HZ = 5.0

# Fail-safe: if confidence stays below this for FAILSAFE_FRAMES ticks, RTH
CONFIDENCE_FAILSAFE_THRESH = 0.3
FAILSAFE_FRAMES = 30   # 6 seconds at 5 Hz

# Assume home is at (0, 0) world coordinates
HOME_POS = (0.0, 0.0)


class NavSafetyNode(Node):
    def __init__(self):
        super().__init__("nav_safety_node")

        self._policy = NavSafetyPolicy()

        self._battery_pct = 100.0
        self._confidence = 1.0
        self._low_conf_count = 0   # Consecutive low-confidence ticks

        # Drone world position — updated from PyBullet or a separate topic
        self._drone_pos = (0.0, 0.0)

        # Subscriptions
        self.create_subscription(Float32, "/skyshade/tracking_confidence",
                                 self._on_confidence, 10)
        self.create_subscription(Float32, "/skyshade/battery_level",
                                 self._on_battery, 10)

        # Publisher
        self._override_pub = self.create_publisher(String, "/skyshade/nav_override", 10)

        self._timer = self.create_timer(1.0 / SAFETY_HZ, self._tick)
        self.get_logger().info("NavSafetyNode started.")

    def _on_confidence(self, msg: Float32):
        self._confidence = float(msg.data)

    def _on_battery(self, msg: Float32):
        self._battery_pct = float(msg.data)

    def _tick(self):
        # Fail-safe: extended low confidence → force RTH
        if self._confidence < CONFIDENCE_FAILSAFE_THRESH:
            self._low_conf_count += 1
        else:
            self._low_conf_count = 0

        if self._low_conf_count >= FAILSAFE_FRAMES:
            override = "RTH"
            self.get_logger().warn("Fail-safe triggered: low tracking confidence → RTH")
        else:
            distance_m = self._distance_to_home()
            action_idx = self._policy.decide(self._battery_pct, distance_m)
            override = ACTION_NAMES[action_idx]

        msg = String()
        msg.data = override
        self._override_pub.publish(msg)

        self.get_logger().debug(
            f"Override={override}  bat={self._battery_pct:.1f}%  "
            f"dist={self._distance_to_home():.1f}m  conf={self._confidence:.2f}"
        )

    def _distance_to_home(self) -> float:
        """Return Euclidean distance from current drone position to home."""
        import math
        dx = self._drone_pos[0] - HOME_POS[0]
        dy = self._drone_pos[1] - HOME_POS[1]
        return math.hypot(dx, dy)


def main(args=None):
    rclpy.init(args=args)
    node = NavSafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
