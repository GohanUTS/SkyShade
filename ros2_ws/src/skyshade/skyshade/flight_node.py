"""
Sub-2 Flight Control — ROS2 node.

Subscribes to:
  /skyshade/user_position   geometry_msgs/Point   — target offset (dx, dy, dz)
  /skyshade/nav_override    std_msgs/String       — CONTINUE | RTH | LAND_NOW

Publishes:
  /skyshade/flight_cmd      geometry_msgs/Twist   — velocity setpoint

The node runs at 20 Hz and uses a PID controller to compute velocity
setpoints from the position error reported by Sub-1.  Velocity is estimated
by finite-differencing successive position messages.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import String

from skyshade.sub2_flight.env.hover_env import TARGET_ALTITUDE

CONTROL_HZ = 20.0
MAX_VEL = 1.0        # m/s output clamp per axis

# PID gains — translate position error to velocity setpoint
KP_XY = 1.2
KD_XY = 0.6
KI_XY = 0.05
KP_Z  = 1.5
KD_Z  = 0.8
KI_Z  = 0.1
INTEGRAL_CLAMP = 1.0


class FlightNode(Node):
    def __init__(self):
        super().__init__("flight_node")

        self._user_offset = np.zeros(3)   # Latest (dx, dy, dz) from Sub-1
        self._prev_error  = np.zeros(3)
        self._integral    = np.zeros(3)
        self._prev_time   = time.monotonic()
        self._override    = "CONTINUE"

        # Subscriptions
        self.create_subscription(Point,  "/skyshade/user_position", self._on_position, 10)
        self.create_subscription(String, "/skyshade/nav_override",   self._on_override, 10)

        # Publisher
        self._cmd_pub = self.create_publisher(Twist, "/skyshade/flight_cmd", 10)

        self._timer = self.create_timer(1.0 / CONTROL_HZ, self._tick)
        self.get_logger().info("FlightNode (PID) started.")

    def _on_position(self, msg: Point):
        self._user_offset = np.array([msg.x, msg.y, msg.z])

    def _on_override(self, msg: String):
        self._override = msg.data

    def _tick(self):
        now = time.monotonic()
        dt  = max(now - self._prev_time, 1e-3)
        self._prev_time = now

        twist = Twist()

        if self._override == "RTH":
            # Fly back toward home (zero out lateral offset)
            twist.linear.x = float(np.clip(-self._user_offset[0] * 0.5, -MAX_VEL, MAX_VEL))
            twist.linear.y = float(np.clip(-self._user_offset[1] * 0.5, -MAX_VEL, MAX_VEL))
            twist.linear.z = 0.0
            self._integral[:] = 0.0

        elif self._override == "LAND_NOW":
            twist.linear.z = -0.5
            self._integral[:] = 0.0

        else:
            # PID hover — drive lateral offset to zero, altitude to TARGET_ALTITUDE
            error = np.array([
                -self._user_offset[0],          # want dx → 0
                -self._user_offset[1],          # want dy → 0
                TARGET_ALTITUDE - self._user_offset[2],  # want dz at target alt
            ])

            d_error = (error - self._prev_error) / dt

            self._integral += error * dt
            self._integral = np.clip(self._integral, -INTEGRAL_CLAMP, INTEGRAL_CLAMP)

            vx = KP_XY * error[0] + KD_XY * d_error[0] + KI_XY * self._integral[0]
            vy = KP_XY * error[1] + KD_XY * d_error[1] + KI_XY * self._integral[1]
            vz = KP_Z  * error[2] + KD_Z  * d_error[2] + KI_Z  * self._integral[2]

            twist.linear.x = float(np.clip(vx, -MAX_VEL, MAX_VEL))
            twist.linear.y = float(np.clip(vy, -MAX_VEL, MAX_VEL))
            twist.linear.z = float(np.clip(vz, -MAX_VEL, MAX_VEL))

            self._prev_error = error.copy()

        self._cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = FlightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
