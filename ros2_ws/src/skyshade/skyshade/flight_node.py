"""
Sub-2 Flight Control — ROS2 node.

Subscribes to:
  /skyshade/user_position   geometry_msgs/Point   — target offset (dx, dy, dz)
  /skyshade/nav_override    std_msgs/String       — CONTINUE | RTH | LAND_NOW

Publishes:
  /skyshade/flight_cmd      geometry_msgs/Twist   — velocity setpoint

The node runs at 20 Hz.  On each tick it:
  1. Checks whether Sub-4 has issued an override command.
  2. If not CONTINUE, hands control to the RTH / LAND_NOW handler.
  3. Otherwise, maps the current (dx, dy, dz, wind) to a discrete state,
     queries the Q-table, and translates the action to a Twist message.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import String

from sub2_flight.policy import FlightPolicy
from sub2_flight.env.hover_env import (
    N_DX, N_DY, N_DZ, N_WIND, DX_EDGES, DY_EDGES, DZ_EDGES,
    TARGET_ALTITUDE, _wind_speed_to_idx, _digitise
)

CONTROL_HZ = 20.0
STEP_VEL = 0.5   # m/s per action step


def _state_from_offset(dx, dy, dz, wind_speed) -> int:
    dx_idx   = int(np.clip(_digitise(dx,  DX_EDGES), 0, N_DX - 1))
    dy_idx   = int(np.clip(_digitise(dy,  DY_EDGES), 0, N_DY - 1))
    dz_idx   = int(np.clip(_digitise(dz,  DZ_EDGES), 0, N_DZ - 1))
    wind_idx = _wind_speed_to_idx(wind_speed)
    return dx_idx * N_DY * N_DZ * N_WIND + dy_idx * N_DZ * N_WIND + dz_idx * N_WIND + wind_idx


# Action index → (vx, vy, vz)
ACTION_VEL = {
    0: ( 0,  STEP_VEL, 0),   # NORTH
    1: ( 0, -STEP_VEL, 0),   # SOUTH
    2: ( STEP_VEL, 0, 0),    # EAST
    3: (-STEP_VEL, 0, 0),    # WEST
    4: ( 0, 0,  STEP_VEL),   # UP
    5: ( 0, 0, -STEP_VEL),   # DOWN
    6: ( 0, 0, 0),           # HOLD
}


class FlightNode(Node):
    def __init__(self):
        super().__init__("flight_node")

        self._policy = FlightPolicy()

        self._user_pos = np.zeros(3)   # Latest (dx, dy, dz) from Sub-1
        self._wind_speed = 0.0         # Estimated wind from environment
        self._override = "CONTINUE"    # Latest override from Sub-4

        # Subscriptions
        self.create_subscription(Point,  "/skyshade/user_position", self._on_position, 10)
        self.create_subscription(String, "/skyshade/nav_override",   self._on_override, 10)

        # Publisher
        self._cmd_pub = self.create_publisher(Twist, "/skyshade/flight_cmd", 10)

        self._timer = self.create_timer(1.0 / CONTROL_HZ, self._tick)
        self.get_logger().info("FlightNode started.")

    def _on_position(self, msg: Point):
        self._user_pos = np.array([msg.x, msg.y, msg.z])

    def _on_override(self, msg: String):
        self._override = msg.data

    def _tick(self):
        twist = Twist()

        if self._override == "RTH":
            # Fly toward home (origin)
            twist.linear.x = float(np.clip(-self._user_pos[0] * 0.5, -STEP_VEL, STEP_VEL))
            twist.linear.y = float(np.clip(-self._user_pos[1] * 0.5, -STEP_VEL, STEP_VEL))
            twist.linear.z = 0.0
        elif self._override == "LAND_NOW":
            # Descend at fixed rate
            twist.linear.z = -STEP_VEL
        else:
            # Normal flight — Q-table lookup
            dx, dy, dz_abs = self._user_pos
            # dz in body frame: positive = user is below (normal hover condition)
            dz_offset = dz_abs - TARGET_ALTITUDE
            state = _state_from_offset(dx, dy, dz_offset, self._wind_speed)
            action = self._policy.select_action(state)
            vx, vy, vz = ACTION_VEL[action]
            twist.linear.x = float(vx)
            twist.linear.y = float(vy)
            twist.linear.z = float(vz)

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
