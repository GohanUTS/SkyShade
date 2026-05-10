"""
SkyShade simulation launch file.

Starts all four subsystem nodes in a single ROS2 launch.  PyBullet is
expected to be already running (or will be started by the perception node
on first camera pull).

Usage (from ros2_ws, with workspace sourced):
    ros2 launch skyshade skyshade_sim.launch.py

Optional arguments:
    render:=true    — open the PyBullet GUI window (default: false)
    episodes:=1     — number of evaluation episodes to run (not used at launch
                      time; consumed by individual test scripts)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    render_arg = DeclareLaunchArgument(
        "render",
        default_value="false",
        description="Open PyBullet GUI window",
    )

    perception_node = Node(
        package="skyshade",
        executable="perception_node",
        name="perception_node",
        output="screen",
        parameters=[{"render": LaunchConfiguration("render")}],
    )

    flight_node = Node(
        package="skyshade",
        executable="flight_node",
        name="flight_node",
        output="screen",
    )

    env_decision_node = Node(
        package="skyshade",
        executable="env_decision_node",
        name="env_decision_node",
        output="screen",
    )

    nav_safety_node = Node(
        package="skyshade",
        executable="nav_safety_node",
        name="nav_safety_node",
        output="screen",
    )

    return LaunchDescription([
        render_arg,
        LogInfo(msg="=== SkyShade simulation starting ==="),
        perception_node,
        flight_node,
        env_decision_node,
        nav_safety_node,
        LogInfo(msg="=== All four nodes launched ==="),
    ])
