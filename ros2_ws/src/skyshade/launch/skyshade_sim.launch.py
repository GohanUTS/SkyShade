"""
SkyShade simulation launch file.

Starts the PyBullet simulation and all four subsystem nodes in a single
ROS2 launch.

Usage, from ros2_ws with workspace sourced:

    ros2 launch skyshade skyshade_sim.launch.py render:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    render_arg = DeclareLaunchArgument(
        "render",
        default_value="false",
        description="Open PyBullet GUI window",
    )

    sim_process = ExecuteProcess(
        cmd=[
            "python3",
            "/mnt/c/Users/Azzaw/OneDrive - UTS/Documents/Uni/Year 4/AI in Robotics/SkyShade/run_sim.py",
        ],
        output="screen",
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
        sim_process,
        perception_node,
        flight_node,
        env_decision_node,
        nav_safety_node,
        LogInfo(msg="=== PyBullet simulation and all four nodes launched ==="),
    ])