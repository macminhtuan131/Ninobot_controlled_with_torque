"""Run the direct straight-line PI baseline without Nav2."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("nino_description"), "launch", "sim.launch.py"]
            )
        ),
        launch_arguments={
            "world": "long_hall.sdf",
            "world_name": "long_hall",
            "headless": LaunchConfiguration("headless"),
            "rviz": "false",
            "sensor_monitor": "false",
            "start_effort_drive": "true",
            "accept_cmd_vel": "true",
            "accept_torque": "false",
            "verbosity": LaunchConfiguration("verbosity"),
        }.items(),
    )
    world_services = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="baseline_world_services",
        output="screen",
        arguments=[
            "/world/long_hall/control@ros_gz_interfaces/srv/ControlWorld",
            "/world/long_hall/set_pose@ros_gz_interfaces/srv/SetEntityPose",
            "/world/long_hall/create@ros_gz_interfaces/srv/SpawnEntity",
            "/world/long_hall/remove@ros_gz_interfaces/srv/DeleteEntity",
        ],
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("rviz", default_value="false"),
            DeclareLaunchArgument("verbosity", default_value="1"),
            simulator,
            world_services,
        ]
    )
