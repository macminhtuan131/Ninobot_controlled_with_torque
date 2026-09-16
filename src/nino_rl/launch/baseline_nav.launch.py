"""Run the same hallway task with the normal Linorobot2/Nav2 controller."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    default_map = PathJoinSubstitution(
        [FindPackageShare("linorobot2_navigation"), "maps", "long_hall.yaml"]
    )
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
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("linorobot2_navigation"),
                    "launch",
                    "navigation.launch.py",
                ]
            )
        ),
        launch_arguments={
            "sim": "true",
            "rviz": LaunchConfiguration("rviz"),
            "map": LaunchConfiguration("map"),
            "initial_pose_x": LaunchConfiguration("start_x"),
            "initial_pose_y": LaunchConfiguration("start_y"),
            "initial_pose_yaw": LaunchConfiguration("start_yaw"),
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
    readiness = Node(
        package="nino_rl",
        executable="wait_for_sim",
        name="nino_sim_readiness",
        output="screen",
    )
    start_navigation_when_ready = RegisterEventHandler(
        OnProcessExit(target_action=readiness, on_exit=[navigation])
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("verbosity", default_value="1"),
            DeclareLaunchArgument("map", default_value=default_map),
            DeclareLaunchArgument("start_x", default_value="0.0"),
            DeclareLaunchArgument("start_y", default_value="0.0"),
            DeclareLaunchArgument("start_yaw", default_value="0.0"),
            simulator,
            world_services,
            start_navigation_when_ready,
            readiness,
        ]
    )
