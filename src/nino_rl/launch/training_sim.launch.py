"""Start Nino, AMCL/Nav2, and the RL-only torque training interfaces."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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
            "linorobot2_mode": "false",
            # Nav2 produces bounded baseline torque; RL adds a bounded residual.
            # Both inputs are required by effort_drive's residual architecture.
            "accept_cmd_vel": "true",
            "accept_torque": "true",
            "verbosity": LaunchConfiguration("verbosity"),
            "max_wheel_torque": LaunchConfiguration("max_wheel_torque"),
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
    reset_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="rl_world_control_bridge",
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
            DeclareLaunchArgument(
                "headless", default_value="true", description="Disable Gazebo GUI while training"
            ),
            DeclareLaunchArgument(
                "rviz", default_value="false", description="Open RViz for debugging"
            ),
            DeclareLaunchArgument(
                "verbosity", default_value="1", description="Gazebo log level (0-4)"
            ),
            DeclareLaunchArgument(
                "max_wheel_torque",
                default_value="4.0",
                description="Symmetric RL motor limit in N.m (also enforced by safety adapter)",
            ),
            DeclareLaunchArgument(
                "map", default_value=default_map, description="Static map; legacy maps are unchanged"
            ),
            DeclareLaunchArgument("start_x", default_value="0.0"),
            DeclareLaunchArgument("start_y", default_value="0.0"),
            DeclareLaunchArgument("start_yaw", default_value="0.0"),
            simulator,
            reset_bridge,
            start_navigation_when_ready,
            readiness,
        ]
    )
