import ast
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import yaml

from nino_rl.ros_interface import RosRobotInterface
from nino_rl.control_v2 import ImuWindow
from nino_rl.core import RobotState


ROOT = Path(__file__).parents[3]


def test_sensor_wait_accepts_complete_snapshot_at_deadline():
    ros = SimpleNamespace(
        _lock=Lock(),
        _received={"odom", "imu", "joint", "scan"},
    )

    # A zero timeout exercises the final atomic snapshot directly. Previously
    # this raised a timeout whose missing-stream list was empty.
    RosRobotInterface.wait_for_sensors(ros, timeout=0.0)


def test_drive_reset_preserves_odometry_published_before_service_response():
    class Client:
        @staticmethod
        def wait_for_service(timeout_sec):
            return True

    ros = SimpleNamespace(
        _lock=Lock(),
        _received={"odom"},
        _received_at={"odom": 1.0},
        reset_odometry=Client(),
    )

    def reset_service(_client, _request_factory, _timeout, _operation):
        # Model the reset node's publication racing ahead of its response.
        ros._received.add("odom")
        ros._received_at["odom"] = 2.0
        return SimpleNamespace(success=True, message="reset")

    ros._call_idempotent_service = reset_service
    RosRobotInterface.reset_drive_state(ros, timeout=0.1)

    assert ros._received_at["odom"] == 2.0


def test_zero_odometry_accepts_post_request_sample_after_slow_service_response():
    ros = SimpleNamespace(
        _lock=Lock(),
        _received_at={"odom": 1.0},
        _state=RobotState(x=0.0, y=0.0, yaw=0.0),
    )
    RosRobotInterface.wait_for_zero_odometry(ros, timeout=0.1)


def test_lockstep_accepts_lost_service_response_immediately_after_clock_proof():
    class Future:
        checks = 0

        def done(self):
            self.checks += 1
            if self.checks == 2:
                ros._sim_clock_stamp = 1.05
            return False

    removed = []
    warnings = []
    ros = SimpleNamespace(
        _lock=Lock(),
        _sim_clock_stamp=1.0,
        physics_step_seconds=0.002,
        world_control=SimpleNamespace(
            wait_for_service=lambda timeout_sec: True,
            call_async=lambda request: Future(),
            remove_pending_request=lambda future: removed.append(future),
        ),
        get_logger=lambda: SimpleNamespace(warn=warnings.append),
    )

    result = RosRobotInterface.advance_world(ros, 25, timeout=1.0)
    assert result == 0.05
    assert len(removed) == 1
    assert not warnings


def test_lockstep_accepts_quiescent_near_complete_clock_interval():
    class Future:
        checks = 0

        def done(self):
            self.checks += 1
            if self.checks == 2:
                ros._sim_clock_stamp = 1.046
            return False

    removed = []
    warnings = []
    ros = SimpleNamespace(
        _lock=Lock(),
        _sim_clock_stamp=1.0,
        physics_step_seconds=0.002,
        world_control=SimpleNamespace(
            wait_for_service=lambda timeout_sec: True,
            call_async=lambda request: Future(),
            remove_pending_request=lambda future: removed.append(future),
        ),
        get_logger=lambda: SimpleNamespace(warn=warnings.append),
    )

    ros.wait_for_clock_quiescence = lambda timeout, quiet_time: ros._sim_clock_stamp
    result = RosRobotInterface.advance_world(ros, 25, timeout=0.01)
    assert result == 0.05
    assert len(removed) == 1
    assert "near-complete atomic chunk" in warnings[0]


def test_pause_accepts_lost_response_when_clock_confirms_state():
    class Future:
        @staticmethod
        def done():
            return False

    removed = []
    warnings = []
    ros = SimpleNamespace(
        _lock=Lock(),
        _sim_clock_stamp=1.0,
        physics_step_seconds=0.002,
        world_control=SimpleNamespace(
            wait_for_service=lambda timeout_sec: True,
            call_async=lambda request: Future(),
            remove_pending_request=lambda future: removed.append(future),
        ),
        get_logger=lambda: SimpleNamespace(warn=warnings.append),
        _wait_future=lambda future, timeout: (_ for _ in ()).throw(
            TimeoutError("lost response")
        ),
        wait_for_clock_quiescence=lambda timeout, quiet_time: 1.0,
    )

    RosRobotInterface.set_world_paused(ros, True, timeout=0.2)
    assert len(removed) == 1
    assert "confirms paused=True" in warnings[0]


def test_unpause_accepts_lost_response_when_clock_advances():
    class Future:
        @staticmethod
        def done():
            return False

    removed = []
    warnings = []
    ros = SimpleNamespace(
        _lock=Lock(),
        _sim_clock_stamp=1.0,
        physics_step_seconds=0.002,
        world_control=SimpleNamespace(
            wait_for_service=lambda timeout_sec: True,
            call_async=lambda request: Future(),
            remove_pending_request=lambda future: removed.append(future),
        ),
        get_logger=lambda: SimpleNamespace(warn=warnings.append),
    )

    def lose_response_after_unpausing(_future, _timeout):
        ros._sim_clock_stamp = 1.01
        raise TimeoutError("lost response")

    ros._wait_future = lose_response_after_unpausing
    RosRobotInterface.set_world_paused(ros, False, timeout=0.2)
    assert len(removed) == 1
    assert "confirms paused=False" in warnings[0]


def test_lockstep_epoch_uses_timestamp_boundary_not_imu_queue_silence():
    ros = SimpleNamespace(
        _lock=Lock(),
        _sim_clock_stamp=10.0,
        physics_step_seconds=0.002,
        _imu_epoch_min_stamp=-float("inf"),
        imu_window=ImuWindow(),
        _received={"imu", "clock"},
        _received_at={"imu": 1.0, "clock": 1.0},
    )
    ros.imu_window.add(9.99, 2.0)
    ros.wait_for_clock_quiescence = lambda timeout: ros._sim_clock_stamp

    def advance(steps, timeout):
        ros._sim_clock_stamp += steps * ros.physics_step_seconds

    ros.advance_world = advance
    result = RosRobotInterface.begin_lockstep_epoch(ros, timeout=0.2)
    assert result == 10.002
    assert ros._imu_epoch_min_stamp == 10.0
    assert not ros.imu_window.samples
    assert "imu" not in ros._received


def test_imu_callback_discards_packets_at_or_before_epoch_boundary():
    received = []
    ros = SimpleNamespace(
        _lock=Lock(),
        _state=RobotState(accel_z=1.0),
        _imu_epoch_min_stamp=10.0,
        imu_window=ImuWindow(),
        imu_includes_gravity=True,
        _mark_received=received.append,
    )

    def message(stamp, accel_z):
        return SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(
                sec=int(stamp), nanosec=int(round((stamp % 1.0) * 1e9))
            )),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            angular_velocity=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            linear_acceleration=SimpleNamespace(x=0.0, y=0.0, z=accel_z),
        )

    RosRobotInterface._imu_callback(ros, message(10.0, 99.0))
    assert ros._state.accel_z == 1.0
    assert not ros.imu_window.samples
    RosRobotInterface._imu_callback(ros, message(10.01, 9.80665))
    assert ros._state.accel_z == 9.80665
    assert ros.imu_window.samples[-1][0] == 10.01
    assert received == ["imu"]


def test_linorobot2_source_and_legacy_maps_are_merged():
    navigation = ROOT / "src" / "linorobot2" / "linorobot2_navigation"
    assert (navigation / "package.xml").exists()
    for stem in ("map", "playground", "turtlebot3_world"):
        assert (navigation / "maps" / f"{stem}.yaml").exists()
        assert (navigation / "maps" / f"{stem}.pgm").exists()
    hall = yaml.safe_load((navigation / "maps" / "long_hall.yaml").read_text())
    assert hall["image"] == "long_hall.pgm"
    assert hall["origin"] == [-3.0, -3.0, 0.0]


def test_nino_description_has_required_frames_effort_joints_and_no_diff_drive():
    xacro_path = ROOT / "src" / "nino_description" / "urdf" / "nino.urdf.xacro"
    text = xacro_path.read_text()
    root = ET.fromstring(text)
    links = {element.attrib["name"] for element in root.findall("link")}
    assert {"base_footprint", "base_link", "imu_link", "laser"} <= links
    assert "gz-sim-diff-drive-system" not in text
    for joint in ("left_wheel_joint", "right_wheel_joint"):
        control_joint = root.find(f".//ros2_control/joint[@name='{joint}']")
        assert control_joint is not None
        assert control_joint.find("command_interface[@name='effort']") is not None


def test_training_uses_direct_straight_baseline_plus_rl_residual_torque():
    launch_dir = ROOT / "src" / "nino_rl" / "launch"
    training = (launch_dir / "training_sim.launch.py").read_text()
    baseline = (launch_dir / "baseline_nav.launch.py").read_text()
    assert '"accept_cmd_vel": "true"' in training
    assert '"accept_torque": "true"' in training
    assert '"accept_cmd_vel": "true"' in baseline
    assert '"accept_torque": "false"' in baseline
    assert "navigation.launch.py" not in training
    assert "navigation.launch.py" not in baseline
    assert "linorobot2_navigation" not in training
    assert "linorobot2_navigation" not in baseline
    assert '"ROS_DOMAIN_ID"' in training
    assert '"ROS_AUTOMATIC_DISCOVERY_RANGE"' in training


def test_nino_python_tools_force_the_same_local_isolated_ros_domain():
    package_init = (
        ROOT / "src" / "nino_rl" / "nino_rl" / "__init__.py"
    ).read_text()
    simulator = (
        ROOT / "src" / "nino_description" / "launch" / "sim.launch.py"
    ).read_text()
    for text in (package_init, simulator):
        assert '"NINO_ROS_DOMAIN_ID", "77"' in text
        assert '"ROS_AUTOMATIC_DISCOVERY_RANGE"' in text


def test_episode_reset_and_step_require_fresh_terrain_preview():
    source = (ROOT / "src" / "nino_rl" / "nino_rl" / "ros_env.py").read_text()
    constructor = source[source.index("    def __init__("):source.index("    def _spin_executor(")]
    reset = source[source.index("    def reset("):source.index("    def step(")]
    step = source[source.index("    def step("):]
    assert "self.ros.set_world_paused(" in constructor
    assert "self.world_is_paused = True" in constructor
    assert "self._stop_executor_spin()" in constructor
    assert "self._start_executor_spin()" in reset
    assert "SingleThreadedExecutor()" in constructor
    assert 'reset_sensor_names = ["ground_truth", "scan"]' in reset
    assert 'reset_sensor_names.append("terrain")' in reset
    assert "refresh_reset_sensors" in reset
    assert 'sensor_markers(["scan"])' not in reset
    assert 'self.ros.sensor_markers(["terrain"])' in step
    assert "terrain_period_steps" in step
    assert "terrain_delivery_timeout = min(" in step
    assert "terrain_markers, terrain_delivery_timeout" in step


def test_preflight_keeps_straight_command_fresh_after_subscriber_discovery():
    source = (ROOT / "src" / "nino_rl" / "nino_rl" / "preflight.py").read_text()
    assert "def straight_reference_ready()" in source
    assert "node.publish_straight_command(0.0)" in source
    assert "node.stale_straight_reference_streams(stale_after)" in source


def test_reset_sensor_refresh_steps_with_stopped_motors_and_bounded_recovery():
    import pytest
    events = []
    waits = []
    def wait(markers, timeout):
        waits.append(markers)
        if len(waits) < 2:
            raise RuntimeError("render delayed")
    ros = SimpleNamespace(
        physics_step_seconds=.002,
        publish_straight_command=lambda speed: events.append(('speed', speed)),
        publish_control=lambda *cmd: events.append(('control', cmd)),
        set_world_paused=lambda value, timeout: events.append(('pause', value)),
        sensor_markers=lambda names: dict.fromkeys(names, 1.0),
        latest_clock_stamp=lambda: 2.0,
        advance_world=lambda steps, timeout: events.append(('step', steps)),
        wait_for_sensor_updates=wait,
        missing_sensor_updates=lambda markers: list(markers),
        get_logger=lambda: SimpleNamespace(warn=lambda msg: None),
    )
    RosRobotInterface.refresh_reset_sensors(ros, ['scan', 'terrain'])
    assert events[:3] == [('speed', 0.0), ('control', (0.0, 0.0, 0.0)), ('pause', True)]
    assert events[3:] == [('step', 50), ('step', 50)]
    assert waits[0] == waits[1] == {'scan': 1.0, 'terrain': 1.0}
    def never_ready(*args, **kwargs):
        raise RuntimeError('missing')
    ros.wait_for_sensor_updates = never_ready
    events.clear()
    with pytest.raises(RuntimeError, match='Reset sensor refresh 3/3.*clock'):
        RosRobotInterface.refresh_reset_sensors(ros, ['scan'])
    assert events.count(('step', 50)) == 3
    events.clear()
    ros.advance_world = never_ready
    with pytest.raises(RuntimeError, match='missing'):
        RosRobotInterface.refresh_reset_sensors(ros, ['scan'])
    assert len(events) == 3  # Failed physics commands are never blindly retried.


def test_ppo_suspends_ros_callbacks_during_optimizer_updates():
    source = (ROOT / "src" / "nino_rl" / "nino_rl" / "train.py").read_text()
    assert "env.resume_callback_dispatch()" in source
    assert "env.suspend_callback_dispatch()" in source


def test_six_phase_curriculum_and_straight_goal_are_configured():
    config_path = ROOT / "src" / "nino_rl" / "config" / "ppo.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["curriculum"]["phase_order"] == [6, 5, 4, 3, 2, 1]
    assert config["curriculum"]["phase_steps"] == 100000
    phases = config["terrain_curriculum"]["phases_hard_to_easy"]
    assert len(phases) == 6
    assert all(phase["diameter_m"] > 0 for phase in phases)
    assert [p["diameter_m"] for p in phases] == sorted(
        [p["diameter_m"] for p in phases], reverse=True
    )
    assert [p["angle_deg"] for p in phases] == sorted(
        [p["angle_deg"] for p in phases], reverse=True
    )
    assert config["navigation"]["goal_pose"][0] == 6.0
    cable_x = config["terrain_curriculum"]["cable_x_m"]
    assert cable_x == 4.0
    assert 0.0 < cable_x < config["navigation"]["goal_pose"][0]
    assert config["navigation"]["goal_pose"][0] - cable_x == 2.0
    assert config["navigation"]["cmd_vel_topic"] == "/cmd_vel"
    assert config["navigation"]["straight_speed_m_s"] == 0.75
    assert config["goal_tolerance_m"] == 0.10
    assert config["goal_capture_on_crossing"] is False
    assert config["goal_require_stopped"] is False
    control_period = 1.0 / config["control_hz"]
    physics_step = config["policy_v2"]["simulation_physics_step_seconds"]
    assert control_period / physics_step == 50
    world = ET.parse(
        ROOT / "src" / "nino_description" / "worlds" / "long_hall.sdf"
    ).getroot()
    assert float(world.find(".//physics/max_step_size").text) == physics_step
    assert config["off_path_hold_seconds"] > 0.0
    assert config["navigation_invalid_hold_seconds"] > 0.0


def test_straight_mode_has_no_nav2_runtime_dependency():
    package = (ROOT / "src" / "nino_rl" / "package.xml").read_text()
    environment = (ROOT / "src" / "nino_rl" / "nino_rl" / "ros_env.py").read_text()
    policy = (ROOT / "src" / "nino_rl" / "nino_rl" / "policy_node.py").read_text()
    assert "nav2_msgs" not in package
    assert "linorobot2_navigation" not in package
    assert "subscribe_plan=False" in environment
    assert "publish_straight_command" in environment
    assert "self.ros.configure_goal_marker(" in environment
    assert 'radius=float(self.config["goal_tolerance_m"])' in environment
    assert "subscribe_plan=False" in policy


def test_goal_marker_is_visible_but_has_no_collision_geometry():
    source_path = ROOT / "src" / "nino_rl" / "nino_rl" / "ros_interface.py"
    module = ast.parse(source_path.read_text())
    interface = next(node for node in module.body if isinstance(node, ast.ClassDef))
    method = next(
        node for node in interface.body
        if isinstance(node, ast.FunctionDef) and node.name == "_goal_marker_sdf"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"), namespace)
    root = ET.fromstring(namespace["_goal_marker_sdf"]("training_goal_marker", 0.10))
    assert len(root.findall(".//visual")) == 3
    assert root.find(".//collision") is None
    assert float(root.find(".//visual[@name='goal_disc']//radius").text) == 0.10

    configure = next(
        node for node in interface.body
        if isinstance(node, ast.FunctionDef) and node.name == "configure_goal_marker"
    )
    configure_source = ast.unparse(configure)
    assert "SetEntityPose.Request" in configure_source
    assert "DeleteEntity.Request" not in configure_source


def test_episode_entities_use_known_names_after_initial_world_adoption():
    class Client:
        srv_name = "/test"

        @staticmethod
        def wait_for_service(timeout_sec):
            return True

    operations = []
    interface = SimpleNamespace(
        delete_entity=Client(),
        spawn_entity=Client(),
        _training_cable_names=None,
        _adaptive_terrain_present=None,
        _cable_spawn_request=lambda name, x, radius, angle: object(),
        _adaptive_terrain_sdf=lambda name, features: "<sdf/>",
    )

    def call(_client, request_factory, _timeout, operation):
        request_factory()
        operations.append(operation)
        return SimpleNamespace(success=True)

    interface._call_idempotent_service = call
    RosRobotInterface.configure_training_cables(
        interface, [(4.0, 0.01, 0.0)]
    )
    initial_cable_operations = len(operations)
    assert initial_cable_operations == 2  # remove legacy model, then spawn

    RosRobotInterface.configure_training_cables(
        interface, [(4.0, 0.01, 0.0)]
    )
    assert len(operations) - initial_cable_operations == 2  # remove/spawn cable 0

    RosRobotInterface.configure_adaptive_terrain(
        interface, [("obstacle", 2.0, 0.5, 0.1)]
    )
    initial_terrain_operations = len(operations)
    RosRobotInterface.configure_adaptive_terrain(interface, [])
    assert len(operations) - initial_terrain_operations == 1  # remove known model
    RosRobotInterface.configure_adaptive_terrain(interface, [])
    assert len(operations) - initial_terrain_operations == 1  # already absent


def test_goal_marker_creates_before_using_the_move_fast_path():
    class Client:
        srv_name = "/test"

        @staticmethod
        def wait_for_service(timeout_sec):
            return True

    operations = []
    interface = SimpleNamespace(
        set_entity_pose=Client(),
        spawn_entity=Client(),
        _goal_marker_present=None,
        _goal_marker_sdf=lambda name, radius: "<sdf/>",
    )

    def call(_client, request_factory, _timeout, operation):
        request_factory()
        operations.append(operation)
        return SimpleNamespace(success=True)

    interface._call_idempotent_service = call
    RosRobotInterface.configure_goal_marker(interface, (6.0, 0.0, 0.0))
    assert operations == ["create goal marker"]
    RosRobotInterface.configure_goal_marker(interface, (5.0, 0.0, 0.0))
    assert operations[-1] == "move goal marker"


def test_robot_has_bridged_downward_terrain_laser():
    urdf = (ROOT / "src" / "nino_description" / "urdf" / "nino.urdf.xacro").read_text()
    bridge = yaml.safe_load(
        (ROOT / "src" / "nino_description" / "config" / "bridge.yaml").read_text()
    )
    assert 'name="terrain_laser"' in urdf
    assert 'type="gpu_lidar"' in urdf
    assert "<topic>terrain_scan</topic>" in urdf
    assert any(item.get("ros_topic_name") == "/terrain_scan" for item in bridge)


def test_adaptive_terrain_sdf_separates_traversable_and_blocking_features():
    source_path = ROOT / "src" / "nino_rl" / "nino_rl" / "ros_interface.py"
    module = ast.parse(source_path.read_text())
    interface = next(node for node in module.body if isinstance(node, ast.ClassDef))
    method = next(
        node for node in interface.body
        if isinstance(node, ast.FunctionDef) and node.name == "_adaptive_terrain_sdf"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"), namespace)
    sdf = namespace["_adaptive_terrain_sdf"]("adaptive", [
        ("pothole", 3.0, 1.0, 0.18),
        ("bump", 4.0, -1.0, 0.12),
        ("cable", 5.0, 1.2, 0.015),
    ])
    root = ET.fromstring(sdf)
    assert len(root.findall(".//link")) == 3
    assert len(root.findall(".//collision")) >= 6
    assert "pothole" in sdf and "bump" in sdf and "cable" in sdf
    bump_collisions = [
        collision for collision in root.findall(".//collision")
        if "bump" in collision.attrib.get("name", "")
    ]
    assert len(bump_collisions) == 5
    assert max(float(item.find("pose").text.split()[2]) for item in bump_collisions) <= 0.0125
    pothole_collisions = [
        collision for collision in root.findall(".//collision")
        if "pothole" in collision.attrib.get("name", "")
    ]
    assert len(pothole_collisions) == 48
    pitches = [abs(float(item.find("pose").text.split()[4])) for item in pothole_collisions]
    assert all(0.15 < pitch < 0.30 for pitch in pitches)
