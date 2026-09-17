import ast
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).parents[3]


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


def test_six_phase_curriculum_and_straight_goal_are_configured():
    config_path = ROOT / "src" / "nino_rl" / "config" / "ppo.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert len(config["curriculum"]["phase_fractions"]) == 6
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
    assert config["goal_tolerance_m"] == 0.01
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
    assert "configure_goal_marker(self.goal_pose)" in environment
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
    root = ET.fromstring(namespace["_goal_marker_sdf"]("training_goal_marker"))
    assert len(root.findall(".//visual")) == 3
    assert root.find(".//collision") is None


def test_adaptive_terrain_sdf_contains_all_three_feature_types():
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
        ("obstacle", 4.0, -1.0, 0.12),
        ("cable", 5.0, 1.2, 0.015),
    ])
    root = ET.fromstring(sdf)
    assert len(root.findall(".//link")) == 3
    assert len(root.findall(".//collision")) >= 6
    assert "pothole" in sdf and "obstacle" in sdf and "cable" in sdf
