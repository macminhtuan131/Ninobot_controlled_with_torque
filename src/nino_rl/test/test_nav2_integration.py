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


def test_training_uses_nav2_baseline_plus_rl_residual_torque():
    launch_dir = ROOT / "src" / "nino_rl" / "launch"
    training = (launch_dir / "training_sim.launch.py").read_text()
    baseline = (launch_dir / "baseline_nav.launch.py").read_text()
    assert '"accept_cmd_vel": "true"' in training
    assert '"accept_torque": "true"' in training
    assert '"accept_cmd_vel": "true"' in baseline
    assert '"accept_torque": "false"' in baseline
    assert 'executable="wait_for_sim"' in training
    assert 'OnProcessExit(target_action=readiness, on_exit=[navigation])' in training
    assert 'executable="wait_for_sim"' in baseline
    assert 'OnProcessExit(target_action=readiness, on_exit=[navigation])' in baseline


def test_six_phase_curriculum_and_nav_goal_are_configured():
    config_path = ROOT / "src" / "nino_rl" / "config" / "ppo.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert len(config["curriculum"]["phase_fractions"]) == 6
    assert config["navigation"]["goal_pose"][0] == 30.0
    assert config["navigation"]["nav_cmd_topic"] == "/cmd_vel_nav"
    assert config["off_path_hold_seconds"] > 0.0
    assert config["navigation_invalid_hold_seconds"] > 0.0


def test_nav2_auto_localizes_and_keeps_reference_alive_for_full_episode():
    navigation_root = ROOT / "src" / "linorobot2" / "linorobot2_navigation"
    nav_config = yaml.safe_load(
        (navigation_root / "config" / "navigation.yaml").read_text()
    )
    rl_config = yaml.safe_load(
        (ROOT / "src" / "nino_rl" / "config" / "ppo.yaml").read_text()
    )
    amcl = nav_config["amcl"]["ros__parameters"]
    assert amcl["set_initial_pose"] is True
    assert amcl["always_reset_initial_pose"] is True
    assert amcl["base_frame_id"] == "base_footprint"
    assert nav_config["bt_navigator"]["ros__parameters"]["default_cancel_timeout"] >= 1000
    assert (
        nav_config["controller_server"]["ros__parameters"]["progress_checker"]
        ["movement_time_allowance"]
        >= rl_config["max_episode_seconds"]
    )
    launch_text = (navigation_root / "launch" / "navigation.launch.py").read_text()
    assert "amcl.ros__parameters.initial_pose.x" in launch_text
    assert "'use_composition': 'False'" in launch_text
