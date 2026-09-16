from math import pi
from pathlib import Path

import numpy as np

from nino_rl.core import (
    OBSERVATION_SIZE,
    NavReference,
    PathTracker,
    RobotState,
    TrackingState,
    catmull_rom_path,
    compute_reward,
    goal_reached,
    is_wrong_direction,
    lidar_sectors,
    load_config,
    make_observation,
    quaternion_to_euler,
    wrap_angle,
    wheel_slip_ratios,
)


LOOKAHEAD = [0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 12.5, 15.0]


def test_path_projection_sign_and_lookahead():
    path = PathTracker([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
    path_s, lateral, heading, _ = path.project(3.0, 1.0)
    assert np.isclose(path_s, 3.0)
    assert np.isclose(lateral, 1.0)
    assert np.isclose(heading, 0.0)
    local, _, _, error = path.local_lookahead(3.0, 1.0, 0.0, [1.0])
    assert np.allclose(local, [[1.0, -1.0]])
    assert np.isclose(error, 0.0)


def test_quaternion_and_angle_wrapping():
    yaw = pi / 2.0
    roll, pitch, measured_yaw = quaternion_to_euler(0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2))
    assert np.allclose([roll, pitch, measured_yaw], [0.0, 0.0, yaw])
    assert np.isclose(wrap_angle(3.0 * pi), -pi)


def test_observation_has_fixed_finite_shape():
    state = RobotState(
        x=1.0,
        y=0.1,
        orientation_z=0.25,
        orientation_w=0.9682458,
        gyro_x=0.3,
        gyro_y=-0.2,
        gyro_z=0.1,
        accel_x=1.0,
        accel_y=-2.0,
        accel_z=9.8,
        lidar_ranges=[float("inf"), 2.0, float("nan"), 1.0, 3.0],
        lidar_range_max=10.0,
    )
    path = PathTracker([(0.0, 0.0), (30.0, 0.0)])
    observation, tracking = make_observation(state, path, LOOKAHEAD, [0.0, 0.0])
    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert np.isclose(tracking.lateral_error, 0.1)
    assert np.allclose(
        observation[24:34],
        [0.0, 0.0, 0.25, 0.9682458, 0.1, -0.2 / 3.0, 0.1 / 3.0, 0.1, -0.2, 0.98],
    )


def test_observation_contains_nav_reference_timer_and_validity():
    state = RobotState(x=1.0, y=0.2)
    path = PathTracker([(0.0, 0.0), (30.0, 0.0)])
    reference = NavReference(
        desired_linear_velocity=0.25,
        desired_angular_velocity=-0.5,
        local_waypoint_distance=4.0,
        final_goal_distance=29.0,
        waypoint_time_remaining_fraction=0.75,
        valid=True,
    )
    observation, _ = make_observation(
        state, path, LOOKAHEAD, [0.0, 0.0], reference
    )
    assert np.isclose(observation[41], 0.5)
    assert np.isclose(observation[42], -0.2)
    assert np.isclose(observation[45], 0.1)
    assert np.isclose(observation[46], 4.0 / 15.0)
    assert np.isclose(observation[47], 29.0 / 35.0)
    assert np.isclose(observation[52], 0.75)
    assert observation[53] == 1.0


def test_wheel_slip_uses_wheel_and_ground_velocity():
    rolling = RobotState(
        left_wheel_velocity=8.0,
        right_wheel_velocity=8.0,
        ground_linear_velocity=0.5,
    )
    slipping = RobotState(
        left_wheel_velocity=12.0,
        right_wheel_velocity=12.0,
        ground_linear_velocity=0.2,
    )
    assert np.allclose(wheel_slip_ratios(rolling), [0.0, 0.0])
    assert max(abs(value) for value in wheel_slip_ratios(slipping)) > 0.5


def test_lidar_empty_and_non_finite_are_safe():
    assert np.allclose(lidar_sectors([], 10.0), np.ones(5))
    sectors = lidar_sectors([float("inf"), float("nan"), -1.0, 5.0, 2.0], 10.0)
    assert np.all((0.0 <= sectors) & (sectors <= 1.0))


def test_catmull_rom_path_preserves_endpoints_and_is_finite():
    controls = [(0.0, 0.0), (2.0, 0.4), (4.0, -0.3), (6.0, 0.0)]
    points = catmull_rom_path(controls, spacing=0.2)
    assert np.allclose(points[0], controls[0])
    assert np.allclose(points[-1], controls[-1])
    assert len(points) > len(controls)
    assert np.all(np.isfinite(points))


def test_reward_prefers_progress_and_penalizes_tracking_error():
    config = {
        "progress_weight": 20.0,
        "rough_progress_weight": 50.0,
        "alignment_weight": 16.0,
        "lateral_weight": 0.8,
        "heading_weight": 0.2,
        "direction_weight": 1.0,
        "smoothness_weight": 0.2,
        "sigma_lateral_m": 0.3,
        "sigma_heading_rad": 0.35,
        "stuck_progress_m": 0.01,
        "stuck_penalty": 10.0,
        "rollover_threshold_deg": 30.0,
        "rollover_weight": 20.0,
        "timeout_distance_weight": 10.0,
        "timeout_constant": 100.0,
        "success_bonus": 100.0,
        "early_finish_bonus": 100.0,
        "target_finish_seconds": 50.0,
        "time_penalty": 0.01,
        "goal_slowdown_distance_m": 2.0,
        "goal_max_speed_m_s": 0.2,
        "goal_max_yaw_rate_rad_s": 0.3,
        "endpoint_motion_weight": 1.0,
        "imu_stability_weight": 10.0,
        "imu_vibration_weight": 0.2,
        "imu_tilt_sigma_rad": 0.2,
        "imu_angular_rate_sigma_rad_s": 1.0,
        "imu_acceleration_change_sigma_m_s2": 2.0,
    }
    previous = TrackingState(0.0, 0.0, 0.0, 10.0, 10.0)
    good = TrackingState(0.1, 0.0, 0.0, 9.9, 9.9)
    bad = TrackingState(0.1, 0.6, 0.7, 9.9, 9.9)
    args = dict(
        state=RobotState(linear_velocity=1.0),
        previous_state=RobotState(linear_velocity=1.0),
        action=[0.2, 0.2],
        previous_action=[0.1, 0.1],
        action_before_previous=[0.0, 0.0],
        dt=0.1,
        elapsed=2.0,
        reward_config=config,
        curriculum_level=0.0,
        timed_out=False,
        succeeded=False,
    )
    good_reward, _ = compute_reward(previous, good, **args)
    bad_reward, _ = compute_reward(previous, bad, **args)
    assert good_reward > bad_reward
    assert good_reward > 0.0


def test_timeout_and_rollover_have_negative_terms():
    config = {
        "progress_weight": 20.0,
        "rough_progress_weight": 50.0,
        "alignment_weight": 16.0,
        "lateral_weight": 0.8,
        "heading_weight": 0.2,
        "direction_weight": 1.0,
        "smoothness_weight": 0.2,
        "sigma_lateral_m": 0.3,
        "sigma_heading_rad": 0.35,
        "stuck_progress_m": 0.01,
        "stuck_penalty": 10.0,
        "rollover_threshold_deg": 30.0,
        "rollover_weight": 20.0,
        "timeout_distance_weight": 10.0,
        "timeout_constant": 100.0,
        "success_bonus": 100.0,
        "early_finish_bonus": 100.0,
        "target_finish_seconds": 50.0,
        "time_penalty": 0.01,
        "goal_slowdown_distance_m": 2.0,
        "goal_max_speed_m_s": 0.2,
        "goal_max_yaw_rate_rad_s": 0.3,
        "endpoint_motion_weight": 1.0,
        "imu_stability_weight": 10.0,
        "imu_vibration_weight": 0.2,
        "imu_tilt_sigma_rad": 0.2,
        "imu_angular_rate_sigma_rad_s": 1.0,
        "imu_acceleration_change_sigma_m_s2": 2.0,
    }
    tracking = TrackingState(0.0, 0.0, 0.0, 5.0, 5.0)
    _, terms = compute_reward(
        tracking,
        tracking,
        RobotState(roll=np.deg2rad(40.0)),
        RobotState(),
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        0.1,
        2.0,
        config,
        1.0,
        timed_out=True,
        succeeded=False,
    )
    assert terms["rollover"] < 0.0
    assert terms["timeout"] == -150.0


def test_reward_prefers_stable_imu_and_desired_direction():
    config = {
        "progress_weight": 20.0,
        "rough_progress_weight": 50.0,
        "alignment_weight": 16.0,
        "lateral_weight": 0.8,
        "heading_weight": 0.2,
        "direction_weight": 1.0,
        "smoothness_weight": 0.2,
        "sigma_lateral_m": 0.3,
        "sigma_heading_rad": 0.35,
        "stuck_progress_m": 0.01,
        "stuck_penalty": 10.0,
        "rollover_threshold_deg": 30.0,
        "rollover_weight": 20.0,
        "timeout_distance_weight": 10.0,
        "timeout_constant": 100.0,
        "success_bonus": 100.0,
        "early_finish_bonus": 100.0,
        "target_finish_seconds": 50.0,
        "time_penalty": 0.01,
        "goal_slowdown_distance_m": 2.0,
        "goal_max_speed_m_s": 0.2,
        "goal_max_yaw_rate_rad_s": 0.3,
        "endpoint_motion_weight": 1.0,
        "imu_stability_weight": 10.0,
        "imu_vibration_weight": 0.2,
        "imu_tilt_sigma_rad": 0.2,
        "imu_angular_rate_sigma_rad_s": 1.0,
        "imu_acceleration_change_sigma_m_s2": 2.0,
    }
    previous_tracking = TrackingState(0.0, 0.0, 0.0, 10.0, 10.0)
    aligned = TrackingState(0.1, 0.0, 0.0, 9.9, 9.9)
    misaligned = TrackingState(0.1, 0.0, 1.0, 9.9, 9.9)
    previous_state = RobotState(accel_z=9.81)
    stable_state = RobotState(linear_velocity=1.0, accel_z=9.81)
    unstable_state = RobotState(
        linear_velocity=1.0,
        roll=0.25,
        pitch=-0.20,
        gyro_x=1.5,
        gyro_y=-1.0,
        accel_x=3.0,
        accel_z=12.0,
    )
    common = dict(
        action=[0.2, 0.2],
        previous_action=[0.2, 0.2],
        action_before_previous=[0.2, 0.2],
        dt=0.1,
        elapsed=2.0,
        reward_config=config,
        curriculum_level=1.0,
        timed_out=False,
        succeeded=False,
    )
    stable_reward, stable_terms = compute_reward(
        previous_tracking,
        aligned,
        stable_state,
        previous_state,
        **common,
    )
    unstable_reward, unstable_terms = compute_reward(
        previous_tracking,
        misaligned,
        unstable_state,
        previous_state,
        **common,
    )
    assert stable_reward > unstable_reward
    assert stable_terms["imu_stability"] > unstable_terms["imu_stability"]
    assert stable_terms["imu_vibration"] > unstable_terms["imu_vibration"]
    assert stable_terms["direction"] > unstable_terms["direction"]


def test_goal_requires_endpoint_accuracy_low_motion_and_low_tilt():
    config = {
        "goal_tolerance_m": 0.45,
        "goal_lateral_tolerance_m": 0.25,
        "goal_heading_tolerance_deg": 12.0,
        "goal_max_speed_m_s": 0.2,
        "goal_max_yaw_rate_rad_s": 0.3,
        "goal_max_tilt_deg": 10.0,
    }
    accurate = TrackingState(29.8, 0.1, np.deg2rad(5.0), 0.2, 0.22)
    settled = RobotState(
        linear_velocity=0.1,
        yaw_rate=0.1,
        roll=np.deg2rad(3.0),
        pitch=np.deg2rad(4.0),
    )
    assert goal_reached(accurate, settled, config)
    assert not goal_reached(
        TrackingState(30.0, 0.1, 0.0, 0.0, 1.0), settled, config
    )
    assert not goal_reached(accurate, RobotState(linear_velocity=0.5), config)
    assert not goal_reached(
        accurate, RobotState(roll=np.deg2rad(15.0)), config
    )


def test_reward_adds_more_points_for_earlier_finish():
    config = {
        "progress_weight": 20.0,
        "rough_progress_weight": 50.0,
        "alignment_weight": 16.0,
        "lateral_weight": 0.8,
        "heading_weight": 0.2,
        "direction_weight": 1.0,
        "smoothness_weight": 0.2,
        "sigma_lateral_m": 0.3,
        "sigma_heading_rad": 0.35,
        "stuck_progress_m": 0.01,
        "stuck_penalty": 10.0,
        "rollover_threshold_deg": 30.0,
        "rollover_weight": 20.0,
        "timeout_distance_weight": 10.0,
        "timeout_constant": 100.0,
        "success_bonus": 100.0,
        "early_finish_bonus": 100.0,
        "target_finish_seconds": 50.0,
        "time_penalty": 0.01,
        "imu_stability_weight": 10.0,
        "imu_vibration_weight": 0.2,
        "imu_tilt_sigma_rad": 0.2,
        "imu_angular_rate_sigma_rad_s": 1.0,
        "imu_acceleration_change_sigma_m_s2": 2.0,
        "goal_slowdown_distance_m": 2.0,
        "goal_max_speed_m_s": 0.2,
        "goal_max_yaw_rate_rad_s": 0.3,
        "endpoint_motion_weight": 1.0,
    }
    previous = TrackingState(29.7, 0.0, 0.0, 0.3, 0.3)
    current = TrackingState(29.8, 0.0, 0.0, 0.2, 0.2)
    common = dict(
        previous=previous,
        current=current,
        state=RobotState(linear_velocity=0.1),
        previous_state=RobotState(linear_velocity=0.1),
        action=[0.0, 0.0],
        previous_action=[0.0, 0.0],
        action_before_previous=[0.0, 0.0],
        dt=0.1,
        reward_config=config,
        curriculum_level=1.0,
        timed_out=False,
        succeeded=True,
    )
    early_reward, early_terms = compute_reward(elapsed=40.0, **common)
    late_reward, late_terms = compute_reward(elapsed=55.0, **common)
    assert early_terms["early_finish"] == 20.0
    assert late_terms["early_finish"] == 0.0
    assert early_reward > late_reward


def test_waypoint_reward_uses_time_margin_without_dominating_goal():
    full = load_config(Path(__file__).parents[1] / "config" / "ppo.yaml")
    reward_config = {
        **full["reward"],
        "target_finish_seconds": full["target_finish_seconds"],
        "goal_max_speed_m_s": full["goal_max_speed_m_s"],
        "goal_max_yaw_rate_rad_s": full["goal_max_yaw_rate_rad_s"],
    }
    previous = TrackingState(4.9, 0.0, 0.0, 25.1, 25.1)
    current = TrackingState(5.0, 0.0, 0.0, 25.0, 25.0)
    common = dict(
        previous=previous,
        current=current,
        state=RobotState(),
        previous_state=RobotState(),
        action=[0.0, 0.0],
        previous_action=[0.0, 0.0],
        action_before_previous=[0.0, 0.0],
        dt=0.1,
        elapsed=10.0,
        reward_config=reward_config,
        curriculum_level=0.0,
        timed_out=False,
        succeeded=False,
        waypoint_reached_count=1,
    )
    early, early_terms = compute_reward(
        waypoint_time_margin_fraction=0.5, **common
    )
    late, late_terms = compute_reward(
        waypoint_time_margin_fraction=-0.5, **common
    )
    assert early_terms["waypoint"] > late_terms["waypoint"]
    assert early > late
    assert early_terms["waypoint"] < reward_config["success_bonus"]


def test_endpoint_motion_penalty_encourages_smooth_stop():
    config = {
        "progress_weight": 0.0,
        "rough_progress_weight": 0.0,
        "alignment_weight": 0.0,
        "lateral_weight": 0.0,
        "heading_weight": 0.0,
        "direction_weight": 0.0,
        "smoothness_weight": 0.0,
        "sigma_lateral_m": 0.3,
        "sigma_heading_rad": 0.35,
        "stuck_progress_m": 0.0,
        "stuck_penalty": 0.0,
        "rollover_threshold_deg": 30.0,
        "rollover_weight": 0.0,
        "timeout_distance_weight": 0.0,
        "timeout_constant": 0.0,
        "success_bonus": 0.0,
        "early_finish_bonus": 0.0,
        "target_finish_seconds": 50.0,
        "time_penalty": 0.0,
        "imu_stability_weight": 0.0,
        "imu_vibration_weight": 0.0,
        "imu_tilt_sigma_rad": 0.2,
        "imu_angular_rate_sigma_rad_s": 1.0,
        "imu_acceleration_change_sigma_m_s2": 2.0,
        "goal_slowdown_distance_m": 2.0,
        "goal_max_speed_m_s": 0.2,
        "goal_max_yaw_rate_rad_s": 0.3,
        "endpoint_motion_weight": 1.0,
    }
    previous = TrackingState(29.0, 0.0, 0.0, 1.0, 1.0)
    current = TrackingState(29.1, 0.0, 0.0, 0.9, 0.9)
    common = dict(
        previous=previous,
        current=current,
        previous_state=RobotState(),
        action=[0.0, 0.0],
        previous_action=[0.0, 0.0],
        action_before_previous=[0.0, 0.0],
        dt=0.1,
        elapsed=10.0,
        reward_config=config,
        curriculum_level=1.0,
        timed_out=False,
        succeeded=False,
    )
    _, smooth_terms = compute_reward(state=RobotState(linear_velocity=0.1), **common)
    _, fast_terms = compute_reward(state=RobotState(linear_velocity=0.7), **common)
    assert smooth_terms["endpoint_motion"] == 0.0
    assert fast_terms["endpoint_motion"] < 0.0


def test_wrong_direction_uses_plan_heading_and_reverse_speed():
    config = {
        "wrong_direction_heading_deg": 60.0,
        "wrong_direction_reverse_speed_m_s": 0.1,
    }
    aligned = TrackingState(1.0, 0.0, np.deg2rad(10.0), 9.0, 9.0)
    turned_away = TrackingState(1.0, 0.0, np.deg2rad(70.0), 9.0, 9.0)
    assert not is_wrong_direction(aligned, RobotState(linear_velocity=0.2), config)
    assert is_wrong_direction(
        turned_away, RobotState(linear_velocity=0.2), config
    )
    assert is_wrong_direction(
        aligned, RobotState(linear_velocity=-0.2), config
    )


def test_reward_encourages_direction_recovery_and_penalizes_failed_attempt():
    full_config = load_config(Path(__file__).parents[1] / "config" / "ppo.yaml")
    reward_config = {
        **full_config["reward"],
        "target_finish_seconds": full_config["target_finish_seconds"],
        "goal_max_speed_m_s": full_config["goal_max_speed_m_s"],
        "goal_max_yaw_rate_rad_s": full_config["goal_max_yaw_rate_rad_s"],
    }
    previous = TrackingState(1.0, 0.0, 0.5, 9.0, 9.0)
    corrected = TrackingState(1.1, 0.0, 0.1, 8.9, 8.9)
    worsened = TrackingState(1.1, 0.0, 0.9, 8.9, 8.9)
    common = dict(
        previous=previous,
        state=RobotState(linear_velocity=0.3),
        previous_state=RobotState(linear_velocity=0.3),
        action=[0.2, 0.1],
        previous_action=[0.1, 0.1],
        action_before_previous=[0.0, 0.0],
        dt=0.1,
        elapsed=5.0,
        reward_config=reward_config,
        curriculum_level=1.0,
        timed_out=False,
        succeeded=False,
    )
    corrected_reward, corrected_terms = compute_reward(
        current=corrected, wrong_direction=False, **common
    )
    worsened_reward, worsened_terms = compute_reward(
        current=worsened, wrong_direction=False, **common
    )
    failed_reward, failed_terms = compute_reward(
        current=worsened, wrong_direction=True, **common
    )
    assert corrected_terms["direction_correction"] > 0.0
    assert worsened_terms["direction_correction"] < 0.0
    assert corrected_reward > worsened_reward
    assert failed_terms["wrong_direction_failure"] == -100.0
    assert failed_reward < worsened_reward


def test_impact_penalty_is_quartic_but_bounded():
    full_config = load_config(Path(__file__).parents[1] / "config" / "ppo.yaml")
    reward_config = {
        **full_config["reward"],
        "target_finish_seconds": full_config["target_finish_seconds"],
        "goal_max_speed_m_s": full_config["goal_max_speed_m_s"],
        "goal_max_yaw_rate_rad_s": full_config["goal_max_yaw_rate_rad_s"],
    }
    tracking = TrackingState(1.0, 0.0, 0.0, 9.0, 9.0)
    common = dict(
        previous=tracking,
        current=tracking,
        previous_state=RobotState(accel_z=9.80665),
        action=[0.0, 0.0],
        previous_action=[0.0, 0.0],
        action_before_previous=[0.0, 0.0],
        dt=0.1,
        elapsed=1.0,
        reward_config=reward_config,
        curriculum_level=1.0,
        timed_out=False,
        succeeded=False,
    )
    _, stable = compute_reward(
        state=RobotState(accel_z=9.80665), **common
    )
    _, impact = compute_reward(
        state=RobotState(accel_z=30.0), **common
    )
    _, extreme = compute_reward(
        state=RobotState(accel_z=3000.0), **common
    )
    assert stable["impact"] == 0.0
    assert impact["impact"] < 0.0
    assert extreme["impact"] == impact["impact"]
    assert np.isfinite(extreme["impact"])


def test_attitude_threshold_and_action_rate_penalties():
    full_config = load_config(Path(__file__).parents[1] / "config" / "ppo.yaml")
    reward_config = {
        **full_config["reward"],
        "target_finish_seconds": full_config["target_finish_seconds"],
        "goal_max_speed_m_s": full_config["goal_max_speed_m_s"],
        "goal_max_yaw_rate_rad_s": full_config["goal_max_yaw_rate_rad_s"],
    }
    tracking = TrackingState(1.0, 0.0, 0.0, 9.0, 9.0)
    common = dict(
        previous=tracking,
        current=tracking,
        previous_state=RobotState(accel_z=9.80665),
        action_before_previous=[0.0, 0.0],
        dt=0.1,
        elapsed=1.0,
        reward_config=reward_config,
        curriculum_level=1.0,
        timed_out=False,
        succeeded=False,
    )
    _, safe = compute_reward(
        state=RobotState(roll=0.19, pitch=0.29, accel_z=9.80665),
        action=[0.0, 0.0],
        previous_action=[0.0, 0.0],
        **common,
    )
    _, unsafe = compute_reward(
        state=RobotState(roll=0.30, pitch=0.40, accel_z=9.80665),
        action=[1.0, -1.0],
        previous_action=[0.0, 0.0],
        **common,
    )
    assert safe["attitude"] == 0.0
    assert unsafe["attitude"] < 0.0
    assert safe["action_rate"] == 0.0
    assert unsafe["action_rate"] < 0.0


def test_terminal_failures_receive_one_explicit_penalty():
    full_config = load_config(Path(__file__).parents[1] / "config" / "ppo.yaml")
    reward_config = {
        **full_config["reward"],
        "target_finish_seconds": full_config["target_finish_seconds"],
        "goal_max_speed_m_s": full_config["goal_max_speed_m_s"],
        "goal_max_yaw_rate_rad_s": full_config["goal_max_yaw_rate_rad_s"],
    }
    tracking = TrackingState(1.0, 0.0, 0.0, 9.0, 9.0)
    _, terms = compute_reward(
        tracking,
        tracking,
        RobotState(accel_z=9.80665),
        RobotState(accel_z=9.80665),
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        0.1,
        1.0,
        reward_config,
        1.0,
        timed_out=False,
        succeeded=False,
        rolled_over=True,
        collision=True,
        off_path=True,
        navigation_invalid=True,
    )
    assert terms["rollover_failure"] == -100.0
    assert terms["collision_failure"] == 0.0
    assert terms["off_path_failure"] == 0.0
    assert terms["navigation_failure"] == 0.0

    expected = {
        "collision": ("collision_failure", -100.0),
        "off_path": ("off_path_failure", -75.0),
        "navigation_invalid": ("navigation_failure", -25.0),
    }
    for flag, (term_name, value) in expected.items():
        _, individual = compute_reward(
            tracking,
            tracking,
            RobotState(accel_z=9.80665),
            RobotState(accel_z=9.80665),
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            0.1,
            1.0,
            reward_config,
            1.0,
            timed_out=False,
            succeeded=False,
            **{flag: True},
        )
        assert individual[term_name] == value
