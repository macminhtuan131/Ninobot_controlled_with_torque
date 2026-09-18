"""Pure math shared by training, evaluation, and deployment.

This module deliberately has no ROS, Gymnasium, or Stable-Baselines3 imports so
its geometry and reward logic can be unit tested on a plain Python install.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, exp, pi, radians, sin, sqrt
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml


OBSERVATION_SIZE = 54


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return config


def wrap_angle(angle: float) -> float:
    return (angle + pi) % (2.0 * pi) - pi


def quaternion_to_euler(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """Return roll, pitch, yaw from a normalized or near-normalized quaternion."""
    norm = sqrt(x * x + y * y + z * z + w * w)
    if norm < 1.0e-12:
        return 0.0, 0.0, 0.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.sign(sinp) * pi / 2.0 if abs(sinp) >= 1.0 else np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(roll), float(pitch), float(atan2(siny_cosp, cosy_cosp))


class PathTracker:
    """Piecewise-linear path projection and look-ahead sampling."""

    def __init__(self, points: Iterable[Sequence[float]]) -> None:
        self.set_points(points)

    def set_points(self, points: Iterable[Sequence[float]]) -> None:
        array = np.asarray(list(points), dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2 or len(array) < 2:
            raise ValueError("A path needs at least two [x, y] waypoints")
        if not np.all(np.isfinite(array)):
            raise ValueError("Path waypoints must be finite")
        delta = np.diff(array, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        keep = np.concatenate(([True], lengths > 1.0e-6))
        array = array[keep]
        if len(array) < 2:
            raise ValueError("Path waypoints cannot all be identical")
        self.points = array
        self.delta = np.diff(array, axis=0)
        self.lengths = np.linalg.norm(self.delta, axis=1)
        self.cumulative = np.concatenate(([0.0], np.cumsum(self.lengths)))
        self.total_length = float(self.cumulative[-1])

    def project(self, x: float, y: float) -> tuple[float, float, float, np.ndarray]:
        position = np.asarray([x, y], dtype=np.float64)
        start = self.points[:-1]
        relative = position - start
        fractions = np.sum(relative * self.delta, axis=1) / (self.lengths**2)
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = start + fractions[:, None] * self.delta
        errors = position - projections
        index = int(np.argmin(np.sum(errors * errors, axis=1)))
        tangent = self.delta[index] / self.lengths[index]
        cross = tangent[0] * errors[index, 1] - tangent[1] * errors[index, 0]
        lateral = float(cross)  # signed perpendicular error; endpoint distance is separate
        path_s = float(self.cumulative[index] + fractions[index] * self.lengths[index])
        heading = atan2(float(tangent[1]), float(tangent[0]))
        return path_s, lateral, heading, projections[index]

    def point_at(self, path_s: float) -> np.ndarray:
        path_s = float(np.clip(path_s, 0.0, self.total_length))
        index = min(int(np.searchsorted(self.cumulative, path_s, side="right") - 1), len(self.lengths) - 1)
        fraction = (path_s - self.cumulative[index]) / self.lengths[index]
        return self.points[index] + fraction * self.delta[index]

    def local_lookahead(
        self, x: float, y: float, yaw: float, distances: Sequence[float]
    ) -> tuple[np.ndarray, float, float, float]:
        path_s, lateral, path_heading, _ = self.project(x, y)
        world_points = np.asarray([self.point_at(path_s + distance) for distance in distances])
        offsets = world_points - np.asarray([x, y])
        rotation = np.asarray([[cos(yaw), sin(yaw)], [-sin(yaw), cos(yaw)]])
        local = offsets @ rotation.T
        return local, path_s, lateral, wrap_angle(path_heading - yaw)


def lidar_sectors(ranges: Sequence[float], range_max: float, count: int = 5) -> np.ndarray:
    values = np.asarray(ranges, dtype=np.float64)
    safe_max = float(range_max) if np.isfinite(range_max) and range_max > 0.0 else 10.0
    if values.size == 0:
        return np.ones(count, dtype=np.float32)
    values = np.nan_to_num(values, nan=safe_max, posinf=safe_max, neginf=0.0)
    values = np.clip(values, 0.0, safe_max)
    sectors = [float(np.min(chunk)) if len(chunk) else safe_max for chunk in np.array_split(values, count)]
    return np.asarray(sectors, dtype=np.float32) / safe_max


def catmull_rom_path(
    control_points: Iterable[Sequence[float]], spacing: float
) -> np.ndarray:
    """Sample an open uniform Catmull-Rom spline through 2D control points."""
    controls = np.asarray(list(control_points), dtype=np.float64)
    if controls.ndim != 2 or controls.shape[1] != 2 or len(controls) < 2:
        raise ValueError("Catmull-Rom needs at least two [x, y] control points")
    if not np.all(np.isfinite(controls)) or not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Control points and spacing must be finite; spacing must be positive")
    padded = np.vstack([controls[0], controls, controls[-1]])
    sampled = []
    for index in range(len(controls) - 1):
        p0, p1, p2, p3 = padded[index : index + 4]
        segment_length = float(np.linalg.norm(p2 - p1))
        count = max(2, int(np.ceil(segment_length / spacing)) + 1)
        endpoint = index == len(controls) - 2
        for t in np.linspace(0.0, 1.0, count, endpoint=endpoint):
            t2, t3 = t * t, t * t * t
            point = 0.5 * (
                2.0 * p1
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            sampled.append(point)
    return np.asarray(sampled, dtype=np.float64)


@dataclass
class RobotState:
    odom_stamp_s: float = 0.0
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    linear_velocity: float = 0.0
    yaw_rate: float = 0.0
    left_wheel_velocity: float = 0.0
    right_wheel_velocity: float = 0.0
    ground_linear_velocity: float = 0.0
    ground_yaw_rate: float = 0.0
    applied_left_torque: float = 0.0
    applied_right_torque: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    orientation_x: float = 0.0
    orientation_y: float = 0.0
    orientation_z: float = 0.0
    orientation_w: float = 1.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    lidar_stamp_s: float = -float("inf")
    lidar_ranges: Sequence[float] = ()
    lidar_range_max: float = 10.0


@dataclass
class TrackingState:
    path_s: float
    lateral_error: float
    heading_error: float
    distance_remaining: float
    endpoint_distance: float


@dataclass
class NavReference:
    """Fresh baseline velocity reference attached to one policy observation."""

    desired_linear_velocity: float = 0.0
    desired_angular_velocity: float = 0.0
    local_waypoint_distance: float = 0.0
    final_goal_distance: float = 0.0
    waypoint_time_remaining_fraction: float = 0.0
    valid: bool = False


def wheel_slip_ratios(
    state: RobotState, wheel_radius: float = 0.0625, wheel_separation: float = 0.34273666
) -> tuple[float, float]:
    """Return bounded longitudinal slip for each wheel using Gazebo truth speed."""
    half_track = 0.5 * wheel_separation
    ground_left = state.ground_linear_velocity - half_track * state.ground_yaw_rate
    ground_right = state.ground_linear_velocity + half_track * state.ground_yaw_rate
    wheel_left = wheel_radius * state.left_wheel_velocity
    wheel_right = wheel_radius * state.right_wheel_velocity

    def ratio(wheel_speed: float, ground_speed: float) -> float:
        scale = max(abs(wheel_speed), abs(ground_speed), 0.10)
        return float(np.clip((wheel_speed - ground_speed) / scale, -5.0, 5.0))

    return ratio(wheel_left, ground_left), ratio(wheel_right, ground_right)


def goal_reached(
    tracking: TrackingState, state: RobotState, config: Mapping[str, float]
) -> bool:
    """Return whether the robot has entered or crossed the goal plane.

    Arrival accuracy is graded in the reward.  It is not a reason to turn a
    completed traversal into a goal-overshoot failure.
    """
    tolerance = float(config["goal_tolerance_m"])
    position_reached = tracking.endpoint_distance <= tolerance
    if config.get("goal_capture_on_crossing", False):
        position_reached = position_reached or tracking.distance_remaining <= tolerance
    return bool(position_reached)


def is_wrong_direction(
    tracking: TrackingState,
    state: RobotState,
    config
) -> bool:
    return (
        abs(tracking.heading_error)
        >= radians(float(config["wrong_direction_heading_deg"]))
        or state.linear_velocity
        <= -float(config.get("wrong_direction_reverse_speed_m_s", 0.10))
    )


def make_observation(
    state: RobotState,
    path: PathTracker,
    lookahead_distances: Sequence[float],
    previous_action: Sequence[float],
    nav_reference: NavReference | None = None,
) -> tuple[np.ndarray, TrackingState]:
    local, path_s, lateral, heading_error = path.local_lookahead(
        state.x, state.y, state.yaw, lookahead_distances
    )
    if len(local) != 9:
        raise ValueError("The paper-derived observation requires exactly 9 look-ahead distances")
    reference = nav_reference or NavReference()
    local_target = local[0]
    local_target_norm = max(float(np.linalg.norm(local_target)), 1.0e-6)
    slip_left, slip_right = wheel_slip_ratios(state)
    observation = np.concatenate(
        [
            (local.reshape(-1) / 15.0),
            [cos(heading_error), sin(heading_error)],
            [state.linear_velocity / 1.0, state.yaw_rate / 3.0],
            [state.left_wheel_velocity / 24.0, state.right_wheel_velocity / 24.0],
            [
                state.orientation_x,
                state.orientation_y,
                state.orientation_z,
                state.orientation_w,
                state.gyro_x / 3.0,
                state.gyro_y / 3.0,
                state.gyro_z / 3.0,
                state.accel_x / 10.0,
                state.accel_y / 10.0,
                state.accel_z / 10.0,
            ],
            lidar_sectors(state.lidar_ranges, state.lidar_range_max),
            np.asarray(previous_action, dtype=np.float64),
            [
                reference.desired_linear_velocity / 0.5,
                reference.desired_angular_velocity / 2.5,
                local_target[0] / local_target_norm,
                local_target[1] / local_target_norm,
                lateral / 2.0,
                reference.local_waypoint_distance / 15.0,
                reference.final_goal_distance / 35.0,
                state.roll / (pi / 2.0),
                state.pitch / (pi / 2.0),
                slip_left,
                slip_right,
                reference.waypoint_time_remaining_fraction,
                1.0 if reference.valid else 0.0,
            ],
        ]
    ).astype(np.float32)
    if observation.shape != (OBSERVATION_SIZE,):
        raise RuntimeError(f"Internal observation shape is {observation.shape}, expected ({OBSERVATION_SIZE},)")
    observation = np.nan_to_num(observation, nan=0.0, posinf=5.0, neginf=-5.0)
    tracking = TrackingState(
        path_s=path_s,
        lateral_error=lateral,
        heading_error=heading_error,
        distance_remaining=max(0.0, path.total_length - path_s),
        endpoint_distance=float(
            np.linalg.norm(np.asarray([state.x, state.y]) - path.points[-1])
        ),
    )
    return np.clip(observation, -5.0, 5.0), tracking


def compute_reward(
    previous: TrackingState,
    current: TrackingState,
    state: RobotState,
    previous_state: RobotState,
    action: Sequence[float],
    previous_action: Sequence[float],
    action_before_previous: Sequence[float],
    dt: float,
    elapsed: float,
    reward_config: Mapping[str, float],
    curriculum_level: float,
    timed_out: bool,
    succeeded: bool,
    wrong_direction: bool = False,
    waypoint_reached_count: int = 0,
    waypoint_time_margin_fraction: float = 0.0,
    rolled_over: bool = False,
    collision: bool = False,
    off_path: bool = False,
    navigation_invalid: bool = False,
) -> tuple[float, dict[str, float]]:
    """Legacy v1 reward retained for comparison tests, NOT used by v2 training.

    Active reward and policy contract live in nino_rl/control_v2.py.
    """
    # Nav2 may replan from the current pose, resetting path_s.  Reduction in
    # remaining arc length is stable across such replans and equals delta_s on
    # a fixed path.
    delta_s = previous.distance_remaining - current.distance_remaining
    progress_weight = (
        float(reward_config["progress_weight"])
        + np.clip(curriculum_level, 0.0, 1.0)
        * (float(reward_config["rough_progress_weight"]) - float(reward_config["progress_weight"]))
    )
    progress = progress_weight * max(0.0, delta_s)
    alignment = (
        float(reward_config["alignment_weight"])
        * max(0.0, state.linear_velocity)
        * dt
        * max(0.0, cos(current.heading_error))
        * exp(-((abs(current.lateral_error) / float(reward_config["sigma_lateral_m"])) ** 2))
    )
    lateral = -float(reward_config["lateral_weight"]) * (
        abs(current.lateral_error) / float(reward_config["sigma_lateral_m"])
    ) ** 2
    heading = -float(reward_config["heading_weight"]) * (
        abs(current.heading_error) / float(reward_config["sigma_heading_rad"])
    ) ** 2
    direction = (
        float(reward_config["direction_weight"])
        * max(0.0, delta_s)
        * max(0.0, cos(current.heading_error))
    )
    direction_correction = float(
        reward_config.get("direction_correction_weight", 0.0)
    ) * (abs(previous.heading_error) - abs(current.heading_error))
    wrong_way = -float(reward_config.get("wrong_direction_weight", 0.0)) * max(
        0.0, -cos(current.heading_error)
    ) * dt
    reverse = -float(reward_config.get("reverse_weight", 0.0)) * max(
        0.0, -state.linear_velocity
    ) * dt
    tilt_sigma = float(reward_config["imu_tilt_sigma_rad"])
    rate_sigma = float(reward_config["imu_angular_rate_sigma_rad_s"])
    acceleration_sigma = float(reward_config["imu_acceleration_change_sigma_m_s2"])
    tilt_energy = (state.roll / tilt_sigma) ** 2 + (state.pitch / tilt_sigma) ** 2
    angular_energy = (state.gyro_x / rate_sigma) ** 2 + (
        state.gyro_y / rate_sigma
    ) ** 2
    acceleration_change = np.asarray(
        [
            state.accel_x - previous_state.accel_x,
            state.accel_y - previous_state.accel_y,
            state.accel_z - previous_state.accel_z,
        ],
        dtype=np.float64,
    )
    acceleration_energy = float(
        np.mean((acceleration_change / acceleration_sigma) ** 2)
    )
    stability_score = exp(-(tilt_energy + angular_energy + acceleration_energy))
    imu_stability = (
        float(reward_config["imu_stability_weight"])
        * max(0.0, delta_s)
        * stability_score
    )
    vibration_energy = min(
        angular_energy + acceleration_energy,
        max(float(reward_config.get("imu_vibration_energy_clip", 25.0)), 0.0),
    )
    imu_vibration = -float(reward_config["imu_vibration_weight"]) * (
        vibration_energy
    ) * dt
    # A raw (a_z - g)^4 term can overwhelm PPO after one noisy IMU sample.
    # Use orientation-independent acceleration magnitude, normalize it, and
    # clip it before the fourth power. This retains strong impact sensitivity
    # while keeping every per-step reward finite and bounded.
    gravity = float(reward_config.get("gravity_m_s2", 9.80665))
    impact_sigma = max(
        float(reward_config.get("impact_acceleration_sigma_m_s2", 2.0)),
        1.0e-6,
    )
    impact_clip = max(float(reward_config.get("impact_clip_sigma", 3.0)), 0.0)
    acceleration_magnitude = float(
        np.linalg.norm([state.accel_x, state.accel_y, state.accel_z])
    )
    normalized_impact = min(
        abs(acceleration_magnitude - gravity) / impact_sigma,
        impact_clip,
    )
    # Phase 1 is flat. Enable the terrain objectives fully when the first
    # cable appears at curriculum level 0.2.
    terrain_scale = float(np.clip(curriculum_level / 0.2, 0.0, 1.0))
    impact = -float(reward_config.get("impact_weight", 0.0)) * (
        normalized_impact**4
    ) * dt * terrain_scale

    body_rate_clip = max(
        float(reward_config.get("body_rate_clip_sigma", 3.0)), 0.0
    )
    normalized_roll_rate = min(
        abs(state.gyro_x) / max(rate_sigma, 1.0e-6), body_rate_clip
    )
    normalized_pitch_rate = min(
        abs(state.gyro_y) / max(rate_sigma, 1.0e-6), body_rate_clip
    )
    body_rate = -float(reward_config.get("body_rate_weight", 0.0)) * (
        normalized_roll_rate**2 + normalized_pitch_rate**2
    ) * dt * terrain_scale

    # Do not penalize normal small body motion. Penalize squared excess beyond
    # separate roll and pitch safety thresholds.
    roll_threshold = max(
        float(reward_config.get("attitude_roll_threshold_rad", 0.20)), 1.0e-6
    )
    pitch_threshold = max(
        float(reward_config.get("attitude_pitch_threshold_rad", 0.30)), 1.0e-6
    )
    roll_excess = max(0.0, abs(state.roll) - roll_threshold) / roll_threshold
    pitch_excess = max(0.0, abs(state.pitch) - pitch_threshold) / pitch_threshold
    attitude = -float(reward_config.get("attitude_weight", 0.0)) * (
        roll_excess**2 + pitch_excess**2
    ) * dt
    slip_left, slip_right = wheel_slip_ratios(state)
    slip = -float(reward_config.get("slip_weight", 0.0)) * (
        slip_left**2 + slip_right**2
    ) * dt
    effort = -float(reward_config.get("effort_weight", 0.0)) * float(
        np.mean(np.asarray(action, dtype=np.float64) ** 2)
    ) * dt
    slowdown_distance = float(reward_config["goal_slowdown_distance_m"])
    slowdown_fraction = max(
        0.0, 1.0 - current.endpoint_distance / max(slowdown_distance, 1.0e-6)
    )
    max_goal_speed = max(float(reward_config["goal_max_speed_m_s"]), 1.0e-6)
    max_goal_yaw_rate = max(
        float(reward_config["goal_max_yaw_rate_rad_s"]), 1.0e-6
    )
    excess_speed = max(0.0, abs(state.linear_velocity) - max_goal_speed)
    excess_yaw_rate = max(0.0, abs(state.yaw_rate) - max_goal_yaw_rate)
    endpoint_motion = -float(reward_config["endpoint_motion_weight"]) * slowdown_fraction * (
        (excess_speed / max_goal_speed) ** 2
        + (excess_yaw_rate / max_goal_yaw_rate) ** 2
    ) * dt
    second_difference = (
        np.asarray(action, dtype=np.float64)
        - 2.0 * np.asarray(previous_action, dtype=np.float64)
        + np.asarray(action_before_previous, dtype=np.float64)
    )
    first_difference = np.asarray(action, dtype=np.float64) - np.asarray(
        previous_action, dtype=np.float64
    )
    action_rate = -float(reward_config.get("action_rate_weight", 0.0)) * float(
        np.mean(first_difference**2)
    )
    smoothness = -float(reward_config["smoothness_weight"]) * float(np.mean(second_difference**2))
    stuck = 0.0
    if elapsed > 1.0 and delta_s < float(reward_config["stuck_progress_m"]):
        stuck = -float(reward_config["stuck_penalty"])
    threshold = np.deg2rad(float(reward_config["rollover_threshold_deg"]))
    excess = max(0.0, abs(state.roll) - threshold) + max(0.0, abs(state.pitch) - threshold)
    rollover = -float(reward_config["rollover_weight"]) * excess
    timeout = 0.0
    if timed_out:
        timeout = -(
            float(reward_config["timeout_distance_weight"]) * current.distance_remaining
            + float(reward_config["timeout_constant"])
        )
    success = float(reward_config["success_bonus"]) if succeeded else 0.0
    # Match the environment's termination precedence so one sample cannot
    # receive several terminal penalties for the same episode ending.
    rollover_failure = (
        -float(reward_config.get("rollover_termination_penalty", 0.0))
        if rolled_over
        else 0.0
    )
    wrong_direction_failure = (
        -float(reward_config.get("wrong_direction_termination_penalty", 0.0))
        if wrong_direction and not rolled_over
        else 0.0
    )
    off_path_failure = (
        -float(reward_config.get("off_path_termination_penalty", 0.0))
        if off_path and not rolled_over and not wrong_direction
        else 0.0
    )
    navigation_failure = (
        -float(reward_config.get("navigation_invalid_termination_penalty", 0.0))
        if navigation_invalid and not rolled_over and not wrong_direction and not off_path
        else 0.0
    )
    collision_failure = (
        -float(reward_config.get("collision_termination_penalty", 0.0))
        if collision
        and not rolled_over
        and not wrong_direction
        and not off_path
        and not navigation_invalid
        else 0.0
    )
    early_finish = 0.0
    if succeeded:
        target_time = float(reward_config["target_finish_seconds"])
        if target_time <= 0.0:
            raise ValueError("target_finish_seconds must be positive")
        early_finish = float(reward_config["early_finish_bonus"]) * max(
            0.0, (target_time - elapsed) / target_time
        )
    waypoint = float(reward_config.get("waypoint_bonus", 0.0)) * max(
        0, waypoint_reached_count
    )
    if waypoint_reached_count:
        waypoint += float(reward_config.get("waypoint_early_bonus", 0.0)) * max(
            0.0, waypoint_time_margin_fraction
        )
        waypoint -= float(reward_config.get("waypoint_late_penalty", 0.0)) * max(
            0.0, -waypoint_time_margin_fraction
        )
    time_cost = -float(reward_config["time_penalty"])
    terms = {
        "progress": float(progress),
        "alignment": float(alignment),
        "lateral": float(lateral),
        "heading": float(heading),
        "direction": float(direction),
        "direction_correction": float(direction_correction),
        "wrong_way": float(wrong_way),
        "reverse": float(reverse),
        "imu_stability": float(imu_stability),
        "imu_vibration": float(imu_vibration),
        "impact": float(impact),
        "body_rate": float(body_rate),
        "attitude": float(attitude),
        "slip": float(slip),
        "effort": float(effort),
        "endpoint_motion": float(endpoint_motion),
        "action_rate": float(action_rate),
        "smoothness": float(smoothness),
        "stuck": float(stuck),
        "rollover": float(rollover),
        "timeout": float(timeout),
        "time": float(time_cost),
        "success": float(success),
        "wrong_direction_failure": float(wrong_direction_failure),
        "rollover_failure": float(rollover_failure),
        "collision_failure": float(collision_failure),
        "off_path_failure": float(off_path_failure),
        "navigation_failure": float(navigation_failure),
        "early_finish": float(early_finish),
        "waypoint": float(waypoint),
    }
    return float(sum(terms.values())), terms


def metrics_dict(
    tracking: TrackingState, state: RobotState, elapsed: float, succeeded: bool
) -> dict[str, float | bool]:
    return {
        "success": succeeded,
        "time_seconds": elapsed,
        "progress_m": tracking.path_s,
        "remaining_m": tracking.distance_remaining,
        "endpoint_distance_m": tracking.endpoint_distance,
        "lateral_error_m": tracking.lateral_error,
        "heading_error_deg": degrees(tracking.heading_error),
        "roll_deg": degrees(state.roll),
        "pitch_deg": degrees(state.pitch),
    }
