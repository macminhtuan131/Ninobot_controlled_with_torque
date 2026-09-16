"""Gymnasium environment backed by the live Nino Gazebo simulation."""

from __future__ import annotations

from copy import deepcopy
from math import ceil, degrees
from threading import Event, Thread
from time import monotonic, sleep, time

import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from rclpy.executors import MultiThreadedExecutor

from nino_rl.core import (
    NavReference,
    PathTracker,
    RobotState,
    goal_reached,
    is_wrong_direction,
    metrics_dict,
    wheel_slip_ratios,
)
from nino_rl.ros_interface import RosRobotInterface
from nino_rl.control_v2 import (
    ACTION_SIZE, BASELINE_ACTION, ObservationHistory, StallWindow,
    make_observation, compute_reward, decode_action,
)


class NinoGazeboEnv(gym.Env):
    """Continuous two-action environment: normalized left and right torque."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict, total_training_steps: int = 1) -> None:
        super().__init__()
        self.config = config
        self.world_paused_for_update = False
        self.total_training_steps = max(1, int(total_training_steps))
        self.control_dt = 1.0 / float(config["control_hz"])
        self.max_steps = int(float(config["max_episode_seconds"]) / self.control_dt)
        self.action_scale = float(config["max_wheel_torque_nm"])
        self.lookahead = list(config["path"]["lookahead_m"])
        self.sensor_timeout = float(config["sensor_timeout_seconds"])
        self.history = ObservationHistory(config["policy_v2"]["history_frames"])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_SIZE,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -5.0, 5.0, shape=(self.history.size,), dtype=np.float32
        )

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=[])
        navigation = config["navigation"]
        self.start_pose = tuple(float(value) for value in navigation["start_pose"])
        self.goal_pose = tuple(float(value) for value in navigation["goal_pose"])
        self.nav_stale_seconds = float(navigation["stale_seconds"])
        self.nav_startup_timeout = float(navigation["startup_timeout_seconds"])
        self.waypoint_spacing = float(navigation["waypoint_spacing_m"])
        self.waypoint_seconds_per_m = float(
            navigation["waypoint_budget_seconds_per_m"]
        )
        self.waypoint_slack = float(navigation["waypoint_slack_seconds"])
        self.ros = RosRobotInterface(
            world_name="long_hall",
            subscribe_plan=True,
            plan_topic=str(navigation.get("plan_topic", "/plan")),
            nav_cmd_topic=str(navigation.get("nav_cmd_topic", "/cmd_vel_nav")),
        )
        self.ros.imu_includes_gravity = bool(config["policy_v2"]["imu_includes_gravity"])
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.ros)
        self.executor_stop = Event()
        self.executor_thread = Thread(target=self._spin_executor, daemon=True)
        self.executor_thread.start()
        self.ros.wait_for_sensors(self.sensor_timeout)

        self.global_steps = 0
        self.attempt_number = 0
        self.episode_steps = 0
        self.wrong_direction_steps = 0
        self.wrong_direction_seconds = 0.0
        self.off_path_seconds = 0.0
        self.nav_invalid_seconds = 0.0
        self.wrong_direction_required_steps = max(
            1,
            ceil(
                float(config["wrong_direction_hold_seconds"])
                / self.control_dt
            ),
        )
        self.off_path_steps = 0
        self.off_path_required_steps = max(
            1, ceil(float(config["off_path_hold_seconds"]) / self.control_dt)
        )
        self.nav_invalid_steps = 0
        self.nav_invalid_required_steps = max(
            1, ceil(float(config["navigation_invalid_hold_seconds"]) / self.control_dt)
        )
        self._randomization = {}
        self.path = self._make_curriculum_path()
        self._episode_nav_path = None
        self.previous_action = BASELINE_ACTION.copy()
        self.action_before_previous = BASELINE_ACTION.copy()
        self.previous_tracking = None
        self.previous_robot_state = None
        self._last_noisy_state: RobotState | None = None
        self.episode_return = 0.0
        self.abs_lateral_sum = 0.0
        self.lateral_square_sum = 0.0
        self.abs_roll_sum = 0.0
        self.abs_pitch_sum = 0.0
        self.imu_angular_xy_sum = 0.0
        self.imu_acceleration_change_sum = 0.0
        self.max_tilt_deg = 0.0
        self.max_path_deviation = 0.0
        self.slip_square_sum = 0.0
        self.max_abs_slip = 0.0
        self.torque_square_sum = 0.0
        self.max_abs_torque = 0.0
        self.accel_square_sum = 0.0
        self.waypoint_arrival_times: list[float] = []
        self.waypoint_targets = np.asarray([], dtype=np.float64)
        self.next_waypoint_index = 0
        self.episode_started_at = monotonic()
        self.episode_start_time_unix = time()

    def _spin_executor(self) -> None:
        while not self.executor_stop.is_set() and rclpy.ok():
            self.executor.spin_once(timeout_sec=0.05)

    def _curriculum_stage(self) -> tuple[int, float, float]:
        curriculum = self.config["curriculum"]
        if "fixed_phase" in curriculum:
            stage = int(np.clip(int(curriculum["fixed_phase"]) - 1, 0, 5))
            return stage, stage / 5.0, float(self.goal_pose[0])
        if not curriculum.get("enabled", True):
            return 5, 1.0, float(self.goal_pose[0])
        fraction = min(1.0, self.global_steps / self.total_training_steps)
        stage = 0
        boundaries = list(curriculum["phase_fractions"])
        for index, boundary in enumerate(boundaries):
            if fraction >= float(boundary):
                stage = index
        stage = min(stage, 5)
        return stage, stage / 5.0, float(self.goal_pose[0])

    def _make_curriculum_path(self) -> PathTracker:
        _, _, goal_x = self._curriculum_stage()
        spacing = float(self.config["path"]["point_spacing_m"])
        start_x, start_y, _ = self.start_pose
        goal_y = float(self.goal_pose[1])
        x_values = np.arange(start_x, goal_x + 0.5 * spacing, spacing)
        if x_values[-1] < goal_x:
            x_values = np.append(x_values, goal_x)
        y_values = np.linspace(start_y, goal_y, len(x_values))
        return PathTracker(zip(x_values, y_values))

    def _curriculum_cables(self, stage: int) -> list[tuple[float, float, float]]:
        """Six phases: flat, fixed, position, angle, diameter, multiple."""
        terrain = self.config["terrain_curriculum"]
        if stage <= 0:
            return []
        x_low, x_high = (float(v) for v in terrain["cable_x_range_m"])
        radius_low, radius_high = (
            0.5 * float(v) for v in terrain["cable_diameter_range_m"]
        )
        fixed_x = float(terrain["fixed_cable_x_m"])
        fixed_radius = 0.5 * float(terrain["fixed_cable_diameter_m"])
        if stage == 1:
            return [(fixed_x, fixed_radius, 0.0)]
        x = float(self.np_random.uniform(x_low, x_high))
        if stage == 2:
            return [(x, fixed_radius, 0.0)]
        max_angle = np.deg2rad(float(terrain["max_cable_angle_deg"]))
        angle = float(self.np_random.uniform(-max_angle, max_angle))
        if stage == 3:
            return [(x, fixed_radius, angle)]
        radius = float(self.np_random.uniform(radius_low, radius_high))
        if stage == 4:
            return [(x, radius, angle)]
        count = int(self.np_random.integers(2, int(terrain["max_cables"]) + 1))
        positions = np.linspace(x_low, x_high, count)
        jitter = self.np_random.uniform(-0.5, 0.5, count)
        return [
            (
                float(np.clip(position + offset, x_low, x_high)),
                float(self.np_random.uniform(radius_low, radius_high)),
                float(self.np_random.uniform(-max_angle, max_angle)),
            )
            for position, offset in zip(positions, jitter)
        ]

    def _sample_randomization(self) -> None:
        cfg = self.config["domain_randomization"]
        if not cfg.get("enabled", True):
            self._randomization = {
                "traction": 1.0,
                "delay": 0.0,
                "torque_noise": 0.0,
                "position_noise": 0.0,
                "heading_noise": 0.0,
                "position_bias": 0.0,
                "heading_bias": 0.0,
                "dropout": 0.0,
            }
            return

        uniform = lambda key: float(self.np_random.uniform(*cfg[key]))
        self._randomization = {
            "traction": uniform("traction_scale"),
            "delay": uniform("motor_delay_ms") / 1000.0,
            "torque_noise": uniform("torque_noise_std_nm"),
            "position_noise": uniform("position_noise_std_m"),
            "heading_noise": np.deg2rad(uniform("heading_noise_std_deg")),
            "position_bias": uniform("position_bias_m") * self.np_random.choice([-1.0, 1.0]),
            "heading_bias": np.deg2rad(uniform("heading_bias_deg")),
            "dropout": uniform("observation_dropout"),
        }

    def _noisy_state(self, truth: RobotState) -> RobotState:
        if (
            self._last_noisy_state is not None
            and self.np_random.random() < self._randomization["dropout"]
        ):
            return deepcopy(self._last_noisy_state)
        noisy = deepcopy(truth)
        sigma_position = self._randomization["position_noise"]
        noisy.x += self._randomization["position_bias"] + self.np_random.normal(0.0, sigma_position)
        noisy.y += self._randomization["position_bias"] + self.np_random.normal(0.0, sigma_position)
        noisy.yaw += self._randomization["heading_bias"] + self.np_random.normal(
            0.0, self._randomization["heading_noise"]
        )
        self._last_noisy_state = deepcopy(noisy)
        return noisy

    def _reference(self, tracking, elapsed: float) -> NavReference:
        desired_linear, desired_angular = self.ros.desired_twist()
        if self.next_waypoint_index < len(self.waypoint_targets):
            waypoint_s = float(self.waypoint_targets[self.next_waypoint_index])
            waypoint_distance = max(0.0, waypoint_s - tracking.path_s)
            budget = self.waypoint_slack + self.waypoint_seconds_per_m * waypoint_s
        else:
            waypoint_distance = tracking.distance_remaining
            budget = float(self.config["target_finish_seconds"])
        time_fraction = np.clip((budget - elapsed) / max(budget, 1.0), -1.0, 1.0)
        return NavReference(
            desired_linear_velocity=desired_linear,
            desired_angular_velocity=desired_angular,
            local_waypoint_distance=waypoint_distance,
            final_goal_distance=tracking.endpoint_distance,
            waypoint_time_remaining_fraction=float(time_fraction),
            valid=self.ros.navigation_valid(self.nav_stale_seconds),
        )

    def _actor_observation(self, state, action, reference):
        preview = self.ros.terrain_preview(
            self.config["policy_v2"]["preview_timeout_seconds"])
        if self.config["policy_v2"]["require_terrain_preview"] and not preview[-1]:
            self.ros.publish_control(0.0, 0.0, 0.0)
            raise RuntimeError("Required terrain preview missing/stale; stopped")
        frame, _ = make_observation(
            state, self.path, self.lookahead, action, reference, preview,
            self.ros.imu_includes_gravity)
        return frame

    def _sim_seconds(self):
        return self.ros.get_clock().now().nanoseconds * 1e-9

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.attempt_number += 1
        self.episode_steps = 0
        self.wrong_direction_steps = 0
        self.wrong_direction_seconds = 0.0
        self.off_path_seconds = 0.0
        self.nav_invalid_seconds = 0.0
        self.off_path_steps = 0
        self.nav_invalid_steps = 0
        self.previous_action = BASELINE_ACTION.copy()
        self.action_before_previous = BASELINE_ACTION.copy()
        self._last_noisy_state = None
        self._sample_randomization()
        stage, level, goal_x = self._curriculum_stage()
        self.ros.reset_episode(start_pose=self.start_pose)
        self.ros.configure_training_cables(self._curriculum_cables(stage))
        self.ros.initialize_navigation(
            self.start_pose, self.goal_pose, timeout=self.nav_startup_timeout
        )
        self.ros.wait_for_sensors(self.sensor_timeout)
        self.ros.wait_for_navigation(
            self.nav_startup_timeout, stale_after=self.nav_stale_seconds
        )
        self.ros.wait_for_v2_controller()
        if not self.ros.ground_truth_ready():
            raise RuntimeError("Training slip reward needs /ground_truth/odom; actor does not")
        self._episode_nav_path = self.ros.nav_path()
        nav_points = self.ros.nav_path_in_odom(self._episode_nav_path)
        self.path = PathTracker(nav_points) if nav_points else self._make_curriculum_path()
        self.waypoint_targets = np.arange(
            self.waypoint_spacing, self.path.total_length, self.waypoint_spacing
        )
        self.next_waypoint_index = 0
        self.waypoint_arrival_times = []
        self.episode_started_at = monotonic()
        self.episode_started_sim = self._sim_seconds()
        self.stall_window = StallWindow()
        self.vertical_square_integral = 0.0
        self.imu_coverage_seconds = 0.0
        self.peak_vertical_acceleration = 0.0
        self.episode_start_time_unix = time()
        self.episode_return = 0.0
        self.abs_lateral_sum = 0.0
        self.lateral_square_sum = 0.0
        self.abs_roll_sum = 0.0
        self.abs_pitch_sum = 0.0
        self.imu_angular_xy_sum = 0.0
        self.imu_acceleration_change_sum = 0.0
        self.max_tilt_deg = 0.0
        self.max_path_deviation = 0.0
        self.slip_square_sum = 0.0
        self.max_abs_slip = 0.0
        self.torque_square_sum = 0.0
        self.max_abs_torque = 0.0
        self.accel_square_sum = 0.0
        truth = self.ros.snapshot()
        self.previous_robot_state = deepcopy(truth)
        _, self.previous_tracking = make_observation(
            truth, self.path, self.lookahead, self.previous_action
        )
        reference = self._reference(self.previous_tracking, 0.0)
        observation = self.history.reset(self._actor_observation(
            self._noisy_state(truth), self.previous_action, reference))
        return observation, {
            "attempt": self.attempt_number,
            "curriculum_stage": stage + 1,
            "curriculum_level": level,
            "goal_x_m": goal_x,
        }

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        scale, torque = decode_action(action, self.action_scale)
        started_sim = self._sim_seconds()
        delay = min(self.control_dt * 0.8, self._randomization["delay"])
        if delay > 0.0:
            sleep(delay)
        torque *= self._randomization["traction"]
        torque += self.np_random.normal(0.0, self._randomization["torque_noise"], size=2)
        torque = np.clip(torque, -self.action_scale, self.action_scale)
        self.ros.publish_control(scale, float(torque[0]), float(torque[1]))
        deadline = monotonic() + self.config["policy_v2"]["simulation_step_wall_timeout_seconds"]
        while self._sim_seconds() < started_sim + self.control_dt:
            if monotonic() > deadline:
                self.ros.publish_control(0.0, 0.0, 0.0)
                raise RuntimeError("Simulation clock paused; aborting instead of generating fake steps")
            sleep(0.002)
        ended_sim = self._sim_seconds()
        step_dt = ended_sim - started_sim
        imu = self.ros.measure_impact(started_sim, ended_sim,
            self.config["reward_v2"]["impact_acceleration_sigma_m_s2"])
        if imu["duration"] < 0.8 * step_dt:
            self.ros.publish_control(0.0, 0.0, 0.0)
            raise RuntimeError("Insufficient timestamped IMU coverage; inspect /imu/data and /clock")
        self.vertical_square_integral += imu["square_integral"]
        self.imu_coverage_seconds += imu["duration"]
        self.peak_vertical_acceleration = max(self.peak_vertical_acceleration, imu["peak"])

        self.episode_steps += 1
        self.global_steps += 1
        elapsed = ended_sim - self.episode_started_sim
        if not self.ros.ground_truth_ready():
            self.ros.publish_control(0.0, 0.0, 0.0)
            raise RuntimeError("Ground truth velocity stale; slip reward cannot be computed")
        truth = self.ros.snapshot()
        # Keep waypoint identities fixed in the map, while applying AMCL's
        # latest correction when comparing against wheel odometry. Freezing
        # the initial map->odom transform produces false endpoint failures.
        nav_points = self.ros.nav_path_in_odom(self._episode_nav_path)
        if nav_points is not None:
            self.path.set_points(nav_points)
            _, self.previous_tracking = make_observation(
                self.previous_robot_state, self.path, self.lookahead,
                self.previous_action,
            )
        _, tracking = make_observation(truth, self.path, self.lookahead, action)

        reached_waypoints = 0
        waypoint_margin = 0.0
        while (
            self.next_waypoint_index < len(self.waypoint_targets)
            and tracking.path_s >= self.waypoint_targets[self.next_waypoint_index]
        ):
            waypoint_s = float(self.waypoint_targets[self.next_waypoint_index])
            budget = self.waypoint_slack + self.waypoint_seconds_per_m * waypoint_s
            waypoint_margin = (budget - elapsed) / max(budget, 1.0)
            self.waypoint_arrival_times.append(elapsed)
            self.next_waypoint_index += 1
            reached_waypoints += 1
        reference = self._reference(tracking, elapsed)
        observation = self.history.append(self._actor_observation(
            self._noisy_state(truth), action, reference))

        min_lidar = min(truth.lidar_ranges, default=truth.lidar_range_max)
        succeeded = goal_reached(tracking, truth, self.config)
        rolled = max(abs(degrees(truth.roll)), abs(degrees(truth.pitch))) >= float(
            self.config["rollover_limit_deg"]
        )
        off_path_sample = abs(tracking.lateral_error) >= float(
            self.config["off_path_limit_m"]
        )
        self.off_path_steps = self.off_path_steps + 1 if off_path_sample else 0
        self.off_path_seconds = self.off_path_seconds + step_dt if off_path_sample else 0.0
        off_path = self.off_path_seconds >= float(self.config["off_path_hold_seconds"])
        nav_invalid_sample = not reference.valid
        self.nav_invalid_steps = self.nav_invalid_steps + 1 if nav_invalid_sample else 0
        self.nav_invalid_seconds = self.nav_invalid_seconds + step_dt if nav_invalid_sample else 0.0
        navigation_invalid = self.nav_invalid_seconds >= float(self.config["navigation_invalid_hold_seconds"])
        collision = np.isfinite(min_lidar) and min_lidar <= float(self.config["lidar_collision_m"])
        wrong_direction_sample = (
            elapsed >= float(self.config["wrong_direction_grace_seconds"])
            and is_wrong_direction(tracking, truth, self.config)
        )
        if wrong_direction_sample:
            self.wrong_direction_steps += 1
        else:
            self.wrong_direction_steps = 0
        self.wrong_direction_seconds = self.wrong_direction_seconds + step_dt if wrong_direction_sample else 0.0
        wrong_direction = self.wrong_direction_seconds >= float(self.config["wrong_direction_hold_seconds"])
        # Infrastructure failures are not a learned collision penalty.
        if navigation_invalid:
            self.ros.publish_control(0.0, 0.0, 0.0)
            raise RuntimeError("Nav2 reference stale for too long; aborting this training run")
        failed = ("rollover" if rolled else "collision" if collision else
                  "off_path" if off_path else "wrong_direction" if wrong_direction else None)
        succeeded = bool(succeeded and not failed)
        timed_out = elapsed >= float(self.config["max_episode_seconds"])
        terminated = bool(
            succeeded
            or rolled
            or wrong_direction
            or off_path
            or navigation_invalid
            or collision
        )
        truncated = bool(timed_out and not terminated)
        _, level, _ = self._curriculum_stage()
        delta_s = self.previous_tracking.distance_remaining - tracking.distance_remaining
        stalled = self.stall_window.update(elapsed, delta_s,
            reference.valid and reference.desired_linear_velocity > 0.05
            and tracking.endpoint_distance > self.config["goal_tolerance_m"])
        reward, reward_terms = compute_reward(
            self.previous_tracking, tracking, truth, action, self.previous_action,
            torque, step_dt, imu,
            {**self.config["reward_v2"], "torque_scale_nm": self.action_scale},
            timed_out=truncated,
            succeeded=succeeded,
            failed=failed, stalled=stalled, impact_scale=min(1.0, 0.25 + level),
        )
        self.episode_return += reward
        self.abs_lateral_sum += abs(tracking.lateral_error)
        self.lateral_square_sum += tracking.lateral_error**2
        self.abs_roll_sum += abs(degrees(truth.roll))
        self.abs_pitch_sum += abs(degrees(truth.pitch))
        self.imu_angular_xy_sum += float(np.hypot(truth.gyro_x, truth.gyro_y))
        self.imu_acceleration_change_sum += float(
            np.linalg.norm(
                np.asarray(
                    [truth.accel_x, truth.accel_y, truth.accel_z], dtype=np.float64
                )
                - np.asarray(
                    [
                        self.previous_robot_state.accel_x,
                        self.previous_robot_state.accel_y,
                        self.previous_robot_state.accel_z,
                    ],
                    dtype=np.float64,
                )
            )
        )
        self.max_tilt_deg = max(
            self.max_tilt_deg, abs(degrees(truth.roll)), abs(degrees(truth.pitch))
        )
        deviation = abs(tracking.lateral_error)
        self.max_path_deviation = max(self.max_path_deviation, deviation)
        slip_left, slip_right = wheel_slip_ratios(truth)
        self.slip_square_sum += 0.5 * (slip_left**2 + slip_right**2)
        self.max_abs_slip = max(self.max_abs_slip, abs(slip_left), abs(slip_right))
        measured_torque = np.asarray(
            [truth.applied_left_torque, truth.applied_right_torque], dtype=np.float64
        )
        self.torque_square_sum += float(np.mean(measured_torque**2))
        self.max_abs_torque = max(
            self.max_abs_torque, float(np.max(np.abs(measured_torque)))
        )
        self.accel_square_sum += float(
            np.mean(np.asarray([truth.accel_x, truth.accel_y, truth.accel_z]) ** 2)
        )
        self.previous_tracking = tracking
        self.previous_robot_state = deepcopy(truth)
        self.action_before_previous = self.previous_action.copy()
        self.previous_action = action.copy()

        info = {
            "attempt": self.attempt_number,
            "reward_terms": reward_terms,
            "applied_torque_nm": torque.tolist(),
            **metrics_dict(tracking, truth, elapsed, succeeded),
        }
        if terminated or truncated:
            reason = (
        "success" if succeeded
        else "rollover" if rolled
        else "wrong_direction" if wrong_direction
        else "off_path" if off_path
        else "navigation_invalid" if navigation_invalid
        else "collision" if collision
        else "timeout"
            )
            self.ros.get_logger().warn(
        f"EPISODE END: {reason} | "
        f"t={elapsed:.2f}s | "
        f"heading_err={degrees(tracking.heading_error):.1f}deg | "
        f"lateral={tracking.lateral_error:.2f}m | "
        f"v={truth.linear_velocity:.2f}m/s | "
        f"nav_valid={reference.valid} | "
        f"lidar_min={min_lidar:.2f}m | "
        f"roll={degrees(truth.roll):.1f}deg | "
        f"pitch={degrees(truth.pitch):.1f}deg"
            )
            count = max(1, self.episode_steps)
            info["episode_metrics"] = {
                **metrics_dict(tracking, truth, elapsed, succeeded),
                "attempt": self.attempt_number,
                "episode_start_time_unix": self.episode_start_time_unix,
                "final_arrival_time_seconds": elapsed if succeeded else None,
                "return": self.episode_return,
                "peak_vertical_acceleration_m_s2": self.peak_vertical_acceleration,
                "rms_vertical_acceleration_m_s2": float(np.sqrt(
                    self.vertical_square_integral / max(self.imu_coverage_seconds, 1e-9))),
                "mean_abs_lateral_error_m": self.abs_lateral_sum / count,
                "rms_path_deviation_m": float(
                    np.sqrt(self.lateral_square_sum / count)
                ),
                "max_path_deviation_m": self.max_path_deviation,
                "mean_abs_roll_deg": self.abs_roll_sum / count,
                "mean_abs_pitch_deg": self.abs_pitch_sum / count,
                "mean_imu_angular_xy_rad_s": self.imu_angular_xy_sum / count,
                "mean_imu_acceleration_change_m_s2": (
                    self.imu_acceleration_change_sum / count
                ),
                "max_tilt_deg": self.max_tilt_deg,
                "rms_wheel_slip": float(np.sqrt(self.slip_square_sum / count)),
                "max_abs_wheel_slip": self.max_abs_slip,
                "rms_wheel_torque_nm": float(np.sqrt(self.torque_square_sum / count)),
                "max_abs_wheel_torque_nm": self.max_abs_torque,
                "rms_imu_acceleration_m_s2": float(
                    np.sqrt(self.accel_square_sum / count)
                ),
                "waypoint_arrival_times_seconds": list(self.waypoint_arrival_times),
                "waypoint_time_budgets_seconds": [
                    self.waypoint_slack + self.waypoint_seconds_per_m * float(value)
                    for value in self.waypoint_targets
                ],
                "finished_within_target_time": bool(
                    succeeded
                    and elapsed <= float(self.config["target_finish_seconds"])
                ),
                "time_margin_seconds": float(
                    self.config["target_finish_seconds"]
                )
                - elapsed,
                "target_finish_seconds": float(
                    self.config["target_finish_seconds"]
                ),
                "final_speed_m_s": truth.linear_velocity,
                "final_yaw_rate_rad_s": truth.yaw_rate,
                "wrong_direction_duration_seconds": (
                    self.wrong_direction_seconds
                ),
                "termination": (
                    "success" if succeeded else "rollover" if rolled else "wrong_direction" if wrong_direction else "off_path" if off_path else "navigation_invalid" if navigation_invalid else "collision" if collision else "timeout"
                ),
            }
            self.ros.publish_control(0.0, 0.0, 0.0)
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            self.ros.publish_control(0.0, 0.0, 0.0)
            if self.world_paused_for_update:
                self.ros.set_world_paused(False)
                self.world_paused_for_update = False
            self.ros.cancel_navigation_goal()
            sleep(0.05)
            self.executor_stop.set()
            self.executor_thread.join(timeout=2.0)
            self.executor.remove_node(self.ros)
            self.executor.shutdown(timeout_sec=2.0)
            self.ros.destroy_node()
        finally:
            if self._owns_rclpy and rclpy.ok():
                rclpy.shutdown()
