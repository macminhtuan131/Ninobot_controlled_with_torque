"""Gymnasium environment backed by the live Nino Gazebo simulation."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from math import ceil, degrees
from threading import Event, Thread
from time import monotonic, sleep, time

import gymnasium as gym
import numpy as np
import rclpy
from gymnasium import spaces
from rclpy.executors import SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions

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
from nino_rl.trajectory_metrics import EpisodeTrajectory
from nino_rl.task_geometry import approach_speed, goal_overshot
from nino_rl.control_v2 import (
    ACTION_SIZE, BASELINE_ACTION, ChallengeRegion, ChallengeTracker,
    ObservationHistory, StallWindow,
    make_observation, compute_reward, decode_action,
)


class NinoGazeboEnv(gym.Env):
    """Straight-line AMR environment: speed scale and wheel residuals."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict, total_training_steps: int = 1) -> None:
        super().__init__()
        self.config = config
        self.world_paused_for_update = False
        self.total_training_steps = max(1, int(total_training_steps))
        self.control_dt = 1.0 / float(config["control_hz"])
        self.sensor_timeout = float(config["sensor_timeout_seconds"])
        self.physics_dt = float(
            config["policy_v2"].get("simulation_physics_step_seconds", 0.005)
        )
        self.simulation_step_timeout = float(
            config["policy_v2"].get(
                "simulation_step_wall_timeout_seconds", self.sensor_timeout
            )
        )
        if self.simulation_step_timeout <= 0.0:
            raise ValueError("simulation step wall timeout must be positive")
        self.lockstep_min_completion_fraction = float(
            config["policy_v2"].get("lockstep_min_completion_fraction", 0.80)
        )
        if not 0.0 < self.lockstep_min_completion_fraction <= 1.0:
            raise ValueError("lockstep_min_completion_fraction must be in (0, 1]")
        step_ratio = self.control_dt / self.physics_dt
        self.physics_steps_per_control = int(round(step_ratio))
        if (self.physics_dt <= 0.0 or self.physics_steps_per_control < 1
                or not np.isclose(step_ratio, self.physics_steps_per_control)):
            raise ValueError(
                "control period must be an integer multiple of the Gazebo physics step"
            )
        self.world_is_paused = False
        self.max_steps = int(float(config["max_episode_seconds"]) / self.control_dt)
        self.action_scale = float(config["max_wheel_torque_nm"])
        self.lookahead = list(config["path"]["lookahead_m"])
        self.history = ObservationHistory(config["policy_v2"]["history_frames"])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACTION_SIZE,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -5.0, 5.0, shape=(self.history.size,), dtype=np.float32
        )

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            # Keep the context alive during Python's KeyboardInterrupt handler
            # so close() can stop the robot and unpause the lockstep world.
            rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
        navigation = config["navigation"]
        self.start_pose = tuple(float(value) for value in navigation["start_pose"])
        self.goal_pose = tuple(float(value) for value in navigation["goal_pose"])
        self.nav_stale_seconds = float(navigation["stale_seconds"])
        self.straight_speed = float(navigation["straight_speed_m_s"])
        self.minimum_approach_speed = float(
            navigation.get("minimum_approach_speed_m_s", 0.03)
        )
        self.goal_slowdown_distance = float(
            navigation["goal_slowdown_distance_m"]
        )
        if not (
            0.0 < self.minimum_approach_speed <= self.straight_speed
            and self.goal_slowdown_distance > 0.0
        ):
            raise ValueError(
                "straight speeds must satisfy 0 < minimum <= cruise and slowdown > 0"
            )
        self.waypoint_spacing = float(navigation["waypoint_spacing_m"])
        self.waypoint_seconds_per_m = float(
            navigation["waypoint_budget_seconds_per_m"]
        )
        self.waypoint_slack = float(navigation["waypoint_slack_seconds"])
        self.ros = RosRobotInterface(
            world_name="long_hall",
            subscribe_plan=False,
            cmd_vel_topic=str(navigation.get("cmd_vel_topic", "/cmd_vel")),
            imu_topic=str(config["policy_v2"].get("imu_topic", "/imu/data")),
            physics_step_seconds=self.physics_dt,
        )
        self.ros.imu_includes_gravity = bool(config["policy_v2"]["imu_includes_gravity"])
        # This executor already has its own dedicated Python thread. Using a
        # MultiThreadedExecutor here makes spin_once submit ready 500 Hz clock
        # callbacks to an internal pool faster than it can execute them,
        # building a minutes-long stale queue. A single-threaded executor
        # applies natural backpressure and preserves callback/service order.
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.ros)
        self.executor_stop = Event()
        self._transport_closed = False
        self.executor_thread: Thread | None = None
        self._start_executor_spin()
        try:
            self.ros.wait_for_sensors(self.sensor_timeout)
            if config["policy_v2"].get("require_terrain_preview", False):
                self.ros.wait_for_terrain_preview(self.sensor_timeout)
            # PPO constructs its policy after the environment. Leaving Gazebo
            # running here floods the Python executor with a 500 Hz clock plus
            # sensor callbacks and can starve CUDA model initialization for
            # minutes. reset() deliberately unpauses for episode setup, so the
            # safe idle state between construction and the first reset is paused.
            self.ros.set_world_paused(
                True, timeout=self.simulation_step_timeout
            )
            self.world_is_paused = True
            self._stop_executor_spin()
        except BaseException:
            # __init__ failures bypass train.py's env.close() finally block.
            # Tear down the executor here so it cannot keep spinning against a
            # destroyed/shutting-down rclpy context and emit a second traceback.
            self._shutdown_ros_transport()
            if self._owns_rclpy and rclpy.ok():
                rclpy.shutdown()
            raise

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
        adaptive = config.get("adaptive_terrain", {})
        self.adaptive_terrain_enabled = bool(adaptive.get("enabled", False))
        self.adaptive_terrain_progress = bool(
            adaptive.get("progress_on_success", True)
        )
        self.max_terrain_features = int(adaptive.get("max_features", 8))
        self.terrain_features_per_success = int(
            adaptive.get("add_per_level", adaptive.get("add_per_success", 1))
        )
        self.terrain_success_window_size = int(
            adaptive.get("consecutive_successes", adaptive.get("rolling_window_episodes", 50))
        )
        self.terrain_advance_success_rate = float(
            1.0 if "consecutive_successes" in adaptive
            else adaptive.get("advance_success_rate", 0.75)
        )
        self.terrain_feature_count = int(adaptive.get("initial_features", 0))
        if (
            self.max_terrain_features < 0
            or self.terrain_features_per_success <= 0
            or self.terrain_success_window_size < 1
            or not 0.0 < self.terrain_advance_success_rate <= 1.0
            or not 0 <= self.terrain_feature_count <= self.max_terrain_features
        ):
            raise ValueError("adaptive terrain feature counts are invalid")
        self.successful_episodes = 0
        self.terrain_episodes_at_level = 0
        self.terrain_success_window = deque(
            maxlen=self.terrain_success_window_size
        )
        self.challenge_tracker = ChallengeTracker()
        self.path = self._make_curriculum_path()
        self.previous_action = BASELINE_ACTION.copy()
        self.action_before_previous = BASELINE_ACTION.copy()
        self.previous_tracking = None
        self.previous_robot_state = None
        self._last_noisy_state: RobotState | None = None
        self.episode_return = 0.0
        self.episode_reward_terms = {}
        self.max_clock_error = 0.0
        self.max_motion_sensor_lag = 0.0
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
        self.speed_scale_sum = 0.0
        self.min_speed_scale = 1.0
        self.max_speed_scale = 0.0
        self.ground_speed_sum = 0.0
        self.waypoint_arrival_times: list[float] = []
        self.waypoint_targets = np.asarray([], dtype=np.float64)
        self.next_waypoint_index = 0
        self.episode_started_at = monotonic()
        self.episode_start_time_unix = time()

    def _spin_executor(self) -> None:
        while not self.executor_stop.is_set() and rclpy.ok():
            self.executor.spin_once(timeout_sec=0.05)

    def _start_executor_spin(self) -> None:
        """Resume callback dispatch before any ROS-dependent environment work."""
        if self._transport_closed:
            raise RuntimeError("Cannot restart a closed ROS transport")
        if self.executor_thread is not None and self.executor_thread.is_alive():
            return
        self.executor_stop.clear()
        self.executor_thread = Thread(target=self._spin_executor, daemon=True)
        self.executor_thread.start()

    def _stop_executor_spin(self) -> None:
        """Suspend callback dispatch while the paused environment is idle."""
        self.executor_stop.set()
        self.executor.wake()
        if self.executor_thread is not None:
            self.executor_thread.join(timeout=2.0)

    def resume_callback_dispatch(self) -> None:
        """Enable ROS callbacks before collecting environment transitions."""
        self._start_executor_spin()

    def suspend_callback_dispatch(self) -> None:
        """Stop ROS callback work while paused PPO optimization runs."""
        self._stop_executor_spin()

    def _curriculum_stage(self) -> tuple[int, float, float]:
        curriculum = self.config["curriculum"]
        if "fixed_phase" in curriculum:
            stage = int(np.clip(int(curriculum["fixed_phase"]) - 1, 0, 5))
            return stage, stage / 5.0, float(self.goal_pose[0])
        if not curriculum.get("enabled", True):
            return 5, 1.0, float(self.goal_pose[0])
        if "phase_steps" in curriculum:
            interval = int(curriculum["phase_steps"])
            order = list(curriculum.get("phase_order", [6, 5, 4, 3, 2, 1]))
            if interval <= 0 or sorted(order) != [1, 2, 3, 4, 5, 6]:
                raise ValueError("Curriculum needs positive phase_steps and six unique phases")
            index = min(self.global_steps // interval, len(order) - 1)
            stage = int(order[index]) - 1
            return stage, stage / 5.0, float(self.goal_pose[0])
        fraction = min(1.0, self.global_steps / self.total_training_steps)
        stage = 0
        boundaries = list(curriculum["phase_fractions"])
        for index, boundary in enumerate(boundaries):
            if fraction >= float(boundary):
                stage = index
        stage = min(stage, 5)
        return stage, stage / 5.0, float(self.goal_pose[0])

    def _make_curriculum_path(self) -> PathTracker:
        spacing = float(self.config["path"]["point_spacing_m"])
        start = np.asarray(self.start_pose[:2], dtype=np.float64)
        goal = np.asarray(self.goal_pose[:2], dtype=np.float64)
        distance = float(np.linalg.norm(goal - start))
        if distance <= 0.0:
            raise ValueError("navigation.goal_pose must differ from start_pose")
        sample_count = max(2, int(np.ceil(distance / spacing)) + 1)
        return PathTracker(np.linspace(start, goal, sample_count))

    def _straight_command(self, tracking) -> float:
        return approach_speed(tracking.endpoint_distance, tracking.distance_remaining,
                              tracking.heading_error, self.config)

    def _curriculum_cables(self, stage: int) -> list[tuple[float, float, float]]:
        """Return the single cable for a hard-to-easy size/angle phase.

        Every phase contains a cable.  Position and cable count stay fixed so
        that phase difficulty is controlled only by diameter and angle.
        """
        terrain = self.config["terrain_curriculum"]
        phases = terrain["phases_hard_to_easy"]
        if len(phases) != 6:
            raise ValueError("terrain_curriculum.phases_hard_to_easy needs six phases")
        diameters = [float(item["diameter_m"]) for item in phases]
        angles = [float(item["angle_deg"]) for item in phases]
        if diameters != sorted(diameters, reverse=True) or angles != sorted(
            angles, reverse=True
        ):
            raise ValueError("Cable phases must be ordered hard-to-easy by size and angle")
        phase = phases[int(np.clip(stage, 0, len(phases) - 1))]
        diameter = float(phase["diameter_m"])
        angle_magnitude = float(phase["angle_deg"])
        if not np.isfinite(diameter) or diameter <= 0.0:
            raise ValueError("Every curriculum phase needs a positive cable diameter")
        if not np.isfinite(angle_magnitude) or not 0.0 <= angle_magnitude <= 90.0:
            raise ValueError("Cable phase angle_deg must be in [0, 90]")
        # Mirroring the angle prevents a left/right bias without changing the
        # configured difficulty magnitude.  Zero remains exactly zero.
        sign = float(self.np_random.choice((-1.0, 1.0))) if angle_magnitude else 1.0
        return [(
            float(terrain["cable_x_m"]),
            0.5 * diameter,
            float(np.deg2rad(sign * angle_magnitude)),
        )]

    def _adaptive_terrain_features(self) -> list[tuple[str, float, float, float]]:
        """Return the proven full-size hazards randomized near the route."""
        if not self.adaptive_terrain_enabled:
            return []
        config = self.config["adaptive_terrain"]
        x_min, x_max = (float(value) for value in config["zone_x_m"])
        max_center_offset = 0.5 * float(config["wheel_separation_m"])
        max_lateral = float(config["max_lateral_center_m"])
        if not x_min < x_max or not 0.0 <= max_center_offset < max_lateral < 1.80:
            raise ValueError("adaptive terrain bounds must fit inside the hallway")
        features = []
        # Keep the established geometry and placement. Every feature is
        # traversable; blocking posts belong to route planning, not this task.
        kinds = tuple(self.np_random.permutation(("pothole", "bump", "cable", "groove")))
        count = max(1, self.terrain_feature_count)
        for index in range(self.terrain_feature_count):
            kind = kinds[index % len(kinds)]
            size = (
                float(config.get("pothole_radius_m", 0.30))
                if kind == "pothole"
                else float(config.get("bump_radius_m", 0.12))
                if kind == "bump"
                else float(config.get("groove_half_width_m", 0.05))
                if kind == "groove"
                else float(config.get("cable_radius_m", 0.015))
            )
            fraction = (index + self.np_random.uniform(0.45, 0.55)) / count
            x = x_min + fraction * (x_max - x_min)
            y = self.np_random.uniform(-max_center_offset, max_center_offset)
            features.append((kind, float(x), float(y), float(size)))
        return features

    def _make_challenge_tracker(
        self,
        cables: list[tuple[float, float, float]],
        features: list[tuple[str, float, float, float]],
    ) -> ChallengeTracker:
        """Build privileged reward regions; none are exposed to the actor."""
        tracking = self.config.get("challenge_tracking", {})
        contact_margin = float(tracking.get("robot_contact_margin_m", 0.20))
        clearance_margin = float(tracking.get("robot_clearance_margin_m", 0.30))
        adaptive_cable_half_length = 0.5 * float(
            self.config["adaptive_terrain"].get("cable_length_m", 0.55)
        )
        groove_half_length = 0.5 * float(
            self.config["adaptive_terrain"].get("groove_length_m", 0.70)
        )
        regions = []
        for index, (x, radius, angle) in enumerate(cables):
            # The spawned cylinder direction is (sin(angle), cos(angle)).
            regions.append(ChallengeRegion(
                name=f"curriculum_cable_{index}",
                kind="cable",
                x=float(x),
                y=0.0,
                radius=float(radius),
                half_length=2.0 / max(np.cos(float(angle)), 0.70),
                angle=float(np.pi / 2.0 - angle),
            ))
        for index, (kind, x, y, size) in enumerate(features):
            if kind not in ("pothole", "bump", "cable", "groove"):
                # Non-traversable clutter is never a rewarded challenge.
                continue
            regions.append(ChallengeRegion(
                name=f"adaptive_{index}_{kind}",
                kind=kind,
                x=float(x),
                y=float(y),
                radius=float(size),
                half_length=(adaptive_cable_half_length if kind == "cable"
                             else groove_half_length if kind == "groove" else 0.0),
                angle=(float(np.pi / 2.0) if kind in ("cable", "groove") else 0.0),
            ))
        return ChallengeTracker(
            regions,
            contact_margin=contact_margin,
            clearance_margin=clearance_margin,
            wheel_separation=float(self.config["adaptive_terrain"].get("wheel_separation_m", .34273666)),
        )

    def adaptive_terrain_state(self) -> dict:
        """Return checkpoint-safe rolling curriculum state."""
        return {
            "schema_version": 1,
            "terrain_feature_count": self.terrain_feature_count,
            "successful_episodes": self.successful_episodes,
            "episodes_at_level": self.terrain_episodes_at_level,
            "rolling_outcomes": [int(value) for value in self.terrain_success_window],
        }

    def restore_adaptive_terrain_state(self, state: dict) -> None:
        """Restore curriculum progress saved alongside a PPO checkpoint."""
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise ValueError("Checkpoint is missing valid adaptive terrain state")
        count = int(state.get("terrain_feature_count", -1))
        successes = int(state.get("successful_episodes", -1))
        episodes = int(state.get("episodes_at_level", -1))
        outcomes = list(state.get("rolling_outcomes", []))
        if (
            not 0 <= count <= self.max_terrain_features
            or successes < 0
            or episodes < 0
            or len(outcomes) > self.terrain_success_window_size
            or any(value not in (0, 1, False, True) for value in outcomes)
        ):
            raise ValueError("Checkpoint adaptive terrain state is invalid")
        self.terrain_feature_count = count
        self.successful_episodes = successes
        self.terrain_episodes_at_level = episodes
        self.terrain_success_window.clear()
        self.terrain_success_window.extend(bool(value) for value in outcomes)

    def _record_adaptive_terrain_outcome(
        self, succeeded: bool
    ) -> tuple[float, int, bool]:
        """Update the rolling gate and return rate, sample count and advance."""
        if not self.adaptive_terrain_progress:
            return 0.0, 0, False
        self.terrain_success_window.append(bool(succeeded))
        self.terrain_episodes_at_level += 1
        window_episodes = len(self.terrain_success_window)
        rolling_success = float(np.mean(self.terrain_success_window))
        level_advanced = False
        if (
            window_episodes == self.terrain_success_window_size
            and rolling_success >= self.terrain_advance_success_rate
            and self.terrain_feature_count < self.max_terrain_features
        ):
            self.terrain_feature_count = min(
                self.max_terrain_features,
                self.terrain_feature_count + self.terrain_features_per_success,
            )
            self.terrain_success_window.clear()
            self.terrain_episodes_at_level = 0
            level_advanced = True
        return rolling_success, window_episodes, level_advanced

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

        phase = self._curriculum_stage()[0]
        scales = cfg.get("phase_scales", [1.0] * 6)
        if len(scales) != 6 or any(not 0.0 <= float(x) <= 1.0 for x in scales):
            raise ValueError("domain_randomization.phase_scales needs six values in [0,1]")
        intensity = float(scales[phase])
        def uniform(key):
            value = float(self.np_random.uniform(*cfg[key]))
            return 1.0 + intensity * (value - 1.0) if key == "traction_scale" else intensity * value
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
            valid=self.ros.straight_reference_valid(self.nav_stale_seconds),
        )

    def _actor_observation(self, state, action, reference):
        # reset() and step() both enforce a new terrain callback barrier before
        # reaching this method. Wall time may then pass while physics remains
        # paused, which does not make the observed terrain physically stale.
        preview = self.ros.terrain_preview(timeout=None)
        if self.config["policy_v2"]["require_terrain_preview"] and not preview[-1]:
            self.ros.publish_control(0.0, 0.0, 0.0)
            raise RuntimeError("Required terrain preview missing/stale; stopped")
        frame, _ = make_observation(
            state, self.path, self.lookahead, action, reference, preview,
            self.ros.imu_includes_gravity)
        return frame

    def reset(self, *, seed=None, options=None):
        self._start_executor_spin()
        super().reset(seed=seed)
        # Episode setup requires a running clock.
        if self.world_is_paused:
            self.ros.set_world_paused(
                False, timeout=self.simulation_step_timeout
            )
            self.world_is_paused = False
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
        self.episode_curriculum_stage = stage
        if getattr(self, "_announced_phase", None) != stage:
            self.ros.get_logger().info(
                f"Curriculum phase {stage + 1} at training step {self.global_steps}"
            )
            self._announced_phase = stage
        self.ros.reset_episode(start_pose=self.start_pose)
        episode_cables = self._curriculum_cables(stage)
        self.ros.configure_training_cables(episode_cables)
        episode_features = self._adaptive_terrain_features()
        self.ros.configure_adaptive_terrain(episode_features)
        self.episode_terrain_layout = episode_features
        self.episode_terrain_height_scale = 1.0
        self.challenge_tracker = self._make_challenge_tracker(
            episode_cables, episode_features
        )
        self.ros.configure_goal_marker(
            self.goal_pose, radius=float(self.config["goal_tolerance_m"])
        )
        # Entity services can briefly hold Gazebo sensor publishers. Capture
        # callback markers only after the last world mutation, then require a
        # view of the completed episode layout and reset pose.
        # A scan captured near the terminal obstacle of the previous episode
        # must never survive the teleport and become the first collision
        # sample of the next one.  Wall-clock freshness cannot detect that
        # case because DDS may deliver the old scan during reset.
        reset_sensor_names = ["ground_truth", "scan"]
        if self.config["policy_v2"].get("require_terrain_preview", False):
            reset_sensor_names.append("terrain")
        self.ros.wait_for_sensors(self.sensor_timeout)
        self.ros.publish_straight_command(0.0)
        self.ros.wait_for_v2_controller()
        # Ground truth and the mandatory terrain view must both describe the
        # newly reset/spawned world, rather than the preceding episode.
        self.ros.refresh_reset_sensors(
            reset_sensor_names, timeout=self.sensor_timeout,
            step_timeout=self.simulation_step_timeout,
        )
        self.world_is_paused = True
        if not self.ros.ground_truth_valid():
            raise RuntimeError(
                "Training slip reward received invalid /ground_truth/odom"
            )
        self.path = self._make_curriculum_path()
        self.waypoint_targets = np.arange(
            self.waypoint_spacing, self.path.total_length, self.waypoint_spacing
        )
        self.next_waypoint_index = 0
        self.waypoint_arrival_times = []
        self.episode_started_at = monotonic()
        self.stall_window = StallWindow()
        self.vertical_square_integral = 0.0
        self.imu_coverage_seconds = 0.0
        self.peak_vertical_acceleration = 0.0
        self.episode_start_time_unix = time()
        self.episode_return = 0.0
        self.episode_reward_terms = {}
        self.max_clock_error = 0.0
        self.max_motion_sensor_lag = 0.0
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
        self.speed_scale_sum = 0.0
        self.min_speed_scale = 1.0
        self.max_speed_scale = 0.0
        self.ground_speed_sum = 0.0
        # Pause before selecting the episode clock origin. DDS may still hold
        # IMU messages generated during reset/navigation; using an earlier
        # sample here makes the first requested interval fall out of the IMU
        # window once that backlog is delivered.
        self.ros.set_world_paused(
            True, timeout=self.simulation_step_timeout
        )
        self.world_is_paused = True
        # Establish time zero from fresh post-reset /clock and IMU messages.
        # A one-physics-step flush here prevents callbacks from the previous
        # epoch shifting every later reward window; it is not paid per action.
        self.lockstep_sim_time = self.ros.begin_lockstep_epoch(
            # This call executes a Gazebo physics step. Entity reset/spawn
            # work can make the first response slower than sensor discovery,
            # so use the dedicated simulation-service deadline.
            timeout=self.simulation_step_timeout
        )
        truth = self.ros.snapshot()
        self.episode_started_sim = self.lockstep_sim_time
        frame, points = "odom", self.path.points
        self.trajectory = EpisodeTrajectory(
            points, frame, truth.odom_stamp_s,
            cable_x=episode_cables[0][0],
            cable_radius=episode_cables[0][1],
            cable_angle=episode_cables[0][2],
        )
        self.trajectory.add(0.0, *self.ros.pose_in_frame(truth, frame))
        self.previous_robot_state = deepcopy(truth)
        _, self.previous_tracking = make_observation(
            truth, self.path, self.lookahead, self.previous_action
        )
        reference = self._reference(self.previous_tracking, 0.0)
        observation = self.history.reset(self._actor_observation(
            self._noisy_state(truth), self.previous_action, reference))
        # From here on, only step() advances simulation. This prevents OS/GPU
        # scheduling delays from stretching a nominal 0.1 s action into a much
        # longer and incompletely sampled interval.
        return observation, {
            "attempt": self.attempt_number,
            "curriculum_stage": stage + 1,
            "curriculum_level": level,
            "goal_x_m": goal_x,
            "cable_count": len(episode_cables),
            "cable_diameter_m": 2.0 * episode_cables[0][1],
            "cable_angle_deg": degrees(episode_cables[0][2]),
            "adaptive_terrain_features": len(episode_features),
            "traversable_challenges": self.challenge_tracker.total,
        }

    def step(self, action):
        if not self.world_is_paused:
            raise RuntimeError("Lockstep invariant violated: Gazebo must be paused")
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        scale, torque = decode_action(action, self.action_scale)
        started_sim = self.lockstep_sim_time
        self.ros.publish_straight_command(
            self._straight_command(self.previous_tracking)
        )
        # The horizontal safety LiDAR remains age-bounded instead of being a
        # hard barrier: one dropped 10 Hz packet must not abort an episode.
        # The 20 Hz downward scan is policy input, however, so require a view
        # produced after this action whenever terrain preview is enabled.
        sensor_names = ["odom", "ground_truth", "joint", "torque"]
        sensor_markers = self.ros.sensor_markers(sensor_names)
        require_terrain = self.config["policy_v2"].get(
            "require_terrain_preview", False
        )
        terrain_markers = (
            self.ros.sensor_markers(["terrain"]) if require_terrain else {}
        )
        reward_reference = self._reference(
            self.previous_tracking, started_sim - self.episode_started_sim)
        delay = min(self.control_dt * 0.8, self._randomization["delay"])
        delay_steps = min(
            self.physics_steps_per_control - 1,
            int(round(delay / self.physics_dt)),
        )
        step_timeout = self.simulation_step_timeout
        # A single long multi_step can outrun ROS/Gazebo sensor publishers.
        # Two 50 ms bursts give odometry, lidar and applied-torque callbacks a
        # scheduling boundary without returning to the old five-call cost.
        max_chunk_steps = max(1, self.physics_steps_per_control // 2)
        advanced_sim = 0.0

        def advance_chunked(steps):
            nonlocal advanced_sim
            remaining = int(steps)
            while remaining > 0:
                chunk = min(max_chunk_steps, remaining)
                advanced_sim += self.ros.advance_world(
                    chunk,
                    timeout=step_timeout,
                    min_completion_fraction=self.lockstep_min_completion_fraction,
                )
                remaining -= chunk

        if delay_steps:
            advance_chunked(delay_steps)
        torque *= self._randomization["traction"]
        torque += self.np_random.normal(0.0, self._randomization["torque_noise"], size=2)
        torque = np.clip(torque, -self.action_scale, self.action_scale)
        if self.config.get("evaluation_baseline", False):
            torque[:] = 0.0
        self.ros.publish_control(scale, float(torque[0]), float(torque[1]))
        advance_chunked(self.physics_steps_per_control - delay_steps)

        # GPU LiDAR rendering can miss the exact paused boundary even though
        # /clock, odometry and controller feedback all advanced. Preserve the
        # fresh-terrain contract by advancing at most one complete 20 Hz scan
        # period, accounting that time as part of this action. This is a rare
        # recovery path, not stale-observation fallback.
        if require_terrain:
            try:
                # Allow the bridge/executor to deliver a scan already rendered
                # by the nominal step before advancing simulation again. Under
                # GPU load, 20 ms was too short and caused an unnecessary extra
                # world-control transaction on many actions.
                terrain_delivery_timeout = min(
                    0.25,
                    float(self.config["policy_v2"].get(
                        "sensor_wait_timeout_seconds", 2.0
                    )),
                )
                self.ros.wait_for_sensor_updates(
                    terrain_markers, terrain_delivery_timeout
                )
            except RuntimeError:
                recovery_markers = self.ros.sensor_markers(sensor_names)
                terrain_period_steps = max(
                    1, int(round(0.05 / self.physics_dt))
                )
                advance_chunked(terrain_period_steps)
                self.ros.wait_for_sensor_updates(
                    {**recovery_markers, **terrain_markers},
                    self.config["policy_v2"].get(
                        "sensor_wait_timeout_seconds", 2.0
                    ),
                )
                self.ros.get_logger().warn(
                    "Terrain scan missed the nominal action boundary; "
                    "advanced one accounted 20 Hz sensor period"
                )

        ended_sim = started_sim + advanced_sim
        self.lockstep_sim_time = ended_sim
        step_dt = ended_sim - started_sim
        if not np.isfinite(step_dt) or step_dt <= 0.0:
            raise RuntimeError(
                f"Lockstep produced invalid simulation interval {step_dt}"
            )
        clock_error = self.ros.latest_clock_stamp() - ended_sim
        if abs(clock_error) > max(0.02, 2.0 * self.physics_dt):
            self.ros.publish_control(0.0, 0.0, 0.0)
            raise RuntimeError(
                f"Episode clock diverged from Gazebo by {clock_error:.6f}s; "
                "refusing to train on incorrect action timing"
            )
        motion_lags = self.ros.wait_for_motion_state(
            ended_sim,
            timeout=self.config["policy_v2"].get("sensor_wait_timeout_seconds", 2.0),
            max_lag=self.config["policy_v2"].get("motion_max_lag_seconds", 0.04),
        )
        self.max_clock_error = max(self.max_clock_error, abs(clock_error))
        self.max_motion_sensor_lag = max(self.max_motion_sensor_lag, *motion_lags.values())
        if self.config["policy_v2"].get("imu_strict_coverage", False):
            imu = self.ros.wait_for_impact(started_sim, ended_sim,
                self.config["reward_v2"]["impact_acceleration_sigma_m_s2"],
                self.config["policy_v2"].get("imu_wait_timeout_seconds", 0.5),
                self.config["policy_v2"].get("imu_min_coverage", 0.8),
                self.config["policy_v2"].get("imu_max_latest_lag_seconds", 0.05))
        else:
            imu = self.ros.estimate_impact(
                started_sim,
                ended_sim,
                self.config["reward_v2"]["impact_acceleration_sigma_m_s2"],
            )
        self.ros.wait_for_sensor_updates(
            sensor_markers,
            self.config["policy_v2"].get("sensor_wait_timeout_seconds", 2.0),
        )
        self.vertical_square_integral += imu["square_integral"]
        self.imu_coverage_seconds += imu["duration"]
        self.peak_vertical_acceleration = max(self.peak_vertical_acceleration, imu["peak"])

        self.episode_steps += 1
        self.global_steps += 1
        elapsed = ended_sim - self.episode_started_sim
        if not self.ros.ground_truth_valid():
            self.ros.publish_control(0.0, 0.0, 0.0)
            raise RuntimeError("Ground truth velocity is invalid; slip reward cannot be computed")
        truth = self.ros.snapshot()
        if not self.ros.applied_torque_valid():
            self.ros.publish_control(0.0, 0.0, 0.0)
            raise RuntimeError("Applied torque feedback is invalid; effort reward cannot be computed")
        lidar_max_lag = float(
            self.config["policy_v2"].get("lidar_max_lag_seconds", 0.25)
        )
        lidar_lag = max(0.0, ended_sim - truth.lidar_stamp_s)
        lidar_fresh = self.ros.sensor_stream_ready("scan", self.nav_stale_seconds)
        # A recently delivered packet can still describe the preceding pose
        # or episode.  Only a scan from this action interval is safe to feed
        # to the policy or use for collision termination.
        lidar_current = (
            lidar_fresh
            and truth.lidar_stamp_s >= started_sim - 0.5 * self.physics_dt
        )
        _, tracking = make_observation(truth, self.path, self.lookahead, action)
        challenge_entry_count, challenge_clear_count = self.challenge_tracker.update(
            (self.previous_robot_state.x, self.previous_robot_state.y),
            (truth.x, truth.y),
            self.previous_robot_state.yaw, truth.yaw,
        )
        # Use the odometry message timestamp, not the end of an IMU wait.
        self.trajectory.add(truth.odom_stamp_s - self.trajectory.clock_origin_sim_s,
                            *self.ros.pose_in_frame(truth, self.trajectory.frame_id))

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
        actor_state = self._noisy_state(truth)
        if not lidar_current:
            # Empty sectors encode maximum/unknown clearance.  A delayed GPU
            # render must neither inject a scan from the previous pose into
            # the policy nor create a false collision termination.
            actor_state = deepcopy(actor_state)
            actor_state.lidar_ranges = ()
        observation = self.history.append(self._actor_observation(
            actor_state, action, reference))

        min_lidar = min(truth.lidar_ranges, default=truth.lidar_range_max)
        succeeded = goal_reached(tracking, truth, self.config)
        missed_goal = not succeeded and goal_overshot(truth, self.path, self.config)
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
        # Conservatively account for forward travel since the latest scan.
        # straight_speed is also the configured wheel-speed ceiling reference.
        # GPU sensor rendering can legitimately trail physics by several scan
        # periods during lockstep bursts even while packets keep arriving.
        # Bound extrapolation to avoid turning render latency into a false
        # collision. Delivery freshness independently decides whether this
        # scan can be used at all.
        collision_scan_lag = min(lidar_lag, lidar_max_lag, step_dt)
        collision_clearance = min_lidar - self.straight_speed * collision_scan_lag
        collision = (
            lidar_current
            and np.isfinite(collision_clearance)
            and collision_clearance <= float(self.config["lidar_collision_m"])
        )
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
        failed = ("rollover" if rolled else "collision" if collision else
                  "off_path" if off_path else "wrong_direction" if wrong_direction else
                  "navigation_invalid" if navigation_invalid else
                  "goal_missed" if missed_goal else None)
        succeeded = bool(succeeded and not failed)
        timed_out = elapsed + 0.5 * self.physics_dt >= float(self.config["max_episode_seconds"])
        terminated = bool(
            succeeded
            or rolled
            or wrong_direction
            or off_path
            or navigation_invalid
            or collision
            or timed_out
            or missed_goal
        )
        # The configured mission deadline is a task failure, not an external
        # rollout cutoff. SB3 must not bootstrap a fictitious continuation.
        truncated = False
        # Keep reward difficulty consistent with geometry until the next reset.
        level = getattr(self, "episode_curriculum_stage", self._curriculum_stage()[0]) / 5.0
        delta_s = self.previous_tracking.distance_remaining - tracking.distance_remaining
        stalled = self.stall_window.update(elapsed, delta_s,
            reference.valid and reference.desired_linear_velocity > 0.05
            and tracking.endpoint_distance > self.config["goal_tolerance_m"])
        completion_fraction = float(np.clip(
            tracking.path_s / self.path.total_length, 0.0, 1.0
        ))
        reward, reward_terms = compute_reward(
            self.previous_tracking, tracking, truth, action, self.previous_action,
            torque, step_dt, imu,
            {**self.config["reward_v2"], "torque_scale_nm": self.action_scale},
            timed_out=timed_out,
            succeeded=succeeded,
            failed=failed, stalled=stalled, impact_scale=min(1.0, 0.25 + level),
            reference=reward_reference, previous_state=self.previous_robot_state,
            completion_fraction=completion_fraction,
            elapsed=elapsed,
            target_finish_seconds=float(self.config["target_finish_seconds"]),
            challenge_entry_count=challenge_entry_count,
            challenge_clear_count=challenge_clear_count,
            challenge_cleared_total=len(self.challenge_tracker.cleared),
            challenge_total=self.challenge_tracker.total,
        )
        self.episode_return += reward
        for name, value in reward_terms.items():
            self.episode_reward_terms[name] = self.episode_reward_terms.get(name, 0.0) + value
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
        self.speed_scale_sum += scale
        self.min_speed_scale = min(self.min_speed_scale, scale)
        self.max_speed_scale = max(self.max_speed_scale, scale)
        self.ground_speed_sum += truth.ground_linear_velocity
        self.previous_tracking = tracking
        self.previous_robot_state = deepcopy(truth)
        self.action_before_previous = self.previous_action.copy()
        self.previous_action = action.copy()

        info = {
            "attempt": self.attempt_number,
            "reward_terms": reward_terms,
            "clock_error_seconds": clock_error,
            "motion_sensor_lag_seconds": max(motion_lags.values()),
            "applied_torque_nm": measured_torque.tolist(),
            "residual_torque_nm": torque.tolist(),
            "speed_scale": scale,
            "lidar_lag_seconds": lidar_lag,
            "lidar_fresh": lidar_fresh,
            "lidar_current": lidar_current,
            "completion_fraction": completion_fraction,
            "difficult_path_chosen": bool(self.challenge_tracker.chosen),
            "challenge_entry_count": challenge_entry_count,
            "challenge_clear_count": challenge_clear_count,
            "challenges_chosen": len(self.challenge_tracker.chosen),
            "challenges_cleared": len(self.challenge_tracker.cleared),
            "traversable_challenges": self.challenge_tracker.total,
            **metrics_dict(tracking, truth, elapsed, succeeded),
        }
        if terminated or truncated:
            episode_feature_count = self.terrain_feature_count
            if succeeded:
                self.successful_episodes += 1
            terrain_level_advanced = False
            terrain_rolling_success = 0.0
            terrain_window_episodes = 0
            (
                terrain_rolling_success,
                terrain_window_episodes,
                terrain_level_advanced,
            ) = self._record_adaptive_terrain_outcome(succeeded)
            reason = (
        "success" if succeeded
        else "rollover" if rolled
        else "wrong_direction" if wrong_direction
        else "off_path" if off_path
        else "navigation_invalid" if navigation_invalid
        else "collision" if collision
        else "goal_missed" if missed_goal
        else "timeout"
            )
            self.ros.get_logger().warn(
        f"EPISODE END: {reason} | "
        f"t={elapsed:.2f}s | "
        f"goal_dist={tracking.endpoint_distance:.2f}m | "
        f"heading_err={degrees(tracking.heading_error):.1f}deg | "
        f"lateral={tracking.lateral_error:.2f}m | "
        f"v={truth.linear_velocity:.2f}m/s | "
        f"reference_valid={reference.valid} | "
        f"challenges={len(self.challenge_tracker.cleared)}/"
        f"{self.challenge_tracker.total} | "
        f"lidar_min={min_lidar:.2f}m | "
        f"roll={degrees(truth.roll):.1f}deg | "
        f"pitch={degrees(truth.pitch):.1f}deg"
            )
            count = max(1, self.episode_steps)
            info["episode_metrics"] = {
                **self.trajectory.metrics(),
                "trajectory_frame": self.trajectory.frame_id,
                "trajectory_pose_source": "wheel_odometry",
                **metrics_dict(tracking, truth, elapsed, succeeded),
                "attempt": self.attempt_number,
                "episode_start_time_unix": self.episode_start_time_unix,
                "final_arrival_time_seconds": elapsed if succeeded else None,
                "goal_reached": succeeded,
                "adaptive_terrain_features": episode_feature_count,
                "next_adaptive_terrain_features": self.terrain_feature_count,
                "adaptive_terrain_rolling_success": terrain_rolling_success,
                "adaptive_terrain_window_episodes": terrain_window_episodes,
                "adaptive_terrain_level_advanced": terrain_level_advanced,
                "successful_episodes": self.successful_episodes,
                "difficult_path_chosen": bool(self.challenge_tracker.chosen),
                "challenges_chosen": len(self.challenge_tracker.chosen),
                "challenges_cleared": len(self.challenge_tracker.cleared),
                "traversable_challenges": self.challenge_tracker.total,
                "challenge_choice_fraction": (
                    len(self.challenge_tracker.chosen)
                    / max(1, self.challenge_tracker.total)
                ),
                "challenge_clear_fraction": (
                    len(self.challenge_tracker.cleared)
                    / max(1, self.challenge_tracker.total)
                ),
                "final_lateral_drift_m": float(tracking.lateral_error),
                "final_abs_lateral_drift_m": abs(float(tracking.lateral_error)),
                "return": self.episode_return,
                "reward_totals": dict(self.episode_reward_terms),
                "clock_elapsed_seconds": elapsed + clock_error,
                "max_clock_error_seconds": self.max_clock_error,
                "max_motion_sensor_lag_seconds": self.max_motion_sensor_lag,
                "curriculum_phase": self.episode_curriculum_stage + 1,
                "terrain_height_scale": self.episode_terrain_height_scale,
                "terrain_layout": self.episode_terrain_layout,
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
                "mean_speed_scale": self.speed_scale_sum / count,
                "min_speed_scale": self.min_speed_scale,
                "max_speed_scale": self.max_speed_scale,
                "mean_ground_speed_m_s": self.ground_speed_sum / count,
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
                    "success" if succeeded else "rollover" if rolled else
                    "wrong_direction" if wrong_direction else
                    "off_path" if off_path else
                    "navigation_invalid" if navigation_invalid else
                    "collision" if collision else "goal_missed" if missed_goal else "timeout"
                ),
            }
            self.ros.publish_control(0.0, 0.0, 0.0)
            self.ros.publish_straight_command(0.0)
        return observation, reward, terminated, truncated, info

    def _shutdown_ros_transport(self) -> None:
        """Stop the executor and node, including after partial construction."""
        if getattr(self, "_transport_closed", False):
            return
        self._transport_closed = True
        self._stop_executor_spin()
        self.executor.remove_node(self.ros)
        self.executor.shutdown(timeout_sec=2.0)
        self.ros.destroy_node()

    def close(self) -> None:
        try:
            if not self._transport_closed:
                self._start_executor_spin()
            self.ros.publish_control(0.0, 0.0, 0.0)
            self.ros.publish_straight_command(0.0)
            if self.world_is_paused or self.world_paused_for_update:
                self.ros.set_world_paused(
                    False, timeout=self.simulation_step_timeout
                )
                self.world_is_paused = False
                self.world_paused_for_update = False
            sleep(0.05)
        finally:
            self._shutdown_ros_transport()
            if self._owns_rclpy and rclpy.ok():
                rclpy.shutdown()
