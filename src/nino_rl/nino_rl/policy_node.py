"""Deploy a trained PPO policy as a safe ROS 2 torque-control node."""

from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path
import sys
from time import monotonic

from ament_index_python.packages import get_package_share_directory
import numpy as np
import rclpy

from nino_rl.core import (
    NavReference,
    PathTracker,
    goal_reached,
    is_wrong_direction,
    load_config,
)
from nino_rl.ros_interface import RosRobotInterface
from nino_rl.task_geometry import approach_speed, goal_overshot
from nino_rl.control_v2 import (
    BASELINE_ACTION, STOP_ACTION, ObservationHistory, make_observation,
    decode_action, validate_model,
)


def arguments() -> argparse.Namespace:
    share = Path(get_package_share_directory("nino_rl"))
    parser = argparse.ArgumentParser(description="Run a trained Nino PPO torque policy")
    parser.add_argument("--model", type=Path,
                        default=share / "models" / "completed_train" / "nino_ppo_final.zip")
    parser.add_argument("--config", type=Path, default=share / "models" / "completed_train" / "ppo.yaml")
    parser.add_argument("--path", type=Path, default=share / "config" / "path.yaml")
    parser.add_argument("--use-sim-time", action="store_true")
    parser.add_argument("--device", default="cpu", help="cuda, cpu, or auto (default: cpu)")
    return parser.parse_args(sys.argv[1:])


class PolicyNode(RosRobotInterface):
    def __init__(self, args: argparse.Namespace) -> None:
        self.config = load_config(args.config)
        super().__init__(
            subscribe_plan=False,
            use_sim_time=args.use_sim_time,
            node_name="nino_rl_policy",
            cmd_vel_topic=str(self.config["navigation"].get("cmd_vel_topic", "/cmd_vel")),
        )
        self.imu_includes_gravity = bool(self.config["policy_v2"]["imu_includes_gravity"])
        self.history = ObservationHistory(self.config["policy_v2"]["history_frames"])
        try:
            from stable_baselines3 import PPO
        except ImportError as error:
            raise RuntimeError("Thiếu stable-baselines3; xem README_VI.md") from error

        path_config = load_config(args.path)
        self.path = PathTracker(path_config["waypoints"])
        self.path_source = "YAML"
        self.path_started_at = None
        self.deadline_reported = False
        self.wrong_direction_reported = False
        self.wrong_direction_steps = 0
        self.wrong_direction_required_steps = max(
            1,
            ceil(
                float(self.config["wrong_direction_hold_seconds"])
                * float(self.config["control_hz"])
            ),
        )
        self.previous_action = BASELINE_ACTION.copy()
        self.straight_speed = float(self.config["navigation"]["straight_speed_m_s"])
        self.minimum_approach_speed = float(
            self.config["navigation"].get("minimum_approach_speed_m_s", 0.03)
        )
        self.goal_slowdown_distance = float(
            self.config["navigation"]["goal_slowdown_distance_m"]
        )
        self.action_scale = float(self.config["max_wheel_torque_nm"])
        self.lookahead = list(self.config["path"]["lookahead_m"])
        self.collision_distance = float(self.config["lidar_collision_m"])
        self.off_path_limit = float(self.config["off_path_limit_m"])
        self.rollover_limit = np.deg2rad(float(self.config["rollover_limit_deg"]))
        device = args.device or str(self.config.get("device", "cuda"))
        self.model = PPO.load(args.model, device=device)
        validate_model(self.model, self.history.size)
        self.timer = self.create_timer(1.0 / float(self.config["control_hz"]), self._control)
        self.get_logger().info(
            f"PPO policy loaded on {self.model.device}; fixed straight path, Nav2 disabled"
        )

    def _control(self) -> None:
        if not self.sensors_ready():
            self._stop_policy()
            return
        if self.path_started_at is None:
            self.path_started_at = self.get_clock().now().nanoseconds * 1e-9
        state = self.snapshot()
        finite_ranges = [value for value in state.lidar_ranges if np.isfinite(value)]
        if finite_ranges and min(finite_ranges) <= self.collision_distance:
            self._stop_policy()
            return
        _, tracking = make_observation(
            state, self.path, self.lookahead, self.previous_action
        )
        speed = approach_speed(tracking.endpoint_distance, tracking.distance_remaining,
                               tracking.heading_error, self.config)
        self.publish_straight_command(speed)
        elapsed = self.get_clock().now().nanoseconds * 1e-9 - self.path_started_at
        desired_linear, desired_angular = self.desired_twist()
        # Match the environment's 5m waypoint reference/budget convention.
        spacing = self.config["navigation"]["waypoint_spacing_m"]
        waypoint_s = min(self.path.total_length, (int(tracking.path_s / spacing) + 1) * spacing)
        budget = (self.config["navigation"]["waypoint_slack_seconds"]
                  + self.config["navigation"]["waypoint_budget_seconds_per_m"] * waypoint_s)
        if waypoint_s >= self.path.total_length:
            budget = self.config["target_finish_seconds"]
        reference = NavReference(
            desired_linear_velocity=desired_linear,
            desired_angular_velocity=desired_angular,
            local_waypoint_distance=max(0.0, waypoint_s - tracking.path_s),
            final_goal_distance=tracking.endpoint_distance,
            waypoint_time_remaining_fraction=float(
                np.clip(
                    (
                        float(budget)
                        - elapsed
                    )
                    / max(float(budget), 1.0),
                    -1.0,
                    1.0,
                )
            ),
            valid=self.straight_reference_valid(
                float(self.config["navigation"]["stale_seconds"])
            ),
        )
        preview = self.terrain_preview(self.config["policy_v2"]["preview_timeout_seconds"])
        if (self.config["policy_v2"]["require_terrain_preview"] and not preview[-1]
                or self.control_publisher.get_subscription_count() == 0):
            self._stop_policy()
            return
        try:
            frame, tracking = make_observation(
                state, self.path, self.lookahead, self.previous_action, reference,
                preview, self.imu_includes_gravity)
        except ValueError:
            self._stop_policy()
            return
        observation = self.history.append(frame)
        timed_out = elapsed >= float(
            self.config["max_episode_seconds"]
        )
        wrong_direction_sample = (
            elapsed
            >= float(self.config["wrong_direction_grace_seconds"])
            and is_wrong_direction(tracking, state, self.config)
        )
        if wrong_direction_sample:
            self.wrong_direction_steps += 1
        else:
            self.wrong_direction_steps = 0
            self.wrong_direction_reported = False
        wrong_direction = (
            self.wrong_direction_steps >= self.wrong_direction_required_steps
        )
        if (
            goal_reached(tracking, state, self.config)
            or goal_overshot(state, self.path, self.config)
            or timed_out
            or wrong_direction
            or abs(tracking.lateral_error) >= self.off_path_limit
            or max(abs(state.roll), abs(state.pitch)) >= self.rollover_limit
            or not reference.valid
        ):
            self._stop_policy()
            if timed_out and not self.deadline_reported:
                self.get_logger().warning(
                    "Quá thời gian chạy tối đa; giữ mô-men hai bánh ở 0"
                )
                self.deadline_reported = True
            if wrong_direction and not self.wrong_direction_reported:
                self.get_logger().error(
                    "Hủy attempt: robot chạy ngược hoặc lệch quá xa hướng /plan"
                )
                self.wrong_direction_reported = True
            return
        action, _ = self.model.predict(observation, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if not np.all(np.isfinite(action)):
            self.get_logger().error("Policy returned non-finite action; commanding zero torque")
            self._stop_policy()
            return
        scale, torque = decode_action(action, self.action_scale)
        self.publish_control(scale, float(torque[0]), float(torque[1]))
        self.previous_action = action

    def _stop_policy(self):
        self.publish_control(0.0, 0.0, 0.0)
        self.publish_straight_command(0.0)
        self.previous_action = STOP_ACTION.copy()
        self.history.values.clear()


def main() -> None:
    args = arguments()
    rclpy.init(args=[])
    node = None
    try:
        node = PolicyNode(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node._stop_policy()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
