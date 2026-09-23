"""Mandatory live-system checks before straight-line PPO training."""

from __future__ import annotations

import argparse
from math import isfinite
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.executors import SingleThreadedExecutor

from nino_rl.core import load_config
from nino_rl.ros_interface import RosRobotInterface


def _wait_until(predicate, timeout: float, failure: str) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.05)
    raise RuntimeError(failure)


def run_preflight(config: dict, timeout: float = 30.0) -> list[str]:
    """Check sensors, straight command transport, TF, and wheel actuation."""
    owns_rclpy = not rclpy.ok()
    if owns_rclpy:
        rclpy.init(args=[])
    nav = config["navigation"]
    node = RosRobotInterface(
        world_name="long_hall",
        subscribe_plan=False,
        node_name="nino_rl_preflight",
        cmd_vel_topic=str(nav.get("cmd_vel_topic", "/cmd_vel")),
        imu_topic=str(config["policy_v2"].get("imu_topic", "/imu/data")),
        physics_step_seconds=float(
            config["policy_v2"].get("simulation_physics_step_seconds", 0.002)
        ),
    )
    # Keep the same ordered, backpressured callback model used by the Gym
    # environment.  A MultiThreadedExecutor can queue the 500 Hz /clock stream
    # faster than its worker pool drains it, making fresh joint feedback look
    # frozen during the actuation check.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    stop = Event()

    def spin() -> None:
        while not stop.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=0.05)

    thread = Thread(target=spin, daemon=True)
    thread.start()
    passed: list[str] = []
    try:
        node.set_world_paused(False, timeout=timeout)
        node.wait_for_sensors(timeout)
        if config["policy_v2"].get("require_terrain_preview", False):
            node.wait_for_terrain_preview(timeout)
        clock_publishers = node.get_publishers_info_by_topic("/clock")
        if len(clock_publishers) != 1:
            publishers = ", ".join(
                f"{item.node_namespace}{item.node_name}"
                for item in clock_publishers
            ) or "none"
            raise RuntimeError(
                "Training requires exactly one /clock publisher; found "
                f"{len(clock_publishers)} ({publishers}). Stop other ROS/Gazebo "
                "graphs and restart training_sim with the Nino isolated domain."
            )
        start = tuple(float(v) for v in nav["start_pose"])
        goal = tuple(float(v) for v in nav["goal_pose"])
        node.reset_episode(timeout=timeout, start_pose=start)
        state = node.snapshot()
        passed.append("1 robot spawned; sensor graph has one isolated clock")

        if not all(isfinite(value) for value in (
            state.orientation_x, state.orientation_y, state.orientation_z,
            state.orientation_w, state.gyro_x, state.gyro_y, state.gyro_z,
            state.accel_x, state.accel_y, state.accel_z,
        )):
            raise RuntimeError("IMU contains non-finite values")
        passed.append("3 IMU publishes finite orientation, gyro, and acceleration")

        if not all(isfinite(v) for v in (
            state.left_wheel_velocity, state.right_wheel_velocity
        )):
            raise RuntimeError("joint_states contains invalid wheel velocities")
        passed.append("4 left/right joint states are valid")
        passed.append("5 wheel odometry is valid")

        finite_scan = [value for value in state.lidar_ranges if isfinite(value)]
        if not finite_scan or min(finite_scan) < 0.0:
            raise RuntimeError("laser scan has no finite non-negative range")
        preview = node.terrain_preview(timeout)
        if config["policy_v2"].get("require_terrain_preview", False) and not preview[-1]:
            raise RuntimeError("downward terrain preview is missing or stale")
        passed.append("6 forward and downward terrain laser scans are valid")

        required_tf = [
            ("odom", "base_footprint"), ("base_footprint", "base_link"),
            ("base_link", "imu_link"), ("base_link", "laser"),
            ("base_link", "terrain_laser"),
            ("base_link", "left_wheel_link"), ("base_link", "right_wheel_link"),
        ]
        for parent, child in required_tf:
            _wait_until(
                lambda p=parent, c=child: node.tf_buffer.can_transform(
                    p, c, rclpy.time.Time()
                ), timeout, f"missing TF {parent} -> {child}"
            )
        passed.append("7 odom TF tree to base, sensors, and wheels is valid")

        node.publish_straight_command(0.0)
        _wait_until(
            lambda: node.cmd_vel_publisher.get_subscription_count() > 0,
            timeout, "no subscriber for the direct /cmd_vel straight command"
        )
        passed.append("8 direct straight command has a controller subscriber")
        if start[:2] == goal[:2]:
            raise RuntimeError("straight trajectory start_pose and goal_pose are identical")
        passed.append("9 fixed straight start and goal are configured")
        if abs(node.desired_twist()[1]) > 0.0:
            raise RuntimeError("straight command unexpectedly contains angular velocity")
        passed.append("10 angular velocity is fixed to zero; Nav2 is not required")
        stale_after = float(nav["stale_seconds"])

        def straight_reference_ready() -> bool:
            # DDS subscriber discovery can outlast the freshness window. Keep
            # the reference alive while waiting for fresh post-reset sensors.
            node.publish_straight_command(0.0)
            return node.straight_reference_valid(stale_after)

        try:
            _wait_until(
                straight_reference_ready,
                timeout,
                "straight reference inputs did not become fresh",
            )
        except RuntimeError as error:
            stale = node.stale_straight_reference_streams(stale_after)
            raise RuntimeError(
                f"straight reference missing/stale streams: {stale}"
            ) from error
        passed.append("11 direct straight reference is observable by RL")

        node.reset_drive_state(timeout=timeout)
        before = node.snapshot()
        test_torque = min(1.0, 0.5 * float(config["max_wheel_torque_nm"]))
        max_left_change = max_right_change = 0.0
        actuation_deadline = monotonic() + min(timeout, 3.0)
        while monotonic() < actuation_deadline:
            # effort_drive deliberately blocks residual torque when there is
            # no fresh baseline motion command. Exercise the same atomic v2
            # path used by training, with a slow straight reference active.
            node.publish_straight_command(0.05)
            node.publish_control(1.0, test_torque, -test_torque)
            sleep(0.05)
            sample = node.snapshot()
            max_left_change = max(max_left_change, abs(
                sample.left_wheel_velocity - before.left_wheel_velocity
            ))
            max_right_change = max(max_right_change, abs(
                sample.right_wheel_velocity - before.right_wheel_velocity
            ))
            if max_left_change >= 0.05 and max_right_change >= 0.05:
                break
        node.publish_control(0.0, 0.0, 0.0)
        node.publish_straight_command(0.0)
        if max_left_change < 0.05 or max_right_change < 0.05:
            raise RuntimeError(
                "direct torque did not rotate both wheels "
                f"(left={max_left_change:.3f}, right={max_right_change:.3f} rad/s)"
            )
        passed.append("2 both wheel joints rotate under bounded v2 residual torque")
        # Training immediately switches from the running preflight world to
        # paused, service-driven stepping. Exercise that exact transition so
        # a stale or overloaded ControlWorld bridge cannot pass preflight and
        # then fail on the first Gym reset.
        lockstep_timeout = float(
            config["policy_v2"].get(
                "simulation_step_wall_timeout_seconds", timeout
            )
        )
        node.set_world_paused(True, timeout=lockstep_timeout)
        epoch = node.begin_lockstep_epoch(timeout=lockstep_timeout)
        chunk_steps = max(1, round(0.05 / node.physics_step_seconds))
        credited = 0.0
        for _ in range(2):
            credited += node.advance_world(chunk_steps, timeout=lockstep_timeout)
            error = node.latest_clock_stamp() - (epoch + credited)
            if abs(error) > max(0.02, 2 * node.physics_step_seconds):
                raise RuntimeError(f"Lockstep credited time differs from /clock by {error:.6f}s")
        node.wait_for_motion_state(epoch + credited, timeout=lockstep_timeout)
        node.set_world_paused(False, timeout=lockstep_timeout)
        passed.append(
            "12 RL residual and lockstep interfaces physically control the world; clock and motion feedback agree"
        )
        return sorted(passed, key=lambda value: int(value.split()[0]))
    finally:
        try:
            node.publish_straight_command(0.0)
            node.publish_control(0.0, 0.0, 0.0)
            node.publish_torque(0.0, 0.0)
            node.set_world_paused(False, timeout=min(timeout, 5.0))
        except (RuntimeError, TimeoutError):
            pass
        stop.set()
        thread.join(timeout=2.0)
        executor.remove_node(node)
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        if owns_rclpy and rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    share = Path(get_package_share_directory("nino_rl"))
    parser = argparse.ArgumentParser(description="Verify straight-line RL prerequisites")
    parser.add_argument("--config", type=Path, default=share / "config" / "ppo.yaml")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    try:
        results = run_preflight(load_config(args.config), args.timeout)
    except (RuntimeError, TimeoutError) as error:
        raise SystemExit(f"PREFLIGHT FAILED: {error}") from error
    for result in results:
        print(f"PASS: {result}")
    print("PASS: all 12 checks; straight-line training is allowed")


if __name__ == "__main__":
    main()
