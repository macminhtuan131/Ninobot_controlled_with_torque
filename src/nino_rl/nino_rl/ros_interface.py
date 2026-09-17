"""ROS 2 transport shared by the Gym environment and deployed policy node."""

from __future__ import annotations

from copy import deepcopy

from math import atan2, cos, sin, sqrt, isfinite

import os

import re

import subprocess

from threading import Lock

from time import monotonic, sleep

import rclpy

from geometry_msgs.msg import PoseWithCovarianceStamped, Twist

from nav_msgs.msg import Odometry, Path

from rclpy.action import ActionClient

from rclpy.node import Node

from rclpy.qos import (

    DurabilityPolicy,

    HistoryPolicy,

    QoSProfile,

    ReliabilityPolicy,

)

from rclpy.time import Time

from ros_gz_interfaces.msg import Entity

from ros_gz_interfaces.srv import ControlWorld, DeleteEntity, SetEntityPose, SpawnEntity

from sensor_msgs.msg import Imu, JointState, LaserScan

from std_msgs.msg import Float64MultiArray

from std_srvs.srv import Trigger

from tf2_ros import Buffer, TransformException, TransformListener

from nino_rl.terrain import cable_layout, terrain_sdf
from nino_rl.core import RobotState, quaternion_to_euler
from nino_rl.control_v2 import ImuWindow, vertical_acceleration



SENSOR_QOS = QoSProfile(

    history=HistoryPolicy.KEEP_LAST,

    depth=1,

    reliability=ReliabilityPolicy.BEST_EFFORT,

)

IMU_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100,
                     reliability=ReliabilityPolicy.BEST_EFFORT)

AMCL_POSE_QOS = QoSProfile(

    history=HistoryPolicy.KEEP_LAST,

    depth=1,

    reliability=ReliabilityPolicy.RELIABLE,

    durability=DurabilityPolicy.TRANSIENT_LOCAL,

)



class RosRobotInterface(Node):

    def __init__(

        self,

        world_name: str = "long_hall",

        subscribe_plan: bool = False,

        use_sim_time: bool = True,

        node_name: str = "nino_rl_interface",

        plan_topic: str = "/plan",

        nav_cmd_topic: str = "/cmd_vel_nav",

    ) -> None:

        super().__init__(

            node_name,

            parameter_overrides=[

                rclpy.parameter.Parameter("use_sim_time", value=use_sim_time)

            ],

        )

        self._lock = Lock()

        self._state = RobotState()
        self.imu_includes_gravity = True
        self.imu_window = ImuWindow()
        self._preview = (0.0, 0.0, 0.0)
        self._preview_received_at = -float("inf")

        self._received = set()

        self._received_at: dict[str, float] = {}

        self._nav_path: tuple[str, list[tuple[float, float]]] | None = None

        self._desired_twist = (0.0, 0.0)

        self._nav_goal_handle = None

        self.world_name = world_name

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.torque_publisher = self.create_publisher(

            Float64MultiArray, "/wheel_torque_commands", 10

        )
        self.control_publisher = self.create_publisher(
            Float64MultiArray, "/nino_rl/control_command", 10
        )
        self.create_subscription(
            Float64MultiArray, "/nino_rl/terrain_preview", self._preview_callback, 10
        )

        self.create_subscription(Odometry, "/odom", self._odom_callback, 10)

        self.create_subscription(

            Odometry, "/ground_truth/odom", self._ground_truth_callback, SENSOR_QOS

        )

        self.create_subscription(Imu, "/imu/data", self._imu_callback, IMU_QOS)

        self.create_subscription(JointState, "/joint_states", self._joint_callback, SENSOR_QOS)

        self.create_subscription(LaserScan, "/scan", self._scan_callback, SENSOR_QOS)

        self.create_subscription(

            Float64MultiArray,

            "/wheel_torque_applied",

            self._applied_torque_callback,

            10,

        )

        if subscribe_plan:

            self.create_subscription(Path, plan_topic, self._plan_callback, 10)

            # Observe the local Nav2 controller before collision_monitor.
            # effort_drive uses Nav2 /cmd_vel as the baseline controller, while
            # RL contributes only residual wheel torque.

            self.create_subscription(Twist, nav_cmd_topic, self._cmd_vel_callback, 10)

            self.create_subscription(

                PoseWithCovarianceStamped,

                "/amcl_pose",

                self._amcl_callback,

                AMCL_POSE_QOS,

            )

            self.initial_pose_publisher = self.create_publisher(

                PoseWithCovarianceStamped, "/initialpose", 10

            )

            try:

                from nav2_msgs.action import NavigateToPose

            except ImportError as error:

                raise RuntimeError(

                    "nav2_msgs is required for Nav2-guided training; install "

                    "ros-jazzy-navigation2 and ros-jazzy-nav2-bringup"

                ) from error

            self._navigate_action_type = NavigateToPose

            self.navigate_to_pose = ActionClient(

                self, NavigateToPose, "/navigate_to_pose"

            )

        self.world_control = self.create_client(

            ControlWorld, f"/world/{world_name}/control"

        )

        self.set_entity_pose = self.create_client(

            SetEntityPose, f"/world/{world_name}/set_pose"

        )

        self.reset_odometry = self.create_client(Trigger, "/reset_wheel_odometry")

        self.spawn_entity = self.create_client(

            SpawnEntity, f"/world/{world_name}/create"

        )

        self.delete_entity = self.create_client(

            DeleteEntity, f"/world/{world_name}/remove"

        )

    def _mark_received(self, name: str) -> None:

        self._received.add(name)

        self._received_at[name] = monotonic()

    def _odom_callback(self, message: Odometry) -> None:

        quaternion = message.pose.pose.orientation

        _, _, yaw = quaternion_to_euler(

            quaternion.x, quaternion.y, quaternion.z, quaternion.w

        )

        with self._lock:

            self._state.odom_stamp_s = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            self._state.x = float(message.pose.pose.position.x)

            self._state.y = float(message.pose.pose.position.y)

            self._state.yaw = yaw

            self._state.linear_velocity = float(message.twist.twist.linear.x)

            self._state.yaw_rate = float(message.twist.twist.angular.z)

            self._mark_received("odom")

    def _ground_truth_callback(self, message: Odometry) -> None:

        with self._lock:

            self._state.ground_linear_velocity = float(message.twist.twist.linear.x)

            self._state.ground_yaw_rate = float(message.twist.twist.angular.z)

            self._mark_received("ground_truth")

    def _imu_callback(self, message: Imu) -> None:

        quaternion = message.orientation

        norm = sqrt(

            quaternion.x * quaternion.x

            + quaternion.y * quaternion.y

            + quaternion.z * quaternion.z

            + quaternion.w * quaternion.w

        )

        if norm < 1.0e-12:

            orientation = (0.0, 0.0, 0.0, 1.0)

        else:

            orientation = (

                quaternion.x / norm,

                quaternion.y / norm,

                quaternion.z / norm,

                quaternion.w / norm,

            )

        roll, pitch, _ = quaternion_to_euler(

            *orientation

        )

        with self._lock:

            self._state.roll = roll

            self._state.pitch = pitch

            self._state.orientation_x = float(orientation[0])

            self._state.orientation_y = float(orientation[1])

            self._state.orientation_z = float(orientation[2])

            self._state.orientation_w = float(orientation[3])

            self._state.gyro_x = float(message.angular_velocity.x)

            self._state.gyro_y = float(message.angular_velocity.y)

            self._state.gyro_z = float(message.angular_velocity.z)

            self._state.accel_x = float(message.linear_acceleration.x)

            self._state.accel_y = float(message.linear_acceleration.y)

            self._state.accel_z = float(message.linear_acceleration.z)
            stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            self.imu_window.add(stamp, vertical_acceleration(
                self._state, self.imu_includes_gravity))

            self._mark_received("imu")

    def _joint_callback(self, message: JointState) -> None:

        velocity = dict(zip(message.name, message.velocity))

        if "left_wheel_joint" not in velocity or "right_wheel_joint" not in velocity:

            return

        with self._lock:

            self._state.left_wheel_velocity = float(velocity["left_wheel_joint"])

            self._state.right_wheel_velocity = float(velocity["right_wheel_joint"])

            self._mark_received("joint")

    def _scan_callback(self, message: LaserScan) -> None:

        with self._lock:

            self._state.lidar_ranges = tuple(float(value) for value in message.ranges)

            self._state.lidar_range_max = float(message.range_max)

            self._mark_received("scan")

    def _applied_torque_callback(self, message: Float64MultiArray) -> None:

        if len(message.data) != 2:

            return

        with self._lock:

            self._state.applied_left_torque = float(message.data[0])

            self._state.applied_right_torque = float(message.data[1])

            self._mark_received("torque")

    def _plan_callback(self, message: Path) -> None:

        points = [(pose.pose.position.x, pose.pose.position.y) for pose in message.poses]

        if len(points) >= 2:

            with self._lock:

                self._nav_path = (message.header.frame_id or "odom", points)

                self._mark_received("plan")

    def _cmd_vel_callback(self, message: Twist) -> None:

        with self._lock:

            self._desired_twist = (

                float(message.linear.x),

                float(message.angular.z),

            )

            self._mark_received("nav_cmd")

    def _amcl_callback(self, _message: PoseWithCovarianceStamped) -> None:

        with self._lock:

            self._mark_received("amcl")

    def snapshot(self) -> RobotState:

        with self._lock:

            return deepcopy(self._state)

    def nav_path(self) -> tuple[str, list[tuple[float, float]]] | None:

        with self._lock:

            return deepcopy(self._nav_path)

    def nav_path_in_odom(

        self, nav_path: tuple[str, list[tuple[float, float]]] | None = None

    ) -> list[tuple[float, float]] | None:

        if nav_path is None:

            nav_path = self.nav_path()

        if nav_path is None:

            return None

        frame, points = nav_path

        if frame in ("", "odom"):

            return points

        try:

            transform = self.tf_buffer.lookup_transform("odom", frame, Time())

        except TransformException:

            return None

        translation = transform.transform.translation

        rotation = transform.transform.rotation

        _, _, yaw = quaternion_to_euler(

            rotation.x, rotation.y, rotation.z, rotation.w

        )

        c, s = cos(yaw), sin(yaw)

        return [

            (

                translation.x + c * x - s * y,

                translation.y + s * x + c * y,

            )

            for x, y in points

        ]

    def desired_twist(self) -> tuple[float, float]:

        with self._lock:

            return self._desired_twist

    def navigation_valid(self, stale_after: float = 2.0) -> bool:

        now = monotonic()

        with self._lock:

            streams_are_fresh = all(

                name in self._received_at

                and now - self._received_at[name] <= stale_after

                for name in (

                    "odom",

                    "imu",

                    "joint",

                    "scan",

                    "plan",

                    "nav_cmd",

                )

            )

        # /amcl_pose is event-driven and may not be republished while the

        # robot is stationary. The live map -> odom transform is the correct

        # localization availability test in that case.

        return streams_are_fresh and self.tf_buffer.can_transform(

            "map", "odom", Time()

        )

    def wait_for_navigation(self, timeout: float, stale_after: float = 2.0) -> None:

        deadline = monotonic() + timeout

        while monotonic() < deadline:

            if self.navigation_valid(stale_after) and self.nav_path_in_odom() is not None:

                return

            sleep(0.05)

        with self._lock:

            stale = [name for name in ("odom", "imu", "joint", "scan", "plan", "nav_cmd")

                     if monotonic() - self._received_at.get(name, 0.0) > stale_after]

        raise TimeoutError(

            f"Nav2 readiness timed out; missing/stale streams: {stale}; "

            f"map->odom available: {self.tf_buffer.can_transform('map', 'odom', Time())}"

        )

    def sensors_ready(self) -> bool:

        with self._lock:

            return {"odom", "imu", "joint", "scan"}.issubset(self._received)

    def wait_for_sensors(self, timeout: float) -> None:

        deadline = monotonic() + timeout

        while monotonic() < deadline:

            if self.sensors_ready():

                return

            sleep(0.05)

        with self._lock:

            missing = sorted({"odom", "imu", "joint", "scan"} - self._received)

        raise TimeoutError(

            "Không nhận đủ dữ liệu ROS trong thời gian chờ; thiếu: " + ", ".join(missing)

        )

    def publish_torque(self, left_nm: float, right_nm: float) -> None:

        message = Float64MultiArray()

        message.data = [float(left_nm), float(right_nm)]

        self.torque_publisher.publish(message)

    def publish_control(self, scale: float, left_nm: float, right_nm: float) -> None:
        """Atomic v2 command: scale Nav2 before PI and add residual torque."""
        message = Float64MultiArray()
        message.data = [float(scale), float(left_nm), float(right_nm)]
        self.control_publisher.publish(message)

    def _preview_callback(self, message) -> None:
        from math import isfinite
        if (len(message.data) != 3 or not all(isfinite(x) for x in message.data)
                or message.data[0] < 0.0):
            return
        with self._lock:
            self._preview = tuple(message.data)
            self._preview_received_at = monotonic()

    def terrain_preview(self, timeout=0.5):
        with self._lock:
            if monotonic() - self._preview_received_at > timeout:
                return [0.0, 0.0, 0.0, 0.0]
            distance, left, right = self._preview
            return [distance / 5.0, left / 0.1, right / 0.1, 1.0]

    def measure_impact(self, start, end, sigma=2.0):
        with self._lock:
            return self.imu_window.measure(start, end, sigma)

    def wait_for_impact(self, start, end, sigma=2.0, timeout=0.5, min_coverage=0.8):
        # /clock and /imu arrive on different DDS queues. Freeze the requested
        # interval while waiting; a real missing interval still raises.
        from nino_rl.timing import wait_for_coverage
        return wait_for_coverage(lambda: self.measure_impact(start, end, sigma),
                                 start, end, timeout, min_coverage)

    def pose_in_frame(self, state, frame):
        if frame == "odom":
            return state.x, state.y, state.yaw
        try:
            transform = self.tf_buffer.lookup_transform(frame, "odom", Time())
        except TransformException as error:
            raise RuntimeError(f"Cannot score trajectory without {frame} <- odom TF") from error
        q = transform.transform.rotation
        _, _, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)
        t = transform.transform.translation
        return (t.x + cos(yaw)*state.x - sin(yaw)*state.y,
                t.y + sin(yaw)*state.x + cos(yaw)*state.y,
                state.yaw + yaw)

    def ground_truth_ready(self, timeout=2.0):
        with self._lock:
            return monotonic() - self._received_at.get("ground_truth", -float("inf")) <= timeout

    def applied_torque_ready(self, timeout=0.5):
        with self._lock:
            return (monotonic() - self._received_at.get("torque", -float("inf")) <= timeout
                    and isfinite(self._state.applied_left_torque)
                    and isfinite(self._state.applied_right_torque))

    def wait_for_v2_controller(self, timeout=5.0):
        deadline = monotonic() + timeout
        while self.control_publisher.get_subscription_count() == 0:
            if monotonic() > deadline:
                raise RuntimeError("No v2 effort_drive: rebuild nino_control and restart simulator")
            sleep(0.05)

    def set_world_paused(self, paused, timeout=5.0):
        if not self.world_control.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("Missing world control bridge for PPO pause/resume")
        request = ControlWorld.Request()
        request.world_control.pause = bool(paused)
        result = self._wait_future(self.world_control.call_async(request), timeout)
        if not result.success:
            raise RuntimeError("Gazebo rejected pause/resume request")

    @staticmethod

    def _wait_future(future, timeout: float):

        deadline = monotonic() + timeout

        while monotonic() < deadline:

            if future.done():

                exception = future.exception()

                if exception is not None:

                    raise RuntimeError(str(exception)) from exception

                return future.result()

            sleep(0.01)

        raise TimeoutError("ROS service call timed out")

    def _clear_navigation_observations(self) -> None:
        """Discard plan/cmd observations belonging to the previous episode."""
        with self._lock:
            self._nav_path = None
            self._desired_twist = (0.0, 0.0)

            for name in ("plan", "nav_cmd"):
                self._received.discard(name)
                self._received_at.pop(name, None)

    def _publish_initial_pose(self, x: float, y: float, yaw: float) -> None:
        message = PoseWithCovarianceStamped()

        # Time(0) asks TF/AMCL to use the latest available transform. Using a
        # just-created sim timestamp can otherwise land a few milliseconds
        # ahead of odom->base_footprint during an episode reset.
        message.header.stamp = Time().to_msg()
        message.header.frame_id = "map"

        message.pose.pose.position.x = float(x)
        message.pose.pose.position.y = float(y)
        message.pose.pose.orientation.z = sin(0.5 * float(yaw))
        message.pose.pose.orientation.w = cos(0.5 * float(yaw))

        message.pose.covariance[0] = 0.04
        message.pose.covariance[7] = 0.04
        message.pose.covariance[35] = 0.03

        self.initial_pose_publisher.publish(message)

    def set_initial_pose(
        self,
        x: float,
        y: float,
        yaw: float,
        timeout: float = 10.0,
    ) -> None:
        """Publish /initialpose until AMCL responds to this reset request.

        Important: an existing map->odom transform may belong to the previous
        episode, so this function deliberately does NOT treat can_transform()
        alone as proof that localization has settled.
        """
        if not hasattr(self, "initial_pose_publisher"):
            raise RuntimeError(
                "This ROS interface was created without Nav2 subscriptions"
            )

        with self._lock:
            self._received.discard("amcl")
            self._received_at.pop("amcl", None)

        deadline = monotonic() + timeout
        next_publish = 0.0
        first_publish_at = None

        while monotonic() < deadline:
            now = monotonic()

            if now >= next_publish:
                if first_publish_at is None:
                    first_publish_at = now
                self._publish_initial_pose(x, y, yaw)
                next_publish = now + 0.25

            with self._lock:
                amcl_received_at = self._received_at.get("amcl")

            # Require an AMCL callback that happened after this reset started.
            if (
                first_publish_at is not None
                and amcl_received_at is not None
                and amcl_received_at >= first_publish_at
            ):
                return

            sleep(0.05)

        raise TimeoutError(
            "AMCL did not acknowledge the new initial pose"
        )

    def wait_for_zero_odometry(
        self,
        timeout: float = 3.0,
        position_tolerance: float = 0.08,
        yaw_tolerance: float = 0.15,
    ) -> None:
        """Wait for a fresh post-reset wheel-odometry sample near zero.

        /reset_wheel_odometry resets the local odom frame to zero even when the
        robot's requested map-frame start pose is non-zero.
        """
        deadline = monotonic() + timeout
        stable_samples = 0
        last_pose = None

        while monotonic() < deadline:
            now = monotonic()

            with self._lock:
                odom_received_at = self._received_at.get("odom")
                x = float(self._state.x)
                y = float(self._state.y)
                yaw = float(self._state.yaw)

            if odom_received_at is None:
                sleep(0.02)
                continue

            age = now - odom_received_at
            position_error = sqrt(x * x + y * y)
            yaw_error = abs(atan2(sin(yaw), cos(yaw)))
            last_pose = (x, y, yaw)

            if (
                age <= 0.5
                and position_error <= position_tolerance
                and yaw_error <= yaw_tolerance
            ):
                stable_samples += 1
                if stable_samples >= 3:
                    return
            else:
                stable_samples = 0

            sleep(0.02)

        raise TimeoutError(
            "Wheel odometry did not settle at zero after reset; "
            f"last odom pose={last_pose}"
        )

    def wait_for_start_pose_in_map(
        self,
        x: float,
        y: float,
        yaw: float,
        timeout: float = 8.0,
        position_tolerance: float = 0.35,
        yaw_tolerance: float = 0.45,
    ) -> None:
        """Wait until live map->base_footprint reflects the new episode pose.

        This closes the reset race where AMCL has accepted /initialpose but
        Nav2 still sees map->base_footprint from the previous episode.
        """
        deadline = monotonic() + timeout
        next_republish = monotonic() + 0.75
        stable_samples = 0
        last_pose = None

        while monotonic() < deadline:
            now = monotonic()

            # If AMCL is taking time to settle, remind it of the intended pose.
            # Do not publish every loop; that would continuously restart AMCL.
            if now >= next_republish:
                self._publish_initial_pose(x, y, yaw)
                next_republish = now + 0.75

            try:
                transform = self.tf_buffer.lookup_transform(
                    "map",
                    "base_footprint",
                    Time(),
                )
            except TransformException:
                stable_samples = 0
                sleep(0.05)
                continue

            translation = transform.transform.translation
            rotation = transform.transform.rotation
            _, _, current_yaw = quaternion_to_euler(
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            )

            tx = float(translation.x)
            ty = float(translation.y)
            last_pose = (tx, ty, current_yaw)

            position_error = sqrt(
                (tx - float(x)) ** 2 + (ty - float(y)) ** 2
            )
            yaw_error = abs(
                atan2(
                    sin(current_yaw - float(yaw)),
                    cos(current_yaw - float(yaw)),
                )
            )

            if (
                position_error <= position_tolerance
                and yaw_error <= yaw_tolerance
            ):
                stable_samples += 1
                if stable_samples >= 3:
                    self.get_logger().info(
                        "Reset TF settled: "
                        f"map->base_footprint=({tx:.2f}, {ty:.2f}), "
                        f"position_error={position_error:.3f} m, "
                        f"yaw_error={yaw_error:.3f} rad"
                    )
                    return
            else:
                stable_samples = 0

            sleep(0.05)

        raise TimeoutError(
            "Reset TF did not settle near the requested start pose "
            f"({x:.2f}, {y:.2f}, {yaw:.2f}); "
            f"last map->base_footprint={last_pose}"
        )

    def initialize_navigation(
        self,
        start_pose: tuple[float, float, float],
        goal_pose: tuple[float, float, float],
        timeout: float = 10.0,
    ) -> None:
        """Initialize AMCL, wait for reset TF to settle, then send Nav2 goal."""
        self.set_initial_pose(*start_pose, timeout=timeout)

        # Do not let Nav2 plan while map->odom still belongs to the previous
        # episode. This is the key reset-race fix.
        self.wait_for_start_pose_in_map(
            *start_pose,
            timeout=timeout,
        )

        # Require a fresh plan and fresh local velocity reference for this
        # goal; never reuse data cached from the previous episode.
        self._clear_navigation_observations()

        self.send_navigation_goal(
            *goal_pose,
            timeout=timeout,
        )

    def send_navigation_goal(

        self, x: float, y: float, yaw: float = 0.0, timeout: float = 10.0

    ) -> None:

        if not hasattr(self, "navigate_to_pose"):

            raise RuntimeError("This ROS interface was created without Nav2 subscriptions")

        if self._nav_goal_handle is not None:

            cancel_future = self._nav_goal_handle.cancel_goal_async()

            try:

                self._wait_future(cancel_future, min(timeout, 2.0))

            except TimeoutError:

                pass

        if not self.navigate_to_pose.wait_for_server(timeout_sec=timeout):

            raise TimeoutError("Missing /navigate_to_pose; start Nav2 before training")

        goal = self._navigate_action_type.Goal()

        goal.pose.header.frame_id = "map"

        goal.pose.header.stamp = self.get_clock().now().to_msg()

        goal.pose.pose.position.x = float(x)

        goal.pose.pose.position.y = float(y)

        goal.pose.pose.orientation.z = sin(0.5 * yaw)

        goal.pose.pose.orientation.w = cos(0.5 * yaw)

        handle = self._wait_future(

            self.navigate_to_pose.send_goal_async(goal), timeout

        )

        if handle is None or not handle.accepted:

            raise RuntimeError("Nav2 rejected the hallway goal")

        self._nav_goal_handle = handle

    def cancel_navigation_goal(self, timeout: float = 2.0) -> None:

        """Cancel the goal owned by this interface, if it still exists."""

        if self._nav_goal_handle is None:

            return

        cancel_future = self._nav_goal_handle.cancel_goal_async()

        try:

            self._wait_future(cancel_future, timeout)

            self._wait_future(self._nav_goal_handle.get_result_async(), timeout)

        except TimeoutError:

            pass

        finally:

            self._nav_goal_handle = None

    def reset_episode(
        self,
        timeout: float = 8.0,
        start_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
        goal_pose: tuple[float, float, float] | None = None,
    ) -> None:
        """Reset Gazebo and wheel odometry without racing Nav2 localization.

        Normal training flow:
            reset_episode(start_pose)
            configure_training_cables(...)
            initialize_navigation(start_pose, goal_pose)

        goal_pose remains supported for callers that want reset+navigation in
        one call.
        """
        # ---------------------------------------------------------
        # 1. Stop the previous Nav2 action and RL residual torque.
        # ---------------------------------------------------------
        self.cancel_navigation_goal()
        self._clear_navigation_observations()

        with self._lock:
            self._received.discard("amcl")
            self._received_at.pop("amcl", None)

        for _ in range(5):
            self.publish_torque(0.0, 0.0)
            sleep(0.02)

        # ---------------------------------------------------------
        # 2. Reset Gazebo model state.
        # ---------------------------------------------------------
        if not self.world_control.wait_for_service(timeout_sec=timeout):
            raise TimeoutError(
                f"Missing {self.world_control.srv_name}; "
                "start training_sim.launch.py"
            )

        request = ControlWorld.Request()
        request.world_control.reset.model_only = True

        response = self._wait_future(
            self.world_control.call_async(request),
            timeout,
        )

        if response is None or not response.success:
            raise RuntimeError(
                "Gazebo rejected the model-only episode reset"
            )

        # Explicitly teleport Nino to the requested start pose. Do this even
        # after model_only reset so the episode start is deterministic.
        if not self.set_entity_pose.wait_for_service(timeout_sec=timeout):
            raise TimeoutError(
                f"Missing {self.set_entity_pose.srv_name}; "
                "rebuild and restart training_sim.launch.py"
            )

        pose_request = SetEntityPose.Request()
        pose_request.entity.name = "nino"
        pose_request.entity.type = Entity.MODEL
        pose_request.pose.position.x = float(start_pose[0])
        pose_request.pose.position.y = float(start_pose[1])
        pose_request.pose.position.z = 0.0
        pose_request.pose.orientation.x = 0.0
        pose_request.pose.orientation.y = 0.0
        pose_request.pose.orientation.z = sin(
            0.5 * float(start_pose[2])
        )
        pose_request.pose.orientation.w = cos(
            0.5 * float(start_pose[2])
        )

        response = self._wait_future(
            self.set_entity_pose.call_async(pose_request),
            timeout,
        )

        if response is None or not response.success:
            raise RuntimeError(
                f"Gazebo rejected reset pose {start_pose}"
            )

        # Give Gazebo one short physics/transport window to expose the new
        # model pose before resetting local wheel odometry.
        sleep(0.10)

        # ---------------------------------------------------------
        # 3. Reset the local wheel-odometry/controller state.
        # ---------------------------------------------------------
        if not self.reset_odometry.wait_for_service(
            timeout_sec=timeout
        ):
            raise TimeoutError(
                "Missing /reset_wheel_odometry from effort_drive"
            )

        # Drop any pre-reset odom marker. We want a new sample generated after
        # the reset service returns.
        with self._lock:
            self._received.discard("odom")
            self._received_at.pop("odom", None)

        response = self._wait_future(
            self.reset_odometry.call_async(Trigger.Request()),
            timeout,
        )

        if response is None or not response.success:
            message = (
                response.message
                if response is not None
                else "no response"
            )
            raise RuntimeError(
                f"Wheel odometry reset failed: {message}"
            )

        # Clear again in case an odom callback raced with the service call,
        # then require several fresh zero-ish samples.
        with self._lock:
            self._received.discard("odom")
            self._received_at.pop("odom", None)

        self.wait_for_zero_odometry(
            timeout=min(timeout, 3.0),
        )

        # ---------------------------------------------------------
        # 4. Optional one-call navigation startup.
        # ---------------------------------------------------------
        if goal_pose is not None:
            self.initialize_navigation(
                start_pose,
                goal_pose,
                timeout=timeout,
            )

        # Small settling delay before terrain replacement or environment read.
        sleep(0.20)

    @staticmethod

    def _cable_sdf(name: str, x: float, radius: float, angle: float) -> str:

        length = 4.0 / max(cos(angle), 0.70)

        return f"""<?xml version='1.0'?>

<sdf version='1.9'><model name='{name}'><static>true</static>

<pose>{x:.6f} 0 {radius:.6f} 1.57079632679 0 {angle:.6f}</pose>

<link name='cable'><collision name='collision'><geometry><cylinder>

<radius>{radius:.6f}</radius><length>{length:.6f}</length>

</cylinder></geometry></collision><visual name='visual'><geometry><cylinder>

<radius>{radius:.6f}</radius><length>{length:.6f}</length>

</cylinder></geometry><material><ambient>0.08 0.08 0.08 1</ambient>

<diffuse>0.12 0.12 0.12 1</diffuse></material></visual></link></model></sdf>"""

    def configure_training_cables(self, cables, timeout=5.0):
        """Compatibility entry point for cable-only callers."""
        self.configure_training_terrain(cable_layout(cables), timeout)

    def configure_training_terrain(self, layout, timeout=5.0):
        """Replace floor and obstacles after resetting onto the permanent pad.

        The end pads belong to terrain_floor_v1 and are never removed. Gazebo
        keeps servicing entity requests without removing support under Nino.
        Fail closed if the old, solid-floor world is still running.
        """
        sdf = terrain_sdf(layout)  # Validate before any scene mutation.
        if not self.delete_entity.wait_for_service(timeout_sec=timeout):
            raise TimeoutError(f"Missing {self.delete_entity.srv_name}")
        if not self.spawn_entity.wait_for_service(timeout_sec=timeout):
            raise TimeoutError(f"Missing {self.spawn_entity.srv_name}")
        scene = subprocess.run(
            ["gz", "service", "-s", f"/world/{self.world_name}/scene/info",
             "--reqtype", "gz.msgs.Empty", "--reptype", "gz.msgs.Scene",
             "--timeout", str(int(timeout * 1000)), "--req", ""],
            capture_output=True, text=True, timeout=timeout + 2.0,
            env={**os.environ, "GZ_IP": "127.0.0.1"})
        models = set(re.findall(r'^model\s*\{\n\s*name: "([^"]+)"', scene.stdout, re.M))
        if scene.returncode != 0 or "nino" not in models:
            raise RuntimeError("Could not query Gazebo models before terrain reset")
        if "terrain_floor_v1" not in models:
            raise RuntimeError("Old solid-floor world: rebuild nino_description and restart Gazebo")
        names = {name for name in models if name in {"training_terrain", "cable_bumps"}
                 or re.fullmatch(r"training_cable_\d+", name)}
        for name in sorted(names):
            request = DeleteEntity.Request()
            request.entity.name = name
            request.entity.type = Entity.MODEL
            response = self._wait_future(self.delete_entity.call_async(request), timeout)
            if response is None or not response.success:
                raise RuntimeError(f"Gazebo failed to remove {name}")
        request = self._terrain_spawn_request(sdf)
        response = self._wait_future(self.spawn_entity.call_async(request), timeout)
        if response is None or not response.success:
            raise RuntimeError("Gazebo failed to spawn training_terrain")

    @staticmethod
    def _terrain_spawn_request(sdf):
        request = SpawnEntity.Request()
        request.entity_factory.name = "training_terrain"
        request.entity_factory.allow_renaming = False
        request.entity_factory.sdf = sdf
        # All geometry uses world coordinates in link-local poses.
        request.entity_factory.pose.orientation.w = 1.0
        return request

    @classmethod

    def _cable_spawn_request(cls, name: str, x: float, radius: float, angle: float):

        request = SpawnEntity.Request()

        request.entity_factory.name = name

        request.entity_factory.allow_renaming = False

        request.entity_factory.sdf = cls._cable_sdf(name, x, radius, angle)

        # EntityFactory.pose overrides the SDF model pose. Leaving its default

        # identity pose spawns an upright cylinder at the robot's origin.

        pose = request.entity_factory.pose

        pose.position.x = float(x)

        pose.position.z = float(radius)

        # Quaternion for roll=pi/2, pitch=0, yaw=angle: cable lies on the floor.

        pose.orientation.x = sqrt(0.5) * cos(0.5 * angle)

        pose.orientation.y = sqrt(0.5) * sin(0.5 * angle)

        pose.orientation.z = sqrt(0.5) * sin(0.5 * angle)

        pose.orientation.w = sqrt(0.5) * cos(0.5 * angle)

        return request
