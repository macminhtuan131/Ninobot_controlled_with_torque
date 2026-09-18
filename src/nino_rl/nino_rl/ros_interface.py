"""ROS 2 transport shared by the Gym environment and deployed policy node."""

from __future__ import annotations

from copy import deepcopy
from math import atan2, cos, sin, sqrt, isfinite
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
from rosgraph_msgs.msg import Clock
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import ControlWorld, DeleteEntity, SetEntityPose, SpawnEntity
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

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

TERRAIN_SENSOR_HEIGHT_M = 0.2325
TERRAIN_SENSOR_PITCH_RAD = 0.45
TERRAIN_HEIGHT_THRESHOLD_M = 0.006
TERRAIN_PREVIEW_RANGE_M = 1.0


class RosRobotInterface(Node):
    def __init__(
        self,
        world_name: str = "long_hall",
        subscribe_plan: bool = False,
        use_sim_time: bool = True,
        node_name: str = "nino_rl_interface",
        plan_topic: str = "/plan",
        nav_cmd_topic: str = "/cmd_vel_nav",
        cmd_vel_topic: str = "/cmd_vel",
        imu_topic: str = "/imu/data",
        physics_step_seconds: float = 0.002,
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
        self._imu_epoch_min_stamp = -float("inf")
        self._sim_clock_stamp = None
        self._preview = (0.0, 0.0, 0.0)
        self._preview_received_at = -float("inf")
        self._received = set()
        self._received_at: dict[str, float] = {}
        self._nav_path: tuple[str, list[tuple[float, float]]] | None = None
        self._desired_twist = (0.0, 0.0)
        self._nav_goal_handle = None
        self.world_name = world_name
        if not isfinite(physics_step_seconds) or physics_step_seconds <= 0.0:
            raise ValueError("physics_step_seconds must be positive and finite")
        self.physics_step_seconds = float(physics_step_seconds)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.torque_publisher = self.create_publisher(
            Float64MultiArray, "/wheel_torque_commands", 10
        )
        self.control_publisher = self.create_publisher(
            Float64MultiArray, "/nino_rl/control_command", 10
        )
        self.cmd_vel_publisher = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.create_subscription(
            Float64MultiArray, "/nino_rl/terrain_preview", self._preview_callback, 10
        )
        self.create_subscription(
            LaserScan, "/terrain_scan", self._terrain_scan_callback, SENSOR_QOS
        )
        self.create_subscription(Odometry, "/odom", self._odom_callback, 10)
        self.create_subscription(
            Odometry, "/ground_truth/odom", self._ground_truth_callback, SENSOR_QOS
        )
        self.create_subscription(Clock, "/clock", self._clock_callback, SENSOR_QOS)
        self.create_subscription(Imu, imu_topic, self._imu_callback, IMU_QOS)
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

    def _clock_callback(self, message: Clock) -> None:
        with self._lock:
            self._sim_clock_stamp = (
                message.clock.sec + message.clock.nanosec * 1e-9
            )
            self._mark_received("clock")

    def _imu_callback(self, message: Imu) -> None:
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
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
            # After an episode reset, DDS may continue delivering queued IMU
            # samples generated before the paused epoch boundary. Ignore the
            # whole sample, including actor state, based on simulation time.
            if stamp <= self._imu_epoch_min_stamp:
                return
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
            self._state.lidar_stamp_s = (
                message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            )
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

    def publish_straight_command(self, linear_m_s: float) -> None:
        """Publish the direct baseline command; angular velocity is always zero."""
        if not isfinite(linear_m_s) or linear_m_s < 0.0:
            raise ValueError("straight speed must be finite and non-negative")
        message = Twist()
        message.linear.x = float(linear_m_s)
        message.angular.z = 0.0
        self.cmd_vel_publisher.publish(message)
        with self._lock:
            self._desired_twist = (float(linear_m_s), 0.0)
            self._mark_received("straight_cmd")

    def straight_reference_valid(self, stale_after: float = 2.0) -> bool:
        """Return whether the state needed by the straight controller is fresh.

        LiDAR is deliberately excluded.  Gazebo's GPU ray sensor is an
        asynchronous, BEST_EFFORT safety aid and must not invalidate the
        wheel/odometry reference when rendering briefly falls behind.
        """
        now = monotonic()
        with self._lock:
            return all(
                name in self._received_at
                and now - self._received_at[name] <= stale_after
                for name in ("odom", "imu", "joint", "straight_cmd")
            )

    def sensor_stream_ready(self, name: str, stale_after: float = 2.0) -> bool:
        """Return whether a named stream has delivered a recent DDS sample."""
        now = monotonic()
        with self._lock:
            return (
                name in self._received_at
                and now - self._received_at[name] <= stale_after
            )

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
        """Atomic v2 command: scale the straight baseline and add residual torque."""
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
            self._mark_received("terrain")

    def _terrain_scan_callback(self, message: LaserScan) -> None:
        """Convert the downward fan into distance and signed left/right relief."""
        samples = []
        for index, value in enumerate(message.ranges):
            distance = float(value)
            if (
                not isfinite(distance)
                or distance < float(message.range_min)
                or distance > float(message.range_max)
            ):
                continue
            angle = float(message.angle_min) + index * float(message.angle_increment)
            horizontal = distance * cos(TERRAIN_SENSOR_PITCH_RAD)
            forward = horizontal * cos(angle)
            lateral = horizontal * sin(angle)
            height = TERRAIN_SENSOR_HEIGHT_M - distance * sin(
                TERRAIN_SENSOR_PITCH_RAD
            )
            if forward > 0.0 and abs(height) >= TERRAIN_HEIGHT_THRESHOLD_M:
                samples.append((forward, lateral, height))

        def strongest(values) -> float:
            return float(max(values, key=lambda item: abs(item), default=0.0))

        if samples:
            preview_distance = min(sample[0] for sample in samples)
            left_height = strongest(
                sample[2] for sample in samples if sample[1] >= 0.0
            )
            right_height = strongest(
                sample[2] for sample in samples if sample[1] < 0.0
            )
        else:
            preview_distance = TERRAIN_PREVIEW_RANGE_M
            left_height = right_height = 0.0
        with self._lock:
            self._preview = (preview_distance, left_height, right_height)
            self._preview_received_at = monotonic()
            self._mark_received("terrain")

    def terrain_preview(self, timeout: float | None = 0.5):
        """Return normalized relief, optionally enforcing wall-time freshness.

        Lockstep training passes ``None`` after its post-step callback barrier:
        a sample cannot become physically stale while Gazebo is paused. Live
        deployment retains a finite watchdog timeout.
        """
        with self._lock:
            if (
                timeout is not None
                and monotonic() - self._preview_received_at > timeout
            ):
                return [0.0, 0.0, 0.0, 0.0]
            distance, left, right = self._preview
            return [
                distance / TERRAIN_PREVIEW_RANGE_M,
                left / 0.1,
                right / 0.1,
                1.0,
            ]

    def wait_for_terrain_preview(self, timeout: float = 5.0) -> None:
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            if self.sensor_stream_ready("terrain", timeout):
                return
            sleep(0.05)
        raise TimeoutError("No downward terrain preview scan was received")

    def measure_impact(self, start, end, sigma=2.0):
        with self._lock:
            return self.imu_window.measure(start, end, sigma)

    def estimate_impact(self, start, end, sigma=2.0):
        with self._lock:
            return self.imu_window.estimate(start, end, sigma)

    def latest_clock_stamp(self):
        with self._lock:
            if self._sim_clock_stamp is None:
                raise RuntimeError("No /clock sample is available")
            return self._sim_clock_stamp

    def wait_for_clock_quiescence(self, timeout=2.0, quiet_time=0.05):
        """Return the actual paused Gazebo time after queued /clock drains."""
        from nino_rl.timing import wait_for_quiescent_timestamp
        return wait_for_quiescent_timestamp(
            self.latest_clock_stamp, timeout, quiet_time
        )

    def begin_lockstep_epoch(self, timeout=2.0):
        """Discard pre-reset timing callbacks and establish a fresh epoch.

        Gazebo model reset and DDS delivery are asynchronous. Queued IMU
        messages are rejected by their simulation timestamp rather than by
        waiting for a high-rate callback queue to become silent. The single
        flush step is episode setup overhead, not per-action overhead.
        """
        paused_clock = self.wait_for_clock_quiescence(timeout=timeout)
        with self._lock:
            self.imu_window.samples.clear()
            self._imu_epoch_min_stamp = paused_clock
            self._received.discard("imu")
            self._received_at.pop("imu", None)

        self.advance_world(1, timeout=timeout)
        target = paused_clock + self.physics_step_seconds
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            with self._lock:
                clock = self._sim_clock_stamp
                ready = (
                    clock is not None
                    and clock >= target - 0.5 * self.physics_step_seconds
                )
            if ready:
                break
            sleep(0.002)
        else:
            raise RuntimeError(
                "No post-boundary /clock sample after lockstep epoch flush"
            )

        return self.wait_for_clock_quiescence(timeout=timeout)

    def wait_for_impact(self, start, end, sigma=2.0, timeout=0.5,
                        min_coverage=0.8, max_latest_lag=0.05):
        # /clock and /imu arrive on different DDS queues. Freeze the requested
        # interval while waiting; a real missing interval still raises.
        from nino_rl.timing import wait_for_coverage
        return wait_for_coverage(lambda: self.measure_impact(start, end, sigma),
                                 start, end, timeout, min_coverage,
                                 max_latest_lag)

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

    def ground_truth_valid(self):
        with self._lock:
            return (
                isfinite(self._state.ground_linear_velocity)
                and isfinite(self._state.ground_yaw_rate)
            )

    def applied_torque_ready(self, timeout=0.5):
        with self._lock:
            return (monotonic() - self._received_at.get("torque", -float("inf")) <= timeout
                    and isfinite(self._state.applied_left_torque)
                    and isfinite(self._state.applied_right_torque))

    def applied_torque_valid(self):
        with self._lock:
            return (
                isfinite(self._state.applied_left_torque)
                and isfinite(self._state.applied_right_torque)
            )

    def sensor_markers(self, names):
        """Return wall-time callback markers used for a lockstep data barrier."""
        with self._lock:
            return {
                name: self._received_at.get(name, -float("inf"))
                for name in names
            }

    def wait_for_sensor_updates(self, previous, timeout=2.0):
        """Wait until every named stream has delivered a post-step message."""
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            with self._lock:
                missing = [
                    name for name, marker in previous.items()
                    if self._received_at.get(name, -float("inf")) <= marker
                ]
            if not missing:
                return
            sleep(0.002)
        raise RuntimeError(
            "Lockstep sensor updates timed out; missing fresh: "
            + ", ".join(missing)
        )

    def wait_for_v2_controller(self, timeout=5.0):
        deadline = monotonic() + timeout
        while self.control_publisher.get_subscription_count() == 0:
            if monotonic() > deadline:
                raise RuntimeError("No v2 effort_drive: rebuild nino_control and restart simulator")
            sleep(0.05)

    def set_world_paused(self, paused, timeout=5.0):
        """Set pause state with idempotent retries and independent clock proof.

        ros_gz_bridge may drop a response after Gazebo has executed the
        request. Reissuing the same pause state is safe, unlike reissuing a
        multi-step request. A missing response is accepted when /clock proves
        the requested state was reached.
        """
        if not self.world_control.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("Missing world control bridge for PPO pause/resume")
        attempts = 3
        last_error = None
        for attempt in range(1, attempts + 1):
            request = ControlWorld.Request()
            request.world_control.pause = bool(paused)
            with self._lock:
                clock_before = self._sim_clock_stamp
            future = self.world_control.call_async(request)
            try:
                result = self._wait_future(future, timeout)
            except TimeoutError as error:
                last_error = error
                if not future.done():
                    try:
                        self.world_control.remove_pending_request(future)
                    except RuntimeError:
                        pass

                confirmed = False
                if paused:
                    try:
                        self.wait_for_clock_quiescence(
                            timeout=min(2.0, max(0.2, timeout)),
                            quiet_time=0.10,
                        )
                        confirmed = True
                    except (RuntimeError, ValueError):
                        pass
                elif clock_before is not None:
                    deadline = monotonic() + min(2.0, max(0.2, timeout))
                    while monotonic() < deadline:
                        with self._lock:
                            clock_after = self._sim_clock_stamp
                        if (
                            clock_after is not None
                            and clock_after
                            > clock_before + 0.5 * self.physics_step_seconds
                        ):
                            confirmed = True
                            break
                        sleep(0.002)

                if confirmed:
                    self.get_logger().warn(
                        "World pause/resume response timed out, but /clock "
                        f"confirms paused={bool(paused)}"
                    )
                    return
                self.get_logger().warn(
                    "World pause/resume response timed out without clock "
                    f"confirmation; retry {attempt}/{attempts}"
                )
                continue

            if result is not None and result.success:
                if not paused:
                    return
                try:
                    self.wait_for_clock_quiescence(
                        timeout=min(2.0, max(0.2, timeout)),
                        quiet_time=0.10,
                    )
                    return
                except (RuntimeError, ValueError) as error:
                    last_error = error
                    self.get_logger().warn(
                        "Gazebo accepted pause but /clock is still advancing; "
                        f"retry {attempt}/{attempts}"
                    )
                    continue
            last_error = RuntimeError("Gazebo rejected pause/resume request")
            self.get_logger().warn(
                f"Gazebo rejected pause={bool(paused)}; retry {attempt}/{attempts}"
            )

        raise RuntimeError(
            f"Could not establish Gazebo paused={bool(paused)} after "
            f"{attempts} idempotent attempts"
        ) from last_error

    def advance_world(
        self, physics_steps, timeout=5.0, min_completion_fraction=0.80
    ):
        """Advance a paused world and return the credited simulation duration.

        The Gazebo multi_step command is atomic. /clock is used only as
        independent evidence because its depth-1 BEST_EFFORT bridge can omit
        the last samples; raw callback timestamps must not redefine the fixed
        policy interval.
        """
        if not isinstance(physics_steps, int) or physics_steps < 1:
            raise ValueError("physics_steps must be a positive integer")
        if not 0.0 < min_completion_fraction <= 1.0:
            raise ValueError("min_completion_fraction must be in (0, 1]")
        if not self.world_control.wait_for_service(timeout_sec=timeout):
            raise RuntimeError("Missing world control bridge for lockstep training")
        request = ControlWorld.Request()
        request.world_control.pause = True
        request.world_control.multi_step = physics_steps
        with self._lock:
            clock_before = self._sim_clock_stamp
        if clock_before is None:
            raise RuntimeError("Cannot step Gazebo without a /clock baseline")
        requested_duration = physics_steps * self.physics_step_seconds
        target = clock_before + requested_duration
        future = self.world_control.call_async(request)
        try:
            result = self._wait_future(future, timeout)
        except TimeoutError as error:
            # ros_gz_bridge can lose or delay the service response after Gazebo
            # has already executed the atomic multi_step request. Never retry
            # blindly: that would advance the policy interval twice. Accept
            # only when /clock independently proves every requested step ran.
            grace_deadline = monotonic() + min(2.0, max(0.1, 0.2 * timeout))
            clock_after = clock_before
            while monotonic() < grace_deadline:
                with self._lock:
                    if self._sim_clock_stamp is not None:
                        clock_after = self._sim_clock_stamp
                if clock_after >= target - 0.5 * self.physics_step_seconds:
                    if not future.done():
                        self.world_control.remove_pending_request(future)
                    self.get_logger().warn(
                        "World-control response timed out, but /clock confirms "
                        f"all {physics_steps} requested physics steps completed"
                    )
                    return requested_duration
                sleep(0.002)

            # A depth-1 BEST_EFFORT /clock subscription can miss the final
            # samples even though the request has stopped with the world
            # paused. Once the clock is quiescent, accept a near-complete
            # interval and let the environment score its observed duration.
            clock_quiescent = False
            try:
                clock_after = self.wait_for_clock_quiescence(
                    timeout=min(2.0, max(0.2, timeout)), quiet_time=0.10
                )
                clock_quiescent = True
            except (RuntimeError, ValueError):
                pass
            advanced = max(0.0, clock_after - clock_before)
            if (
                clock_quiescent
                and advanced >= min_completion_fraction * requested_duration
            ):
                if not future.done():
                    try:
                        self.world_control.remove_pending_request(future)
                    except RuntimeError:
                        pass
                self.get_logger().warn(
                    "World-control response timed out; /clock verified a "
                    f"paused near-complete atomic chunk "
                    f"{advanced:.6f}/{requested_duration:.6f}s"
                )
                return requested_duration
            raise TimeoutError(
                "World-control response timed out and verified progress was "
                "below the safe completion threshold "
                f"(before={clock_before:.6f}, after={clock_after:.6f}, "
                f"target={target:.6f}, minimum_fraction="
                f"{min_completion_fraction:.3f})"
            ) from error
        if not result.success:
            raise RuntimeError("Gazebo rejected lockstep physics request")
        return requested_duration

    def reset_drive_state(self, timeout=5.0):
        """Reset wheel odometry and release any stale v2 command ownership."""
        if not self.reset_odometry.wait_for_service(timeout_sec=timeout):
            raise TimeoutError("Missing /reset_wheel_odometry from effort_drive")
        response = self._call_idempotent_service(
            self.reset_odometry,
            Trigger.Request,
            timeout,
            "wheel odometry reset",
        )
        if response is None or not response.success:
            message = response.message if response is not None else "no response"
            raise RuntimeError(f"Wheel odometry reset failed: {message}")

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

    def _call_idempotent_service(
        self, client, request_factory, timeout: float, operation: str,
        attempts: int = 3,
    ):
        """Retry a state-setting service whose duplicate execution is safe."""
        last_error = None
        for attempt in range(1, attempts + 1):
            future = client.call_async(request_factory())
            try:
                return self._wait_future(future, timeout)
            except TimeoutError as error:
                last_error = error
                if not future.done():
                    try:
                        client.remove_pending_request(future)
                    except RuntimeError:
                        pass
                self.get_logger().warn(
                    f"{operation} response timed out; retry "
                    f"{attempt}/{attempts}"
                )
        raise TimeoutError(
            f"{operation} timed out after {attempts} idempotent attempts"
        ) from last_error

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

        deadline = monotonic() + timeout
        attempts = 0
        while monotonic() < deadline:
            attempts += 1
            goal = self._navigate_action_type.Goal()
            goal.pose.header.frame_id = "map"
            goal.pose.header.stamp = self.get_clock().now().to_msg()
            goal.pose.pose.position.x = float(x)
            goal.pose.pose.position.y = float(y)
            goal.pose.pose.orientation.z = sin(0.5 * yaw)
            goal.pose.pose.orientation.w = cos(0.5 * yaw)
            try:
                handle = self._wait_future(
                    self.navigate_to_pose.send_goal_async(goal),
                    max(0.1, deadline - monotonic()),
                )
            except TimeoutError:
                handle = None
            if handle is not None and handle.accepted:
                self._nav_goal_handle = handle
                return
            # A cancellation from the previous rapid check_env/reset cycle can
            # still be completing inside bt_navigator. Rejection is transient;
            # retry with a fresh goal stamp while the action server remains up.
            sleep(0.10)

        raise RuntimeError(
            f"Nav2 rejected the hallway goal {attempts} times during reset"
        )

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

        response = self._call_idempotent_service(
            self.world_control,
            lambda: deepcopy(request),
            timeout,
            "Gazebo model reset",
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

        response = self._call_idempotent_service(
            self.set_entity_pose,
            lambda: deepcopy(pose_request),
            timeout,
            "Nino start-pose reset",
        )

        if response is None or not response.success:
            raise RuntimeError(
                f"Gazebo rejected reset pose {start_pose}"
            )

        # A preceding lockstep request leaves Gazebo paused. Although the
        # environment requests an unpause before reset, ControlWorld's model
        # reset can race that state transition. Reassert running state after
        # teleport so controller timers can publish fresh zero odometry.
        self.set_world_paused(False, timeout=timeout)

        # Give Gazebo one short physics/transport window to expose the new
        # model pose before resetting local wheel odometry.
        sleep(0.10)

        # ---------------------------------------------------------
        # 3. Reset the local wheel-odometry/controller state.
        # ---------------------------------------------------------
        # Drop any pre-reset odom marker. We want a new sample generated after
        # the reset service returns.
        with self._lock:
            self._received.discard("odom")
            self._received_at.pop("odom", None)

        self.reset_drive_state(timeout)

        # Keep simulation time moving after the drive reset as well. Without
        # this, a late pause transition can leave the odometry marker empty
        # forever even though both reset services succeeded.
        self.set_world_paused(False, timeout=timeout)

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

    def configure_training_cables(
        self, cables: list[tuple[float, float, float]], timeout: float = 5.0
    ) -> None:
        """Replace the legacy dense cable model with this episode's curriculum."""
        if not self.delete_entity.wait_for_service(timeout_sec=timeout):
            raise TimeoutError(f"Missing {self.delete_entity.srv_name}")

        # The external `gz service /scene/info` CLI is unreliable during rapid
        # check_env resets. The curriculum uses a finite namespace, so deleting
        # every possible cable name is idempotent and avoids that discovery race.
        terrain_names = {"cable_bumps", *[f"training_cable_{i}" for i in range(8)]}

        def remove_if_present(name):
            def request():
                value = DeleteEntity.Request()
                value.entity.name = name
                value.entity.type = Entity.MODEL
                return value
            response = self._call_idempotent_service(
                self.delete_entity, request, timeout, f"delete {name}"
            )
            # Gazebo reports success=False when the entity is already absent;
            # that is the desired postcondition for an idempotent reset.
            return response is not None and response.success

        for name in sorted(terrain_names):
            remove_if_present(name)

        if not cables:
            return

        if not self.spawn_entity.wait_for_service(timeout_sec=timeout):
            raise TimeoutError(f"Missing {self.spawn_entity.srv_name}")

        for index, (x, radius, angle) in enumerate(cables):
            name = f"training_cable_{index}"
            request = self._cable_spawn_request(name, x, radius, angle)
            response = self._call_idempotent_service(
                self.spawn_entity, lambda: deepcopy(request), timeout, f"spawn {name}"
            )
            if response is None or not response.success:
                # A prior asynchronous removal may have raced the first create.
                # Remove this exact name once more and retry with a fresh request.
                remove_if_present(name)
                request = self._cable_spawn_request(name, x, radius, angle)
                response = self._call_idempotent_service(
                    self.spawn_entity,
                    lambda: deepcopy(request),
                    timeout,
                    f"spawn {name} after removal",
                )
                if response is None or not response.success:
                    raise RuntimeError(f"Gazebo failed to spawn {name} after retry")

    @staticmethod
    def _goal_marker_sdf(name: str, radius: float = 0.10) -> str:
        """Return a bright, visual-only goal beacon that cannot affect physics."""
        return f"""<?xml version='1.0'?>
<sdf version='1.9'><model name='{name}'><static>true</static><link name='marker'>
<visual name='goal_disc'><pose>0 0 0.01 0 0 0</pose><geometry><cylinder>
<radius>{radius:.6f}</radius><length>0.02</length></cylinder></geometry><material>
<ambient>0.05 1 0.05 1</ambient><diffuse>0.05 1 0.05 1</diffuse>
<emissive>0 0.6 0 1</emissive></material></visual>
<visual name='goal_pole'><pose>0 0 0.50 0 0 0</pose><geometry><cylinder>
<radius>0.025</radius><length>1.0</length></cylinder></geometry><material>
<ambient>0.05 1 0.05 1</ambient><diffuse>0.05 1 0.05 1</diffuse>
<emissive>0 0.6 0 1</emissive></material></visual>
<visual name='goal_flag'><pose>0.14 0 0.82 0 0 0</pose><geometry><box>
<size>0.28 0.02 0.20</size></box></geometry><material>
<ambient>0.05 1 0.05 1</ambient><diffuse>0.05 1 0.05 1</diffuse>
<emissive>0 0.6 0 1</emissive></material></visual>
</link></model></sdf>"""

    def configure_goal_marker(
        self, goal_pose: tuple[float, float, float], timeout: float = 5.0,
        radius: float = 0.10,
    ) -> None:
        """Create the goal marker once, then move it without visual gaps."""
        name = "training_goal_marker"

        def pose_request():
            request = SetEntityPose.Request()
            request.entity.name = name
            request.entity.type = Entity.MODEL
            request.pose.position.x = float(goal_pose[0])
            request.pose.position.y = float(goal_pose[1])
            request.pose.orientation.z = sin(0.5 * float(goal_pose[2]))
            request.pose.orientation.w = cos(0.5 * float(goal_pose[2]))
            return request

        if not self.set_entity_pose.wait_for_service(timeout_sec=timeout):
            raise TimeoutError(f"Missing {self.set_entity_pose.srv_name}")
        response = self._call_idempotent_service(
            self.set_entity_pose,
            pose_request,
            timeout,
            "move goal marker",
        )
        if response is not None and response.success:
            return

        if not self.spawn_entity.wait_for_service(timeout_sec=timeout):
            raise TimeoutError(f"Missing {self.spawn_entity.srv_name}")

        def spawn_request():
            request = SpawnEntity.Request()
            request.entity_factory.name = name
            request.entity_factory.allow_renaming = False
            request.entity_factory.sdf = self._goal_marker_sdf(name, radius)
            request.entity_factory.pose.position.x = float(goal_pose[0])
            request.entity_factory.pose.position.y = float(goal_pose[1])
            request.entity_factory.pose.orientation.z = sin(0.5 * float(goal_pose[2]))
            request.entity_factory.pose.orientation.w = cos(0.5 * float(goal_pose[2]))
            return request

        response = self._call_idempotent_service(
            self.spawn_entity,
            spawn_request,
            timeout,
            "create goal marker",
        )
        if response is None or not response.success:
            # Another asynchronous caller may have created the fixed-name
            # marker between set_pose and create. Moving it is idempotent and
            # avoids the delete/recreate interval that made it disappear.
            response = self._call_idempotent_service(
                self.set_entity_pose,
                pose_request,
                timeout,
                "move goal marker after create race",
            )
            if response is None or not response.success:
                raise RuntimeError("Gazebo failed to create or move the visual goal marker")

    @staticmethod
    def _adaptive_terrain_sdf(
        name: str, features: list[tuple[str, float, float, float]]
    ) -> str:
        """Build one static model containing randomized path hazards."""
        from math import atan2, cos, pi, sin

        links = []
        for index, (kind, x, y, size) in enumerate(features):
            prefix = f"feature_{index}_{kind}"
            if kind == "obstacle":
                height = 0.35
                body = f"""
<collision name='{prefix}_collision'><pose>{x:.6f} {y:.6f} {0.5 * height:.6f} 0 0 0</pose>
<geometry><cylinder><radius>{size:.6f}</radius><length>{height:.6f}</length></cylinder></geometry></collision>
<visual name='{prefix}_visual'><pose>{x:.6f} {y:.6f} {0.5 * height:.6f} 0 0 0</pose>
<geometry><cylinder><radius>{size:.6f}</radius><length>{height:.6f}</length></cylinder></geometry>
<material><ambient>0.85 0.20 0.04 1</ambient><diffuse>1 0.28 0.05 1</diffuse></material></visual>"""
            elif kind == "cable":
                length = 0.55
                body = f"""
<collision name='{prefix}_collision'><pose>{x:.6f} {y:.6f} {size:.6f} 1.57079632679 0 1.57079632679</pose>
<geometry><cylinder><radius>{size:.6f}</radius><length>{length:.6f}</length></cylinder></geometry></collision>
<visual name='{prefix}_visual'><pose>{x:.6f} {y:.6f} {size:.6f} 1.57079632679 0 1.57079632679</pose>
<geometry><cylinder><radius>{size:.6f}</radius><length>{length:.6f}</length></cylinder></geometry>
<material><ambient>0.08 0.08 0.08 1</ambient><diffuse>0.12 0.12 0.12 1</diffuse></material></visual>"""
            elif kind == "pothole":
                # Runtime spawning cannot subtract the hall's box floor. Use a
                # round basin surrogate: 24 overlapping outer/inner ramps make
                # a 30 mm relative depression with about a 12.5 degree grade.
                # The 3 mm leading edge and broad 0.60 m diameter are traversable
                # by the 16 mm casters while still demanding wheel effort.
                segments = 24
                ramp_width = 0.45 * size
                rim_height = 0.10 * size
                thickness = 0.006
                slope = atan2(rim_height, ramp_width)
                rim_parts = []
                for part in range(segments):
                    angle = 2.0 * pi * part / segments
                    for side, radius, pitch in (
                        ("outer", size - 0.5 * ramp_width, slope),
                        ("inner", size - 1.5 * ramp_width, -slope),
                    ):
                        cx = x + radius * cos(angle)
                        cy = y + radius * sin(angle)
                        tangent_width = 2.10 * max(
                            size - ramp_width, radius
                        ) * sin(pi / segments)
                        pose = (
                            f"{cx:.6f} {cy:.6f} {0.5 * rim_height:.6f} "
                            f"0 {pitch:.6f} {angle:.6f}"
                        )
                        geometry = (
                            f"<geometry><box><size>{ramp_width:.6f} "
                            f"{tangent_width:.6f} {thickness:.6f}</size></box></geometry>"
                        )
                        rim_parts.append(f"""
<collision name='{prefix}_{side}_{part}_collision'><pose>{pose}</pose>{geometry}
<surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
<visual name='{prefix}_{side}_{part}_visual'><pose>{pose}</pose>{geometry}
<material><ambient>0.20 0.12 0.05 1</ambient><diffuse>0.28 0.16 0.06 1</diffuse></material></visual>""")
                body = f"""
<visual name='{prefix}_depression'><pose>{x:.6f} {y:.6f} 0.001 0 0 0</pose>
<geometry><cylinder><radius>{size:.6f}</radius><length>0.002</length></cylinder></geometry>
<material><ambient>0.015 0.015 0.018 1</ambient><diffuse>0.025 0.025 0.03 1</diffuse></material></visual>
{''.join(rim_parts)}"""
            else:
                raise ValueError(f"Unknown adaptive terrain kind: {kind}")
            links.append(f"<link name='{prefix}'>{body}</link>")
        return (
            "<?xml version='1.0'?><sdf version='1.9'><model name='"
            + name + "'><static>true</static>" + "".join(links) + "</model></sdf>"
        )

    def configure_adaptive_terrain(
        self,
        features: list[tuple[str, float, float, float]],
        timeout: float = 5.0,
    ) -> None:
        """Replace adaptive side clutter with one efficiently spawned model."""
        name = "adaptive_training_terrain"
        if not self.delete_entity.wait_for_service(timeout_sec=timeout):
            raise TimeoutError(f"Missing {self.delete_entity.srv_name}")

        def remove_if_present() -> None:
            def request():
                value = DeleteEntity.Request()
                value.entity.name = name
                value.entity.type = Entity.MODEL
                return value
            self._call_idempotent_service(
                self.delete_entity, request, timeout, f"delete {name}"
            )

        remove_if_present()
        if not features:
            return
        if not self.spawn_entity.wait_for_service(timeout_sec=timeout):
            raise TimeoutError(f"Missing {self.spawn_entity.srv_name}")

        def spawn_request():
            request = SpawnEntity.Request()
            request.entity_factory.name = name
            request.entity_factory.allow_renaming = False
            request.entity_factory.sdf = self._adaptive_terrain_sdf(name, features)
            return request

        response = self._call_idempotent_service(
            self.spawn_entity,
            spawn_request,
            timeout,
            f"spawn {name}",
        )
        if response is None or not response.success:
            remove_if_present()
            response = self._call_idempotent_service(
                self.spawn_entity,
                spawn_request,
                timeout,
                f"spawn {name} after removal",
            )
            if response is None or not response.success:
                raise RuntimeError("Gazebo failed to spawn adaptive training terrain")

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
