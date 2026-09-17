"""Closed-loop /cmd_vel to wheel-torque adapter for JointGroupEffortController.

Control architecture:
    direct straight-line /cmd_vel
        -> PI wheel-speed controller
        -> baseline wheel torque
        + RL residual torque from /wheel_torque_commands
        -> /wheel_effort_controller/commands

The RL command is a residual correction. It does not override the baseline.
"""

from math import cos, isfinite, sin

import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from nino_control.kinematics import (
    clamp,
    integrate_wheel_odometry,
    limit_effort_commands,
    wheel_angular_targets,
)


class EffortDrive(Node):
    """Track differential wheel speeds with bounded baseline + residual torque."""

    def __init__(self) -> None:
        super().__init__("effort_drive")

        # ---------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------
        self.declare_parameter("left_wheel_joint", "left_wheel_joint")
        self.declare_parameter("right_wheel_joint", "right_wheel_joint")
        self.declare_parameter("wheel_radius", 0.0625)
        self.declare_parameter("wheel_separation", 0.34273666)

        # The URDF hard limit is 24 rad/s. Keep a margin so direct torque
        # cannot drive controller_manager into that hard limit.
        self.declare_parameter("max_wheel_speed", 12.0)
        self.declare_parameter("max_wheel_acceleration", 12.0)
        self.declare_parameter("max_wheel_torque", 12.0)

        # Maximum torque produced by the baseline velocity PI controller.
        self.declare_parameter("max_velocity_control_torque", 2.0)

        self.declare_parameter("max_effort_rate", 10.0)
        self.declare_parameter("velocity_kp", 0.30)
        self.declare_parameter("velocity_ki", 0.10)
        self.declare_parameter("integral_limit", 4.0)

        self.declare_parameter("command_timeout", 0.5)
        self.declare_parameter("torque_command_timeout", 0.25)

        self.declare_parameter("control_rate", 500.0)
        self.declare_parameter("odom_publish_rate", 50.0)
        self.declare_parameter("torque_status_publish_rate", 50.0)

        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_footprint")
        self.declare_parameter("publish_odom_tf", True)

        # Training mode should set both to True:
        #   accept_cmd_vel=True  -> /cmd_vel supplies the baseline motion
        #   accept_torque=True   -> RL supplies residual torque
        self.declare_parameter("accept_cmd_vel", True)
        self.declare_parameter("accept_torque", True)

        # ---------------------------------------------------------
        # Read parameters
        # ---------------------------------------------------------
        self.left_joint = str(self.get_parameter("left_wheel_joint").value)
        self.right_joint = str(self.get_parameter("right_wheel_joint").value)

        self.wheel_radius = float(self.get_parameter("wheel_radius").value)
        self.wheel_separation = float(self.get_parameter("wheel_separation").value)

        self.max_wheel_speed = float(self.get_parameter("max_wheel_speed").value)
        self.max_wheel_acceleration = float(
            self.get_parameter("max_wheel_acceleration").value
        )

        requested_max_torque = float(self.get_parameter("max_wheel_torque").value)
        if not isfinite(requested_max_torque) or requested_max_torque <= 0.0:
            raise ValueError("max_wheel_torque must be finite and positive")
        self.max_torque = min(requested_max_torque, 12.0)

        requested_velocity_torque = float(
            self.get_parameter("max_velocity_control_torque").value
        )
        if (
            not isfinite(requested_velocity_torque)
            or requested_velocity_torque <= 0.0
        ):
            raise ValueError(
                "max_velocity_control_torque must be finite and positive"
            )
        self.max_velocity_torque = min(
            requested_velocity_torque, self.max_torque
        )

        self.max_effort_rate = float(self.get_parameter("max_effort_rate").value)
        self.kp = float(self.get_parameter("velocity_kp").value)
        self.ki = float(self.get_parameter("velocity_ki").value)
        self.integral_limit = float(self.get_parameter("integral_limit").value)

        self.command_timeout = float(self.get_parameter("command_timeout").value)
        self.torque_timeout = float(
            self.get_parameter("torque_command_timeout").value
        )

        self.control_rate = float(self.get_parameter("control_rate").value)
        self.odom_publish_rate = float(
            self.get_parameter("odom_publish_rate").value
        )
        self.torque_status_publish_rate = float(
            self.get_parameter("torque_status_publish_rate").value
        )

        self.odom_frame = str(self.get_parameter("odom_frame_id").value)
        self.base_frame = str(self.get_parameter("base_frame_id").value)
        self.publish_odom_tf = bool(
            self.get_parameter("publish_odom_tf").value
        )

        self.accept_cmd_vel = bool(
            self.get_parameter("accept_cmd_vel").value
        )
        self.accept_torque = bool(
            self.get_parameter("accept_torque").value
        )

        # ---------------------------------------------------------
        # Validate parameters
        # ---------------------------------------------------------
        positive_parameters = {
            "wheel_radius": self.wheel_radius,
            "wheel_separation": self.wheel_separation,
            "max_wheel_speed": self.max_wheel_speed,
            "max_wheel_acceleration": self.max_wheel_acceleration,
            "max_wheel_torque": self.max_torque,
            "max_velocity_control_torque": self.max_velocity_torque,
            "max_effort_rate": self.max_effort_rate,
            "control_rate": self.control_rate,
            "odom_publish_rate": self.odom_publish_rate,
            "torque_status_publish_rate": self.torque_status_publish_rate,
        }

        for name, value in positive_parameters.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive, got {value}"
                )

        for name, value in {
            "velocity_kp": self.kp,
            "velocity_ki": self.ki,
        }.items():
            if not isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative, got {value}"
                )

        if requested_max_torque > 12.0:
            self.get_logger().info(
                "max_wheel_torque is capped at the URDF limit of 12.0 N.m"
            )

        # ---------------------------------------------------------
        # Publishers
        # ---------------------------------------------------------
        self.effort_publisher = self.create_publisher(
            Float64MultiArray,
            "/wheel_effort_controller/commands",
            10,
        )

        self.applied_torque_publisher = self.create_publisher(
            Float64MultiArray,
            "/wheel_torque_applied",
            10,
        )

        self.odom_publisher = self.create_publisher(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            10,
        )

        self.tf_broadcaster = (
            TransformBroadcaster(self) if self.publish_odom_tf else None
        )

        # ---------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------
        self.create_subscription(
            Twist,
            "/cmd_vel",
            self._cmd_vel_callback,
            10,
        )

        self.create_subscription(
            Float64MultiArray,
            "/wheel_torque_commands",
            self._torque_callback,
            10,
        )
        self.create_subscription(
            Float64MultiArray, "/nino_rl/control_command", self._control_v2_callback, 10
        )

        wheel_state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            wheel_state_qos,
        )

        # ---------------------------------------------------------
        # Services
        # ---------------------------------------------------------
        self.create_service(
            Trigger,
            "/reset_wheel_odometry",
            self._reset_odometry,
        )

        # ---------------------------------------------------------
        # Controller state
        # ---------------------------------------------------------
        now_ns = self.get_clock().now().nanoseconds

        self.last_control_ns = now_ns
        self.last_cmd_ns = 0
        self.last_torque_ns = 0
        self.last_odom_publish_ns = 0
        self.last_torque_status_publish_ns = 0

        self.requested_linear = 0.0
        self.requested_angular = 0.0

        # RL residual torque command.
        self.override_torque = [0.0, 0.0]
        self.v2_active = False
        self.speed_scale = 1.0
        self.filtered_speed_scale = 1.0

        # Final torque currently sent to the wheel effort controller.
        self.applied_effort = [0.0, 0.0]

        self.wheel_velocity = [0.0, 0.0]
        self.have_wheel_state = False

        # PI wheel-speed controller state.
        self.target_velocity = [0.0, 0.0]
        self.error_integral = [0.0, 0.0]

        # Locally integrated wheel odometry.
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        # ---------------------------------------------------------
        # Main control timer
        # ---------------------------------------------------------
        self.timer = self.create_timer(
            1.0 / self.control_rate,
            self._control_update,
        )

        self.get_logger().info(
            "Effort drive ready: Nav2 /cmd_vel baseline + "
            "RL /wheel_torque_commands residual -> "
            "/wheel_effort_controller/commands"
        )

    def _reset_odometry(self, _request, response):
        """Reset wheel odometry and controller state after an episode reset."""
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.requested_linear = 0.0
        self.requested_angular = 0.0

        self.override_torque = [0.0, 0.0]
        self.v2_active = False
        self.speed_scale = 1.0
        self.filtered_speed_scale = 1.0
        self.applied_effort = [0.0, 0.0]

        self.target_velocity = [0.0, 0.0]
        self.error_integral = [0.0, 0.0]

        self.last_cmd_ns = 0
        self.last_torque_ns = 0
        self.last_control_ns = self.get_clock().now().nanoseconds

        response.success = True
        response.message = (
            "Wheel odometry and controller state reset to zero"
        )
        return response

    def _cmd_vel_callback(self, message: Twist) -> None:
        """Receive the velocity command used by the baseline controller."""
        if not self.accept_cmd_vel:
            return

        if (
            not isfinite(message.linear.x)
            or not isfinite(message.angular.z)
        ):
            self.get_logger().error(
                "Ignoring non-finite /cmd_vel command"
            )
            return

        self.requested_linear = float(message.linear.x)
        self.requested_angular = float(message.angular.z)
        self.last_cmd_ns = self.get_clock().now().nanoseconds

    def _torque_callback(
        self,
        message: Float64MultiArray,
    ) -> None:
        """Receive RL residual torque [left_Nm, right_Nm].

        IMPORTANT:
        This callback does not clear the baseline velocity command.
        The torque is added to the velocity PI baseline in _control_update().
        """
        if not self.accept_torque or self.v2_active:
            return

        if len(message.data) != 2:
            self.get_logger().error(
                "/wheel_torque_commands requires [left_Nm, right_Nm]"
            )
            return

        if not all(isfinite(value) for value in message.data):
            self.get_logger().error(
                "Ignoring non-finite wheel torque command"
            )
            return

        self.override_torque = [
            clamp(
                float(message.data[0]),
                -self.max_torque,
                self.max_torque,
            ),
            clamp(
                float(message.data[1]),
                -self.max_torque,
                self.max_torque,
            ),
        ]

        self.last_torque_ns = self.get_clock().now().nanoseconds

        # Do NOT reset last_cmd_ns here.
        # RL is residual; the velocity PI must keep controlling baseline motion.

    def _control_v2_callback(self, message):
        if not self.accept_torque:
            return
        if len(message.data) != 3 or not all(isfinite(x) for x in message.data):
            self.get_logger().error("Invalid v2 command; stopping scaled reference")
            self.v2_active = True
            self.speed_scale = 0.0
            self.override_torque = [0.0, 0.0]
            self.last_torque_ns = 0
            return
        self.v2_active = True
        self.speed_scale = clamp(float(message.data[0]), 0.0, 1.0)
        self.override_torque = [clamp(float(v), -self.max_torque, self.max_torque)
                                for v in message.data[1:]]
        self.last_torque_ns = self.get_clock().now().nanoseconds

    def _joint_state_callback(
        self,
        message: JointState,
    ) -> None:
        """Read left/right wheel angular velocity."""
        velocity_by_name = dict(
            zip(message.name, message.velocity)
        )

        if (
            self.left_joint not in velocity_by_name
            or self.right_joint not in velocity_by_name
        ):
            return

        self.wheel_velocity[0] = float(
            velocity_by_name[self.left_joint]
        )
        self.wheel_velocity[1] = float(
            velocity_by_name[self.right_joint]
        )
        self.have_wheel_state = True

    def _control_update(self) -> None:
        """Run baseline velocity PI control and add fresh RL residual torque."""
        now = self.get_clock().now()
        now_ns = now.nanoseconds

        dt = (now_ns - self.last_control_ns) * 1.0e-9
        self.last_control_ns = now_ns

        if dt <= 0.0 or dt > 0.25:
            dt = 1.0 / self.control_rate

        # ---------------------------------------------------------
        # Is the RL residual torque command fresh?
        # ---------------------------------------------------------
        residual_torque_active = (
            self.accept_torque
            and self.last_torque_ns > 0
            and (now_ns - self.last_torque_ns) * 1.0e-9
            <= self.torque_timeout
        )

        # ---------------------------------------------------------
        # Without wheel feedback, do not drive the robot.
        # ---------------------------------------------------------
        if not self.have_wheel_state:
            efforts = [0.0, 0.0]

        else:
            # -----------------------------------------------------
            # 1. Nav2 baseline controller
            # -----------------------------------------------------
            command_is_fresh = (
                self.accept_cmd_vel
                and self.last_cmd_ns > 0
                and (now_ns - self.last_cmd_ns) * 1.0e-9
                <= self.command_timeout
            )

            linear = (
                self.requested_linear
                if command_is_fresh
                else 0.0
            )
            angular = (
                self.requested_angular
                if command_is_fresh
                else 0.0
            )

            if self.v2_active:
                # A lost policy heartbeat stops the reference, never resumes
                # full-speed baseline. Reset service releases v2 ownership.
                scale = self.speed_scale if residual_torque_active else 0.0
                # Preserve in-place Nav2 turns except explicit stop/watchdog.
                if abs(linear) < 1e-6 and abs(angular) > 1e-6 and scale > 0.0:
                    scale = max(scale, 0.25)
                if scale == 0.0:
                    self.filtered_speed_scale = 0.0
                else:
                    self.filtered_speed_scale += clamp(
                        scale - self.filtered_speed_scale, -2.0 * dt, 2.0 * dt)
                linear *= self.filtered_speed_scale
                angular *= self.filtered_speed_scale
                if self.filtered_speed_scale < 1.0:
                    # Bleed accumulated PI torque while the policy slows down.
                    self.error_integral = [i * max(0.0, 1.0 - 5.0 * dt)
                                           for i in self.error_integral]

            if (
                abs(linear) < 1.0e-9
                and abs(angular) < 1.0e-9
            ):
                # Do not let stored integral torque push the robot
                # after Nav2 stops or the command becomes stale.
                self.error_integral = [0.0, 0.0]

            requested_targets = wheel_angular_targets(
                linear,
                angular,
                self.wheel_radius,
                self.wheel_separation,
            )

            max_step = self.max_wheel_acceleration * dt
            base_efforts = []

            for index, requested in enumerate(
                requested_targets
            ):
                requested = clamp(
                    requested,
                    -self.max_wheel_speed,
                    self.max_wheel_speed,
                )

                delta = clamp(
                    requested - self.target_velocity[index],
                    -max_step,
                    max_step,
                )

                self.target_velocity[index] += delta

                error = (
                    self.target_velocity[index]
                    - self.wheel_velocity[index]
                )

                self.error_integral[index] = clamp(
                    self.error_integral[index] + error * dt,
                    -self.integral_limit,
                    self.integral_limit,
                )

                effort = (
                    self.kp * error
                    + self.ki * self.error_integral[index]
                )

                base_efforts.append(
                    clamp(
                        effort,
                        -self.max_velocity_torque,
                        self.max_velocity_torque,
                    )
                )

            # -----------------------------------------------------
            # 2. RL residual torque
            #
            # tau_motor = tau_nav2_baseline + delta_tau_RL
            # -----------------------------------------------------
            nav_is_commanding_motion = (
                command_is_fresh
                and (
                    abs(linear) > 1.0e-6
                    or abs(angular) > 1.0e-6
                )
            )

            if (
                residual_torque_active
                and nav_is_commanding_motion
            ):
                efforts = [
                    base_efforts[0]
                    + self.override_torque[0],
                    base_efforts[1]
                    + self.override_torque[1],
                ]
            else:
                efforts = base_efforts

        # ---------------------------------------------------------
        # Final torque/rate/speed safety limits
        # ---------------------------------------------------------
        efforts = limit_effort_commands(
            efforts,
            self.applied_effort,
            self.wheel_velocity,
            dt,
            self.max_torque,
            self.max_effort_rate,
            self.max_wheel_speed,
        )

        self.applied_effort = efforts

        # ---------------------------------------------------------
        # Publish wheel torque
        # ---------------------------------------------------------
        command = Float64MultiArray()
        command.data = efforts
        self.effort_publisher.publish(command)

        torque_status_period_ns = int(
            1.0e9 / self.torque_status_publish_rate
        )

        if (
            now_ns - self.last_torque_status_publish_ns
            >= torque_status_period_ns
        ):
            self.applied_torque_publisher.publish(command)
            self.last_torque_status_publish_ns = now_ns

        # ---------------------------------------------------------
        # Wheel odometry
        # ---------------------------------------------------------
        if self.have_wheel_state:
            (
                self.x,
                self.y,
                self.yaw,
                odom_linear,
                odom_angular,
            ) = integrate_wheel_odometry(
                self.x,
                self.y,
                self.yaw,
                self.wheel_velocity[0],
                self.wheel_velocity[1],
                self.wheel_radius,
                self.wheel_separation,
                dt,
            )

            odom_period_ns = int(
                1.0e9 / self.odom_publish_rate
            )

            if (
                now_ns - self.last_odom_publish_ns
                >= odom_period_ns
            ):
                self._publish_odometry(
                    now,
                    odom_linear,
                    odom_angular,
                )
                self.last_odom_publish_ns = now_ns

    def _publish_odometry(
        self,
        stamp,
        linear: float,
        angular: float,
    ) -> None:
        """Publish wheel odometry and optional odom -> base TF."""
        half_yaw = 0.5 * self.yaw
        orientation_z = sin(half_yaw)
        orientation_w = cos(half_yaw)

        message = Odometry()
        message.header.stamp = stamp.to_msg()
        message.header.frame_id = self.odom_frame
        message.child_frame_id = self.base_frame

        message.pose.pose.position.x = self.x
        message.pose.pose.position.y = self.y
        message.pose.pose.orientation.z = orientation_z
        message.pose.pose.orientation.w = orientation_w

        message.twist.twist.linear.x = linear
        message.twist.twist.angular.z = angular

        message.pose.covariance[0] = 0.02
        message.pose.covariance[7] = 0.02
        message.pose.covariance[14] = 1.0e6
        message.pose.covariance[21] = 1.0e6
        message.pose.covariance[28] = 1.0e6
        message.pose.covariance[35] = 0.05

        message.twist.covariance[0] = 0.02
        message.twist.covariance[7] = 0.02
        message.twist.covariance[14] = 1.0e6
        message.twist.covariance[21] = 1.0e6
        message.twist.covariance[28] = 1.0e6
        message.twist.covariance[35] = 0.05

        self.odom_publisher.publish(message)

        if self.tf_broadcaster is not None:
            transform = TransformStamped()
            transform.header = message.header
            transform.child_frame_id = self.base_frame

            transform.transform.translation.x = self.x
            transform.transform.translation.y = self.y
            transform.transform.rotation.z = orientation_z
            transform.transform.rotation.w = orientation_w

            self.tf_broadcaster.sendTransform(transform)

    def stop(self) -> None:
        """Send zero torque during a clean shutdown."""
        if not self.context.ok():
            return

        message = Float64MultiArray()
        message.data = [0.0, 0.0]

        try:
            self.effort_publisher.publish(message)
        except RuntimeError:
            # SIGINT may invalidate the context between ok() and publish().
            pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EffortDrive()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        # A subscription can be invalidated between SIGINT and executor exit.
        # Preserve real runtime failures while treating that shutdown race as clean.
        if rclpy.ok():
            raise
    finally:
        if rclpy.ok():
            node.stop()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
