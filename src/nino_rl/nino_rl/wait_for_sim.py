"""Exit only after the simulator can supply the interfaces Nav2 requires."""

from __future__ import annotations

from time import monotonic

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu, JointState, LaserScan
from tf2_ros import Buffer, TransformListener


class SimulationReadiness(Node):
    """Gate Nav2 startup until controllers, sensors, odometry, and TF exist."""

    def __init__(self) -> None:
        super().__init__("nino_sim_readiness")
        self.received = {
            "odom": False,
            "joint_states": False,
            "imu": False,
            "scan": False,
        }
        self.create_subscription(
            Odometry, "/odom", lambda _: self._mark("odom"), qos_profile_sensor_data
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            lambda _: self._mark("joint_states"),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu, "/imu/data", lambda _: self._mark("imu"), qos_profile_sensor_data
        )
        self.create_subscription(
            LaserScan, "/scan", lambda _: self._mark("scan"), qos_profile_sensor_data
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _mark(self, name: str) -> None:
        self.received[name] = True

    def missing(self) -> list[str]:
        missing = [name for name, ready in self.received.items() if not ready]
        if not self.tf_buffer.can_transform("odom", "base_footprint", Time()):
            missing.append("TF odom->base_footprint")
        return missing


def main() -> None:
    rclpy.init(args=None)
    node = SimulationReadiness()
    next_report = monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            missing = node.missing()
            if not missing:
                node.get_logger().info(
                    "Simulator controllers, sensors, odometry, and TF are ready; starting Nav2"
                )
                return
            now = monotonic()
            if now >= next_report:
                node.get_logger().info("Waiting for " + ", ".join(missing))
                next_report = now + 5.0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
