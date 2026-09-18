"""Reinforcement-learning tools for Nino's differential wheel torques."""

import os


# Keep this simulator isolated from unrelated ROS/Gazebo clocks discovered on
# the LAN. All nino_rl console entry points import this package before rclpy
# creates a DDS participant. Override only through the Nino-specific variable
# so generic shell ROS_DOMAIN_ID settings cannot accidentally merge graphs.
os.environ["ROS_DOMAIN_ID"] = os.environ.get("NINO_ROS_DOMAIN_ID", "77")
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"

__version__ = "0.1.0"
