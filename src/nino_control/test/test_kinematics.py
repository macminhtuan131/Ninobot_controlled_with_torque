from math import isclose, pi
from types import SimpleNamespace

from nino_control.effort_drive import EffortDrive
from nino_control.kinematics import (
    clamp,
    integrate_wheel_odometry,
    limit_effort_commands,
    wheel_angular_targets,
)


def test_clamp() -> None:
    assert clamp(2.0, -1.0, 1.0) == 1.0
    assert clamp(-2.0, -1.0, 1.0) == -1.0
    assert clamp(0.5, -1.0, 1.0) == 0.5


def test_straight_wheel_targets_are_equal() -> None:
    left, right = wheel_angular_targets(0.5, 0.0, 0.1, 0.4)
    assert left == right == 5.0


def test_rotation_wheel_targets_are_opposite() -> None:
    left, right = wheel_angular_targets(0.0, 1.0, 0.1, 0.4)
    assert left == -2.0
    assert right == 2.0


def test_straight_odometry() -> None:
    x, y, yaw, linear, angular = integrate_wheel_odometry(
        0.0, 0.0, 0.0, 5.0, 5.0, 0.1, 0.4, 2.0
    )
    assert isclose(x, 1.0)
    assert isclose(y, 0.0)
    assert isclose(yaw, 0.0)
    assert isclose(linear, 0.5)
    assert isclose(angular, 0.0)


def test_yaw_is_normalized() -> None:
    _, _, yaw, _, _ = integrate_wheel_odometry(
        0.0, 0.0, pi - 0.01, -1.0, 1.0, 0.1, 0.4, 1.0
    )
    assert -pi <= yaw <= pi


def test_effort_is_slew_limited() -> None:
    output = limit_effort_commands(
        [12.0, -12.0], [0.0, 0.0], [0.0, 0.0], 0.001, 12.0, 20.0, 14.0
    )
    assert output == [0.02, -0.02]


def test_soft_speed_limit_blocks_acceleration_but_allows_braking() -> None:
    output = limit_effort_commands(
        [0.8, -0.8], [0.8, -0.8], [14.0, 14.0], 0.001, 12.0, 20.0, 14.0
    )
    assert output == [0.0, -0.8]


def test_negative_soft_speed_limit_is_symmetric() -> None:
    output = limit_effort_commands(
        [-0.8, 0.8], [-0.8, 0.8], [-14.0, -14.0], 0.001, 12.0, 20.0, 14.0
    )
    assert output == [0.0, 0.8]


def test_odometry_reset_publishes_authoritative_zero_sample() -> None:
    stamp = SimpleNamespace(nanoseconds=123)
    published = []
    drive = SimpleNamespace(
        x=2.0,
        y=-1.0,
        yaw=0.5,
        requested_linear=1.0,
        requested_angular=0.2,
        override_torque=[1.0, -1.0],
        v2_active=True,
        speed_scale=0.5,
        filtered_speed_scale=0.5,
        applied_effort=[1.0, -1.0],
        target_velocity=[2.0, 2.0],
        error_integral=[0.2, 0.3],
        last_cmd_ns=10,
        last_torque_ns=11,
        last_control_ns=12,
        last_odom_publish_ns=13,
        get_clock=lambda: SimpleNamespace(now=lambda: stamp),
        _publish_odometry=lambda now, linear, angular: published.append(
            (now, linear, angular)
        ),
    )
    response = SimpleNamespace(success=False, message="")

    assert EffortDrive._reset_odometry(drive, None, response) is response
    assert (drive.x, drive.y, drive.yaw) == (0.0, 0.0, 0.0)
    assert published == [(stamp, 0.0, 0.0)]
    assert drive.last_odom_publish_ns == stamp.nanoseconds
    assert response.success
