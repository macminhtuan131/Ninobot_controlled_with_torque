"""Numerical and actuator regression checks runnable without ROS/Gazebo."""
import ast
from copy import deepcopy
from math import degrees
from math import cos, sin, isfinite
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

from nino_rl.core import RobotState, PathTracker, TrackingState, NavReference, goal_reached, is_wrong_direction, metrics_dict, wheel_slip_ratios
from nino_rl.control_v2 import (
    BASELINE_ACTION, STOP_ACTION, FRAME_SIZE, ImuWindow, ObservationHistory,
    StallWindow, compute_reward, decode_action, make_observation,
    validate_model, vertical_acceleration,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG = yaml.safe_load((ROOT / "src/nino_rl/config/ppo.yaml").read_text())


class TestV2(unittest.TestCase):
    def test_gravity_compensation_when_tilted_and_stationary(self):
        for roll, pitch in [(0, 0), (0.2, 0.4), (-0.3, -0.5)]:
            state = RobotState(roll=roll, pitch=pitch,
                accel_x=-9.80665 * sin(pitch),
                accel_y=9.80665 * cos(pitch) * sin(roll),
                accel_z=9.80665 * cos(pitch) * cos(roll))
            self.assertAlmostEqual(vertical_acceleration(state), 0.0)
        self.assertEqual(vertical_acceleration(RobotState(accel_z=2), False), 2)

    def test_short_impact_between_control_ticks_is_retained(self):
        window = ImuWindow()
        for stamp, az in [(0, 0), (.02, 4), (.04, 0), (.06, 0), (.08, 0), (.1, 0)]:
            window.add(stamp, az)
        metrics = window.measure(0, .1)
        self.assertAlmostEqual(metrics["duration"], .1)
        self.assertAlmostEqual(metrics["impact_integral"], .32)
        self.assertEqual(metrics["peak"], 4)

    def test_imu_gap_reset_and_clipping(self):
        window = ImuWindow()
        window.add(0, 1e6)
        result = window.measure(0, 1)
        self.assertAlmostEqual(result["duration"], .1)
        self.assertAlmostEqual(result["impact_integral"], 8.1)
        self.assertEqual(result["peak"], 1e6)
        window.add(-1, 0)
        self.assertEqual(len(window.samples), 1)

    def test_action_mapping_baseline_brake_and_yaw(self):
        scale, torque = decode_action(BASELINE_ACTION)
        self.assertEqual(scale, 1)
        np.testing.assert_array_equal(torque, [0, 0])
        self.assertEqual(decode_action(STOP_ACTION)[0], 0)
        np.testing.assert_array_equal(decode_action([0, 0, 1])[1], [-.5, .5])
        for invalid in ([0, 0], [0, np.nan, 0]):
            with self.assertRaises(ValueError):
                decode_action(invalid)

    def test_actor_is_independent_of_ground_truth_slip(self):
        state = RobotState(accel_z=9.80665)
        path = PathTracker([(0, 0), (30, 0)])
        args = (path, CONFIG["path"]["lookahead_m"], BASELINE_ACTION)
        obs, _ = make_observation(state, *args)
        state.ground_linear_velocity = 50
        state.ground_yaw_rate = 10
        changed, _ = make_observation(state, *args)
        np.testing.assert_array_equal(obs, changed)
        self.assertEqual(obs.shape, (FRAME_SIZE,))
        self.assertEqual(obs[-1], 0)  # no fabricated terrain preview

    def test_history_reset_prevents_cross_episode_leak(self):
        history = ObservationHistory(5)
        self.assertEqual(history.reset(np.zeros(FRAME_SIZE)).shape, (300,))
        history.append(np.ones(FRAME_SIZE))
        np.testing.assert_array_equal(history.reset(np.zeros(FRAME_SIZE)), np.zeros(300))

    def test_checkpoint_rejection(self):
        model = SimpleNamespace(observation_space=SimpleNamespace(shape=(54,)),
                                action_space=SimpleNamespace(shape=(2,)))
        with self.assertRaisesRegex(ValueError, "NEW v2"):
            validate_model(model, 300)

    def test_stall_uses_window_and_clears_during_nav_stop(self):
        window = StallWindow()
        for i in range(30):
            self.assertFalse(window.update(i / 10, 0, True))
        self.assertTrue(window.update(3, 0, True))
        self.assertFalse(window.update(3.1, 0, False))
        for i in range(40):
            self.assertFalse(window.update(4 + i / 10, .01, True))

    def reward(self, delta=0, **kwargs):
        previous = TrackingState(0, 0, 0, 30, 30)
        current = TrackingState(delta, 0, 0, 30-delta, 30-delta)
        return compute_reward(previous, current, RobotState(),
            BASELINE_ACTION, BASELINE_ACTION, [0, 0], .1,
            {"impact_integral": 0},
            {**CONFIG["reward_v2"], "torque_scale_nm": .5}, **kwargs)

    def test_forward_reverse_reward_has_no_positive_loop(self):
        _, f = self.reward(.02)
        _, b = self.reward(-.02)
        self.assertAlmostEqual(f["progress"] + b["progress"], 0)
        self.assertGreater(f["progress"], 0)
        self.assertLess(b["progress"], 0)

    def test_terminal_precedence_and_single_penalty(self):
        self.assertEqual(self.reward(succeeded=True)[1]["terminal"], 100)
        self.assertEqual(self.reward(succeeded=True, failed="collision",
                                     timed_out=True)[1]["terminal"], -100)
        self.assertEqual(self.reward(failed="off_path")[1]["terminal"], -75)
        self.assertEqual(self.reward(timed_out=True)[1]["terminal"], -50)


class TestActuator(unittest.TestCase):
    """Execute actual production controller methods with a fake ROS clock/I/O."""
    def setUp(self):
        from nino_control.kinematics import clamp, wheel_angular_targets, limit_effort_commands
        source = ROOT / "src/nino_control/nino_control/effort_drive.py"
        cls = next(x for x in ast.parse(source.read_text()).body if isinstance(x, ast.ClassDef))
        methods = [x for x in cls.body if isinstance(x, ast.FunctionDef)
                   and x.name in ("_control_update", "_control_v2_callback", "_torque_callback")]
        namespace = dict(isfinite=isfinite, clamp=clamp,
                         wheel_angular_targets=wheel_angular_targets,
                         limit_effort_commands=limit_effort_commands,
                         Float64MultiArray=SimpleNamespace)
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
        self.methods = namespace
        self.now = 1_000_000_000
        self.commands = []
        self.drive = SimpleNamespace(
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=self.now)),
            get_logger=lambda: SimpleNamespace(error=lambda _: None),
            accept_torque=True, accept_cmd_vel=True, v2_active=False,
            speed_scale=1., filtered_speed_scale=1., last_control_ns=self.now-100_000_000,
            last_torque_ns=0, last_cmd_ns=self.now, torque_timeout=.25, command_timeout=.5,
            control_rate=10., have_wheel_state=True, requested_linear=.5, requested_angular=0.,
            error_integral=[0., 0.], target_velocity=[0., 0.], wheel_velocity=[0., 0.],
            wheel_radius=.0625, wheel_separation=.34273666, max_wheel_acceleration=100.,
            max_wheel_speed=12., kp=.3, ki=.1, integral_limit=4., max_velocity_torque=2.,
            max_torque=12., override_torque=[0., 0.], applied_effort=[0., 0.],
            max_effort_rate=100., torque_status_publish_rate=50., last_torque_status_publish_ns=self.now,
        )
        def publish(message):
            self.commands.append(message.data)
            # Avoid irrelevant odometry/TF in this actuator-only harness.
            self.drive.have_wheel_state = False
        self.drive.effort_publisher = SimpleNamespace(publish=publish)

    def command(self, values):
        self.methods["_control_v2_callback"](self.drive, SimpleNamespace(data=values))

    def update(self):
        self.methods["_control_update"](self.drive)

    def test_scaling_changes_pi_wheel_targets_not_only_observation(self):
        self.command([.5, 0, 0])
        self.drive.filtered_speed_scale = .5
        self.update()
        np.testing.assert_allclose(self.drive.target_velocity, [4, 4])
        self.assertGreater(self.commands[-1][0], 0)

    def test_expired_policy_does_not_revert_to_full_speed_nav2(self):
        self.command([1, .5, .5])
        self.drive.last_torque_ns = self.now - 300_000_000
        self.update()
        np.testing.assert_allclose(self.drive.target_velocity, [0, 0])
        np.testing.assert_allclose(self.commands[-1], [0, 0])

    def test_stop_blocks_residual_and_clears_integral(self):
        self.command([0, .5, .5])
        self.drive.error_integral = [2., 2.]
        self.update()
        np.testing.assert_allclose(self.commands[-1], [0, 0])

    def test_legacy_command_cannot_override_v2_ownership(self):
        self.command([0, 0, 0])
        self.methods["_torque_callback"](self.drive, SimpleNamespace(data=[5., 5.]))
        self.assertEqual(self.drive.override_torque, [0., 0.])

    def test_nan_command_stops_reference(self):
        self.command([np.nan, 0, 0])
        self.assertEqual(self.drive.speed_scale, 0)
        self.update()
        np.testing.assert_allclose(self.commands[-1], [0, 0])


class TestEnvironmentContract(unittest.TestCase):
    def run_step(self, collision=False, timed_out=False, torque_fresh=True):
        source = ROOT / "src/nino_rl/nino_rl/ros_env.py"
        cls = next(x for x in ast.parse(source.read_text()).body if isinstance(x, ast.ClassDef))
        step = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "step")
        clock = [1.0]
        commands = []
        def sleep(dt):
            clock[0] += dt
        namespace = dict(np=np, sleep=sleep, monotonic=lambda: clock[0],
            degrees=degrees, deepcopy=deepcopy, decode_action=decode_action,
            make_observation=make_observation, compute_reward=compute_reward,
            goal_reached=goal_reached, is_wrong_direction=is_wrong_direction,
            metrics_dict=metrics_dict, wheel_slip_ratios=wheel_slip_ratios)
        exec(compile(ast.Module(body=[step], type_ignores=[]), str(source), "exec"), namespace)
        state = RobotState(x=.02, accel_z=9.80665,
                           lidar_ranges=[.05 if collision else 3.0])
        path = PathTracker([(0, 0), (30, 0)])
        history = ObservationHistory(5)
        history.reset(np.zeros(FRAME_SIZE))
        reference = NavReference(desired_linear_velocity=.3, valid=True)
        ros = SimpleNamespace(
            publish_control=lambda *args: commands.append(args),
            snapshot=lambda: state, ground_truth_ready=lambda: True,
            applied_torque_ready=lambda: torque_fresh,
            nav_path_in_odom=lambda _: None,
            measure_impact=lambda start, end, sigma: dict(duration=end-start,
                square_integral=0., impact_integral=0., peak=0.),
            get_logger=lambda: SimpleNamespace(warn=lambda _: None))
        env = SimpleNamespace(config=deepcopy(CONFIG), action_scale=.5, control_dt=.1,
            _randomization={"delay": 0., "traction": 1., "torque_noise": 0.},
            _sim_seconds=lambda: clock[0], ros=ros, episode_steps=0, global_steps=0,
            episode_started_sim=1., episode_start_time_unix=0.,
            np_random=np.random.default_rng(42), _episode_nav_path=None,
            path=path, lookahead=CONFIG["path"]["lookahead_m"],
            previous_tracking=TrackingState(0, 0, 0, 30, 30),
            previous_robot_state=RobotState(accel_z=9.80665),
            previous_action=BASELINE_ACTION.copy(), action_before_previous=BASELINE_ACTION.copy(),
            waypoint_targets=np.array([]), next_waypoint_index=0, waypoint_arrival_times=[],
            waypoint_slack=3., waypoint_seconds_per_m=2.5,
            _reference=lambda *args: reference, _noisy_state=lambda truth: truth,
            _actor_observation=lambda truth, action, ref: make_observation(
                truth, path, CONFIG["path"]["lookahead_m"], action, ref)[0],
            history=history, _curriculum_stage=lambda: (0, 0., 30), stall_window=StallWindow(),
            off_path_steps=0, off_path_seconds=0., nav_invalid_steps=0, nav_invalid_seconds=0.,
            wrong_direction_steps=0, wrong_direction_seconds=0., attempt_number=1)
        for key in ("vertical_square_integral", "imu_coverage_seconds", "peak_vertical_acceleration",
                    "episode_return", "abs_lateral_sum", "lateral_square_sum", "abs_roll_sum",
                    "abs_pitch_sum", "imu_angular_xy_sum", "imu_acceleration_change_sum",
                    "max_tilt_deg", "max_path_deviation", "slip_square_sum", "max_abs_slip",
                    "torque_square_sum", "max_abs_torque", "accel_square_sum"):
            setattr(env, key, 0.)
        if timed_out:
            env.config["max_episode_seconds"] = .05
        return namespace["step"](env, BASELINE_ACTION.copy()), commands

    def test_step_emits_300_values_and_atomic_3_value_command(self):
        (obs, reward, terminated, truncated, info), commands = self.run_step()
        self.assertEqual(obs.shape, (300,))
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(terminated or truncated)
        self.assertEqual(commands[0], (1., 0., 0.))
        self.assertIn("impact", info["reward_terms"])

    def test_collision_terminal_stops_pi_and_logs_impact_metrics(self):
        (_, _, terminated, _, info), commands = self.run_step(collision=True)
        self.assertTrue(terminated)
        self.assertEqual(info["reward_terms"]["terminal"], -100)
        self.assertEqual(info["episode_metrics"]["termination"], "collision")
        self.assertEqual(commands[-1], (0., 0., 0.))
        self.assertIn("rms_vertical_acceleration_m_s2", info["episode_metrics"])

    def test_mission_deadline_does_not_bootstrap(self):
        (_, _, terminated, truncated, info), commands = self.run_step(timed_out=True)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["reward_terms"]["terminal"], -50)
        self.assertEqual(info["episode_metrics"]["termination"], "timeout")
        self.assertEqual(commands[-1], (0., 0., 0.))

    def test_missing_effort_feedback_aborts(self):
        with self.assertRaisesRegex(RuntimeError, "torque feedback stale"):
            self.run_step(torque_fresh=False)


if __name__ == "__main__":
    unittest.main()
