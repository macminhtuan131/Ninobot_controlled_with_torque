"""Behavioral tests for staged perturbations, torque envelope and frame transforms."""
import ast
from copy import deepcopy
from math import cos, sin
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from nino_rl.core import RobotState, TrackingState, quaternion_to_euler
from nino_rl.control_v2 import compute_reward, BASELINE_ACTION

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / 'config/ppo.yaml').read_text())


def method(filename, name, namespace):
    source = ROOT / 'nino_rl' / filename
    cls = next(x for x in ast.parse(source.read_text()).body if isinstance(x, ast.ClassDef))
    node = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[name]


def test_optional_randomization_decreases_from_hard_to_easy():
    sample = method('ros_env.py', '_sample_randomization', dict(np=np))
    env = SimpleNamespace(config=deepcopy(CONFIG), np_random=np.random.default_rng(42),
                          _curriculum_stage=lambda: (0, 0., 30.))
    env.config['domain_randomization']['enabled'] = True
    sample(env)
    assert .3 <= env._randomization['traction'] <= 1
    assert .01 <= env._randomization['delay'] <= .04
    assert env._randomization['position_noise'] > 0
    env._curriculum_stage = lambda: (5, 1., 30.)
    sample(env)
    assert env._randomization['traction'] == 1
    assert all(value == 0 for key, value in env._randomization.items() if key != 'traction')


def test_every_phase_has_one_cable_and_size_angle_get_easier():
    cables = method('ros_env.py', '_curriculum_cables', dict(np=np))
    env = SimpleNamespace(config=deepcopy(CONFIG), np_random=np.random.default_rng(42))
    phases = [cables(env, stage) for stage in range(6)]
    assert all(len(phase) == 1 for phase in phases)
    diameters = [2.0 * phase[0][1] for phase in phases]
    angles = [abs(np.degrees(phase[0][2])) for phase in phases]
    assert diameters == sorted(diameters, reverse=True)
    assert angles == sorted(angles, reverse=True)
    assert all(diameter > 0.0 for diameter in diameters)


def test_straight_command_cruises_slows_crawls_and_stops_at_one_cm():
    command = method('ros_env.py', '_straight_command', dict(np=np))
    env = SimpleNamespace(
        config={"goal_tolerance_m": 0.01},
        straight_speed=0.4,
        minimum_approach_speed=0.03,
        goal_slowdown_distance=1.0,
    )
    assert command(env, 2.0) == pytest.approx(0.4)
    assert command(env, 0.51) == pytest.approx(0.2)
    assert command(env, 0.02) == pytest.approx(0.03)
    assert command(env, 0.01) == 0.0


def test_adaptive_terrain_adds_mixed_features_outside_clear_path():
    features = method('ros_env.py', '_adaptive_terrain_features', dict(np=np))
    env = SimpleNamespace(
        adaptive_terrain_enabled=True,
        terrain_feature_count=20,
        config={"adaptive_terrain": {
            "zone_x_m": [2.8, 5.2],
            "path_clearance_m": 0.60,
            "max_lateral_center_m": 1.55,
        }},
    )
    generated = features(env)
    assert len(generated) == 20
    assert {feature[0] for feature in generated} == {"pothole", "obstacle", "cable"}
    for kind, x, y, size in generated:
        assert 2.8 <= x <= 5.2
        assert abs(y) - size >= 0.60
        assert abs(y) + size < 1.80


def test_operating_envelope_saturation_cost_and_dt():
    cfg = {**CONFIG['reward_v2'], 'torque_scale_nm': .5}
    tracking = TrackingState(0, 0, 0, 30, 30)
    def cost(torque, dt):
        state = RobotState(applied_left_torque=torque, applied_right_torque=-torque)
        return compute_reward(tracking, tracking, state, BASELINE_ACTION, BASELINE_ACTION,
                              [0, 0], dt, {'impact_integral': 0}, cfg)[1]['saturation']
    assert cost(4.5, .1) == 0
    assert cost(5.0, .1) == pytest.approx(-.03)
    assert cost(20, .1) == pytest.approx(-.03)
    assert cost(4.8, .1) == pytest.approx(2 * cost(4.8, .05))


def test_pose_and_path_use_consistent_transform_direction():
    pose = method('ros_interface.py', 'pose_in_frame',
                  dict(cos=cos, sin=sin, quaternion_to_euler=quaternion_to_euler,
                       TransformException=RuntimeError, Time=lambda: None))
    angle = np.pi / 2
    rotation = SimpleNamespace(x=0, y=0, z=sin(angle / 2), w=cos(angle / 2))
    calls = []
    transform = SimpleNamespace(transform=SimpleNamespace(rotation=rotation,
                                            translation=SimpleNamespace(x=1, y=2)))
    def lookup(target, source, stamp):
        calls.append((target, source))
        return transform
    ros = SimpleNamespace(tf_buffer=SimpleNamespace(lookup_transform=lookup))
    np.testing.assert_allclose(pose(ros, RobotState(x=1, y=0, yaw=.1), 'map'), [1, 3, angle + .1])
    assert calls == [('map', 'odom')]
    assert pose(ros, RobotState(x=1, y=2, yaw=.1), 'odom') == (1, 2, .1)
