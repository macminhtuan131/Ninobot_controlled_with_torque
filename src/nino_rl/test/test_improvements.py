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


def test_phase_one_is_nominal_phase_six_perturbs():
    sample = method('ros_env.py', '_sample_randomization', dict(np=np))
    env = SimpleNamespace(config=deepcopy(CONFIG), np_random=np.random.default_rng(42),
                          _curriculum_stage=lambda: (0, 0., 30.))
    sample(env)
    assert env._randomization['traction'] == 1
    assert all(value == 0 for key, value in env._randomization.items() if key != 'traction')
    env._curriculum_stage = lambda: (5, 1., 30.)
    sample(env)
    assert .3 <= env._randomization['traction'] <= 1
    assert .01 <= env._randomization['delay'] <= .04
    assert env._randomization['position_noise'] > 0


def test_operating_envelope_saturation_cost_and_dt():
    cfg = {**CONFIG['reward_v2'], 'torque_scale_nm': .5}
    tracking = TrackingState(0, 0, 0, 30, 30)
    def cost(torque, dt):
        state = RobotState(applied_left_torque=torque, applied_right_torque=-torque)
        return compute_reward(tracking, tracking, state, BASELINE_ACTION, BASELINE_ACTION,
                              [0, 0], dt, {'impact_integral': 0}, cfg)[1]['saturation']
    assert cost(2.25, .1) == 0
    assert cost(2.5, .1) == pytest.approx(-.03)
    assert cost(20, .1) == pytest.approx(-.03)
    assert cost(2.4, .1) == pytest.approx(2 * cost(2.4, .05))


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
