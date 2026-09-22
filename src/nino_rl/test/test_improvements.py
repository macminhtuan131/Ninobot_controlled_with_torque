"""Behavioral tests for staged perturbations, torque envelope and frame transforms."""
import ast
from collections import deque
from copy import deepcopy
from math import cos, sin
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from nino_rl.core import RobotState, TrackingState, quaternion_to_euler
from nino_rl.control_v2 import (
    BASELINE_ACTION, ChallengeRegion, ChallengeTracker, compute_reward,
)

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


def test_adaptive_terrain_randomizes_mixed_features_across_path():
    features = method('ros_env.py', '_adaptive_terrain_features', dict(np=np))
    env = SimpleNamespace(
        adaptive_terrain_enabled=True,
        terrain_feature_count=20,
        np_random=np.random.default_rng(42),
        config={"adaptive_terrain": {
            "zone_x_m": [1.2, 5.2],
            "max_center_offset_m": 0.10,
            "max_lateral_center_m": 1.55,
        }},
    )
    generated = features(env)
    env.np_random = np.random.default_rng(43)
    regenerated = features(env)
    assert len(generated) == 20
    assert {feature[0] for feature in generated} == {"pothole", "bump", "cable"}
    for kind, x, y, size in generated:
        assert 1.2 <= x <= 5.2
        assert abs(y) <= 0.10
        assert abs(y) + size < 1.80
    assert generated != regenerated


def test_challenge_tracker_includes_only_traversable_episode_geometry():
    build = method(
        'ros_env.py',
        '_make_challenge_tracker',
        dict(np=np, ChallengeRegion=ChallengeRegion, ChallengeTracker=ChallengeTracker),
    )
    env = SimpleNamespace(config={
        "challenge_tracking": {
            "robot_contact_margin_m": 0.20,
            "robot_clearance_margin_m": 0.30,
        },
        "adaptive_terrain": {"cable_length_m": 0.55},
    })
    tracker = build(
        env,
        [(4.0, 0.01, np.deg2rad(30.0))],
        [
            ("pothole", 2.0, 0.0, 0.30),
            ("bump", 3.0, 0.0, 0.12),
            ("cable", 5.0, 0.0, 0.015),
            ("obstacle", 3.5, 0.0, 0.12),
        ],
    )
    assert tracker.total == 4
    assert {region.kind for region in tracker.regions} == {"pothole", "bump", "cable"}


def test_adaptive_terrain_advances_only_after_rolling_success_gate():
    record = method('ros_env.py', '_record_adaptive_terrain_outcome', dict(np=np))
    env = SimpleNamespace(
        adaptive_terrain_progress=True,
        terrain_success_window=deque(maxlen=4),
        terrain_success_window_size=4,
        terrain_advance_success_rate=.75,
        terrain_episodes_at_level=0,
        terrain_feature_count=1,
        terrain_features_per_success=1,
        max_terrain_features=20,
    )
    assert record(env, True) == (1.0, 1, False)
    assert record(env, False) == (.5, 2, False)
    assert record(env, True) == pytest.approx((2 / 3, 3, False))
    assert record(env, True) == (.75, 4, True)
    assert env.terrain_feature_count == 2
    assert env.terrain_episodes_at_level == 0
    assert not env.terrain_success_window


def test_adaptive_terrain_checkpoint_state_round_trip():
    state = method('ros_env.py', 'adaptive_terrain_state', {})
    restore = method('ros_env.py', 'restore_adaptive_terrain_state', {})
    source = SimpleNamespace(
        terrain_feature_count=3,
        successful_episodes=41,
        terrain_episodes_at_level=12,
        terrain_success_window=deque([True, False, True], maxlen=50),
    )
    saved = state(source)
    target = SimpleNamespace(
        max_terrain_features=20,
        terrain_success_window_size=50,
        terrain_success_window=deque(maxlen=50),
    )
    restore(target, saved)
    assert state(target) == saved


def test_automatic_phase_schedule_boundaries_and_resume():
    stage = method('ros_env.py', '_curriculum_stage', dict(np=np))
    env = SimpleNamespace(config=deepcopy(CONFIG), goal_pose=(6, 0, 0),
                          global_steps=0, total_training_steps=600000)
    for steps, phase in [(0, 6), (99999, 6), (100000, 5), (200000, 4),
                         (300000, 3), (400000, 2), (499999, 2),
                         (500000, 1), (600000, 1), (900000, 1)]:
        env.global_steps = steps
        assert stage(env)[0] + 1 == phase
    env.global_steps = 250000
    env.total_training_steps = 350000  # Resume budget must not shift boundaries.
    assert stage(env)[0] + 1 == 4
    env.config['curriculum']['fixed_phase'] = 1
    assert stage(env)[0] + 1 == 1


def test_five_success_streak_failure_reset_cap_and_checkpoint():
    record = method('ros_env.py', '_record_adaptive_terrain_outcome', dict(np=np))
    save = method('ros_env.py', 'adaptive_terrain_state', {})
    restore = method('ros_env.py', 'restore_adaptive_terrain_state', {})
    env = SimpleNamespace(adaptive_terrain_progress=True,
        terrain_success_window=deque(maxlen=5), terrain_success_window_size=5,
        terrain_advance_success_rate=1.0, terrain_episodes_at_level=0,
        terrain_feature_count=7, terrain_features_per_success=1,
        max_terrain_features=8, successful_episodes=0)
    for outcome in [True]*4 + [False] + [True]*4:
        assert not record(env, outcome)[2]
    saved = save(env)
    env.terrain_success_window.clear()
    restore(env, saved)
    assert record(env, True)[2]
    assert env.terrain_feature_count == 8
    assert not env.terrain_success_window
    for _ in range(10):
        assert not record(env, True)[2]
    assert env.terrain_feature_count == 8


def test_default_field_capacity_randomization_and_spacing():
    generate = method('ros_env.py', '_adaptive_terrain_features', dict(np=np))
    env = SimpleNamespace(config=deepcopy(CONFIG), adaptive_terrain_enabled=True,
                          terrain_feature_count=8)
    layouts = []
    for seed in range(30):
        env.np_random = np.random.default_rng(seed)
        features = generate(env)
        assert len(features) == 8
        assert {f[0] for f in features} == {'bump', 'pothole', 'cable'}
        assert min(np.diff([f[1] for f in features])) >= .45
        layouts.append(tuple(features))
    assert len(set(layouts)) == 30


def test_downward_scan_produces_advance_terrain_preview():
    callback = method(
        'ros_interface.py', '_terrain_scan_callback',
        dict(
            LaserScan=object,
            isfinite=np.isfinite,
            cos=cos,
            sin=sin,
            monotonic=lambda: 10.0,
            TERRAIN_SENSOR_HEIGHT_M=.2325,
            TERRAIN_SENSOR_PITCH_RAD=.45,
            TERRAIN_HEIGHT_THRESHOLD_M=.006,
            TERRAIN_PREVIEW_RANGE_M=1.0,
        ),
    )
    flat_range = .2325 / sin(.45)
    raised_range = (.2325 - .03) / sin(.45)
    message = SimpleNamespace(
        ranges=[flat_range, raised_range, flat_range],
        range_min=.05,
        range_max=2.0,
        angle_min=-.1,
        angle_increment=.1,
    )
    class Lock:
        def __enter__(self): return self
        def __exit__(self, *args): return False
    received = []
    ros = SimpleNamespace(
        _lock=Lock(),
        _preview=None,
        _preview_received_at=0.0,
        _mark_received=received.append,
    )
    callback(ros, message)
    assert ros._preview[0] < .5
    assert ros._preview[1] == pytest.approx(.03)
    assert ros._preview[2] == 0.0
    assert received == ["terrain"]


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
