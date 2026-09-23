"""Behavioral checks for reaching the goal while practicing wheel control."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from nino_rl.core import PathTracker, RobotState, TrackingState
from nino_rl.control_v2 import BASELINE_ACTION, ChallengeRegion, ChallengeTracker, compute_reward
from nino_rl.task_geometry import approach_speed, goal_overshot

CONFIG = yaml.safe_load((Path(__file__).parents[1] / 'config/ppo.yaml').read_text())


def test_final_approach_never_accelerates_after_missing_goal():
    assert approach_speed(.5, .5, 0., CONFIG) > .03
    for error in (.2, .5, 1., 3.):
        assert approach_speed(error, 0., 0., CONFIG) == .03
    assert approach_speed(.09, .08, 0., CONFIG) == 0.
    # Inside the position circle but not yet aligned: retain steering authority.
    assert approach_speed(.09, .08, .3, CONFIG) == .03


def test_overshoot_is_a_failure_boundary_not_a_false_success():
    path = PathTracker([(0., 0.), (6., 0.)])
    assert not goal_overshot(RobotState(x=6.2, y=.5), path, CONFIG)
    assert goal_overshot(RobotState(x=6.31, y=.5), path, CONFIG)
    assert not goal_overshot(RobotState(x=5.8, y=.5), path, CONFIG)


def test_nominal_approach_allows_initial_policy_to_finish_before_target():
    # Kinematic feasibility only: Gazebo validation covers dynamics/terrain.
    remaining = 6.0
    dt = 1. / CONFIG['control_hz']
    for step in range(round(CONFIG['target_finish_seconds'] / dt)):
        remaining -= CONFIG['ppo']['initial_speed_scale'] * approach_speed(
            remaining, remaining, 0., CONFIG) * dt
        if remaining <= CONFIG['goal_tolerance_m']:
            break
    assert remaining <= CONFIG['goal_tolerance_m']


def progress(before, after):
    # Path projection is clamped at 6 m in both cases; endpoint distance isn't.
    _, terms = compute_reward(TrackingState(6., 0., 0., 0., before),
        TrackingState(6., 0., 0., 0., after), RobotState(), BASELINE_ACTION,
        BASELINE_ACTION, [0., 0.], .1, {'impact_integral': 0.},
        {**CONFIG['reward_v2'], 'torque_scale_nm': 2.})
    return terms['progress']


def test_endpoint_reward_penalizes_leaving_goal_and_rewards_approaching():
    assert progress(.2, .3) < 0.
    assert progress(.3, .2) > 0.
    assert progress(.2, .3) + progress(.3, .2) == pytest.approx(0.)


def test_chassis_passing_over_tiny_bump_does_not_count_as_wheel_challenge():
    center_bump = ChallengeRegion('center', 'bump', 2., 0., .12)
    tracker = ChallengeTracker([center_bump], wheel_separation=.34273666)
    for x in np.linspace(1.5, 2.6, 50):
        tracker.update((x-.03, 0.), (x, 0.))
    assert not tracker.chosen and not tracker.cleared
    wheel_bump = ChallengeRegion('left', 'bump', 2., .34273666/2, .12)
    tracker = ChallengeTracker([wheel_bump], wheel_separation=.34273666)
    for x in np.linspace(1.5, 2.6, 50):
        tracker.update((x-.03, 0.), (x, 0.))
    assert tracker.chosen == tracker.cleared == {'left'}
