"""Analytic examples and failure cases for automatic trajectory scoring."""
import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from nino_rl.trajectory_metrics import (
    EpisodeTrajectory, path_metrics, timed_metrics, read_track, time_weights, write_csv,
)
from nino_rl.evaluation import benchmark_id, compare_summaries, summarize, METRICS
from nino_rl.timing import wait_for_coverage, wait_for_quiescent_timestamp


def test_constant_offset_and_endpoint_overshoot():
    result = path_metrics([0, 1, 2], [[0, .3], [1, .3], [2, .3]], [[0, 0], [2, 0]], [0, 0, 0])
    assert result['path_rmse_m'] == pytest.approx(.3)
    assert result['cross_track_rmse_m'] == pytest.approx(.3)
    assert result['heading_rmse_deg'] == 0
    assert result['final_progress_fraction'] == 1
    # Beyond the endpoint the perpendicular component is zero, but distance is not.
    result = path_metrics([0, 1], [[3, 0], [3, 0]], [[0, 0], [2, 0]])
    assert result['path_rmse_m'] == 1
    assert result['cross_track_rmse_m'] == 0


def test_stationary_robot_cannot_appear_complete():
    result = path_metrics([0, 1, 2], [[0, 0]] * 3, [[0, 0], [30, 0]])
    assert result['path_rmse_m'] == 0
    assert result['final_progress_fraction'] == 0
    assert result['endpoint_error_m'] == 30


def test_irregular_time_weighting_and_angle_wrap():
    times = [0, .01, 1]
    np.testing.assert_allclose(time_weights(times), [.005, .5, .495])
    result = path_metrics(times, [[0, 1], [0, 0], [0, 0]], [[1, 0], [-1, 0]], [-np.pi] * 3)
    assert result['path_rmse_m'] == pytest.approx(np.sqrt(.005))
    assert result['heading_rmse_deg'] == 0


def test_timed_rmse_exact_integration_and_no_frame_alignment():
    # Error grows linearly 0 -> 1. Integral(error^2) is 1/3, not trapezoid 1/2.
    result = timed_metrics([0, 1], [[0, 0], [2, 0]], [0, .5, 1], [[0, 0], [.5, 0], [1, 0]], max_gap_s=1)
    assert result['position_rmse_m'] == pytest.approx(np.sqrt(1 / 3))
    # Constant 3-4 offset is preserved, not aligned away by a rigid fit.
    result = timed_metrics([0, 1], [[3, 4], [4, 4]], [0, 1], [[0, 0], [1, 0]], max_gap_s=1)
    assert result['position_rmse_m'] == 5


def test_timed_overlap_and_heading_wrap():
    result = timed_metrics([1, 2], [[1, 0], [2, 0]], [0, 1, 2, 3], [[0, 0], [1, 0], [2, 0], [3, 0]],
                           [-np.pi + .01] * 2, [np.pi - .01] * 4, max_gap_s=1)
    assert result['reference_time_coverage'] == pytest.approx(1 / 3)
    assert result['position_rmse_m'] == 0
    assert result['timed_heading_rmse_deg'] == pytest.approx(np.degrees(.02))


@pytest.mark.parametrize('times', [[0], [0, 0], [1, 0], [0, np.nan]])
def test_reject_invalid_timestamps(times):
    with pytest.raises(ValueError):
        time_weights(times)


def test_reject_gaps_and_no_overlap():
    with pytest.raises(ValueError, match='gap'):
        timed_metrics([0, 2], [[0, 0], [2, 0]], [0, 2], [[0, 0], [2, 0]])
    with pytest.raises(ValueError, match='overlap'):
        timed_metrics([0, 1], [[0, 0], [1, 0]], [2, 3], [[2, 0], [3, 0]], max_gap_s=1)


def test_export_roundtrip_cli_and_frame_rejection(tmp_path):
    trace = EpisodeTrajectory([[0, 0], [1, 0]], 'map')
    trace.add(0, 0, .1, 0)
    trace.add(.1, 1, .1, 0)
    trace.save(tmp_path)
    assert (tmp_path / 'trajectory.png').stat().st_size > 0
    assert (tmp_path / 'trajectory.csv').stat().st_size > 0
    actual = read_track(tmp_path / 'actual.csv')
    assert actual['frame'] == 'map'
    output = tmp_path / 'result.json'
    args = [sys.executable, '-m', 'nino_rl.trajectory_metrics', '--actual', str(tmp_path / 'actual.csv'),
            '--reference', str(tmp_path / 'reference.csv'), '--output', str(output)]
    subprocess.run(args, check=True, capture_output=True)
    assert json.loads(output.read_text())['path_rmse_m'] == pytest.approx(.1)
    text = (tmp_path / 'reference.csv').read_text().replace('map', 'odom')
    (tmp_path / 'reference.csv').write_text(text)
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'frame_id mismatch' in result.stderr


def test_late_imu_arrives_before_timeout_but_real_gap_fails():
    clock = [0.]
    def pause(dt):
        clock[0] += dt
    def measure():
        return dict(duration=.1 if clock[0] > .02 else .02,
                    latest_stamp=.1 if clock[0] > .02 else .02)
    result = wait_for_coverage(measure, 0, .1, now=lambda: clock[0], pause=pause)
    assert result['duration'] == .1
    # A 50 Hz sensor can naturally be one sample behind an arbitrary endpoint.
    result = wait_for_coverage(
        lambda: dict(duration=.1, latest_stamp=.087), 0, .1,
        now=lambda: clock[0], pause=pause,
    )
    assert result['latest_stamp'] == .087
    # Gazebo lockstep may expose one held sample for the full 100 ms action.
    result = wait_for_coverage(
        lambda: dict(duration=.1, latest_stamp=.04), 0, .1,
        max_latest_lag=.1, now=lambda: clock[0], pause=pause,
    )
    assert result['latest_stamp'] == .04
    clock[0] = 0
    # Coverage alone is insufficient: don't call extrapolated old data fresh.
    with pytest.raises(RuntimeError, match='Insufficient'):
        wait_for_coverage(lambda: dict(duration=.1, latest_stamp=.01), 0, .1, timeout=.01,
                          now=lambda: clock[0], pause=pause)
    assert clock[0] < .02


def test_coverage_threshold_tolerates_only_floating_point_roundoff():
    result = wait_for_coverage(
        lambda: dict(duration=.08 - 1e-12, latest_stamp=.1),
        0, .1, min_coverage=.8,
    )
    assert result["duration"] < .08


def test_paused_episode_clock_waits_for_queued_imu_callbacks():
    clock = [0.0]
    samples = iter([2040.82, 2070.0, 2081.78])
    stamp = [2040.82]

    def pause(dt):
        clock[0] += dt
        try:
            stamp[0] = next(samples)
        except StopIteration:
            pass

    result = wait_for_quiescent_timestamp(
        lambda: stamp[0], timeout=.2, quiet_time=.01,
        now=lambda: clock[0], pause=pause,
    )
    assert result == 2081.78


def test_comparison_requires_matching_completed_experiments():
    config = {'path': [1, 2], 'reward_v2': {'a': 1}, 'evaluation_baseline': True}
    changed = deepcopy(config)
    changed['reward_v2']['a'] = 2
    changed['evaluation_baseline'] = False
    assert benchmark_id(config) == benchmark_id(changed)
    row = {m: 1 for m in METRICS}
    row.update(success=True, finished_within_target_time=True, termination='success')
    metadata = dict(schema_version=1, phase=1, seed=10000, randomized=False,
                    benchmark_id=benchmark_id(config), pose_source='odom', complete=True)
    baseline = summarize([row], metadata)
    candidate = deepcopy(baseline)
    assert compare_summaries(baseline, candidate)['metrics']['path_rmse_m']['delta'] == 0
    candidate['seed'] += 1
    with pytest.raises(ValueError, match='seed'):
        compare_summaries(baseline, candidate)
    candidate = deepcopy(baseline)
    candidate['complete'] = False
    with pytest.raises(ValueError, match='incomplete'):
        compare_summaries(baseline, candidate)


def test_trace_rejects_repeated_sensor_stamp_and_records_origin():
    trace = EpisodeTrajectory([[0, 0], [1, 0]], 'map', 123.4)
    trace.add(0, 0, 0, 0)
    with pytest.raises(ValueError, match='clock'):
        trace.add(0, .1, 0, 0)
    trace.add(.1, .1, 0, 0)
    assert trace.metrics()['clock_origin_sim_s'] == 123.4
