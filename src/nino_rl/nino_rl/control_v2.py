"""Version 2 policy contract and reward; no ROS or learning dependencies.

Actor inputs never include Gazebo truth velocity or generated cable positions.
All time integrals use simulation timestamps. Old 54-input/2-action models are
intentionally incompatible with this contract.
"""
from collections import deque
from math import cos, sin, exp

import numpy as np

from nino_rl.core import make_observation as legacy_observation, wheel_slip_ratios

FRAME_SIZE = 60
ACTION_SIZE = 3
BASELINE_ACTION = np.array([1.0, 0.0, 0.0], dtype=np.float32)
STOP_ACTION = np.array([-1.0, 0.0, 0.0], dtype=np.float32)


def decode_action(action, max_torque=0.5):
    u = np.asarray(action, dtype=np.float64)
    if u.shape != (ACTION_SIZE,) or not np.all(np.isfinite(u)):
        raise ValueError("v2 needs three finite actions: speed, forward, yaw")
    u = np.clip(u, -1.0, 1.0)
    torque = max_torque * np.clip([u[1] - u[2], u[1] + u[2]], -1.0, 1.0)
    return float((u[0] + 1.0) / 2.0), torque


def vertical_acceleration(state, includes_gravity=True):
    # Third row of body->world rotation, REP-103 z-up. IMU axes must be
    # aligned with base_link (as in this robot's fixed imu joint).
    value = (-sin(state.pitch) * state.accel_x
             + cos(state.pitch) * sin(state.roll) * state.accel_y
             + cos(state.pitch) * cos(state.roll) * state.accel_z)
    return float(value - 9.80665 if includes_gravity else value)


def make_observation(state, path, lookahead, previous_action, nav_reference=None,
                     preview=None, includes_gravity=True):
    _, torque = decode_action(previous_action, 1.0)
    legacy, tracking = legacy_observation(state, path, lookahead, torque, nav_reference)
    # Drop indices 50,51: slip computed using simulator-only truth velocity.
    # Preview is [distance/5m, left height/0.1m, right height/0.1m, valid].
    terrain = np.zeros(4) if preview is None else np.asarray(preview)
    observation = np.concatenate((legacy[:50], legacy[52:], previous_action,
                                  [vertical_acceleration(state, includes_gravity) / 10.0],
                                  terrain)).astype(np.float32)
    if observation.shape != (FRAME_SIZE,) or not np.all(np.isfinite(observation)):
        raise ValueError("Invalid v2 sensor observation")
    return np.clip(observation, -5.0, 5.0), tracking


class ObservationHistory:
    def __init__(self, frames=5):
        if not isinstance(frames, int) or frames < 1:
            raise ValueError("history_frames must be a positive integer")
        self.frames = frames
        self.values = deque(maxlen=frames)

    @property
    def size(self):
        return FRAME_SIZE * self.frames

    def reset(self, observation):
        self.values.clear()
        self.values.extend(np.array(observation, copy=True) for _ in range(self.frames))
        return np.concatenate(self.values).astype(np.float32)

    def append(self, observation):
        if not self.values:
            return self.reset(observation)
        self.values.append(np.array(observation, copy=True))
        return np.concatenate(self.values).astype(np.float32)


class ImuWindow:
    """Zero-order hold integrals over *all* timestamped IMU samples in a step.

    Duplicate/non-monotonic samples are discarded; a backwards clock clears
    history. No sample is extrapolated across a gap >0.1s. Missing coverage is
    an infrastructure error, never interpreted as a smooth ride.
    """
    def __init__(self):
        self.samples = deque(maxlen=2000)

    def add(self, stamp, az):
        if not np.isfinite(stamp) or not np.isfinite(az):
            return
        if self.samples and stamp < self.samples[-1][0]:
            self.samples.clear()
        if not self.samples or stamp > self.samples[-1][0]:
            self.samples.append((float(stamp), float(az)))

    def measure(self, start, end, sigma=2.0):
        if end <= start or sigma <= 0:
            raise ValueError("Invalid IMU window")
        duration = square = fourth = peak = 0.0
        values = list(self.samples)
        for i, (stamp, az) in enumerate(values):
            next_stamp = values[i + 1][0] if i + 1 < len(values) else end
            lo, hi = max(start, stamp), min(end, next_stamp, stamp + 0.1)
            if hi <= lo:
                continue
            width = hi - lo
            duration += width
            square += az * az * width
            fourth += min((abs(az) / sigma) ** 4, 81.0) * width
            peak = max(peak, abs(az))
        return {"latest_stamp": values[-1][0] if values else -float("inf"),
                "duration": duration, "square_integral": square,
                "impact_integral": fourth, "peak": peak}

    def estimate(self, start, end, sigma=2.0, max_sample_age=0.2):
        """Return a full-window estimate without hiding stale/missing data.

        Gazebo can defer IMU publication while executing a large multi_step
        request. If part of the action is covered, preserve its measured mean
        energy and normalize it over the requested duration. With no overlap,
        only a recent held sample is used; otherwise the estimate is zero.
        The original coverage is reported for evaluation diagnostics.
        """
        result = self.measure(start, end, sigma)
        window = end - start
        coverage = result["duration"]
        result["coverage_fraction"] = coverage / window
        result["estimated"] = coverage + 1e-9 < window
        if coverage > 1e-9:
            scale = window / coverage
            result["square_integral"] *= scale
            result["impact_integral"] *= scale
            result["duration"] = window
            return result
        if self.samples and self.samples[-1][0] >= start - max_sample_age:
            az = self.samples[-1][1]
            result.update(
                duration=window,
                square_integral=az * az * window,
                impact_integral=min((abs(az) / sigma) ** 4, 81.0) * window,
                peak=abs(az),
            )
            return result
        result.update(
            duration=window,
            square_integral=0.0,
            impact_integral=0.0,
            peak=0.0,
        )
        return result


class StallWindow:
    def __init__(self, seconds=3.0, progress=0.05):
        self.seconds, self.progress = seconds, progress
        self.values = deque()
        self.total = 0.0

    def update(self, elapsed, delta_s, commanded_forward):
        if not commanded_forward:
            self.values.clear()
            self.total = 0.0
            return False
        self.total += delta_s
        self.values.append((elapsed, self.total))
        while len(self.values) > 1 and self.values[1][0] <= elapsed - self.seconds:
            self.values.popleft()
        return (elapsed - self.values[0][0] >= self.seconds - 1e-6
                and self.total - self.values[0][1] < self.progress)


def compute_reward(previous, current, state, action, previous_action, torque,
                   dt, imu, cfg, *, succeeded=False, failed=None,
                   timed_out=False, stalled=False, impact_scale=1.0,
                   reference=None, previous_state=None):
    """AMR reward with v2 I/O; see REWARD_POLICY_UPDATE.md for the objective.

    Tracking uses an unscaled pre-action baseline reference and simulation truth
    speeds. Never use the action-scaled reference: stopping would then remove
    its own tracking error. Total torque feedback is the controller's limited
    effort command, not a motor current measurement or energy measurement.
    """
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Reward requires positive simulation time")
    if not 0.0 <= cfg.get("saturation_fraction", 0.9) < 1.0:
        raise ValueError("saturation_fraction must be in [0, 1)")
    h = dt / 0.1
    cap = lambda x: min(float(x) ** 2, 9.0)
    delta = previous.distance_remaining - current.distance_remaining
    slip = wheel_slip_ratios(state)
    applied = np.asarray([state.applied_left_torque, state.applied_right_torque])
    torque_scale = float(cfg.get("applied_torque_scale_nm", cfg["torque_scale_nm"]))
    if not np.isfinite(torque_scale) or torque_scale <= 0:
        raise ValueError("applied_torque_scale_nm must be positive and finite")
    kernel_cost = lambda error, sigma: -np.expm1(-min((error / sigma) ** 2, 81.0))
    velocity_cost = yaw_cost = 0.0
    if reference is not None and reference.valid:
        v_ref = reference.desired_linear_velocity
        # Along the commanded direction, overspeed costs more than slowing for
        # a bump. A zero command penalizes motion in either direction equally.
        e_v = state.ground_linear_velocity - v_ref
        overspeed = abs(v_ref) < 1e-6 or e_v * v_ref > 0.0
        weight = cfg.get("overspeed_weight" if overspeed else "velocity_weight", 0.0)
        velocity_cost = -h * weight * kernel_cost(e_v, cfg.get("velocity_sigma_m_s", 0.20))
        yaw_cost = -h * cfg.get("yaw_tracking_weight", 0.0) * kernel_cost(
            state.ground_yaw_rate - reference.desired_angular_velocity,
            cfg.get("yaw_sigma_rad_s", 0.50))
    goal_gate = exp(-(current.endpoint_distance / cfg.get("goal_braking_distance_m", 0.8)) ** 2)
    torque_rate = 0.0
    if previous_state is not None:
        old_torque = np.asarray([previous_state.applied_left_torque,
                                 previous_state.applied_right_torque])
        torque_rate = -cfg.get("torque_rate_weight", 0.0) * float(np.mean(
            ((applied - old_torque) / torque_scale) ** 2)) / h
    terms = {
        # No asymmetric clipping across a closed forward/backward path: raw
        # signed differences telescope for a fixed path (undiscounted).
        "progress": cfg["progress_weight"] * float(delta),
        "lateral": -h * cfg["lateral_weight"] * cap(current.lateral_error / 0.30),
        "heading": -h * cfg["heading_weight"] * cap(current.heading_error / 0.35),
        "impact": -impact_scale * cfg["impact_weight"] * imu["impact_integral"] / 0.1,
        "body_rate": -h * cfg["body_rate_weight"] * (cap(state.gyro_x) + cap(state.gyro_y)),
        "attitude": -h * cfg["attitude_weight"] * (
            cap(max(0.0, abs(state.roll) - 0.20) / 0.15)
            + cap(max(0.0, abs(state.pitch) - 0.30) / 0.15)),
        "slip": -h * cfg["slip_weight"] * sum(cap(s / 0.30) for s in slip),
        "smoothness": -cfg["smoothness_weight"] * float(np.sum(
            (np.asarray(action) - np.asarray(previous_action)) ** 2)) / h,
        "saturation": -h * cfg.get("saturation_weight", 0.0) * float(np.mean(
            np.clip((np.abs(applied) / torque_scale - cfg.get("saturation_fraction", 0.9))
                    / (1.0 - cfg.get("saturation_fraction", 0.9)), 0.0, 1.0) ** 2)),
        "effort": -h * cfg["effort_weight"] * float(np.mean((applied / torque_scale) ** 2)),
        "residual_effort": -h * cfg.get("residual_effort_weight", 0.0) * float(np.mean(
            (np.asarray(torque) / cfg["torque_scale_nm"]) ** 2)),
        "torque_rate": torque_rate,
        "velocity": velocity_cost,
        "yaw_tracking": yaw_cost,
        "goal_braking": -h * cfg.get("goal_braking_weight", 0.0) * goal_gate * (
            cap(state.ground_linear_velocity / cfg.get("goal_speed_sigma_m_s", 0.20))
            + cap(state.ground_yaw_rate / cfg.get("goal_yaw_sigma_rad_s", 0.30))),
        "time": -h * cfg["time_penalty"],
        "stall": -h * cfg["stall_penalty"] if stalled else 0.0,
        "terminal": 0.0,
    }
    # Failure has precedence, including at a goal or time limit.
    if failed:
        terms["terminal"] = -cfg["off_path_penalty"] if failed == "off_path" else -cfg["failure_penalty"]
    elif succeeded:
        terms["terminal"] = cfg["success_bonus"]
    elif timed_out:
        terms["terminal"] = -cfg["timeout_penalty"]
    reward = float(sum(terms.values()))
    if not np.isfinite(reward):
        raise ValueError("Non-finite v2 reward")
    return reward, terms


def validate_model(model, history_size):
    if (tuple(model.observation_space.shape) != (history_size,)
            or tuple(model.action_space.shape) != (ACTION_SIZE,)):
        raise ValueError("Incompatible checkpoint: v2 requires stacked 60-value frames "
                         "and 3 actions. Train a NEW v2 model; do not resume a v1 checkpoint.")
