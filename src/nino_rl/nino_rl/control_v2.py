"""Version 2 policy contract and reward; no ROS or learning dependencies.

Actor inputs never include Gazebo truth velocity or generated cable positions.
All time integrals use simulation timestamps. Old 54-input/2-action models are
intentionally incompatible with this contract.
"""
from collections import deque
from dataclasses import dataclass
from math import cos, sin, exp

import numpy as np

from nino_rl.core import make_observation as legacy_observation, wheel_slip_ratios

FRAME_SIZE = 60
ACTION_SIZE = 3
BASELINE_ACTION = np.array([1.0, 0.0, 0.0], dtype=np.float32)
STOP_ACTION = np.array([-1.0, 0.0, 0.0], dtype=np.float32)


@dataclass(frozen=True)
class ChallengeRegion:
    """A traversable circular hazard or finite cable segment in odom."""

    name: str
    kind: str
    x: float
    y: float
    radius: float
    half_length: float = 0.0
    angle: float = 0.0

    def __post_init__(self):
        values = (self.x, self.y, self.radius, self.half_length, self.angle)
        if (
            not self.name
            or self.kind not in ("pothole", "bump", "cable", "groove")
            or not np.all(np.isfinite(values))
            or self.radius <= 0.0
            or self.half_length < 0.0
        ):
            raise ValueError("Invalid traversable challenge region")

    def _closest_point(self, x: float, y: float) -> tuple[float, float]:
        if self.half_length == 0.0:
            return self.x, self.y
        direction_x, direction_y = cos(self.angle), sin(self.angle)
        along = np.clip(
            (x - self.x) * direction_x + (y - self.y) * direction_y,
            -self.half_length,
            self.half_length,
        )
        return (
            self.x + float(along) * direction_x,
            self.y + float(along) * direction_y,
        )

    def touches(self, x: float, y: float, margin: float) -> bool:
        closest_x, closest_y = self._closest_point(x, y)
        return bool(
            np.hypot(x - closest_x, y - closest_y)
            <= self.radius + margin
        )

    def clearance_x(self, y: float, margin: float) -> float:
        """Return the forward edge local to the robot's current lateral line."""
        crossing_x = self.x
        if self.half_length > 0.0:
            direction_x, direction_y = cos(self.angle), sin(self.angle)
            if abs(direction_y) > 1e-6:
                along = np.clip(
                    (y - self.y) / direction_y,
                    -self.half_length,
                    self.half_length,
                )
                crossing_x += float(along) * direction_x
            else:
                crossing_x += self.half_length
        return crossing_x + self.radius + margin


class ChallengeTracker:
    """One-shot difficult-path flags that cannot be farmed by oscillating."""

    def __init__(self, regions=(), contact_margin=0.20, clearance_margin=0.30,
                 wheel_separation=None, wheel_width=0.047):
        self.regions = tuple(regions)
        if (
            len({region.name for region in self.regions}) != len(self.regions)
            or not np.isfinite(contact_margin)
            or not np.isfinite(clearance_margin)
            or contact_margin < 0.0
            or clearance_margin < contact_margin
        ):
            raise ValueError("Invalid challenge tracker geometry")
        self.contact_margin = float(contact_margin)
        self.clearance_margin = float(clearance_margin)
        if wheel_separation is not None and (
            not np.isfinite(wheel_separation) or wheel_separation <= 0
            or not np.isfinite(wheel_width) or wheel_width <= 0
        ):
            raise ValueError("Wheel challenge tracking requires positive dimensions")
        self.wheel_separation = wheel_separation
        self.wheel_width = wheel_width
        self.chosen: set[str] = set()
        self.cleared: set[str] = set()

    @property
    def total(self) -> int:
        return len(self.regions)

    def update(self, previous_xy, current_xy, previous_yaw=0., current_yaw=0.) -> tuple[int, int]:
        previous = np.asarray(previous_xy, dtype=float)
        current = np.asarray(current_xy, dtype=float)
        if (
            previous.shape != (2,)
            or current.shape != (2,)
            or not np.all(np.isfinite(previous))
            or not np.all(np.isfinite(current))
        ):
            raise ValueError("Challenge tracking needs two finite XY poses")
        midpoint = 0.5 * (previous + current)
        points = (previous, midpoint, current)
        margin = self.contact_margin
        if self.wheel_separation is not None:
            if not np.isfinite([previous_yaw, current_yaw]).all():
                raise ValueError("Wheel tracking requires finite headings")
            def wheels(center, yaw):
                offset = .5 * self.wheel_separation * np.array([-sin(yaw), cos(yaw)])
                return center - offset, center + offset
            before, after = wheels(previous, previous_yaw), wheels(current, current_yaw)
            points = (*before, *(0.5 * (a+b) for a, b in zip(before, after)), *after)
            margin = self.wheel_width / 2.
        newly_chosen = newly_cleared = 0
        for region in self.regions:
            if region.name not in self.chosen and any(
                region.touches(float(point[0]), float(point[1]), margin)
                for point in points
            ):
                self.chosen.add(region.name)
                newly_chosen += 1
            if (
                region.name in self.chosen
                and region.name not in self.cleared
                and current[0] > previous[0]
                and current[0]
                >= region.clearance_x(float(current[1]), self.clearance_margin)
            ):
                self.cleared.add(region.name)
                newly_cleared += 1
        return newly_chosen, newly_cleared


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
    # Preview is [distance/1m, left height/0.1m, right height/0.1m, valid].
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
                   reference=None, previous_state=None,
                   completion_fraction=0.0, elapsed=None,
                   target_finish_seconds=None, challenge_entry_count=0,
                   challenge_clear_count=0, challenge_cleared_total=0,
                   challenge_total=0):
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
    challenge_counts = (
        challenge_entry_count,
        challenge_clear_count,
        challenge_cleared_total,
        challenge_total,
    )
    if (
        any(int(value) != value or value < 0 for value in challenge_counts)
        or challenge_entry_count > challenge_total
        or challenge_clear_count > challenge_total
        or challenge_cleared_total > challenge_total
    ):
        raise ValueError("Invalid traversable challenge counts")
    h = dt / 0.1
    cap = lambda x: min(float(x) ** 2, 9.0)
    progress_metric = cfg.get("progress_metric", "path")
    if progress_metric not in ("endpoint", "path"):
        raise ValueError("progress_metric must be endpoint or path")
    if progress_metric == "endpoint":
        delta = previous.endpoint_distance - current.endpoint_distance
    else:
        delta = previous.distance_remaining - current.distance_remaining
    # Credit forward motion fully only when it follows the straight
    # centerline. Reverse motion retains its full penalty so this gate cannot
    # be exploited by driving back while misaligned.
    progress_quality = exp(-(
        current.lateral_error / cfg.get("progress_lateral_sigma_m", 0.25)
    ) ** 2) * exp(-(
        current.heading_error / cfg.get("progress_heading_sigma_rad", 0.35)
    ) ** 2)
    credited_progress = float(delta) * (
        progress_quality if delta > 0.0 else 1.0
    )
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
    challenge_denominator = max(1, int(challenge_total))
    challenge_step_allowed = not failed and not timed_out
    terms = {
        # No asymmetric clipping across a closed forward/backward path: raw
        # signed differences telescope for a fixed path (undiscounted).
        "progress": cfg["progress_weight"] * credited_progress,
        "lateral": -h * cfg["lateral_weight"] * cap(
            current.lateral_error / cfg.get("lateral_sigma_m", 0.30)
        ),
        "heading": -h * cfg["heading_weight"] * cap(
            current.heading_error / cfg.get("heading_sigma_rad", 0.35)
        ),
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
        # Each region pays at most once through ChallengeTracker. Normalizing
        # by the episode count keeps the maximum shaping return fixed as the
        # adaptive curriculum adds hazards.
        "challenge_entry": (
            cfg.get("challenge_entry_bonus", 0.0)
            * int(challenge_entry_count)
            / challenge_denominator
            if challenge_step_allowed else 0.0
        ),
        "challenge_clear": (
            cfg.get("challenge_clear_bonus", 0.0)
            * int(challenge_clear_count)
            / challenge_denominator
            if challenge_step_allowed else 0.0
        ),
        "challenge_goal": 0.0,
        "success_position": 0.0,
        "success_heading": 0.0,
        "on_time_success": 0.0,
        "terminal": 0.0,
    }
    # Failure has precedence, including at a goal or time limit.
    if failed:
        terms["terminal"] = -cfg["off_path_penalty"] if failed == "off_path" else -cfg["failure_penalty"]
    elif succeeded:
        terms["terminal"] = cfg["success_bonus"]
        terms["challenge_goal"] = (
            cfg.get("challenge_goal_bonus", 0.0)
            * int(challenge_cleared_total)
            / challenge_denominator
        )
        terms["success_position"] = -cfg.get(
            "success_position_penalty", 0.0
        ) * kernel_cost(
            current.endpoint_distance,
            cfg.get("success_position_sigma_m", 0.25),
        )
        terms["success_heading"] = -cfg.get(
            "success_heading_penalty", 0.0
        ) * kernel_cost(
            current.heading_error,
            cfg.get("success_heading_sigma_rad", 0.21),
        )
        if elapsed is not None and target_finish_seconds is not None:
            target = float(target_finish_seconds)
            if not np.isfinite(target) or target <= 0.0:
                raise ValueError("target_finish_seconds must be positive and finite")
            time_margin = float(np.clip(
                (target - float(elapsed)) / target, 0.0, 1.0
            ))
            terms["on_time_success"] = (
                cfg.get("on_time_success_bonus", 0.0) * time_margin
            )
    elif timed_out:
        completion = float(np.clip(completion_fraction, 0.0, 1.0))
        if cfg.get("timeout_completion_scaling", False):
            minimum = float(cfg.get("timeout_minimum_fraction", 0.0))
            if not 0.0 <= minimum <= 1.0:
                raise ValueError("timeout_minimum_fraction must be in [0, 1]")
            timeout_scale = minimum + (1.0 - minimum) * (1.0 - completion)
        else:
            timeout_scale = 1.0
        terms["terminal"] = -cfg["timeout_penalty"] * timeout_scale
    reward = float(sum(terms.values()))
    if not np.isfinite(reward):
        raise ValueError("Non-finite v2 reward")
    return reward, terms


def validate_model(model, history_size):
    if (tuple(model.observation_space.shape) != (history_size,)
            or tuple(model.action_space.shape) != (ACTION_SIZE,)):
        raise ValueError("Incompatible checkpoint: v2 requires stacked 60-value frames "
                         "and 3 actions. Train a NEW v2 model; do not resume a v1 checkpoint.")
