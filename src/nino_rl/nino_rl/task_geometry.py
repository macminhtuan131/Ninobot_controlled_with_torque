"""Goal approach geometry shared by training and deployment."""
import numpy as np


def approach_speed(endpoint_distance, path_remaining, heading_error, config):
    """Keep steering authority near arrival; never accelerate after overshoot."""
    nav = config["navigation"]
    tolerance = float(config["goal_tolerance_m"])
    cruise = float(nav["straight_speed_m_s"])
    minimum = float(nav.get("minimum_approach_speed_m_s", .03))
    slowdown = max(float(nav["goal_slowdown_distance_m"]), tolerance)
    if endpoint_distance <= tolerance and abs(heading_error) <= np.deg2rad(
        config["goal_heading_tolerance_deg"]
    ):
        return 0.0
    # A lateral miss must not make the cruise command grow again. Keep a
    # small forward command so residual steering is still enabled by the PI
    # controller until success or the overshoot failure boundary.
    remaining = min(float(endpoint_distance), max(0., float(path_remaining)))
    return float(max(minimum, cruise * np.clip((remaining - tolerance) / slowdown, 0., 1.)))


def goal_overshot(state, path, config):
    """A forward-only mission cannot recover an arbitrarily missed goal."""
    tangent = path.delta[-1] / path.lengths[-1]
    beyond = float(np.dot(np.array([state.x, state.y]) - path.points[-1], tangent))
    return beyond > float(config.get("goal_overshoot_limit_m", .30))

