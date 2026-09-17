"""Bounded asynchronous sensor synchronization, independently testable."""
from time import monotonic, sleep


def wait_for_quiescent_timestamp(sample, timeout=2.0, quiet_time=0.05,
                                 *, now=monotonic, pause=sleep):
    """Return a timestamp only after its asynchronous callback queue drains."""
    if timeout <= 0 or quiet_time <= 0 or quiet_time >= timeout:
        raise ValueError("Invalid timestamp quiescence request")
    deadline = now() + timeout
    value = sample()
    quiet_since = now()
    while True:
        current = sample()
        if current < value:
            raise RuntimeError("Timestamp moved backwards while draining callbacks")
        if current > value:
            value = current
            quiet_since = now()
        if now() - quiet_since >= quiet_time:
            return value
        if now() >= deadline:
            raise RuntimeError(
                "Timestamp callback queue did not become quiescent after pause"
            )
        pause(0.002)


def wait_for_coverage(measure, start, end, timeout=0.5, min_coverage=0.8,
                      max_latest_lag=0.05, *, now=monotonic, pause=sleep):
    if (end <= start or timeout <= 0 or not 0 < min_coverage <= 1
            or max_latest_lag < 0):
        raise ValueError("Invalid IMU coverage request")
    deadline = now() + timeout
    while True:
        result = measure()
        # Timestamp subtraction at large simulation times can undershoot the
        # exact threshold by a few ulps (for example 0.07999999999998 vs
        # 0.08). This tolerance is numerical only; freshness remains strict.
        if (result["duration"] + 1e-9 >= min_coverage * (end - start)
                and result["latest_stamp"] >= end - max_latest_lag - 1e-9):
            return result
        if now() >= deadline:
            raise RuntimeError(
                f"Insufficient timestamped IMU coverage: {result['duration']:.4f}s "
                f"of {end-start:.4f}s, latest={result['latest_stamp']:.4f}, "
                f"end={end:.4f}, allowed_lag={max_latest_lag:.4f}s; "
                "inspect /imu/data and /clock")
        pause(0.002)
