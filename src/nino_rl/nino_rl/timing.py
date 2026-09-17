"""Bounded asynchronous sensor synchronization, independently testable."""
from time import monotonic, sleep


def wait_for_coverage(measure, start, end, timeout=0.5, min_coverage=0.8,
                      *, now=monotonic, pause=sleep):
    if end <= start or timeout <= 0 or not 0 < min_coverage <= 1:
        raise ValueError("Invalid IMU coverage request")
    deadline = now() + timeout
    while True:
        result = measure()
        if (result["duration"] >= min_coverage * (end - start)
                and result["latest_stamp"] >= end):
            return result
        if now() >= deadline:
            raise RuntimeError(
                f"Insufficient timestamped IMU coverage: {result['duration']:.4f}s "
                f"of {end-start:.4f}s, latest={result['latest_stamp']:.4f}, "
                f"end={end:.4f}; inspect /imu/data and /clock")
        pause(0.002)
