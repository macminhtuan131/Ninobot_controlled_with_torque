"""Frame-explicit path and timed-trajectory metrics, independent of ROS/SB3."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from nino_rl.core import PathTracker


def time_weights(times):
    """Trapezoidal integration weights; irregular samples cannot dominate RMSE."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)):
        raise ValueError("Need at least two finite timestamps")
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("Timestamps must be strictly increasing; split resets into episodes")
    weights = np.concatenate(([dt[0] / 2], (dt[:-1] + dt[1:]) / 2, [dt[-1] / 2]))
    return weights / (times[-1] - times[0])


def weighted_percentile(values, weights, quantile=0.95):
    order = np.argsort(values)
    index = np.searchsorted(np.cumsum(weights[order]), quantile, side="left")
    return float(np.asarray(values)[order[min(index, len(order) - 1)]])


def _finite_xy(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("Positions must be finite N x 2 [x_m, y_m]")
    return points


def path_metrics(times, actual_xy, reference_xy, yaw=None):
    """Distance to polyline (including endpoints), cross-track, heading and progress.

    These are geometry metrics, not time-aligned position RMSE or localization
    accuracy. Nearest-segment progress is ambiguous at self-intersections.
    """
    actual = _finite_xy(actual_xy)
    weights = time_weights(times)
    if len(actual) != len(weights):
        raise ValueError("Timestamp/position lengths differ")
    path = PathTracker(_finite_xy(reference_xy))
    projection = [path.project(*xy) for xy in actual]
    progress = np.asarray([p[0] for p in projection])
    headings = np.asarray([p[2] for p in projection])
    errors = actual - np.asarray([p[3] for p in projection])
    distance = np.linalg.norm(errors, axis=1)
    # Signed perpendicular component differs from distance beyond endpoints.
    cross = -np.sin(headings) * errors[:, 0] + np.cos(headings) * errors[:, 1]
    result = {
        "duration_s": float(times[-1] - times[0]),
        "samples": len(actual),
        "path_rmse_m": float(np.sqrt(weights @ distance**2)),
        "cross_track_rmse_m": float(np.sqrt(weights @ cross**2)),
        "path_mae_m": float(weights @ distance),
        "path_p95_m": weighted_percentile(distance, weights),
        "path_max_m": float(distance.max()),
        "endpoint_error_m": float(np.linalg.norm(actual[-1] - path.points[-1])),
        "final_progress_fraction": float(progress[-1] / path.total_length),
        "max_progress_fraction": float(progress.max() / path.total_length),
        "backtracking_m": float(np.maximum(-np.diff(progress), 0).sum()),
        "reference_length_m": path.total_length,
        "travelled_distance_m": float(np.linalg.norm(np.diff(actual, axis=0), axis=1).sum()),
    }
    if yaw is not None:
        yaw = np.asarray(yaw, dtype=float)
        if yaw.shape != headings.shape or not np.all(np.isfinite(yaw)):
            raise ValueError("Yaw must be finite and match positions")
        angle = (yaw - headings + np.pi) % (2 * np.pi) - np.pi
        result["heading_rmse_deg"] = float(np.degrees(np.sqrt(weights @ angle**2)))
    return result


def timed_metrics(actual_t, actual_xy, reference_t, reference_xy, actual_yaw=None,
                  reference_yaw=None, max_gap_s=0.5):
    """Interpolate both tracks on their common timestamps; never align or extrapolate.

    Position MSE is integrated exactly for the piecewise-linear difference.
    Clock origin and frame alignment are the caller's responsibility.
    """
    a_t, r_t = np.asarray(actual_t, float), np.asarray(reference_t, float)
    time_weights(a_t)
    time_weights(r_t)
    a_xy, r_xy = _finite_xy(actual_xy), _finite_xy(reference_xy)
    if len(a_t) != len(a_xy) or len(r_t) != len(r_xy):
        raise ValueError("Timestamp/position lengths differ")
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive")
    if max(np.diff(a_t).max(), np.diff(r_t).max()) > max_gap_s:
        raise ValueError("Timestamp gap exceeds --max-gap; do not interpolate missing motion")
    lo, hi = max(a_t[0], r_t[0]), min(a_t[-1], r_t[-1])
    if hi <= lo:
        raise ValueError("Trajectories have no nonzero time overlap")
    grid = np.unique(np.r_[lo, a_t[(a_t > lo) & (a_t < hi)],
                          r_t[(r_t > lo) & (r_t < hi)], hi])
    interp = lambda t, xy: np.column_stack([np.interp(grid, t, xy[:, j]) for j in range(2)])
    error = interp(a_t, a_xy) - interp(r_t, r_xy)
    dt = np.diff(grid)
    integral = np.sum(dt[:, None] * (error[:-1]**2 + error[:-1]*error[1:] + error[1:]**2) / 3, axis=0)
    mse = integral / (hi - lo)
    result = {
        "position_rmse_m": float(np.sqrt(mse.sum())),
        "x_rmse_m": float(np.sqrt(mse[0])), "y_rmse_m": float(np.sqrt(mse[1])),
        "time_overlap_s": float(hi - lo),
        "reference_time_coverage": float((hi - lo) / (r_t[-1] - r_t[0])),
        "actual_time_coverage": float((hi - lo) / (a_t[-1] - a_t[0])),
    }
    if actual_yaw is not None and reference_yaw is not None:
        a_yaw, r_yaw = np.asarray(actual_yaw, float), np.asarray(reference_yaw, float)
        if (a_yaw.shape != a_t.shape or r_yaw.shape != r_t.shape
                or not np.all(np.isfinite(a_yaw)) or not np.all(np.isfinite(r_yaw))):
            raise ValueError("Invalid timestamped yaw")
        e = np.interp(grid, a_t, np.unwrap(a_yaw)) - np.interp(grid, r_t, np.unwrap(r_yaw))
        e = (e + np.pi) % (2 * np.pi) - np.pi
        result["timed_heading_rmse_deg"] = float(np.degrees(np.sqrt(time_weights(grid) @ e**2)))
    return result


def read_track(filename, require_time=True):
    with Path(filename).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    required = {"x_m", "y_m", "frame_id"} | ({"time_s"} if require_time else set())
    if not rows or not required <= fields:
        raise ValueError(f"{filename}: need nonempty CSV columns {sorted(required)}")
    frames = {r["frame_id"].strip() for r in rows}
    if len(frames) != 1 or not next(iter(frames)):
        raise ValueError("Each file must use one nonempty frame_id")
    data = {"frame": next(iter(frames)),
            "xy": _finite_xy([[float(r["x_m"]), float(r["y_m"])] for r in rows])}
    for column, key in (("time_s", "time"), ("yaw_rad", "yaw")):
        if column in fields:
            values = np.asarray([float(r[column]) for r in rows])
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite {column}")
            data[key] = values
    if "time" in data:
        time_weights(data["time"])
    return data


def write_csv(filename, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write empty trajectory")
    with Path(filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class EpisodeTrajectory:
    """Collect estimated poses in the reference frame; never leak truth to actor."""
    def __init__(self, reference_xy, frame_id, clock_origin_sim_s=0.0):
        self.reference = _finite_xy(reference_xy)
        PathTracker(self.reference)
        if not frame_id or not np.isfinite(clock_origin_sim_s):
            raise ValueError("Need a frame and finite clock origin")
        self.frame_id = frame_id
        self.clock_origin_sim_s = float(clock_origin_sim_s)
        self.rows = []

    def add(self, time_s, x, y, yaw):
        if not np.all(np.isfinite([time_s, x, y, yaw])):
            raise ValueError("Invalid trajectory sample")
        if self.rows and time_s <= self.rows[-1]["time_s"]:
            raise ValueError("Trajectory clock did not advance")
        self.rows.append(dict(time_s=float(time_s), x_m=float(x), y_m=float(y),
                              yaw_rad=float(yaw), frame_id=self.frame_id))

    def metrics(self):
        return {"clock_origin_sim_s": self.clock_origin_sim_s,
                **path_metrics([r["time_s"] for r in self.rows],
                               [[r["x_m"], r["y_m"]] for r in self.rows], self.reference,
                               [r["yaw_rad"] for r in self.rows])}

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_csv(directory / "actual.csv", self.rows)
        write_csv(directory / "reference.csv", [dict(x_m=float(x), y_m=float(y),
                  frame_id=self.frame_id) for x, y in self.reference])
        (directory / "metrics.json").write_text(json.dumps(self.metrics(), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--mode", choices=("path", "timed"), default="path")
    parser.add_argument("--max-gap", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("trajectory_metrics.json"))
    parser.add_argument("--plot", type=Path, help="Optional XY overlay PNG/PDF/SVG")
    args = parser.parse_args()
    try:
        actual = read_track(args.actual)
        reference = read_track(args.reference, require_time=args.mode == "timed")
        if actual["frame"] != reference["frame"]:
            raise ValueError("frame_id mismatch; transform into one frame before comparison")
        if args.mode == "timed":
            result = timed_metrics(actual["time"], actual["xy"], reference["time"],
                reference["xy"], actual.get("yaw"), reference.get("yaw"), args.max_gap)
            if np.any(np.linalg.norm(np.diff(reference["xy"], axis=0), axis=1) > 1e-6):
                result.update(path_metrics(actual["time"], actual["xy"], reference["xy"], actual.get("yaw")))
        else:
            result = path_metrics(actual["time"], actual["xy"], reference["xy"], actual.get("yaw"))
        result.update(frame_id=actual["frame"], mode=args.mode,
                      actual=str(args.actual), reference=str(args.reference))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(*reference["xy"].T, "--", label="Reference", color="#1d4ed8")
            ax.plot(*actual["xy"].T, label="Actual", color="#dc2626")
            ax.set(xlabel="x [m]", ylabel="y [m]", title=f"Trajectory in {actual['frame']}")
            ax.axis("equal")
            ax.grid(alpha=.25)
            ax.legend()
            fig.tight_layout()
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.plot, dpi=160)
            plt.close(fig)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
