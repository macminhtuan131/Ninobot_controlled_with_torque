"""Shared baseline/PPO evaluation and provenance-checked report comparison."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from nino_rl.core import load_config
from nino_rl.control_v2 import BASELINE_ACTION, validate_model
from nino_rl.trajectory_metrics import write_csv

METRICS = ("path_rmse_m", "cross_track_rmse_m", "path_p95_m", "path_max_m",
           "heading_rmse_deg", "endpoint_error_m", "final_progress_fraction",
           "backtracking_m", "time_seconds", "rms_vertical_acceleration_m_s2",
           "peak_vertical_acceleration_m_s2", "rms_wheel_slip", "rms_wheel_torque_nm")


def benchmark_id(config):
    # Reward/PPO differences are allowed; geometry, sensing, task and test
    # perturbations must match. This is not a hash of external Gazebo binaries.
    ignored = {"reward", "reward_v2", "ppo", "device", "seed", "evaluation_baseline"}
    task = {k: v for k, v in config.items() if k not in ignored}
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()


def summarize(rows, metadata):
    successes = [r for r in rows if r["success"]]
    summary = {
        **metadata, "episodes": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "on_time_success_rate": float(np.mean([r["finished_within_target_time"] for r in rows])),
        "termination_counts": {reason: sum(r["termination"] == reason for r in rows)
                               for reason in sorted({r["termination"] for r in rows})},
        "metrics_all_episodes": {}, "metrics_successful_episodes": {},
    }
    for name in METRICS:
        values = np.asarray([r[name] for r in rows], float)
        summary["metrics_all_episodes"][name] = {
            "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max()),
        }
        summary["metrics_successful_episodes"][name] = (
            float(np.mean([r[name] for r in successes])) if successes else None)
    return summary


def run(baseline=False):
    from ament_index_python.packages import get_package_share_directory
    parser = argparse.ArgumentParser(description="Evaluate PI baseline or deterministic PPO on matching seeds")
    parser.add_argument("--config", type=Path,
        default=Path(get_package_share_directory("nino_rl")) / "config/ppo.yaml")
    if not baseline:
        parser.add_argument("--model", required=True, type=Path)
        parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--phase", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--randomized", action="store_true",
                        help="Enable full-strength residual/sensor perturbations")
    parser.add_argument("--output", type=Path, default=Path("rl_runs/baseline" if baseline else "rl_runs/evaluation"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    config = deepcopy(load_config(args.config))
    config["curriculum"]["fixed_phase"] = args.phase
    config["domain_randomization"]["enabled"] = args.randomized
    config["domain_randomization"]["phase_scales"] = [1.0] * 6
    config["evaluation_baseline"] = baseline
    adaptive = config.get("adaptive_terrain", {})
    if adaptive.get("enabled", False):
        adaptive["progress_on_success"] = False
        adaptive["initial_features"] = int(adaptive.get("evaluation_features", 8))
    model = None
    if not baseline:
        import torch
        from stable_baselines3 import PPO
        device = "cuda" if args.device == "auto" else args.device
        if device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA unavailable; run ros2 run nino_rl check_cuda")
        model = PPO.load(args.model, device=device)
        validate_model(model, 60 * config["policy_v2"]["history_frames"])
    output = args.output.expanduser().resolve() / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    import yaml
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = {
        "schema_version": 1, "controller": "straight_pi_baseline" if baseline else "ppo",
        "model": str(args.model.resolve()) if not baseline else None,
        "phase": args.phase, "seed": args.seed, "randomized": args.randomized,
        "benchmark_id": benchmark_id(config),
        "pose_source": "wheel_odometry",
        "metric_weighting": "simulation_time_trapezoid",
        "reference": "fixed straight line from configured start_pose to goal_pose in odom",
        "per_episode_plot": "trajectory.png; raw samples are in trajectory.csv",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Create ROS only after validating the model/config.
    from nino_rl.ros_env import NinoGazeboEnv
    env = NinoGazeboEnv(config, total_training_steps=1)
    rows = []
    try:
        for episode in range(args.episodes):
            observation, reset_info = env.reset(seed=args.seed + episode)
            while True:
                action = BASELINE_ACTION.copy() if baseline else model.predict(observation, deterministic=True)[0]
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            row = dict(info["episode_metrics"])
            row.update(episode=episode + 1, seed=args.seed + episode,
                       controller=metadata["controller"], phase=args.phase,
                       cable_count=reset_info["cable_count"],
                       cable_diameter_m=reset_info["cable_diameter_m"],
                       cable_angle_deg=reset_info["cable_angle_deg"])
            rows.append(row)
            env.trajectory.save(output / f"episode-{episode+1:03d}")
            # Persist each completed episode, so a later transport failure does
            # not lose previous measurements. No success row for a broken run.
            write_csv(output / "episodes.csv", rows)
            print(
                f"{episode+1}: reached={row['goal_reached']}; "
                f"time={row['time_seconds']:.2f}s, "
                f"endpoint={row['endpoint_distance_m']:.3f}m, "
                f"lateral drift={row['final_abs_lateral_drift_m']:.3f}m, "
                f"path RMSE={row['path_rmse_m']:.3f}m"
            )
    finally:
        env.close()
        if rows:
            summary = summarize(rows, {**metadata, "complete": len(rows) == args.episodes,
                                       "requested_episodes": args.episodes})
            (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved evaluation: {output}")


def compare_summaries(baseline, candidate):
    for report in (baseline, candidate):
        if not report.get("complete"):
            raise ValueError("Do not compare incomplete evaluations")
    for key in ("schema_version", "benchmark_id", "phase", "seed", "episodes", "randomized", "pose_source"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"Evaluation mismatch: {key}; rerun with identical test settings")
    result = {"phase": baseline["phase"], "episodes": baseline["episodes"],
              "delta_convention": "candidate minus baseline; negative error/time is better",
              "success_rate": {"baseline": baseline["success_rate"], "candidate": candidate["success_rate"],
                               "delta": candidate["success_rate"] - baseline["success_rate"]},
              "metrics": {}}
    for name in METRICS:
        b = baseline["metrics_all_episodes"][name]["mean"]
        c = candidate["metrics_all_episodes"][name]["mean"]
        result["metrics"][name] = {"baseline": b, "candidate": c, "delta": c-b}
    result["interpretation"] = (
        "Compare success first, then errors and comfort. Early failure can lower RMSE/time. "
        "Seeds match terrain draws; ROS/Gazebo transport is still asynchronous. "
        "No statistical significance or physical ground-truth accuracy is claimed.")
    return result


def compare_main():
    parser = argparse.ArgumentParser(description="Compare complete, matching evaluation summaries")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("comparison.json"))
    args = parser.parse_args()
    try:
        result = compare_summaries(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
