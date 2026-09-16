"""Evaluate normal Nav2 actuation with the same task and metrics as RL."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import numpy as np

from nino_rl.core import load_config
from nino_rl.control_v2 import BASELINE_ACTION


def main() -> None:
    share = Path(get_package_share_directory("nino_rl"))
    parser = argparse.ArgumentParser(description="Evaluate the normal Nav2 baseline")
    parser.add_argument("--config", type=Path, default=share / "config" / "ppo.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--phase", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--randomized", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("rl_runs/baseline"))
    args = parser.parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")

    from nino_rl.ros_env import NinoGazeboEnv

    config = load_config(args.config)
    config["curriculum"]["fixed_phase"] = args.phase
    config["domain_randomization"]["enabled"] = args.randomized
    env = NinoGazeboEnv(config, total_training_steps=1)
    rows = []
    try:
        for episode in range(args.episodes):
            _, _ = env.reset(seed=args.seed + episode)
            done = False
            info = {}
            while not done:
                _, _, terminated, truncated, info = env.step(
                    BASELINE_ACTION.copy()
                )
                done = terminated or truncated
            row = dict(info["episode_metrics"])
            row["episode"] = episode + 1
            row["controller"] = "nav2_baseline"
            rows.append(row)
    finally:
        env.close()

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = output / f"baseline-{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "seed": args.seed,
        "max_vertical_acceleration_m_s2": float(np.max([
            row["peak_vertical_acceleration_m_s2"] for row in rows])),
        "mean_rms_vertical_acceleration_m_s2": float(np.mean([
            row["rms_vertical_acceleration_m_s2"] for row in rows])),
        "controller": "nav2_baseline",
        "phase": args.phase,
        "episodes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_completion_time_seconds": float(
            np.mean([row["time_seconds"] for row in rows])
        ),
        "mean_rms_path_deviation_m": float(
            np.mean([row["rms_path_deviation_m"] for row in rows])
        ),
        "max_path_deviation_m": float(
            np.max([row["max_path_deviation_m"] for row in rows])
        ),
        "mean_rms_wheel_slip": float(
            np.mean([row["rms_wheel_slip"] for row in rows])
        ),
    }
    json_path = output / f"baseline-summary-{stamp}.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved {csv_path}\nSaved {json_path}")


if __name__ == "__main__":
    main()
