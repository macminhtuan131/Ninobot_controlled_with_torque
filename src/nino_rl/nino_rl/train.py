"""Train a PPO wheel-torque policy against the running Gazebo world."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import yaml

from ament_index_python.packages import get_package_share_directory
import numpy as np

from nino_rl.core import load_config
from nino_rl.control_v2 import validate_model


def arguments() -> argparse.Namespace:
    default_config = Path(get_package_share_directory("nino_rl")) / "config" / "ppo.yaml"
    parser = argparse.ArgumentParser(description="Train PPO for Nino wheel torques")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--output", type=Path, default=Path("rl_runs"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=25_000)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--phase", type=int, choices=range(1, 7),
                        help="Fixed terrain phase; advance after held-out evaluation")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--preflight-timeout", type=float, default=30.0)
    return parser.parse_args(sys.argv[1:])


def main() -> None:
    args = arguments()
    if args.timesteps <= 0:
        raise SystemExit("--timesteps phải lớn hơn 0")
    try:
        import torch as th
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
    except ImportError as error:
        raise SystemExit(
            "Thiếu thư viện RL. Kích hoạt .venv và chạy: "
            "pip install -r src/nino_rl/requirements.txt"
        ) from error

    config = load_config(args.config)
    if args.phase is not None:
        config["curriculum"]["fixed_phase"] = args.phase
    device = args.device or str(config.get("device", "cpu"))
    if device.startswith("cuda") and not th.cuda.is_available():
        raise SystemExit(
            "Cấu hình yêu cầu CUDA nhưng torch.cuda.is_available() = False. "
            "Chạy `ros2 run nino_rl check_cuda` để chẩn đoán."
        )

    from nino_rl.ros_env import NinoGazeboEnv
    from nino_rl.preflight import run_preflight

    print("Running mandatory 12-point Nav2/RL preflight...")
    try:
        preflight_results = run_preflight(config, args.preflight_timeout)
    except (RuntimeError, TimeoutError) as error:
        raise SystemExit(f"PREFLIGHT FAILED; training was not started: {error}") from error
    for result in preflight_results:
        print(f"PASS: {result}")

    class TrainingMetricsCallback(BaseCallback):
        """Expose reward components and endpoint metrics in TensorBoard."""

        def __init__(self) -> None:
            super().__init__()
            self.reward_terms: dict[str, list[float]] = {}

        def _on_rollout_start(self) -> None:
            self.reward_terms.clear()
            if env.world_paused_for_update:
                env.ros.set_world_paused(False)
                env.world_paused_for_update = False

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                for name, value in info.get("reward_terms", {}).items():
                    self.reward_terms.setdefault(name, []).append(float(value))
                metrics = info.get("episode_metrics")
                if metrics is not None:
                    self.logger.record_mean(
                        "episode/wrong_direction_failure",
                        float(metrics["termination"] == "wrong_direction"),
                    )
                    for name in (
                        "success",
                        "finished_within_target_time",
                        "time_seconds",
                        "endpoint_distance_m",
                        "mean_abs_lateral_error_m",
                        "rms_path_deviation_m",
                        "max_path_deviation_m",
                        "max_tilt_deg",
                        "rms_wheel_slip",
                        "rms_wheel_torque_nm",
                        "mean_imu_angular_xy_rad_s",
                        "peak_vertical_acceleration_m_s2",
                        "rms_vertical_acceleration_m_s2",
                    ):
                        self.logger.record_mean(f"episode/{name}", float(metrics[name]))
            return True

        def _on_rollout_end(self) -> None:
            # Do not leave the policy driving during an arbitrarily long PPO update.
            env.ros.publish_control(0.0, 0.0, 0.0)
            # Optimizer wall time must not consume episode simulation time.
            env.ros.set_world_paused(True)
            env.world_paused_for_update = True
            for name, values in self.reward_terms.items():
                if values:
                    self.logger.record(f"reward_terms/{name}", float(np.mean(values)))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output.expanduser().resolve() / stamp
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    config["device"] = device
    with (run_dir / "ppo.yaml").open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    env = NinoGazeboEnv(config, total_training_steps=args.timesteps)
    try:
        if args.check_env:
            check_env(env, warn=True)
        monitored = Monitor(env, filename=str(run_dir / "monitor.csv"))
        ppo = config["ppo"]
        if args.resume:
            model = PPO.load(args.resume, device=device)
            validate_model(model, env.history.size)
            model.set_env(monitored)
            model.tensorboard_log = str(tensorboard_dir)
            env.global_steps = int(model.num_timesteps)
            env.total_training_steps = int(model.num_timesteps) + args.timesteps
            reset_num_timesteps = False
        else:
            activation = {"elu": th.nn.ELU, "relu": th.nn.ReLU, "tanh": th.nn.Tanh}[
                str(ppo["activation"]).lower()
            ]
            layers = [int(value) for value in ppo["policy_layers"]]
            model = PPO(
                "MlpPolicy",
                monitored,
                learning_rate=float(ppo["learning_rate"]),
                gamma=float(ppo["gamma"]),
                gae_lambda=float(ppo["gae_lambda"]),
                n_steps=int(ppo["n_steps"]),
                batch_size=int(ppo["batch_size"]),
                n_epochs=int(ppo["n_epochs"]),
                clip_range=float(ppo["clip_range"]),
                ent_coef=float(ppo["ent_coef"]),
                vf_coef=float(ppo["vf_coef"]),
                max_grad_norm=float(ppo["max_grad_norm"]),
                policy_kwargs={
                    "activation_fn": activation,
                    "net_arch": {"pi": layers, "vf": layers},
                },
                tensorboard_log=str(tensorboard_dir),
                device=device,
                seed=int(config["seed"]),
                verbose=1,
            )
            reset_num_timesteps = True

        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, int(args.checkpoint_every)),
            save_path=str(checkpoint_dir),
            name_prefix="nino_ppo",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
        print(f"Bắt đầu train trên {model.device}; kết quả: {run_dir}")
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, TrainingMetricsCallback()],
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
        final_path = run_dir / "nino_ppo_final"
        model.save(final_path)
        print(f"Đã lưu policy: {final_path}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
