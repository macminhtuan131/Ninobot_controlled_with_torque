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
from nino_rl.training_contract import training_contract, validate_resume


def arguments() -> argparse.Namespace:
    default_config = Path(get_package_share_directory("nino_rl")) / "config" / "ppo.yaml"
    parser = argparse.ArgumentParser(description="Train PPO for Nino wheel torques")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--output", type=Path, default=Path("rl_runs"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=25_000)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--phase", type=int, choices=range(1, 7),
                        help="Fixed hard-to-easy cable phase (1=hardest, 6=easiest)")
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
        from nino_rl.policies import policy_spec
    except ImportError as error:
        raise SystemExit(
            "Thiếu thư viện RL. Kích hoạt .venv và chạy: "
            "pip install -r src/nino_rl/requirements.txt"
        ) from error

    config = load_config(args.config)
    if args.phase is not None:
        config["curriculum"]["fixed_phase"] = args.phase
    # auto means CUDA for this GPU training workflow; never silently fall back.
    device = args.device or str(config.get("device", "cuda"))
    if device == "auto":
        device = "cuda"
    if device.startswith("cuda") and not th.cuda.is_available():
        raise SystemExit(
            "Cấu hình yêu cầu CUDA nhưng torch.cuda.is_available() = False. "
            "Chạy `ros2 run nino_rl check_cuda` để chẩn đoán."
        )
    if device.startswith("cuda"):
        try:
            # Fail before starting ROS/Gazebo if the installed wheel and NVIDIA
            # driver cannot actually execute a CUDA kernel.
            probe = th.ones((32, 32), device=device)
            _ = probe @ probe
            th.cuda.synchronize()
        except (RuntimeError, AssertionError) as error:
            raise SystemExit(f"CUDA was detected but a CUDA operation failed: {error}") from error
        print(
            f"CUDA ready: {th.cuda.get_device_name(th.cuda.current_device())}",
            flush=True,
        )

    from nino_rl.ros_env import NinoGazeboEnv
    from nino_rl.preflight import run_preflight

    print("Running mandatory 12-point straight-line RL preflight...", flush=True)
    try:
        preflight_results = run_preflight(config, args.preflight_timeout)
    except (RuntimeError, TimeoutError) as error:
        raise SystemExit(f"PREFLIGHT FAILED; training was not started: {error}") from error
    for result in preflight_results:
        print(f"PASS: {result}", flush=True)

    class TrainingMetricsCallback(BaseCallback):
        """Expose reward components and endpoint metrics in TensorBoard."""

        def __init__(self) -> None:
            super().__init__()
            self.reward_terms: dict[str, list[float]] = {}

        def _on_rollout_start(self) -> None:
            env.resume_callback_dispatch()
            self.reward_terms.clear()

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                for name, value in info.get("reward_terms", {}).items():
                    self.reward_terms.setdefault(name, []).append(float(value))
                metrics = info.get("episode_metrics")
                if metrics is not None:
                    # Preserve each outcome instead of only rollout averages;
                    # this also survives an interrupted/incomplete PPO rollout.
                    with (run_dir / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"training_step": self.num_timesteps,
                                                 **metrics}) + "\n")
                    for reason in ("timeout", "off_path", "collision", "rollover",
                                   "wrong_direction", "navigation_invalid", "goal_missed"):
                        self.logger.record_mean(
                            f"episode/{reason}_failure", float(metrics["termination"] == reason))
                    for name, value in metrics["reward_totals"].items():
                        self.logger.record_mean(f"episode_reward/{name}", float(value))
                    for name in (
                        "return",
                        "clock_elapsed_seconds",
                        "max_clock_error_seconds",
                        "max_motion_sensor_lag_seconds",
                        "curriculum_phase",
                        "terrain_height_scale",
                        "success",
                        "finished_within_target_time",
                        "time_seconds",
                        "endpoint_distance_m",
                        "heading_error_deg",
                        "final_abs_lateral_drift_m",
                        "final_speed_m_s",
                        "mean_abs_lateral_error_m",
                        "rms_path_deviation_m",
                        "max_path_deviation_m",
                        "path_rmse_m",
                        "path_p95_m",
                        "heading_rmse_deg",
                        "final_progress_fraction",
                        "max_tilt_deg",
                        "rms_wheel_slip",
                        "rms_wheel_torque_nm",
                        "mean_imu_angular_xy_rad_s",
                        "peak_vertical_acceleration_m_s2",
                        "rms_vertical_acceleration_m_s2",
                        "adaptive_terrain_features",
                        "next_adaptive_terrain_features",
                        "adaptive_terrain_rolling_success",
                        "adaptive_terrain_window_episodes",
                        "adaptive_terrain_level_advanced",
                        "difficult_path_chosen",
                        "challenges_chosen",
                        "challenges_cleared",
                        "traversable_challenges",
                        "challenge_choice_fraction",
                        "challenge_clear_fraction",
                        "mean_speed_scale",
                        "min_speed_scale",
                        "max_speed_scale",
                        "mean_ground_speed_m_s",
                    ):
                        self.logger.record_mean(f"episode/{name}", float(metrics[name]))
            return True

        def _on_rollout_end(self) -> None:
            for index, name in enumerate(("speed", "common_torque", "steering")):
                self.logger.record(f"policy/std_{name}", float(self.model.policy.log_std[index].detach().exp().cpu()))
            # Do not leave the policy driving during an arbitrarily long PPO update.
            env.ros.publish_control(0.0, 0.0, 0.0)
            # The lockstep environment is already paused between every action,
            # so optimizer wall time cannot consume episode simulation time.
            env.suspend_callback_dispatch()
            for name, values in self.reward_terms.items():
                if values:
                    self.logger.record(f"reward_terms/{name}", float(np.mean(values)))

    def sync_adaptive_terrain_state(model, environment) -> None:
        model.nino_adaptive_terrain_state = environment.adaptive_terrain_state()

    class AdaptiveCheckpointCallback(CheckpointCallback):
        """Keep rolling curriculum progress inside each normal PPO checkpoint."""

        def _on_step(self) -> bool:
            if self.n_calls % self.save_freq == 0:
                sync_adaptive_terrain_state(self.model, env)
            return super()._on_step()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = args.output.expanduser().resolve() / stamp
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    config["device"] = device
    with (run_dir / "ppo.yaml").open("w") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    import json
    import platform
    from importlib.metadata import version
    (run_dir / "run_metadata.json").write_text(json.dumps({
        "argv": sys.argv, "python": platform.python_version(), "device": device,
        "gpu": th.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "packages": {name: version(name) for name in ("torch", "stable-baselines3", "gymnasium", "numpy")},
    }, indent=2) + "\n")
    env = NinoGazeboEnv(config, total_training_steps=args.timesteps)
    try:
        if args.check_env:
            initial_adaptive_state = env.adaptive_terrain_state()
            check_env(env, warn=True)
            env.restore_adaptive_terrain_state(initial_adaptive_state)
            env.global_steps = 0  # API validation does not consume the curriculum.
        monitored = Monitor(env, filename=str(run_dir / "monitor.csv"))
        ppo = config["ppo"]
        if args.resume:
            model = PPO.load(args.resume, device=device)
            validate_model(model, env.history.size)
            validate_resume(model, config)
            env.restore_adaptive_terrain_state(
                getattr(model, "nino_adaptive_terrain_state", None)
            )
            model.set_env(monitored)
            model.tensorboard_log = str(tensorboard_dir)
            env.global_steps = int(model.num_timesteps)
            env.total_training_steps = int(model.num_timesteps) + args.timesteps
            reset_num_timesteps = False
        else:
            policy_class, policy_kwargs = policy_spec(config)
            model = PPO(
                policy_class,
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
                target_kl=float(ppo["target_kl"]),
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(tensorboard_dir),
                device=device,
                seed=int(config["seed"]),
                verbose=1,
            )
            model.nino_training_contract = training_contract(config)
            sync_adaptive_terrain_state(model, env)
            reset_num_timesteps = True

        checkpoint_callback = AdaptiveCheckpointCallback(
            save_freq=max(1, int(args.checkpoint_every)),
            save_path=str(checkpoint_dir),
            name_prefix="nino_ppo",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
        sync_adaptive_terrain_state(model, env)
        print(
            f"Training is active on {model.device}; results: {run_dir}",
            flush=True,
        )
        print(
            f"Collecting {model.n_steps} environment steps before each PPO "
            "update; episode-end lines are live rollout progress.",
            flush=True,
        )
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, TrainingMetricsCallback()],
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
        final_path = run_dir / "nino_ppo_final"
        sync_adaptive_terrain_state(model, env)
        model.save(final_path)
        print(f"Đã lưu policy: {final_path}.zip")
    except KeyboardInterrupt:
        if "model" in locals():
            interrupted = run_dir / "nino_ppo_interrupted"
            sync_adaptive_terrain_state(model, env)
            model.save(interrupted)
            print(f"Saved {interrupted}.zip; unfinished rollout is discarded on resume.")
        print("Training interrupted cleanly; robot stopped and simulator released.")
    except (RuntimeError, TimeoutError):
        if "model" in locals():
            interrupted = run_dir / "nino_ppo_interrupted"
            sync_adaptive_terrain_state(model, env)
            model.save(interrupted)
            print(f"Saved {interrupted}.zip; unfinished rollout is discarded on resume.")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
